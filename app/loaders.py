from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from config import (
    ANOVA_RANKING_PATH,
    ALL48_EXPERIMENT_PATH,
    BASELINE_SUMMARY_PATH,
    BENCHMARK_HOLDOUT_PATH,
    DATASET_PARQUET_PATH,
    DATASET_PATH,
    EDA_CATEGORICAL_BY_TARGET_PATH,
    EDA_CATEGORICAL_SUMMARY_PATH,
    EDA_NUMERIC_BY_TARGET_PATH,
    EDA_NUMERIC_SUMMARY_PATH,
    EDA_TARGET_CORRELATIONS_PATH,
    EDA_TARGET_SUMMARY_PATH,
    FEATURE_RANKING_PATH,
    FEATURE_SELECTION_SUMMARY_PATH,
    FEATURE_SELECTION_TOP30_PATH,
    LEAKAGE_RISK_PATH,
    MODEL_INVENTORY_PATH,
    OFFICIAL_MODEL_OPTIONS,
    REPRESENTATIVE_IMPORTANCE_ALIASES,
    SELECTKBEST_SUMMARY_PATH,
    TARGET_COL,
    TOP30_OFFICIAL_FEATURES,
    TOP10_RECALL_PATH,
    TOP30_EXPERIMENT_PATH,
    TOP30_FE_EXPERIMENT_PATH,
    TUNING_SUMMARY_PATH,
)
from feature_config import TOP30_FULL
from ui_helpers import (
    baseline_alias,
    feature_display_name,
    scenario_label,
    tuned_alias,
    tuned_story_name,
)


def _empty_df(columns: list[str] | None = None) -> pd.DataFrame:
    return pd.DataFrame(columns=columns or [])


def _safe_csv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return _empty_df(columns)
    return pd.read_csv(path)


def _safe_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _top30_features() -> list[str]:
    df = _safe_csv(FEATURE_SELECTION_TOP30_PATH)
    if not df.empty and "variable" in df.columns:
        features = df["variable"].dropna().astype(str).tolist()
        if features:
            return features
    return list(TOP30_OFFICIAL_FEATURES or TOP30_FULL)


def _representative_inventory() -> pd.DataFrame:
    inventory = load_model_inventory_safe()
    if inventory.empty:
        return pd.DataFrame()

    inventory = inventory.copy()
    if "feature_set" in inventory.columns:
        inventory = inventory.loc[inventory["feature_set"] == "top30_full"]

    selected_rows: list[pd.Series] = []
    used_index: set[int] = set()
    for family, aliases in REPRESENTATIVE_IMPORTANCE_ALIASES.items():
        for alias in aliases:
            matches = inventory.loc[inventory["alias"] == alias]
            if not matches.empty:
                row = matches.iloc[0]
                if row.name not in used_index:
                    selected_rows.append(row)
                    used_index.add(row.name)
                break

    if not selected_rows:
        return pd.DataFrame()
    return pd.DataFrame(selected_rows)


def _get_pipeline_parts(pipeline) -> tuple[Any | None, Any | None]:
    named_steps = getattr(pipeline, "named_steps", {})
    return named_steps.get("preprocessor"), named_steps.get("model")


def _map_transformed_feature(feature_name: str, categorical_features: list[str]) -> str:
    if feature_name.startswith("num__"):
        return feature_name.replace("num__", "", 1)

    if feature_name.startswith("cat__"):
        remainder = feature_name.replace("cat__", "", 1)
        for base_feature in sorted(categorical_features, key=len, reverse=True):
            if remainder == base_feature or remainder.startswith(f"{base_feature}_"):
                return base_feature
        return remainder

    return feature_name


def _extract_grouped_importances(pipeline) -> pd.Series:
    preprocessor, model = _get_pipeline_parts(pipeline)
    if preprocessor is None or model is None:
        return pd.Series(dtype=float)

    transformed_features = list(preprocessor.get_feature_names_out())
    categorical_features: list[str] = []
    for name, _, cols in getattr(preprocessor, "transformers_", []):
        if name == "cat":
            categorical_features = list(cols)
            break

    if hasattr(model, "coef_"):
        raw_importances = np.abs(np.asarray(model.coef_)).ravel()
    elif hasattr(model, "feature_importances_"):
        raw_importances = np.abs(np.asarray(model.feature_importances_)).ravel()
    else:
        return pd.Series(dtype=float)

    if len(transformed_features) != len(raw_importances):
        return pd.Series(dtype=float)

    grouped: dict[str, float] = {}
    for transformed_feature, importance in zip(transformed_features, raw_importances):
        base_feature = _map_transformed_feature(transformed_feature, categorical_features)
        grouped[base_feature] = grouped.get(base_feature, 0.0) + float(importance)

    series = pd.Series(grouped, dtype=float)
    max_value = float(series.max()) if not series.empty else 0.0
    if max_value > 0:
        series = series / max_value
    return series.sort_values(ascending=False)


def _predict_scores(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)
        if isinstance(scores, np.ndarray) and scores.ndim == 2:
            return scores[:, 1]
        return np.asarray(scores)

    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X))

    return np.asarray(model.predict(X))


def _family_display_name(model) -> str:
    family = type(model).__name__ if model is not None else "Modelo"
    mapping = {
        "LogisticRegression": "LogisticRegression",
        "RandomForestClassifier": "RandomForestClassifier",
        "DecisionTreeClassifier": "DecisionTreeClassifier",
        "XGBClassifier": "XGBoostClassifier",
        "XGBoostClassifier": "XGBoostClassifier",
        "LGBMClassifier": "LightGBMClassifier",
        "LightGBMClassifier": "LightGBMClassifier",
        "SVC": "SVC",
    }
    return mapping.get(family, family)


@st.cache_data
def load_dataset_safe() -> pd.DataFrame:
    if DATASET_PATH.exists():
        return pd.read_csv(DATASET_PATH)
    if DATASET_PARQUET_PATH.exists():
        return pd.read_parquet(DATASET_PARQUET_PATH)
    return _empty_df()


@st.cache_data
def load_baseline_safe() -> pd.DataFrame:
    df = _safe_csv(BASELINE_SUMMARY_PATH)
    if df.empty:
        return df

    df = _coerce_numeric(
        df,
        [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "precision_pobre",
            "recall_pobre",
            "f1_pobre",
            "support_pobre",
        ],
    )
    if "alias" not in df.columns and {"model", "feature_set"}.issubset(df.columns):
        df["alias"] = df.apply(baseline_alias, axis=1)
    if "escenario_label" not in df.columns and "feature_set" in df.columns:
        df["escenario_label"] = df["feature_set"].map(scenario_label)
    return df


@st.cache_data
def load_tuning_safe() -> pd.DataFrame:
    df = _safe_csv(TUNING_SUMMARY_PATH)
    if df.empty:
        return df

    df = _coerce_numeric(
        df,
        [
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "precision_cv",
            "recall_cv",
            "f1_cv",
            "roc_auc_cv",
            "pr_auc_cv",
            "tn",
            "fp",
            "fn",
            "tp",
            "train_rows",
            "test_rows",
        ],
    )
    if "alias" not in df.columns and {"modelo_familia", "escenario", "threshold"}.issubset(df.columns):
        df["alias"] = df.apply(tuned_alias, axis=1)
    if "escenario_label" not in df.columns and "escenario" in df.columns:
        df["escenario_label"] = df["escenario"].map(scenario_label)
    if "story_name" not in df.columns and "alias" in df.columns:
        df["story_name"] = df.apply(tuned_story_name, axis=1)
    return df


@st.cache_data
def load_top10_recall_safe() -> pd.DataFrame:
    df = _safe_csv(TOP10_RECALL_PATH)
    if df.empty:
        return df

    df = _coerce_numeric(
        df,
        ["threshold", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "tn", "fp", "fn", "tp"],
    )
    if "alias" not in df.columns and {"modelo_familia", "escenario", "threshold"}.issubset(df.columns):
        df["alias"] = df.apply(tuned_alias, axis=1)
    if "escenario_label" not in df.columns and "escenario" in df.columns:
        df["escenario_label"] = df["escenario"].map(scenario_label)
    return df


@st.cache_data
def load_target_summary_safe() -> pd.DataFrame:
    return _safe_csv(EDA_TARGET_SUMMARY_PATH)


@st.cache_data
def load_eda_numeric_summary_safe() -> pd.DataFrame:
    return _safe_csv(EDA_NUMERIC_SUMMARY_PATH)


@st.cache_data
def load_eda_categorical_summary_safe() -> pd.DataFrame:
    return _safe_csv(EDA_CATEGORICAL_SUMMARY_PATH)


@st.cache_data
def load_eda_numeric_by_target_safe() -> pd.DataFrame:
    return _safe_csv(EDA_NUMERIC_BY_TARGET_PATH)


@st.cache_data
def load_eda_categorical_by_target_safe() -> pd.DataFrame:
    return _safe_csv(EDA_CATEGORICAL_BY_TARGET_PATH)


@st.cache_data
def load_eda_target_correlations_safe() -> pd.DataFrame:
    return _safe_csv(EDA_TARGET_CORRELATIONS_PATH)


@st.cache_data
def load_feature_selection_top30_safe() -> pd.DataFrame:
    return _safe_csv(FEATURE_SELECTION_TOP30_PATH)


@st.cache_data
def load_feature_selection_summary_safe() -> dict[str, Any]:
    return _safe_json(FEATURE_SELECTION_SUMMARY_PATH, {})


@st.cache_data
def load_anova_ranking_safe() -> pd.DataFrame:
    df = _safe_csv(ANOVA_RANKING_PATH)
    if df.empty:
        return df
    return _coerce_numeric(
        df,
        [
            "expanded_features",
            "f_score_max",
            "f_score_mean",
            "p_value_min",
            "constant_expanded_features",
            "rank_f_score",
        ],
    )


@st.cache_data
def load_selectkbest_summary_safe() -> dict[str, Any]:
    return _safe_json(SELECTKBEST_SUMMARY_PATH, {})


@st.cache_data
def load_leakage_risk_safe() -> pd.DataFrame:
    return _safe_csv(LEAKAGE_RISK_PATH)


@st.cache_data
def load_top30_experiment_safe() -> pd.DataFrame:
    return _safe_csv(TOP30_EXPERIMENT_PATH)


@st.cache_data
def load_all48_experiment_safe() -> pd.DataFrame:
    return _safe_csv(ALL48_EXPERIMENT_PATH)


@st.cache_data
def load_top30_fe_experiment_safe() -> pd.DataFrame:
    return _safe_csv(TOP30_FE_EXPERIMENT_PATH)


@st.cache_data
def load_benchmark_holdout_safe() -> pd.DataFrame:
    df = _safe_csv(BENCHMARK_HOLDOUT_PATH)
    if df.empty:
        return df

    return _coerce_numeric(
        df,
        [
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "tn",
            "fp",
            "fn",
            "tp",
        ],
    )


@st.cache_data
def load_model_inventory_safe() -> pd.DataFrame:
    return _safe_csv(MODEL_INVENTORY_PATH)


@st.cache_data
def load_feature_ranking_safe() -> pd.DataFrame:
    return _safe_csv(FEATURE_RANKING_PATH)


@st.cache_data
def load_top10_feature_influence_safe() -> dict[str, Any]:
    inventory = _representative_inventory()
    if inventory.empty:
        return {"matrix": pd.DataFrame(), "long": pd.DataFrame(), "sources": pd.DataFrame()}

    columns_data: dict[str, pd.Series] = {}
    source_rows: list[dict[str, str]] = []

    for _, row in inventory.iterrows():
        path = Path(str(row["path"]))
        pipeline = load_model_safe(path)
        if pipeline is None:
            continue

        importance_series = _extract_grouped_importances(pipeline)
        if importance_series.empty:
            continue

        _, model = _get_pipeline_parts(pipeline)
        family = _family_display_name(model)
        columns_data[family] = importance_series
        source_rows.append(
            {
                "modelo": family,
                "alias_usado": str(row.get("alias", "")),
                "grupo": str(row.get("group", "")),
                "fuente_modelo": str(row.get("model_label", "")),
            }
        )

    if not columns_data:
        return {"matrix": pd.DataFrame(), "long": pd.DataFrame(), "sources": pd.DataFrame(source_rows)}

    matrix = pd.concat(columns_data, axis=1).fillna(0.0)
    matrix["promedio"] = matrix.mean(axis=1)
    top10 = matrix.sort_values("promedio", ascending=False).head(10)
    top10 = top10.reset_index().rename(columns={"index": "variable"})
    top10["variable_label"] = top10["variable"].map(feature_display_name)

    model_columns = [col for col in top10.columns if col not in {"variable", "variable_label", "promedio"}]
    long_df = top10.melt(
        id_vars=["variable", "variable_label"],
        value_vars=model_columns,
        var_name="modelo",
        value_name="influencia",
    )

    return {
        "matrix": top10[["variable", "variable_label"] + model_columns],
        "long": long_df,
        "sources": pd.DataFrame(source_rows),
    }


@st.cache_data
def load_roc_curve_data_safe() -> dict[str, Any]:
    dataset = load_dataset_safe()
    if dataset.empty or TARGET_COL not in dataset.columns:
        return {"curves": pd.DataFrame(), "summary": pd.DataFrame()}

    features = [feature for feature in _top30_features() if feature in dataset.columns]
    if not features:
        return {"curves": pd.DataFrame(), "summary": pd.DataFrame()}

    filtered = dataset.dropna(subset=[TARGET_COL]).copy()
    X = filtered[features]
    y = filtered[TARGET_COL]
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    curve_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for label, option in OFFICIAL_MODEL_OPTIONS.items():
        pipeline = load_model_safe(option["path"])
        if pipeline is None:
            continue

        scores = _predict_scores(pipeline, X_test)
        fpr, tpr, _ = roc_curve(y_test, scores)
        auc_value = roc_auc_score(y_test, scores)
        for fpr_value, tpr_value in zip(fpr, tpr):
            curve_rows.append(
                {
                    "modelo": label,
                    "story": option["story"],
                    "fpr": float(fpr_value),
                    "tpr": float(tpr_value),
                }
            )
        summary_rows.append(
            {
                "modelo": label,
                "story": option["story"],
                "auc": float(auc_value),
            }
        )

    return {
        "curves": pd.DataFrame(curve_rows),
        "summary": pd.DataFrame(summary_rows),
    }


@st.cache_resource
def load_model_safe(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)
