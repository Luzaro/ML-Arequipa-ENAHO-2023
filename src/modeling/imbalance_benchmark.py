from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.io_utils import save_json
from utils.logging_utils import configure_logging
from utils.paths import build_paths, ensure_directories
from modeling.baseline import LEAKAGE_VARS, RANDOM_STATE, TARGET_COL, resolve_top20_path


DATASET_CSV = "enaho_arequipa_escenario_b_clean.csv"
DATASET_PARQUET = "enaho_arequipa_escenario_b_clean.parquet"
OUT_DIRNAME = "scenario_b_modeling_imbalance"
TEST_SIZE = 0.3
CV_SPLITS = 5
THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
MIN_ACCEPTABLE_PRECISION = 0.25


@dataclass(frozen=True)
class ScenarioWinner:
    scenario_label: str
    objective: str
    feature_set: str
    strategy: str
    model_name: str
    threshold: float


def load_dataset(paths: Any) -> pd.DataFrame:
    csv_path = paths.data_processed / DATASET_CSV
    parquet_path = paths.data_processed / DATASET_PARQUET
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.read_parquet(parquet_path)


def load_top20(paths: Any) -> list[str]:
    top20_df = pd.read_csv(resolve_top20_path(paths))
    return top20_df["variable"].dropna().astype(str).tolist()


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    if {"num_habitaciones", "tam_hogar"}.issubset(work.columns):
        household = pd.to_numeric(work["tam_hogar"], errors="coerce")
        rooms = pd.to_numeric(work["num_habitaciones"], errors="coerce").replace({0: np.nan})
        work["personas_por_habitacion"] = household / rooms

    if {"num_ninos_0_5", "num_ninos_6_16", "tam_hogar"}.issubset(work.columns):
        children = (
            pd.to_numeric(work["num_ninos_0_5"], errors="coerce").fillna(0)
            + pd.to_numeric(work["num_ninos_6_16"], errors="coerce").fillna(0)
        )
        denom = pd.to_numeric(work["tam_hogar"], errors="coerce").replace({0: np.nan})
        work["proporcion_ninos_hogar"] = children / denom

    if {"num_miembros_sin_atencion_salud", "tam_hogar"}.issubset(work.columns):
        unattended = pd.to_numeric(work["num_miembros_sin_atencion_salud"], errors="coerce")
        denom = pd.to_numeric(work["tam_hogar"], errors="coerce").replace({0: np.nan})
        work["proporcion_sin_atencion_salud"] = unattended / denom

    transfer_cols = [
        "monto_juntos",
        "monto_bono_gas",
        "monto_pension_65",
        "monto_programa_contigo",
        "monto_otras_transferencias_publicas",
        "monto_bono_electricidad",
    ]
    available_transfer_cols = [col for col in transfer_cols if col in work.columns]
    if available_transfer_cols:
        transfer_frame = work[available_transfer_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        work["transferencias_publicas_total"] = transfer_frame.sum(axis=1)

    service_cols = [
        "acceso_agua_red",
        "acceso_desague_red",
        "agua_potable_reportada",
        "acceso_internet_hogar_derivado",
    ]
    available_service_cols = [col for col in service_cols if col in work.columns]
    if available_service_cols:
        service_frame = work[available_service_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        work["indice_servicios_basicos"] = service_frame.sum(axis=1)

    return work


def build_feature_sets(df: pd.DataFrame, top20_full: list[str]) -> dict[str, list[str]]:
    top30_full = [col for col in top20_full if col in df.columns]
    top30_sin_leakage = [col for col in top30_full if col not in LEAKAGE_VARS]

    engineered_candidates = [
        "personas_por_habitacion",
        "proporcion_ninos_hogar",
        "proporcion_sin_atencion_salud",
        "transferencias_publicas_total",
        "indice_servicios_basicos",
    ]
    engineered_available = [col for col in engineered_candidates if col in df.columns]

    return {
        "top30_full": top30_full,
        "top30_full_plus_fe": top30_full + engineered_available,
        "top30_sin_leakage": top30_sin_leakage,
        "top30_sin_leakage_plus_fe": top30_sin_leakage + engineered_available,
    }


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )


def positive_weight(y: pd.Series) -> float:
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0:
        return 1.0
    return negatives / positives


def build_models(pos_weight: float) -> dict[str, dict[str, Any]]:
    return {
        "Logistic": {
            "weighted": LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                solver="liblinear",
            ),
            "smote": LogisticRegression(
                max_iter=3000,
                random_state=RANDOM_STATE,
                solver="liblinear",
            ),
        },
        "Tree": {
            "weighted": DecisionTreeClassifier(
                max_depth=5,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "smote": DecisionTreeClassifier(
                max_depth=5,
                min_samples_leaf=10,
                random_state=RANDOM_STATE,
            ),
        },
        "RandomForest": {
            "weighted": RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "smote": RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
        },
        "XGBoost": {
            "weighted": XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                objective="binary:logistic",
                scale_pos_weight=pos_weight,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "smote": XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                objective="binary:logistic",
                scale_pos_weight=1.0,
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
        },
        "LightGBM": {
            "weighted": LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbose=-1,
            ),
            "smote": LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                class_weight=None,
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbose=-1,
            ),
        },
    }


def make_pipeline(X: pd.DataFrame, estimator: Any, strategy: str) -> Any:
    preprocessor = build_preprocessor(X)
    if strategy == "smote":
        return ImbPipeline(
            [
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("model", estimator),
            ]
        )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def normalize_scores(prob: np.ndarray) -> np.ndarray:
    minimum = float(np.nanmin(prob))
    maximum = float(np.nanmax(prob))
    return (prob - minimum) / (maximum - minimum + 1e-9)


def predict_scores(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        raw = estimator.decision_function(X)
        return normalize_scores(np.asarray(raw))
    raise ValueError("El estimador no soporta probabilidad ni decision_function.")


def evaluate_threshold(y_true: pd.Series, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
    }


def cross_validate_models(
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []

    for feature_set_name, features in feature_sets.items():
        X = df[features].copy()
        y = pd.to_numeric(df[TARGET_COL], errors="coerce").astype(int)
        cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

        for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            models = build_models(positive_weight(y_train))

            for model_name, strategies in models.items():
                for strategy_name, estimator in strategies.items():
                    pipeline = make_pipeline(X_train, clone(estimator), strategy_name)
                    pipeline.fit(X_train, y_train)
                    y_score = predict_scores(pipeline, X_valid)

                    for threshold in THRESHOLDS:
                        metrics = evaluate_threshold(y_valid, y_score, threshold)
                        record = {
                            "feature_set": feature_set_name,
                            "fold": fold,
                            "model": model_name,
                            "strategy": strategy_name,
                            "threshold": threshold,
                            **metrics,
                        }
                        rows.append(record)
                        fold_rows.append(record)

    detailed = pd.DataFrame(fold_rows)
    summary = (
        detailed.groupby(["feature_set", "model", "strategy", "threshold"], as_index=False)[
            ["recall", "precision", "f1", "roc_auc", "pr_auc"]
        ]
        .mean()
    )
    return summary, detailed


def select_winners(summary: pd.DataFrame) -> list[ScenarioWinner]:
    maxima_detection = (
        summary.loc[summary["precision"] >= MIN_ACCEPTABLE_PRECISION]
        .sort_values(["recall", "pr_auc", "f1"], ascending=False)
        .iloc[0]
    )
    best_balance = summary.sort_values(["f1", "pr_auc", "recall"], ascending=False).iloc[0]
    no_leakage = (
        summary.loc[summary["feature_set"].isin(["top30_sin_leakage", "top30_sin_leakage_plus_fe"])]
        .sort_values(["pr_auc", "f1", "recall"], ascending=False)
        .iloc[0]
    )

    return [
        ScenarioWinner(
            scenario_label="maxima_deteccion",
            objective="Maximizar recall con precision minima aceptable",
            feature_set=str(maxima_detection["feature_set"]),
            strategy=str(maxima_detection["strategy"]),
            model_name=str(maxima_detection["model"]),
            threshold=float(maxima_detection["threshold"]),
        ),
        ScenarioWinner(
            scenario_label="mejor_equilibrio",
            objective="Maximizar f1 manteniendo buen pr_auc",
            feature_set=str(best_balance["feature_set"]),
            strategy=str(best_balance["strategy"]),
            model_name=str(best_balance["model"]),
            threshold=float(best_balance["threshold"]),
        ),
        ScenarioWinner(
            scenario_label="sin_leakage",
            objective="Maximizar pr_auc/f1 usando variables sin leakage",
            feature_set=str(no_leakage["feature_set"]),
            strategy=str(no_leakage["strategy"]),
            model_name=str(no_leakage["model"]),
            threshold=float(no_leakage["threshold"]),
        ),
    ]


def fit_final_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    strategy: str,
) -> Any:
    models = build_models(positive_weight(y_train))
    estimator = clone(models[model_name][strategy])
    pipeline = make_pipeline(X_train, estimator, strategy)
    pipeline.fit(X_train, y_train)
    return pipeline


def error_profile(df_errors: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    if df_errors.empty:
        return {"n_rows": 0}

    numeric_cols = [col for col in feature_columns if col in df_errors.columns and pd.api.types.is_numeric_dtype(df_errors[col])]
    categorical_cols = [col for col in feature_columns if col in df_errors.columns and col not in numeric_cols]

    numeric_summary = (
        df_errors[numeric_cols]
        .mean(numeric_only=True)
        .round(4)
        .sort_values(ascending=False)
        .head(8)
        .to_dict()
        if numeric_cols
        else {}
    )
    categorical_summary = {}
    for col in categorical_cols[:8]:
        mode = df_errors[col].mode(dropna=True)
        categorical_summary[col] = None if mode.empty else mode.iloc[0]

    return {
        "n_rows": int(df_errors.shape[0]),
        "numeric_means_top": numeric_summary,
        "categorical_modes": categorical_summary,
    }


def evaluate_winner(
    df: pd.DataFrame,
    winner: ScenarioWinner,
    out_dir: Path,
) -> dict[str, Any]:
    feature_sets = build_feature_sets(df, load_top20(build_paths(Path(__file__).resolve().parents[2])))
    features = feature_sets[winner.feature_set]

    X = df[features].copy()
    y = pd.to_numeric(df[TARGET_COL], errors="coerce").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = fit_final_pipeline(X_train, y_train, winner.model_name, winner.strategy)
    y_score = predict_scores(pipeline, X_test)
    y_pred = (y_score >= winner.threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    test_frame = X_test.copy()
    test_frame["y_true"] = y_test.values
    test_frame["y_score"] = y_score
    test_frame["y_pred"] = y_pred

    fp_df = test_frame[(test_frame["y_true"] == 0) & (test_frame["y_pred"] == 1)].copy()
    fn_df = test_frame[(test_frame["y_true"] == 1) & (test_frame["y_pred"] == 0)].copy()
    fp_df.to_csv(out_dir / f"{winner.scenario_label}_false_positives.csv", index=False, encoding="utf-8-sig")
    fn_df.to_csv(out_dir / f"{winner.scenario_label}_false_negatives.csv", index=False, encoding="utf-8-sig")
    joblib.dump(pipeline, out_dir / f"{winner.scenario_label}_pipeline.joblib")

    return {
        "scenario_label": winner.scenario_label,
        "objective": winner.objective,
        "feature_set": winner.feature_set,
        "model": winner.model_name,
        "strategy": winner.strategy,
        "threshold": winner.threshold,
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "pr_auc": average_precision_score(y_test, y_score),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "false_positive_profile": error_profile(fp_df, features),
        "false_negative_profile": error_profile(fn_df, features),
    }


def build_markdown_report(
    output_path: Path,
    class_distribution: dict[str, Any],
    winners_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
) -> None:
    lines = [
        "# Benchmark de desbalance",
        "",
        "## Diagnostico",
        f"- Hogares totales: {class_distribution['n_rows']}",
        f"- Hogares pobres: {class_distribution['positives']}",
        f"- Tasa positiva: {class_distribution['positive_rate']:.4f}",
        "- Se compararon class_weight/scale_pos_weight y SMOTE dentro de validacion cruzada estratificada.",
        "- Las metricas principales fueron recall, precision, f1, roc_auc y pr_auc.",
        "",
        "## Ganadores por escenario (CV)",
    ]

    for _, row in winners_df.iterrows():
        lines.extend(
            [
                f"### {row['scenario_label']}",
                f"- Modelo: {row['model']}",
                f"- Estrategia: {row['strategy']}",
                f"- Feature set: {row['feature_set']}",
                f"- Threshold: {row['threshold']}",
                f"- Recall CV: {row['recall']:.4f}",
                f"- Precision CV: {row['precision']:.4f}",
                f"- F1 CV: {row['f1']:.4f}",
                f"- ROC AUC CV: {row['roc_auc']:.4f}",
                f"- PR AUC CV: {row['pr_auc']:.4f}",
                "",
            ]
        )

    lines.append("## Validacion holdout")
    for _, row in holdout_df.iterrows():
        lines.extend(
            [
                f"### {row['scenario_label']}",
                f"- Modelo final: {row['model']} ({row['strategy']})",
                f"- Recall: {row['recall']:.4f}",
                f"- Precision: {row['precision']:.4f}",
                f"- F1: {row['f1']:.4f}",
                f"- ROC AUC: {row['roc_auc']:.4f}",
                f"- PR AUC: {row['pr_auc']:.4f}",
                f"- Matriz de confusion: TN={row['tn']} | FP={row['fp']} | FN={row['fn']} | TP={row['tp']}",
                "",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    paths = build_paths(root)
    ensure_directories(paths)

    logger = configure_logging(paths.logs / f"imbalance_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    out_dir = paths.reports / OUT_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(paths)
    df = add_engineered_features(df)
    top20_full = load_top20(paths)
    feature_sets = build_feature_sets(df, top20_full)

    class_distribution = {
        "n_rows": int(df.shape[0]),
        "positives": int((df[TARGET_COL] == 1).sum()),
        "negatives": int((df[TARGET_COL] == 0).sum()),
        "positive_rate": float((df[TARGET_COL] == 1).mean()),
    }

    logger.info("Iniciando benchmark de desbalance con distribucion: %s", class_distribution)
    summary_df, detailed_df = cross_validate_models(df, feature_sets)
    detailed_df.to_csv(out_dir / "cv_detailed_results.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "cv_summary_results.csv", index=False, encoding="utf-8-sig")

    winners = select_winners(summary_df)
    winners_rows = []
    for winner in winners:
        row = summary_df[
            (summary_df["feature_set"] == winner.feature_set)
            & (summary_df["model"] == winner.model_name)
            & (summary_df["strategy"] == winner.strategy)
            & (summary_df["threshold"] == winner.threshold)
        ].iloc[0].to_dict()
        row["scenario_label"] = winner.scenario_label
        row["objective"] = winner.objective
        winners_rows.append(row)

    winners_df = pd.DataFrame(winners_rows)
    winners_df.to_csv(out_dir / "scenario_winners_cv.csv", index=False, encoding="utf-8-sig")

    holdout_rows = [evaluate_winner(df, winner, out_dir) for winner in winners]
    holdout_df = pd.DataFrame(holdout_rows)
    holdout_df.to_csv(out_dir / "scenario_winners_holdout.csv", index=False, encoding="utf-8-sig")

    save_json(class_distribution, out_dir / "class_distribution.json")
    save_json(
        {
            "feature_sets": feature_sets,
            "thresholds": THRESHOLDS,
            "cv_splits": CV_SPLITS,
            "min_acceptable_precision": MIN_ACCEPTABLE_PRECISION,
        },
        out_dir / "benchmark_metadata.json",
    )

    build_markdown_report(out_dir / "benchmark_report.md", class_distribution, winners_df, holdout_df)
    logger.info("Benchmark finalizado. Resultados en %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
