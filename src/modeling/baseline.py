from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.io_utils import save_json
from utils.logging_utils import configure_logging
from utils.paths import build_paths, ensure_directories


DATASET = "enaho_arequipa_escenario_b_clean.parquet"
TOP20_FILE = "selected_features_top30.csv"
TARGET_COL = "target_pobreza_monetaria_bin"
LEAKAGE_VARS = ["ingreso_neto_total_hogar", "monto_juntos", "monto_bono_gas"]
TEST_SIZE = 0.3
RANDOM_STATE = 42


def load_inputs(paths: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = pd.read_parquet(paths.data_processed / DATASET)
    top20 = pd.read_csv(resolve_top20_path(paths))
    return dataset, top20


def resolve_top20_path(paths: Any) -> Path:
    candidates = [
        paths.reports / "feature_selection" / TOP20_FILE,
        paths.reports / "scenario_b_selectkbest" / TOP20_FILE,
        paths.root / "archive" / "legacy_reports" / "scenario_b_selectkbest" / TOP20_FILE,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No se encontro {TOP20_FILE} en las rutas esperadas.")


def detect_feature_sets(dataset: pd.DataFrame, top20_df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    if "variable" not in top20_df.columns:
        raise ValueError(f"{TOP20_FILE} no contiene la columna 'variable'.")

    top30_full = [col for col in top20_df["variable"].dropna().astype(str).tolist() if col in dataset.columns]
    missing = [col for col in top20_df["variable"].dropna().astype(str).tolist() if col not in dataset.columns]
    return top30_full, [], missing


def build_xy(dataset: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.Series, int]:
    if TARGET_COL not in dataset.columns:
        raise ValueError(f"No se encontrÃ³ la variable target '{TARGET_COL}' en el dataset.")

    work = dataset.copy()
    target_null_rows = int(work[TARGET_COL].isna().sum())
    if target_null_rows > 0:
        work = work.loc[work[TARGET_COL].notna()].copy()

    y = pd.to_numeric(work[TARGET_COL], errors="coerce")
    valid_mask = y.notna()
    if not valid_mask.all():
        work = work.loc[valid_mask].copy()
        y = y.loc[valid_mask]

    y = y.astype(int)
    X = work[features].copy()
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype(object)
            X[col] = X[col].where(pd.notna(X[col]), np.nan)
    return X, y, target_null_rows


def detect_variable_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )


def build_models() -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {
        "LogisticRegression": {
            "estimator": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "use_scaler": True,
        },
        "RandomForestClassifier": {
            "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
            "use_scaler": False,
        },
        "DecisionTreeClassifier": {
            "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "use_scaler": False,
        },
        "SVC": {
            "estimator": SVC(random_state=RANDOM_STATE),
            "use_scaler": True,
        },
    }

    if importlib.util.find_spec("xgboost"):
        from xgboost import XGBClassifier

        models["XGBoostClassifier"] = {
            "estimator": XGBClassifier(
                random_state=RANDOM_STATE,
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
            ),
            "use_scaler": False,
        }
    return models


def build_model_pipeline(model_config: dict[str, Any], preprocessor: ColumnTransformer) -> Pipeline:
    steps: list[tuple[str, Any]] = [("preprocessor", preprocessor)]
    if model_config.get("use_scaler"):
        steps.append(("scaler", StandardScaler(with_mean=False)))
    steps.append(("model", model_config["estimator"]))
    return Pipeline(steps=steps)


def evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    details: dict[str, dict[str, Any]] = {}

    for model_name, model_config in build_models().items():
        pipeline = build_model_pipeline(model_config, preprocessor)

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_text = classification_report(y_test, y_pred, zero_division=0)

        row = {
            "model": model_name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "precision_pobre": report_dict.get("1", {}).get("precision", 0.0),
            "recall_pobre": report_dict.get("1", {}).get("recall", 0.0),
            "f1_pobre": report_dict.get("1", {}).get("f1-score", 0.0),
            "support_pobre": report_dict.get("1", {}).get("support", 0.0),
        }
        rows.append(row)

        details[model_name] = {
            "confusion_matrix": cm.tolist(),
            "classification_report_text": report_text,
            "classification_report_dict": report_dict,
        }

    summary_df = pd.DataFrame(rows).sort_values(
        ["recall_pobre", "f1_pobre", "accuracy"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return summary_df, details


def run_feature_set(
    dataset: pd.DataFrame,
    feature_set_name: str,
    features: list[str],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, Any]]:
    X, y, target_null_rows = build_xy(dataset, features)
    numeric_cols, categorical_cols = detect_variable_types(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    summary_df, details = evaluate_models(X_train, X_test, y_train, y_test, preprocessor)
    summary_df.insert(0, "feature_set", feature_set_name)

    metadata = {
        "feature_set": feature_set_name,
        "features": features,
        "target_null_rows_dropped": target_null_rows,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "models_run": summary_df["model"].tolist(),
        "best_model_by_recall_pobre": summary_df.iloc[0]["model"],
    }
    return summary_df, details, metadata


def write_detail_reports(out_dir: Path, details: dict[str, dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for model_name, payload in details.items():
        safe_name = model_name.lower()
        (out_dir / f"{safe_name}_classification_report.txt").write_text(
            payload["classification_report_text"],
            encoding="utf-8",
        )
        save_json(
            {
                "confusion_matrix": payload["confusion_matrix"],
                "classification_report": payload["classification_report_dict"],
            },
            out_dir / f"{safe_name}_metrics.json",
        )


def write_summary_report(
    path: Path,
    dataset_name: str,
    target_col: str,
    top30_full: list[str],
    summary_df: pd.DataFrame,
    metadata_full: dict[str, Any],
) -> None:
    best_model = summary_df.iloc[0]
    lines = [
        "# Modelado inicial escenario B",
        "",
        f"- Dataset detectado: {dataset_name}",
        f"- Target detectado: {target_col}",
        f"- Total top30_full: {len(top30_full)}",
        "",
        "## top30_full",
        "",
        *[f"- {col}" for col in top30_full],
        "",
        "## Resumen de modelos",
        "",
        *[
            f"- {row.feature_set} | {row.model}: accuracy={row.accuracy:.4f}, precision={row.precision:.4f}, recall={row.recall:.4f}, f1={row.f1_score:.4f}, recall_pobre={row.recall_pobre:.4f}"
            for row in summary_df.itertuples()
        ],
        "",
        "## Variables por escenario",
        "",
        f"- top30_full: {len(metadata_full['features'])} variables ({len(metadata_full['numeric_columns'])} numericas, {len(metadata_full['categorical_columns'])} categoricas)",
        "",
        "## Mejor modelo",
        "",
        f"- Mejor desempeno priorizando recall de la clase pobre: {best_model['feature_set']} | {best_model['model']}",
        f"- recall_pobre={best_model['recall_pobre']:.4f}, f1_pobre={best_model['f1_pobre']:.4f}, accuracy={best_model['accuracy']:.4f}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    paths = build_paths(root)
    ensure_directories(paths)
    logger = configure_logging(paths.logs / f"scenario_b_modeling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    dataset, top20_df = load_inputs(paths)
    top30_full, _, missing = detect_feature_sets(dataset, top20_df)
    summary_df, details_full, meta_full = run_feature_set(dataset, "top30_full", top30_full)
    summary_df = summary_df.sort_values(["recall_pobre", "f1_pobre", "accuracy"], ascending=[False, False, False]).reset_index(drop=True)

    out_dir = paths.reports / "scenario_b_modeling"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(out_dir / "model_comparison_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"top30_full": pd.Series(top30_full)}).to_csv(out_dir / "feature_sets_used.csv", index=False, encoding="utf-8-sig")
    write_detail_reports(out_dir / "top30_full", details_full)
    write_summary_report(
        out_dir / "modeling_summary.md",
        DATASET,
        TARGET_COL,
        top30_full,
        summary_df,
        meta_full,
    )

    save_json(
        {
            "dataset_detected": DATASET,
            "target_detected": TARGET_COL,
            "top30_full": top30_full,
            "missing_top20_columns": missing,
            "top30_full_metadata": meta_full,
            "models_run": summary_df["model"].tolist(),
            "best_model_by_recall_pobre": {
                "feature_set": summary_df.iloc[0]["feature_set"],
                "model": summary_df.iloc[0]["model"],
            },
        },
        out_dir / "modeling_metadata.json",
    )

    validation = {
        "phase": "SCENARIO_B_MODELING",
        "passed": True,
        "dataset_detected": DATASET,
        "target_detected": TARGET_COL,
        "n_top30_full": len(top30_full),
        "n_numeric_full": len(meta_full["numeric_columns"]),
        "n_categorical_full": len(meta_full["categorical_columns"]),
        "models_run": summary_df["model"].tolist(),
        "best_model_by_recall_pobre": {
            "feature_set": summary_df.iloc[0]["feature_set"],
            "model": summary_df.iloc[0]["model"],
        },
    }
    save_json(validation, out_dir / "modeling_validation.json")
    logger.info("SCENARIO_B_MODELING validacion: %s", validation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

