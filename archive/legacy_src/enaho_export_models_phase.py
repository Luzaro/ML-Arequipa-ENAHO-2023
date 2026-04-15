from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from .enaho_etl_phase01 import build_paths, configure_logging, ensure_directories, save_json
from .enaho_scenario_b_modeling_phase import (
    DATASET,
    TOP20_FILE,
    build_model_pipeline,
    build_preprocessor,
    build_xy,
    detect_feature_sets,
    detect_variable_types,
)


DATASET_CSV = "enaho_arequipa_escenario_b_clean.csv"


def load_dataset(paths: Any) -> pd.DataFrame:
    csv_path = paths.data_processed / DATASET_CSV
    parquet_path = paths.data_processed / DATASET
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.read_parquet(parquet_path)


def load_top20(paths: Any) -> pd.DataFrame:
    return pd.read_csv(paths.reports / "scenario_b_selectkbest" / TOP20_FILE)


def baseline_exports() -> list[dict[str, Any]]:
    exports: list[dict[str, Any]] = [
        {
            "group": "baseline",
            "name": "logistic_regression_top20_full_pipeline.pkl",
            "feature_set": "top20_full",
            "threshold": 0.50,
            "model_label": "LogisticRegression baseline top20_full",
            "model_config": {
                "estimator": LogisticRegression(max_iter=1000, random_state=42),
                "use_scaler": True,
            },
        },
        {
            "group": "baseline",
            "name": "decision_tree_top20_sin_leakage_pipeline.pkl",
            "feature_set": "top20_sin_leakage",
            "threshold": 0.50,
            "model_label": "DecisionTree baseline top20_sin_leakage",
            "model_config": {
                "estimator": DecisionTreeClassifier(random_state=42),
                "use_scaler": False,
            },
        },
    ]

    if importlib.util.find_spec("xgboost"):
        from xgboost import XGBClassifier

        exports.append(
            {
                "group": "baseline",
                "name": "xgboost_top20_full_pipeline.pkl",
                "feature_set": "top20_full",
                "threshold": 0.50,
                "model_label": "XGBoost baseline top20_full",
                "model_config": {
                    "estimator": XGBClassifier(
                        random_state=42,
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="logloss",
                    ),
                    "use_scaler": False,
                },
            }
        )

    return exports


def tuned_exports() -> list[dict[str, Any]]:
    return [
        {
            "group": "tuned",
            "name": "decision_tree_balanceado_top20_full_pipeline.pkl",
            "feature_set": "top20_full",
            "threshold": 0.40,
            "model_label": "DecisionTree balanceado top20_full",
            "model_config": {
                "estimator": DecisionTreeClassifier(
                    random_state=42,
                    max_depth=3,
                    min_samples_leaf=10,
                    class_weight=None,
                ),
                "use_scaler": False,
            },
        },
        {
            "group": "tuned",
            "name": "random_forest_recall_top20_full_pipeline.pkl",
            "feature_set": "top20_full",
            "threshold": 0.30,
            "model_label": "RandomForest recall top20_full",
            "model_config": {
                "estimator": RandomForestClassifier(
                    random_state=42,
                    n_estimators=200,
                    max_depth=5,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    n_jobs=1,
                ),
                "use_scaler": False,
            },
        },
        {
            "group": "tuned",
            "name": "random_forest_recall_top20_sin_leakage_pipeline.pkl",
            "feature_set": "top20_sin_leakage",
            "threshold": 0.30,
            "model_label": "RandomForest recall top20_sin_leakage",
            "model_config": {
                "estimator": RandomForestClassifier(
                    random_state=42,
                    n_estimators=200,
                    max_depth=5,
                    min_samples_leaf=1,
                    class_weight="balanced",
                    n_jobs=1,
                ),
                "use_scaler": False,
            },
        },
    ]


def train_and_export(
    dataset: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    export_cfg: dict[str, Any],
    models_root: Path,
) -> dict[str, Any]:
    features = feature_sets[export_cfg["feature_set"]]
    X, y, target_null_rows = build_xy(dataset, features)
    numeric_cols, categorical_cols = detect_variable_types(X)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    pipeline = build_model_pipeline(export_cfg["model_config"], preprocessor)
    pipeline.fit(X, y)

    out_dir = models_root / export_cfg["group"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / export_cfg["name"]
    joblib.dump(pipeline, out_path)

    return {
        "group": export_cfg["group"],
        "file_name": export_cfg["name"],
        "model_label": export_cfg["model_label"],
        "feature_set": export_cfg["feature_set"],
        "threshold": export_cfg["threshold"],
        "n_features": len(features),
        "n_rows_trained": int(X.shape[0]),
        "target_null_rows_dropped": target_null_rows,
        "path": str(out_path),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    paths = build_paths(root)
    ensure_directories(paths)
    logger = configure_logging(paths.logs / f"export_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    dataset = load_dataset(paths)
    top20_df = load_top20(paths)
    top20_full, top20_sin_leakage, missing = detect_feature_sets(dataset, top20_df)
    feature_sets = {
        "top20_full": top20_full,
        "top20_sin_leakage": top20_sin_leakage,
    }

    models_root = root / "models"
    exported: list[dict[str, Any]] = []
    for export_cfg in baseline_exports() + tuned_exports():
        payload = train_and_export(dataset, feature_sets, export_cfg, models_root)
        exported.append(payload)
        logger.info("Modelo exportado: %s", payload["path"])

    inventory_df = pd.DataFrame(exported)
    inventory_df.to_csv(models_root / "model_inventory.csv", index=False, encoding="utf-8-sig")

    save_json(
        {
            "dataset_source": DATASET_CSV if (paths.data_processed / DATASET_CSV).exists() else DATASET,
            "missing_top20_columns": missing,
            "models_exported": [
                {
                    "group": row["group"],
                    "file_name": row["file_name"],
                    "feature_set": row["feature_set"],
                    "threshold": row["threshold"],
                }
                for row in exported
            ],
        },
        models_root / "model_inventory.json",
    )

    validation = {
        "phase": "EXPORT_MODELS",
        "passed": True,
        "dataset_source": DATASET_CSV if (paths.data_processed / DATASET_CSV).exists() else DATASET,
        "n_models_exported": len(exported),
        "missing_top20_columns": missing,
        "baseline_models": [row["file_name"] for row in exported if row["group"] == "baseline"],
        "tuned_models": [row["file_name"] for row in exported if row["group"] == "tuned"],
    }
    save_json(validation, models_root / "model_export_validation.json")
    logger.info("EXPORT_MODELS validacion: %s", validation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
