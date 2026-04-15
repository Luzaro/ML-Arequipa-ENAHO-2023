from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lightgbm import LGBMClassifier

from utils.io_utils import save_json
from utils.logging_utils import configure_logging
from utils.paths import build_paths, ensure_directories
from modeling.baseline import RANDOM_STATE, TARGET_COL, build_xy, detect_feature_sets, load_inputs
from modeling.imbalance_benchmark import add_engineered_features, make_pipeline, positive_weight


def baseline_exports() -> list[dict[str, Any]]:
    exports: list[dict[str, Any]] = [
        {
            "group": "baseline",
            "alias": "LR-30",
            "name": "logistic_regression_top30_full_pipeline.pkl",
            "feature_set": "top30_full",
            "threshold": 0.50,
            "model_label": "LogisticRegression baseline top30_full",
            "estimator": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "strategy": "weighted",
        },
        {
            "group": "baseline",
            "alias": "RF-30",
            "name": "random_forest_top30_full_pipeline.pkl",
            "feature_set": "top30_full",
            "threshold": 0.50,
            "model_label": "RandomForest baseline top30_full",
            "estimator": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            "strategy": "weighted",
        },
    ]

    if importlib.util.find_spec("xgboost"):
        from xgboost import XGBClassifier

        exports.append(
            {
                "group": "baseline",
                "alias": "XGB-30",
                "name": "xgboost_top30_full_pipeline.pkl",
                "feature_set": "top30_full",
                "threshold": 0.50,
                "model_label": "XGBoost baseline top30_full",
                "estimator": XGBClassifier(
                    random_state=RANDOM_STATE,
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    n_jobs=1,
                ),
                "strategy": "weighted",
            }
        )

    return exports


def tuned_exports(pos_weight: float) -> list[dict[str, Any]]:
    exports: list[dict[str, Any]] = [
        {
            "group": "tuned",
            "alias": "RADAR MAX",
            "name": "logistic_smote_deteccion_top30_full_pipeline.pkl",
            "feature_set": "top30_full",
            "threshold": 0.20,
            "model_label": "Logistic smote maxima deteccion top30_full",
            "estimator": LogisticRegression(
                max_iter=3000,
                random_state=RANDOM_STATE,
                solver="liblinear",
            ),
            "strategy": "smote",
        },
        {
            "group": "tuned",
            "alias": "XGB BALANCE",
            "name": "xgboost_weighted_equilibrio_top30_full_pipeline.pkl",
            "feature_set": "top30_full",
            "threshold": 0.40,
            "model_label": "XGBoost weighted equilibrio top30_full",
            "strategy": "weighted",
        },
        {
            "group": "tuned",
            "alias": "LGBM PRECISO",
            "name": "lightgbm_smote_precision_top30_full_pipeline.pkl",
            "feature_set": "top30_full",
            "threshold": 0.35,
            "model_label": "LightGBM smote precision top30_full",
            "estimator": LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                class_weight=None,
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbose=-1,
            ),
            "strategy": "smote",
        },
    ]

    if importlib.util.find_spec("xgboost"):
        from xgboost import XGBClassifier

        exports[1]["estimator"] = XGBClassifier(
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
        )
    else:
        exports.pop(1)

    return exports


def train_and_export(
    dataset: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    export_cfg: dict[str, Any],
    models_root: Path,
) -> dict[str, Any]:
    features = feature_sets[export_cfg["feature_set"]]
    X, y, target_null_rows = build_xy(dataset, features)
    pipeline = make_pipeline(X, export_cfg["estimator"], export_cfg["strategy"])
    pipeline.fit(X, y)

    out_dir = models_root / export_cfg["group"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / export_cfg["name"]
    joblib.dump(pipeline, out_path)

    return {
        "group": export_cfg["group"],
        "alias": export_cfg["alias"],
        "file_name": export_cfg["name"],
        "model_label": export_cfg["model_label"],
        "feature_set": export_cfg["feature_set"],
        "threshold": export_cfg["threshold"],
        "strategy": export_cfg["strategy"],
        "n_features": len(features),
        "n_rows_trained": int(X.shape[0]),
        "target_null_rows_dropped": target_null_rows,
        "path": str(out_path),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    paths = build_paths(root)
    ensure_directories(paths)
    logger = configure_logging(paths.logs / f"export_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    dataset, top20_df = load_inputs(paths)
    dataset = add_engineered_features(dataset)
    top30_full, _, missing = detect_feature_sets(dataset, top20_df)
    feature_sets = {"top30_full": top30_full}

    models_root = root / "models"
    exported: list[dict[str, Any]] = []
    X_full, y_full, _ = build_xy(dataset, top30_full)
    pos_weight = positive_weight(y_full)

    for export_cfg in baseline_exports() + tuned_exports(pos_weight):
        payload = train_and_export(dataset, feature_sets, export_cfg, models_root)
        exported.append(payload)
        logger.info("Modelo exportado: %s", payload["path"])

    inventory_df = pd.DataFrame(exported)
    inventory_df.to_csv(models_root / "model_inventory.csv", index=False, encoding="utf-8-sig")

    save_json(
        {
            "dataset_source": "enaho_arequipa_escenario_b_clean",
            "feature_set_official": "top30_full",
            "missing_top20_columns": missing,
            "models_exported": [
                {
                    "group": row["group"],
                    "alias": row["alias"],
                    "file_name": row["file_name"],
                    "feature_set": row["feature_set"],
                    "threshold": row["threshold"],
                    "strategy": row["strategy"],
                }
                for row in exported
            ],
        },
        models_root / "model_inventory.json",
    )

    validation = {
        "phase": "EXPORT_MODELS",
        "passed": True,
        "feature_set_official": "top30_full",
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
