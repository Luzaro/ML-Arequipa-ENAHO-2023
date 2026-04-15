from __future__ import annotations

import importlib.util
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .enaho_etl_phase01 import build_paths, configure_logging, ensure_directories, save_json
from .enaho_scenario_b_modeling_phase import (
    DATASET,
    LEAKAGE_VARS,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
    TOP20_FILE,
    build_xy,
    detect_feature_sets,
    detect_variable_types,
    load_inputs,
)


THRESHOLDS = [0.50, 0.45, 0.40, 0.35, 0.30]


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


def build_pipeline(model: Any, preprocessor: ColumnTransformer, use_scaler: bool) -> Pipeline:
    steps: list[tuple[str, Any]] = [("preprocessor", preprocessor)]
    if use_scaler:
        steps.append(("scaler", StandardScaler(with_mean=False)))
    steps.append(("model", model))
    return Pipeline(steps=steps)


def get_scale_pos_weight(y_train: pd.Series) -> float:
    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    if positives == 0:
        return 1.0
    return negatives / positives


def build_model_candidates(scale_pos_weight: float) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    for c_value, class_weight in product([0.1, 1.0, 3.0], [None, "balanced"]):
        estimator = LogisticRegression(
            C=c_value,
            class_weight=class_weight,
            max_iter=2000,
            random_state=RANDOM_STATE,
            solver="liblinear",
        )
        candidates.append(
            {
                "model_family": "LogisticRegression",
                "model_label": f"LogisticRegression__C_{c_value}__class_weight_{class_weight or 'none'}",
                "estimator": estimator,
                "use_scaler": True,
                "params": {
                    "C": c_value,
                    "class_weight": class_weight or "none",
                },
            }
        )

    for max_depth, min_samples_leaf, class_weight in product([3, 5, 8, None], [1, 5, 10], [None, "balanced"]):
        estimator = DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
        )
        candidates.append(
            {
                "model_family": "DecisionTreeClassifier",
                "model_label": (
                    f"DecisionTreeClassifier__max_depth_{max_depth or 'none'}"
                    f"__min_leaf_{min_samples_leaf}__class_weight_{class_weight or 'none'}"
                ),
                "estimator": estimator,
                "use_scaler": False,
                "params": {
                    "max_depth": max_depth or "none",
                    "min_samples_leaf": min_samples_leaf,
                    "class_weight": class_weight or "none",
                },
            }
        )

    for n_estimators, max_depth, min_samples_leaf, class_weight in product(
        [200, 400],
        [5, 10, None],
        [1, 5],
        [None, "balanced"],
    ):
        estimator = RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            n_jobs=1,
        )
        candidates.append(
            {
                "model_family": "RandomForestClassifier",
                "model_label": (
                    f"RandomForestClassifier__n_{n_estimators}__max_depth_{max_depth or 'none'}"
                    f"__min_leaf_{min_samples_leaf}__class_weight_{class_weight or 'none'}"
                ),
                "estimator": estimator,
                "use_scaler": False,
                "params": {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth or "none",
                    "min_samples_leaf": min_samples_leaf,
                    "class_weight": class_weight or "none",
                },
            }
        )

    if importlib.util.find_spec("xgboost"):
        from xgboost import XGBClassifier

        for n_estimators, max_depth, learning_rate, pos_weight in product(
            [200, 400],
            [3, 5],
            [0.05, 0.10],
            [1.0, scale_pos_weight],
        ):
            estimator = XGBClassifier(
                random_state=RANDOM_STATE,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                objective="binary:logistic",
                scale_pos_weight=pos_weight,
                n_jobs=1,
            )
            candidates.append(
                {
                    "model_family": "XGBoostClassifier",
                    "model_label": (
                        f"XGBoostClassifier__n_{n_estimators}__max_depth_{max_depth}"
                        f"__lr_{learning_rate}__scale_pos_weight_{round(pos_weight, 3)}"
                    ),
                    "estimator": estimator,
                    "use_scaler": False,
                    "params": {
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "learning_rate": learning_rate,
                        "scale_pos_weight": round(pos_weight, 6),
                    },
                }
            )

    return candidates


def evaluate_thresholds(
    y_true: pd.Series,
    y_score: pd.Series,
    threshold: float,
) -> dict[str, Any]:
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_score),
        "confusion_matrix": cm.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def run_scenario(
    dataset: pd.DataFrame,
    scenario_name: str,
    features: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
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
    scale_pos_weight = get_scale_pos_weight(y_train)
    rows: list[dict[str, Any]] = []

    for candidate in build_model_candidates(scale_pos_weight):
        pipeline = build_pipeline(candidate["estimator"], preprocessor, candidate["use_scaler"])
        pipeline.fit(X_train, y_train)
        y_score = pd.Series(pipeline.predict_proba(X_test)[:, 1], index=y_test.index)

        for threshold in THRESHOLDS:
            metrics = evaluate_thresholds(y_test, y_score, threshold)
            rows.append(
                {
                    "escenario": scenario_name,
                    "modelo_familia": candidate["model_family"],
                    "modelo": candidate["model_label"],
                    "threshold": metrics["threshold"],
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "pr_auc": metrics["pr_auc"],
                    "confusion_matrix": json_dumps(metrics["confusion_matrix"]),
                    "tn": metrics["tn"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "tp": metrics["tp"],
                    "parametros": json_dumps(candidate["params"]),
                }
            )

    summary_df = (
        pd.DataFrame(rows)
        .sort_values(["recall", "f1", "precision"], ascending=[False, False, False])
        .reset_index(drop=True)
    )

    metadata = {
        "escenario": scenario_name,
        "features": features,
        "target_null_rows_dropped": target_null_rows,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "scale_pos_weight_train": scale_pos_weight,
        "thresholds_tested": THRESHOLDS,
    }
    return summary_df, metadata


def json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)


def build_final_report(
    out_path: Path,
    dataset_name: str,
    target_name: str,
    top10_df: pd.DataFrame,
    best_recall_row: pd.Series,
    best_balanced_row: pd.Series,
    best_noleak_row: pd.Series,
) -> None:
    lines = [
        "# Segunda ronda de modelado enfocada en recall",
        "",
        f"- Dataset detectado: {dataset_name}",
        f"- Target detectado: {target_name}",
        "",
        "## Top 10 corridas por recall",
        "",
    ]

    for row in top10_df.itertuples():
        lines.append(
            f"- {row.escenario} | {row.modelo} | threshold={row.threshold:.2f} | "
            f"recall={row.recall:.4f} | f1={row.f1:.4f} | precision={row.precision:.4f} | pr_auc={row.pr_auc:.4f}"
        )

    lines.extend(
        [
            "",
            "## Mejor modelo por recall",
            "",
            f"- {best_recall_row['escenario']} | {best_recall_row['modelo']} | threshold={best_recall_row['threshold']:.2f}",
            f"- recall={best_recall_row['recall']:.4f}, f1={best_recall_row['f1']:.4f}, precision={best_recall_row['precision']:.4f}, pr_auc={best_recall_row['pr_auc']:.4f}",
            f"- confusion_matrix={best_recall_row['confusion_matrix']}",
            "",
            "## Mejor modelo balanceado",
            "",
            f"- {best_balanced_row['escenario']} | {best_balanced_row['modelo']} | threshold={best_balanced_row['threshold']:.2f}",
            f"- recall={best_balanced_row['recall']:.4f}, f1={best_balanced_row['f1']:.4f}, precision={best_balanced_row['precision']:.4f}, pr_auc={best_balanced_row['pr_auc']:.4f}",
            f"- confusion_matrix={best_balanced_row['confusion_matrix']}",
            "",
            "## Mejor modelo sin leakage",
            "",
            f"- {best_noleak_row['escenario']} | {best_noleak_row['modelo']} | threshold={best_noleak_row['threshold']:.2f}",
            f"- recall={best_noleak_row['recall']:.4f}, f1={best_noleak_row['f1']:.4f}, precision={best_noleak_row['precision']:.4f}, pr_auc={best_noleak_row['pr_auc']:.4f}",
            f"- confusion_matrix={best_noleak_row['confusion_matrix']}",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    paths = build_paths(root)
    ensure_directories(paths)
    logger = configure_logging(paths.logs / f"scenario_b_modeling_recall_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    dataset, top20_df = load_inputs(paths)
    top20_full, top20_sin_leakage, missing = detect_feature_sets(dataset, top20_df)

    summary_full, meta_full = run_scenario(dataset, "top20_full", top20_full)
    summary_noleak, meta_noleak = run_scenario(dataset, "top20_sin_leakage", top20_sin_leakage)
    summary_df = (
        pd.concat([summary_full, summary_noleak], ignore_index=True)
        .sort_values(["recall", "f1", "precision"], ascending=[False, False, False])
        .reset_index(drop=True)
    )

    top10_df = summary_df.head(10).copy()
    best_recall_row = summary_df.iloc[0]
    best_balanced_row = summary_df.sort_values(["f1", "recall", "precision"], ascending=[False, False, False]).iloc[0]
    noleak_mask = summary_df["escenario"] == "top20_sin_leakage"
    best_noleak_row = summary_df.loc[noleak_mask].sort_values(
        ["recall", "f1", "precision"], ascending=[False, False, False]
    ).iloc[0]

    out_dir = paths.reports / "scenario_b_modeling_recall"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(out_dir / "model_recall_optimization_summary.csv", index=False, encoding="utf-8-sig")
    top10_df.to_csv(out_dir / "top10_by_recall.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "top20_full": pd.Series(top20_full),
            "top20_sin_leakage": pd.Series(top20_sin_leakage),
        }
    ).to_csv(out_dir / "feature_sets_used.csv", index=False, encoding="utf-8-sig")

    build_final_report(
        out_dir / "modeling_recall_summary.md",
        DATASET,
        TARGET_COL,
        top10_df,
        best_recall_row,
        best_balanced_row,
        best_noleak_row,
    )

    save_json(
        {
            "dataset_detected": DATASET,
            "target_detected": TARGET_COL,
            "top20_full": top20_full,
            "top20_sin_leakage": top20_sin_leakage,
            "missing_top20_columns": missing,
            "top20_full_metadata": meta_full,
            "top20_sin_leakage_metadata": meta_noleak,
            "best_model_by_recall": {
                "escenario": best_recall_row["escenario"],
                "modelo": best_recall_row["modelo"],
                "threshold": best_recall_row["threshold"],
            },
            "best_model_balanced": {
                "escenario": best_balanced_row["escenario"],
                "modelo": best_balanced_row["modelo"],
                "threshold": best_balanced_row["threshold"],
            },
            "best_model_without_leakage": {
                "escenario": best_noleak_row["escenario"],
                "modelo": best_noleak_row["modelo"],
                "threshold": best_noleak_row["threshold"],
            },
        },
        out_dir / "modeling_recall_metadata.json",
    )

    validation = {
        "phase": "SCENARIO_B_MODELING_RECALL",
        "passed": True,
        "dataset_detected": DATASET,
        "target_detected": TARGET_COL,
        "n_top20_full": len(top20_full),
        "n_top20_sin_leakage": len(top20_sin_leakage),
        "n_results": int(summary_df.shape[0]),
        "models_tested": sorted(summary_df["modelo_familia"].unique().tolist()),
        "best_model_by_recall": {
            "escenario": best_recall_row["escenario"],
            "modelo": best_recall_row["modelo"],
            "threshold": float(best_recall_row["threshold"]),
            "recall": float(best_recall_row["recall"]),
        },
    }
    save_json(validation, out_dir / "modeling_recall_validation.json")
    logger.info("SCENARIO_B_MODELING_RECALL validacion: %s", validation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
