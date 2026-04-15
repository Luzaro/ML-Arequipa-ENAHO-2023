from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.io_utils import save_json
from utils.logging_utils import configure_logging
from utils.paths import build_paths, ensure_directories
from modeling.baseline import RANDOM_STATE, TARGET_COL
from modeling.imbalance_benchmark import (
    TEST_SIZE,
    add_engineered_features,
    build_feature_sets,
    cross_validate_models,
    fit_final_pipeline,
    load_dataset,
    load_top20,
    predict_scores,
)


OFFICIAL_CANDIDATES = [
    {
        "scenario_label": "maxima_deteccion",
        "story_name": "Maxima deteccion",
        "alias": "RADAR MAX",
        "objective": "Maximizar recall para no dejar hogares pobres fuera.",
        "feature_set": "top30_full",
        "model_name": "Logistic",
        "model_family": "LogisticRegression",
        "strategy": "smote",
        "threshold": 0.20,
    },
    {
        "scenario_label": "mejor_equilibrio",
        "story_name": "Mejor equilibrio",
        "alias": "XGB BALANCE",
        "objective": "Equilibrar recall y precision con mejor F1 global.",
        "feature_set": "top30_full",
        "model_name": "XGBoost",
        "model_family": "XGBoostClassifier",
        "strategy": "weighted",
        "threshold": 0.40,
    },
    {
        "scenario_label": "precision_controlada",
        "story_name": "Mayor precision",
        "alias": "LGBM PRECISO",
        "objective": "Elevar precision sin perder sensibilidad util.",
        "feature_set": "top30_full",
        "model_name": "LightGBM",
        "model_family": "LightGBMClassifier",
        "strategy": "smote",
        "threshold": 0.35,
    },
]


def holdout_metrics(
    df: pd.DataFrame,
    features: list[str],
    model_name: str,
    strategy: str,
    threshold: float,
) -> dict[str, Any]:
    X = df[features].copy()
    y = pd.to_numeric(df[TARGET_COL], errors="coerce").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = fit_final_pipeline(X_train, y_train, model_name, strategy)
    y_score = predict_scores(pipeline, X_test)
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "pipeline": pipeline,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "confusion_matrix": json.dumps(cm.tolist(), ensure_ascii=False),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
    }


def build_top10_view(summary_df: pd.DataFrame) -> pd.DataFrame:
    top10 = (
        summary_df.sort_values(["recall", "f1", "precision"], ascending=[False, False, False])
        .head(10)
        .copy()
    )
    top10["escenario"] = top10["feature_set"]
    top10["modelo_familia"] = top10["model"].map(
        {
            "Logistic": "LogisticRegression",
            "Tree": "DecisionTreeClassifier",
            "RandomForest": "RandomForestClassifier",
            "XGBoost": "XGBoostClassifier",
            "LightGBM": "LightGBMClassifier",
        }
    )
    top10["modelo"] = top10["model"] + "__" + top10["strategy"]
    top10["accuracy"] = pd.NA
    top10["confusion_matrix"] = pd.NA
    top10["tn"] = pd.NA
    top10["fp"] = pd.NA
    top10["fn"] = pd.NA
    top10["tp"] = pd.NA
    return top10[
        [
            "escenario",
            "modelo_familia",
            "modelo",
            "strategy",
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "confusion_matrix",
            "tn",
            "fp",
            "fn",
            "tp",
        ]
    ].rename(columns={"strategy": "estrategia"})


def build_markdown_report(
    out_path: Path,
    selected_df: pd.DataFrame,
) -> None:
    lines = [
        "# Tuning oficial con SMOTE",
        "",
        "El tuning oficial del proyecto quedo concentrado en modelos con variables monetarias (`top30_full`) y con tratamiento explicito del desbalance.",
        "",
        "## Finalistas oficiales",
        "",
    ]

    for row in selected_df.itertuples():
        lines.extend(
            [
                f"### {row.story_name} | {row.alias}",
                f"- Modelo: {row.modelo_familia}",
                f"- Estrategia: {row.estrategia}",
                f"- Threshold: {row.threshold:.2f}",
                f"- Accuracy holdout: {row.accuracy:.4f}",
                f"- Precision holdout: {row.precision:.4f}",
                f"- Recall holdout: {row.recall:.4f}",
                f"- F1 holdout: {row.f1:.4f}",
                f"- ROC AUC holdout: {row.roc_auc:.4f}",
                f"- PR AUC CV: {row.pr_auc_cv:.4f}",
                f"- Matriz de confusion: TN={row.tn} | FP={row.fp} | FN={row.fn} | TP={row.tp}",
                "",
            ]
        )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    paths = build_paths(root)
    ensure_directories(paths)
    logger = configure_logging(paths.logs / f"scenario_b_modeling_recall_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    df = add_engineered_features(load_dataset(paths))
    top30_full = load_top20(paths)
    feature_sets = build_feature_sets(df, top30_full)
    full_features = feature_sets["top30_full"]

    cv_summary, _ = cross_validate_models(df, {"top30_full": full_features})
    full_summary = cv_summary.loc[cv_summary["feature_set"] == "top30_full"].copy()
    top10_df = build_top10_view(full_summary)

    selected_rows: list[dict[str, Any]] = []
    for candidate in OFFICIAL_CANDIDATES:
        cv_row = full_summary[
            (full_summary["model"] == candidate["model_name"])
            & (full_summary["strategy"] == candidate["strategy"])
            & (full_summary["threshold"] == candidate["threshold"])
        ].iloc[0]
        holdout = holdout_metrics(
            df=df,
            features=full_features,
            model_name=candidate["model_name"],
            strategy=candidate["strategy"],
            threshold=candidate["threshold"],
        )

        selected_rows.append(
            {
                "scenario_label": candidate["scenario_label"],
                "story_name": candidate["story_name"],
                "alias": candidate["alias"],
                "objective": candidate["objective"],
                "escenario": candidate["feature_set"],
                "modelo_familia": candidate["model_family"],
                "modelo": f"{candidate['model_name']}__{candidate['strategy']}",
                "estrategia": candidate["strategy"],
                "threshold": candidate["threshold"],
                "accuracy": holdout["accuracy"],
                "precision": holdout["precision"],
                "recall": holdout["recall"],
                "f1": holdout["f1"],
                "roc_auc": holdout["roc_auc"],
                "pr_auc": float(cv_row["pr_auc"]),
                "precision_cv": float(cv_row["precision"]),
                "recall_cv": float(cv_row["recall"]),
                "f1_cv": float(cv_row["f1"]),
                "roc_auc_cv": float(cv_row["roc_auc"]),
                "pr_auc_cv": float(cv_row["pr_auc"]),
                "confusion_matrix": holdout["confusion_matrix"],
                "tn": holdout["tn"],
                "fp": holdout["fp"],
                "fn": holdout["fn"],
                "tp": holdout["tp"],
                "train_rows": holdout["train_rows"],
                "test_rows": holdout["test_rows"],
            }
        )

    selected_df = pd.DataFrame(selected_rows)

    out_dir = paths.reports / "scenario_b_modeling_recall"
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_df.to_csv(out_dir / "model_recall_optimization_summary.csv", index=False, encoding="utf-8-sig")
    top10_df.to_csv(out_dir / "top10_by_recall.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"top30_full": pd.Series(full_features)}).to_csv(
        out_dir / "feature_sets_used.csv",
        index=False,
        encoding="utf-8-sig",
    )

    build_markdown_report(out_dir / "modeling_recall_summary.md", selected_df)

    metadata = {
        "dataset": "enaho_arequipa_escenario_b_clean",
        "target": TARGET_COL,
        "test_size": TEST_SIZE,
        "feature_set_official": "top30_full",
        "official_candidates": OFFICIAL_CANDIDATES,
    }
    save_json(metadata, out_dir / "modeling_recall_metadata.json")

    validation = {
        "phase": "SCENARIO_B_MODELING_RECALL",
        "passed": True,
        "feature_set_official": "top30_full",
        "n_selected_models": int(selected_df.shape[0]),
        "selected_aliases": selected_df["alias"].tolist(),
    }
    save_json(validation, out_dir / "modeling_recall_validation.json")
    logger.info("SCENARIO_B_MODELING_RECALL validacion: %s", validation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
