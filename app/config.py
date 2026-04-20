from __future__ import annotations

from pathlib import Path

from feature_config import TOP30_FULL


APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "enaho_arequipa_escenario_b_clean.csv"
DATASET_PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "enaho_arequipa_escenario_b_clean.parquet"

BASELINE_SUMMARY_PATH = PROJECT_ROOT / "reports" / "scenario_b_modeling" / "model_comparison_summary.csv"
TUNING_SUMMARY_PATH = PROJECT_ROOT / "reports" / "scenario_b_modeling_recall" / "model_recall_optimization_summary.csv"
TOP10_RECALL_PATH = PROJECT_ROOT / "reports" / "scenario_b_modeling_recall" / "top10_by_recall.csv"

EDA_TARGET_SUMMARY_PATH = PROJECT_ROOT / "reports" / "eda" / "eda_target_summary.csv"
EDA_NUMERIC_SUMMARY_PATH = PROJECT_ROOT / "reports" / "eda" / "eda_numeric_summary.csv"
EDA_CATEGORICAL_SUMMARY_PATH = PROJECT_ROOT / "reports" / "eda" / "eda_categorical_summary.csv"
EDA_NUMERIC_BY_TARGET_PATH = PROJECT_ROOT / "reports" / "eda" / "eda_numeric_by_target.csv"
EDA_CATEGORICAL_BY_TARGET_PATH = PROJECT_ROOT / "reports" / "eda" / "eda_categorical_by_target.csv"
EDA_TARGET_CORRELATIONS_PATH = PROJECT_ROOT / "reports" / "eda" / "eda_target_correlations.csv"

FEATURE_SELECTION_TOP30_PATH = PROJECT_ROOT / "reports" / "feature_selection" / "selected_features_top30.csv"
FEATURE_SELECTION_SUMMARY_PATH = PROJECT_ROOT / "reports" / "feature_selection" / "feature_selection_summary.json"
FEATURE_RANKING_PATH = PROJECT_ROOT / "reports" / "feature_selection" / "feature_ranking.csv"
ANOVA_RANKING_PATH = PROJECT_ROOT / "reports" / "feature_selection" / "variable_ranking_anova.csv"
SELECTKBEST_SUMMARY_PATH = PROJECT_ROOT / "reports" / "feature_selection" / "scenario_b_selectkbest_summary.json"
LEAKAGE_RISK_PATH = PROJECT_ROOT / "reports" / "riesgo_fuga_informacion.csv"

TOP30_EXPERIMENT_PATH = PROJECT_ROOT / "reports" / "scenario_b_modeling_top30_experiment" / "holdout_top20_vs_top30.csv"
ALL48_EXPERIMENT_PATH = PROJECT_ROOT / "reports" / "scenario_b_modeling_all48_experiment" / "holdout_top20_vs_all48.csv"
TOP30_FE_EXPERIMENT_PATH = PROJECT_ROOT / "reports" / "scenario_b_modeling_top30_fe_experiment" / "holdout_top30_vs_top30_fe.csv"
BENCHMARK_HOLDOUT_PATH = PROJECT_ROOT / "reports" / "scenario_b_modeling_imbalance" / "scenario_winners_holdout.csv"

MODEL_INVENTORY_PATH = PROJECT_ROOT / "models" / "model_inventory.csv"

TARGET_COL = "target_pobreza_monetaria_bin"
TARGET_LABEL = "Pobreza monetaria oficial del INEI"
SOURCE_LABEL = "ENAHO 2023"
REGION_LABEL = "Arequipa"
UNIT_LABEL = "Hogar"
OFFICIAL_FEATURE_SET = "top30_full"
OFFICIAL_FEATURE_SET_LABEL = "Top 30 Full"
OFFICIAL_MODEL_LABEL = "Modelo balanceado recomendado"
OFFICIAL_MODEL_FICHA = {
    "Algoritmo": "XGBoostClassifier",
    "Estrategia de balanceo": "Ponderación de clases",
    "Conjunto de variables": "Top 30 Full",
    "Threshold de decisión": "0.40",
}
OFFICIAL_MODEL_RESULTS = {
    "Precisión": "0.416",
    "Recall": "0.597",
    "F1": "0.490",
}

ETL_MODULES = [
    "Módulo01: Vivienda y hogar",
    "Módulo02: Miembros del hogar",
    "Módulo03: Educación",
    "Módulo04: Salud",
    "Módulo05: Empleo e ingresos",
    "Módulo34: Sumaria",
]

OFFICIAL_MODEL_OPTIONS = {
    "Modelo de máxima detección": {
        "path": PROJECT_ROOT / "models" / "tuned" / "logistic_smote_deteccion_top30_full_pipeline.pkl",
        "threshold": 0.20,
        "story": "Máxima detección",
        "note": "Escenario agresivo para ampliar cobertura de hogares pobres.",
        "algorithm": "LogisticRegression",
        "strategy_label": "SMOTE",
        "feature_set_label": "Top 30 Full",
        "technical_name": "LogisticRegression + SMOTE | Top 30 Full | Threshold 0.20",
    },
    "Modelo balanceado recomendado": {
        "path": PROJECT_ROOT / "models" / "tuned" / "xgboost_weighted_equilibrio_top30_full_pipeline.pkl",
        "threshold": 0.40,
        "story": "Mejor equilibrio",
        "note": "Escenario recomendado por balance entre precisión, recall y F1.",
        "algorithm": "XGBoostClassifier",
        "strategy_label": "Ponderación de clases",
        "feature_set_label": "Top 30 Full",
        "technical_name": "XGBoostClassifier + Class Weight / Scale Pos Weight | Top 30 Full | Threshold 0.40",
    },
    "Modelo de mayor precisión relativa": {
        "path": PROJECT_ROOT / "models" / "tuned" / "lightgbm_smote_precision_top30_full_pipeline.pkl",
        "threshold": 0.35,
        "story": "Mayor precisión relativa",
        "note": "Escenario más selectivo para reducir sobre-alertas.",
        "algorithm": "LightGBMClassifier",
        "strategy_label": "SMOTE",
        "feature_set_label": "Top 30 Full",
        "technical_name": "LightGBMClassifier + SMOTE | Top 30 Full | Threshold 0.35",
    },
}

REPRESENTATIVE_IMPORTANCE_ALIASES = {
    "LogisticRegression": ["RADAR MAX", "LR-30"],
    "RandomForestClassifier": ["RF-30"],
    "XGBoostClassifier": ["XGB BALANCE", "XGB-30"],
    "LightGBMClassifier": ["LGBM PRECISO"],
}

TOP30_OFFICIAL_FEATURES = TOP30_FULL
