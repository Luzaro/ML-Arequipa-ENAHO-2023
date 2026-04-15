from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from feature_config import FEATURE_CONFIG


APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "enaho_arequipa_escenario_b_clean.csv"
BASELINE_REPORT_PATH = PROJECT_ROOT / "reports" / "scenario_b_modeling" / "model_comparison_summary.csv"
TUNING_REPORT_PATH = PROJECT_ROOT / "reports" / "scenario_b_modeling_recall" / "model_recall_optimization_summary.csv"
TOP10_RECALL_PATH = PROJECT_ROOT / "reports" / "scenario_b_modeling_recall" / "top10_by_recall.csv"
MODEL_INVENTORY_PATH = PROJECT_ROOT / "models" / "model_inventory.csv"
BASELINE_CONFUSIONS = {
    "LR-30": PROJECT_ROOT / "reports" / "scenario_b_modeling" / "top30_full" / "logisticregression_metrics.json",
    "RF-30": PROJECT_ROOT / "reports" / "scenario_b_modeling" / "top30_full" / "randomforestclassifier_metrics.json",
    "XGB-30": PROJECT_ROOT / "reports" / "scenario_b_modeling" / "top30_full" / "xgboostclassifier_metrics.json",
}

PALETTE = {
    "navy": "#14213D",
    "slate": "#33415C",
    "teal": "#0E7490",
    "gold": "#C98B2B",
    "coral": "#D1603D",
    "cream": "#F6F1E8",
    "ink": "#0B172A",
    "muted": "#667085",
    "success": "#2F6F4F",
    "sand": "#E8D9BF",
    "ice": "#E7F1F5",
}

TARGET_LABELS = {0: "No pobre", 1: "Pobre"}
MODEL_FULL_NAMES = {
    "LogisticRegression": "Regresion logistica",
    "DecisionTreeClassifier": "Arbol de decision",
    "RandomForestClassifier": "Bosque aleatorio",
    "XGBoostClassifier": "XGBoost",
    "LightGBMClassifier": "LightGBM",
    "SVC": "Maquina de soporte vectorial",
}
STRATEGY_FULL_NAMES = {
    "smote": "SMOTE",
    "weighted": "Ponderacion de clases",
}


def ensure_project_on_path() -> None:
    project_root_str = str(PROJECT_ROOT)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


def inject_global_styles() -> None:
    st.markdown(
        f"""
        <style>
        .block-container {{
            padding-top: 1.2rem;
            padding-bottom: 2.5rem;
            max-width: 1240px;
        }}
        .stApp {{
            background:
                radial-gradient(circle at top right, rgba(201,139,43,.18), transparent 24%),
                radial-gradient(circle at top left, rgba(14,116,144,.10), transparent 18%),
                linear-gradient(180deg, #fffdf9 0%, #f5efe5 52%, #eef4f6 100%);
            color: {PALETTE["ink"]};
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #f6f0e4 0%, #edf4f6 100%);
            border-right: 1px solid rgba(22,50,79,.08);
        }}
        [data-testid="stSidebarNav"] > ul {{
            gap: .25rem;
        }}
        [data-testid="stSidebarNav"] a {{
            border-radius: 12px;
            padding: .35rem .45rem;
        }}
        [data-testid="stSidebarNav"] a:hover {{
            background: rgba(20,33,61,.06);
        }}
        .hero-block {{
            padding: 1.65rem 1.7rem;
            border-radius: 26px;
            background:
                radial-gradient(circle at top right, rgba(201,139,43,.22), transparent 22%),
                linear-gradient(135deg, rgba(20,33,61,.99) 0%, rgba(51,65,92,.97) 56%, rgba(14,116,144,.95) 100%);
            color: #fffef9;
            border: 1px solid rgba(255,255,255,.09);
            box-shadow: 0 20px 54px rgba(11,23,42,.18);
            margin-bottom: 1.1rem;
        }}
        .hero-kicker {{
            letter-spacing: .18rem;
            text-transform: uppercase;
            font-size: .74rem;
            font-weight: 700;
            color: rgba(255,255,255,.66);
            margin-bottom: .4rem;
        }}
        .hero-title {{
            font-size: 2.7rem;
            line-height: 1.08;
            font-weight: 900;
            margin: 0 0 .45rem 0;
        }}
        .hero-subtitle {{
            font-size: 1.02rem;
            line-height: 1.7;
            color: rgba(255,255,255,.86);
            margin: 0;
            max-width: 980px;
        }}
        .section-intro {{
            border-radius: 20px;
            padding: 1rem 1.15rem;
            background: linear-gradient(180deg, rgba(255,255,255,.82) 0%, rgba(255,255,255,.66) 100%);
            border: 1px solid rgba(20,33,61,.08);
            box-shadow: 0 14px 30px rgba(11,23,42,.05);
            margin: .45rem 0 1rem 0;
        }}
        .section-kicker {{
            text-transform: uppercase;
            letter-spacing: .12rem;
            font-size: .72rem;
            font-weight: 800;
            color: {PALETTE["gold"]};
            margin-bottom: .25rem;
        }}
        .section-title {{
            font-size: 1.3rem;
            font-weight: 800;
            color: {PALETTE["navy"]};
            margin-bottom: .3rem;
        }}
        .section-body {{
            color: {PALETTE["slate"]};
            line-height: 1.65;
            font-size: .96rem;
        }}
        .metric-card {{
            border-radius: 20px;
            padding: 1rem 1.1rem;
            background: linear-gradient(180deg, rgba(255,255,255,.92) 0%, rgba(255,255,255,.80) 100%);
            border: 1px solid rgba(20,33,61,.08);
            box-shadow: 0 12px 26px rgba(11,23,42,.06);
            margin-bottom: .7rem;
            position: relative;
            overflow: hidden;
        }}
        .metric-card::before {{
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 5px;
            background: linear-gradient(90deg, {PALETTE["gold"]} 0%, {PALETTE["coral"]} 45%, {PALETTE["teal"]} 100%);
        }}
        .metric-label {{
            font-size: .78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: .06rem;
            color: {PALETTE["muted"]};
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: 800;
            color: {PALETTE["navy"]};
            margin-top: .2rem;
        }}
        .metric-note {{
            font-size: .92rem;
            color: {PALETTE["slate"]};
            margin-top: .3rem;
            line-height: 1.45;
        }}
        .story-card {{
            border: 1px solid rgba(20,33,61,.08);
            background: linear-gradient(180deg, rgba(255,255,255,.88) 0%, rgba(255,255,255,.76) 100%);
            padding: 1rem 1.1rem;
            border-radius: 18px;
            box-shadow: 0 12px 24px rgba(11,23,42,.05);
            margin-bottom: .85rem;
        }}
        .story-title {{
            font-size: 1rem;
            font-weight: 800;
            color: {PALETTE["navy"]};
            margin-bottom: .3rem;
        }}
        .story-body {{
            color: {PALETTE["slate"]};
            line-height: 1.55;
            font-size: .95rem;
        }}
        .highlight-card {{
            border-radius: 22px;
            padding: 1rem 1.15rem;
            color: white;
            min-height: 195px;
            box-shadow: 0 18px 34px rgba(11,23,42,.13);
            border: 1px solid rgba(255,255,255,.08);
        }}
        .highlight-title {{
            font-size: .82rem;
            text-transform: uppercase;
            letter-spacing: .06rem;
            font-weight: 800;
            opacity: .8;
        }}
        .highlight-model {{
            font-size: 1.45rem;
            font-weight: 900;
            margin-top: .5rem;
            margin-bottom: .6rem;
        }}
        .highlight-metric {{
            font-size: 1.02rem;
            font-weight: 700;
            margin-top: .15rem;
        }}
        .section-label {{
            text-transform: uppercase;
            letter-spacing: .08rem;
            color: {PALETTE["coral"]};
            font-size: .78rem;
            font-weight: 800;
            margin-top: .8rem;
        }}
        .small-note {{
            color: {PALETTE["muted"]};
            font-size: .92rem;
            line-height: 1.5;
        }}
        .prediction-card {{
            border-radius: 22px;
            padding: 1.2rem 1.3rem;
            background: linear-gradient(135deg, rgba(201,139,43,.15) 0%, rgba(14,116,144,.12) 100%);
            border: 1px solid rgba(20,33,61,.08);
            box-shadow: 0 16px 32px rgba(11,23,42,.08);
        }}
        .sidebar-block {{
            border-radius: 18px;
            padding: 1rem;
            background: rgba(255,255,255,.72);
            border: 1px solid rgba(20,33,61,.08);
            margin-bottom: .8rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(kicker: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero-block">
            <div class="hero-kicker">{kicker}</div>
            <div class="hero-title">{title}</div>
            <p class="hero-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, note: str = "") -> None:
    note_html = f'<div class="metric-note">{note}</div>' if note else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_intro(title: str, body: str, kicker: str = "Lectura ejecutiva") -> None:
    st.markdown(
        f"""
        <div class="section-intro">
            <div class="section-kicker">{kicker}</div>
            <div class="section-title">{title}</div>
            <div class="section-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_story_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="story-card">
            <div class="story-title">{title}</div>
            <div class="story-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_highlight_card(title: str, alias: str, metrics: list[str], tone: str) -> None:
    gradients = {
        "navy": "linear-gradient(135deg, #16324F 0%, #2F4858 100%)",
        "coral": "linear-gradient(135deg, #E76F51 0%, #F4A261 100%)",
        "teal": "linear-gradient(135deg, #1F7A73 0%, #2A9D8F 100%)",
    }
    metrics_html = "".join([f'<div class="highlight-metric">{item}</div>' for item in metrics])
    st.markdown(
        f"""
        <div class="highlight-card" style="background: {gradients.get(tone, gradients['navy'])};">
            <div class="highlight-title">{title}</div>
            <div class="highlight-model">{alias}</div>
            {metrics_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_summary(df: pd.DataFrame) -> None:
    pobreza_rate = float(df["target_pobreza_monetaria_bin"].mean()) if "target_pobreza_monetaria_bin" in df.columns else 0.0
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-block">
                <div class="metric-label">Proyecto</div>
                <div style="font-size:1.15rem;font-weight:800;color:#16324F;margin-top:.2rem;">ENAHO Arequipa Pobreza 2023 ML</div>
                <div class="small-note" style="margin-top:.55rem;">
                    Un tablero de presentacion para explicar el problema, la solucion analitica y el resultado final sin perder trazabilidad.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("### Ficha Rapida")
        st.write("**Fuente:** ENAHO 2023 Anual")
        st.write("**Ambito:** Arequipa")
        st.write("**Unidad:** Hogar")
        st.write("**Target:** Pobreza monetaria oficial")
        st.metric("Hogares", f"{len(df):,}".replace(",", "."))
        st.metric("Columnas", f"{df.shape[1]}")
        st.metric("Tasa de pobreza", format_pct(pobreza_rate, digits=1))


def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def feature_display_name(feature: str) -> str:
    if feature in FEATURE_CONFIG:
        label = FEATURE_CONFIG[feature]["label"]
        return label.replace("¿", "").replace("?", "")
    return feature.replace("_", " ").title()


def format_pct(value: float | int, digits: int = 1) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value) * 100:.{digits}f}%"


def scenario_label(value: str) -> str:
    return {
        "top30_full": "Top 30 oficial",
    }.get(value, value)


def full_model_name(value: str) -> str:
    return MODEL_FULL_NAMES.get(value, value)


def family_short(value: str) -> str:
    return {
        "LogisticRegression": "LR",
        "DecisionTreeClassifier": "TREE",
        "RandomForestClassifier": "RF",
        "XGBoostClassifier": "XGB",
        "LightGBMClassifier": "LGBM",
        "SVC": "SVM",
    }.get(value, value)


def baseline_alias(row: pd.Series) -> str:
    alias_map = {
        ("LogisticRegression", "top30_full"): "LR-30",
        ("DecisionTreeClassifier", "top30_full"): "TREE-30",
        ("RandomForestClassifier", "top30_full"): "RF-30",
        ("XGBoostClassifier", "top30_full"): "XGB-30",
        ("SVC", "top30_full"): "SVM-30",
    }
    return alias_map.get((row["model"], row["feature_set"]), f"{family_short(row['model'])}-?")


def baseline_display_name(row: pd.Series) -> str:
    return f"{full_model_name(row['model'])} | {scenario_label(row['feature_set'])}"


def tuned_alias(row: pd.Series) -> str:
    family = row["modelo_familia"]
    scenario = row["escenario"]
    threshold = float(row["threshold"])
    strategy = str(row.get("estrategia", ""))

    if family == "LogisticRegression" and scenario == "top30_full" and strategy == "smote" and math.isclose(threshold, 0.20, abs_tol=1e-6):
        return "RADAR MAX"
    if family == "XGBoostClassifier" and scenario == "top30_full" and strategy == "weighted" and math.isclose(threshold, 0.40, abs_tol=1e-6):
        return "XGB BALANCE"
    if family == "LightGBMClassifier" and scenario == "top30_full" and strategy == "smote" and math.isclose(threshold, 0.35, abs_tol=1e-6):
        return "LGBM PRECISO"

    return f"{family_short(family)}-30-T{int(threshold * 100)}"


def tuned_story_name(row: pd.Series) -> str:
    if row["alias"] == "RADAR MAX":
        return "Maxima deteccion"
    if row["alias"] == "XGB BALANCE":
        return "Mejor equilibrio"
    if row["alias"] == "LGBM PRECISO":
        return "Mayor precision"
    return row["alias"]


def tuned_display_name(row: pd.Series) -> str:
    alias = str(row.get("alias", ""))
    if alias == "RADAR MAX":
        return "Modelo de maxima deteccion"
    if alias == "XGB BALANCE":
        return "Modelo de mejor equilibrio"
    if alias == "LGBM PRECISO":
        return "Modelo de mayor precision relativa"

    family = full_model_name(str(row["modelo_familia"]))
    strategy = STRATEGY_FULL_NAMES.get(str(row.get("estrategia", "")), str(row.get("estrategia", "")))
    scenario = scenario_label(str(row["escenario"]))
    threshold = float(row["threshold"])
    return f"{family} + {strategy} | {scenario} | Threshold {threshold:.2f}"


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATASET_PATH)


@st.cache_data
def load_baseline_summary() -> pd.DataFrame:
    df = pd.read_csv(BASELINE_REPORT_PATH)
    numeric_cols = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "precision_pobre",
        "recall_pobre",
        "f1_pobre",
        "support_pobre",
    ]
    df = _coerce_numeric(df, numeric_cols)
    df["alias"] = df.apply(baseline_alias, axis=1)
    df["escenario_label"] = df["feature_set"].map(scenario_label)
    return df


@st.cache_data
def load_tuning_summary() -> pd.DataFrame:
    df = pd.read_csv(TUNING_REPORT_PATH)
    numeric_cols = [
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    df = _coerce_numeric(df, numeric_cols)
    df["alias"] = df.apply(tuned_alias, axis=1)
    df["escenario_label"] = df["escenario"].map(scenario_label)
    df["story_name"] = df.apply(tuned_story_name, axis=1)
    return df


@st.cache_data
def load_top10_recall() -> pd.DataFrame:
    df = pd.read_csv(TOP10_RECALL_PATH)
    numeric_cols = ["threshold", "accuracy", "precision", "recall", "f1", "pr_auc", "tn", "fp", "fn", "tp"]
    df = _coerce_numeric(df, numeric_cols)
    df["alias"] = df.apply(tuned_alias, axis=1)
    df["escenario_label"] = df["escenario"].map(scenario_label)
    return df


@st.cache_data
def load_model_inventory() -> pd.DataFrame:
    if not MODEL_INVENTORY_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(MODEL_INVENTORY_PATH)
    if "threshold" in df.columns:
        df["threshold"] = pd.to_numeric(df["threshold"], errors="coerce")
    return df


@st.cache_data
def load_baseline_confusions() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for alias, path in BASELINE_CONFUSIONS.items():
        if path.exists():
            result[alias] = json.loads(path.read_text(encoding="utf-8"))
    return result


def parse_confusion_matrix(cm_value: str | list[list[int]]) -> list[list[int]]:
    if isinstance(cm_value, str):
        return json.loads(cm_value)
    return cm_value


def metric_heatmap(df: pd.DataFrame, id_col: str, metric_map: dict[str, str], sort_by: str | None = None) -> alt.Chart:
    source = df.copy()
    if sort_by and sort_by in source.columns:
        source = source.sort_values(sort_by, ascending=False)

    melted = source[[id_col] + list(metric_map)].melt(
        id_vars=id_col,
        value_vars=list(metric_map),
        var_name="metric",
        value_name="score",
    )
    melted["metric_label"] = melted["metric"].map(metric_map)
    melted["score_label"] = melted["score"].map(lambda x: f"{x:.2f}")
    max_value = float(melted["score"].max()) if not melted.empty else 1.0
    text_cutoff = max_value * 0.72

    heat = (
        alt.Chart(melted)
        .mark_rect(cornerRadius=6)
        .encode(
            x=alt.X("metric_label:N", title=None, sort=list(metric_map.values())),
            y=alt.Y(f"{id_col}:N", title=None, sort=list(source[id_col])),
            color=alt.Color(
                "score:Q",
                scale=alt.Scale(range=["#F6F1E8", "#E7D3AA", "#D98C3F", "#355070"]),
                legend=None,
            ),
        )
    )
    dark_text = (
        alt.Chart(melted)
        .transform_filter(alt.datum.score < text_cutoff)
        .mark_text(fontSize=12, fontWeight="bold", color="#102235")
        .encode(
            x=alt.X("metric_label:N", sort=list(metric_map.values())),
            y=alt.Y(f"{id_col}:N", sort=list(source[id_col])),
            text="score_label:N",
        )
    )
    light_text = (
        alt.Chart(melted)
        .transform_filter(alt.datum.score >= text_cutoff)
        .mark_text(fontSize=12, fontWeight="bold", color="white")
        .encode(
            x=alt.X("metric_label:N", sort=list(metric_map.values())),
            y=alt.Y(f"{id_col}:N", sort=list(source[id_col])),
            text="score_label:N",
        )
    )
    return (heat + dark_text + light_text).properties(height=max(220, len(source) * 36))


def grouped_metric_chart(df: pd.DataFrame, id_col: str, metric_map: dict[str, str], sort_by: str) -> alt.Chart:
    source = df.copy().sort_values(sort_by, ascending=False)
    melted = source[[id_col] + list(metric_map)].melt(
        id_vars=id_col,
        value_vars=list(metric_map),
        var_name="metric",
        value_name="score",
    )
    melted["metric_label"] = melted["metric"].map(metric_map)
    return (
        alt.Chart(melted)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(f"{id_col}:N", sort=list(source[id_col]), title=None),
            y=alt.Y("score:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "metric_label:N",
                title=None,
                scale=alt.Scale(
                    domain=list(metric_map.values()),
                    range=[PALETTE["teal"], PALETTE["gold"], PALETTE["coral"], PALETTE["navy"]][: len(metric_map)],
                ),
            ),
            tooltip=[id_col, "metric_label", alt.Tooltip("score:Q", format=".3f")],
            xOffset="metric_label:N",
        )
        .properties(height=340)
    )


def performance_scatter(df: pd.DataFrame, label_col: str, color_col: str) -> alt.Chart:
    source = df.copy()
    return (
        alt.Chart(source)
        .mark_circle(opacity=0.78, stroke="white", strokeWidth=1.2)
        .encode(
            x=alt.X("recall:Q", title="Recall", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("precision:Q", title="Precision", scale=alt.Scale(domain=[0, 1])),
            size=alt.Size("f1:Q", title="F1", scale=alt.Scale(range=[80, 1000])),
            color=alt.Color(
                f"{color_col}:N",
                title=None,
                scale=alt.Scale(range=[PALETTE["navy"], PALETTE["teal"], PALETTE["coral"], PALETTE["gold"]]),
            ),
            tooltip=[label_col, color_col, alt.Tooltip("recall:Q", format=".3f"), alt.Tooltip("precision:Q", format=".3f"), alt.Tooltip("f1:Q", format=".3f")],
        )
        .properties(height=360)
    )


def confusion_matrix_chart(cm_value: str | list[list[int]], title: str | None = None) -> alt.Chart:
    matrix = parse_confusion_matrix(cm_value)
    cm_df = pd.DataFrame(
        [
            {"Real": "No pobre", "Prediccion": "No pobre", "Conteo": matrix[0][0]},
            {"Real": "No pobre", "Prediccion": "Pobre", "Conteo": matrix[0][1]},
            {"Real": "Pobre", "Prediccion": "No pobre", "Conteo": matrix[1][0]},
            {"Real": "Pobre", "Prediccion": "Pobre", "Conteo": matrix[1][1]},
        ]
    )
    heat = (
        alt.Chart(cm_df)
        .mark_rect(cornerRadius=10)
        .encode(
            x=alt.X("Prediccion:N", title=None),
            y=alt.Y("Real:N", title=None),
            color=alt.Color("Conteo:Q", scale=alt.Scale(range=[PALETTE["cream"], PALETTE["gold"], PALETTE["coral"], PALETTE["navy"]]), legend=None),
        )
    )
    text = heat.mark_text(fontSize=18, fontWeight="bold", color="white").encode(text="Conteo:Q")
    chart = (heat + text).properties(width=260, height=220)
    if title:
        chart = chart.properties(title=title)
    return chart


def target_distribution_chart(df: pd.DataFrame) -> alt.Chart:
    counts = (
        df["target_pobreza_monetaria_bin"]
        .value_counts(dropna=False)
        .rename_axis("target")
        .reset_index(name="hogares")
    )
    counts["segmento"] = counts["target"].map(TARGET_LABELS)
    counts["share"] = counts["hogares"] / counts["hogares"].sum()
    return (
        alt.Chart(counts)
        .mark_arc(innerRadius=65, outerRadius=120)
        .encode(
            theta="hogares:Q",
            color=alt.Color(
                "segmento:N",
                scale=alt.Scale(domain=["No pobre", "Pobre"], range=[PALETTE["navy"], PALETTE["coral"]]),
                legend=alt.Legend(title=None),
            ),
            tooltip=["segmento", "hogares", alt.Tooltip("share:Q", format=".1%")],
        )
        .properties(height=300)
    )


def numeric_distribution_chart(df: pd.DataFrame, feature: str) -> alt.Chart:
    source = df[[feature, "target_pobreza_monetaria_bin"]].dropna().copy()
    source["grupo"] = source["target_pobreza_monetaria_bin"].map(TARGET_LABELS)
    return (
        alt.Chart(source)
        .mark_bar(opacity=0.8)
        .encode(
            x=alt.X(f"{feature}:Q", bin=alt.Bin(maxbins=25), title=feature_display_name(feature)),
            y=alt.Y("count():Q", title="Hogares"),
            color=alt.Color("grupo:N", scale=alt.Scale(domain=["No pobre", "Pobre"], range=[PALETTE["navy"], PALETTE["coral"]]), title=None),
            tooltip=[alt.Tooltip(f"{feature}:Q", bin=True), alt.Tooltip("count():Q", title="Hogares"), "grupo:N"],
        )
        .properties(height=320)
    )


def categorical_distribution_chart(df: pd.DataFrame, feature: str, top_n: int = 8) -> alt.Chart:
    source = (
        df[[feature]]
        .fillna("Sin dato")
        .value_counts()
        .reset_index(name="hogares")
        .rename(columns={feature: "categoria"})
        .head(top_n)
    )
    return (
        alt.Chart(source)
        .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
        .encode(
            x=alt.X("hogares:Q", title="Hogares"),
            y=alt.Y("categoria:N", sort="-x", title=feature_display_name(feature)),
            color=alt.value(PALETTE["teal"]),
            tooltip=["categoria:N", "hogares:Q"],
        )
        .properties(height=320)
    )


def poverty_rate_by_category_chart(df: pd.DataFrame, feature: str, top_n: int = 8) -> alt.Chart:
    source = (
        df[[feature, "target_pobreza_monetaria_bin"]]
        .dropna()
        .groupby(feature, dropna=False)
        .agg(hogares=("target_pobreza_monetaria_bin", "size"), pobreza_rate=("target_pobreza_monetaria_bin", "mean"))
        .reset_index()
        .sort_values("hogares", ascending=False)
        .head(top_n)
        .rename(columns={feature: "categoria"})
    )
    return (
        alt.Chart(source)
        .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
        .encode(
            x=alt.X("pobreza_rate:Q", title="Tasa de pobreza", axis=alt.Axis(format="%")),
            y=alt.Y("categoria:N", sort="-x", title=feature_display_name(feature)),
            color=alt.value(PALETTE["coral"]),
            tooltip=["categoria:N", alt.Tooltip("pobreza_rate:Q", format=".1%"), "hogares:Q"],
        )
        .properties(height=320)
    )


def split_features(features: list[str]) -> tuple[list[str], list[str]]:
    midpoint = math.ceil(len(features) / 2)
    return features[:midpoint], features[midpoint:]


def dynamic_display_map(df: pd.DataFrame, col: str) -> dict[str, object]:
    values = df[col].dropna().astype(str).sort_values().unique().tolist()
    return {value: value for value in values}


def render_feature_input(feature_name: str, cfg: dict[str, Any], df_reference: pd.DataFrame) -> Any:
    label = cfg["label"]
    input_type = cfg["input_type"]

    if input_type == "number_input":
        return st.number_input(
            label,
            min_value=cfg.get("min_value", 0),
            step=cfg.get("step", 1),
            format=cfg.get("format"),
            key=f"input_{feature_name}",
        )

    if input_type == "selectbox":
        options = list(cfg["display_map"].keys())
        selected = st.selectbox(label, options, key=f"input_{feature_name}")
        return cfg["display_map"][selected]

    if input_type == "selectbox_dynamic":
        display_map = dynamic_display_map(df_reference, feature_name)
        options = list(display_map.keys())
        selected = st.selectbox(label, options, key=f"input_{feature_name}")
        return display_map[selected]

    raise ValueError(f"Tipo de input no soportado para {feature_name}: {input_type}")
