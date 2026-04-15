from __future__ import annotations

import ast
import altair as alt
import pandas as pd
from pathlib import Path
from typing import Iterable

import streamlit as st
from config import PROJECT_ROOT

from ui_helpers import (
    PALETTE,
    baseline_display_name,
    categorical_distribution_chart,
    confusion_matrix_chart,
    feature_display_name,
    format_pct,
    grouped_metric_chart,
    inject_global_styles,
    metric_heatmap,
    numeric_distribution_chart,
    performance_scatter,
    poverty_rate_by_category_chart,
    render_feature_input,
    render_hero,
    render_metric_card,
    render_section_intro,
    render_sidebar_summary,
    render_story_card,
    split_features,
    target_distribution_chart,
    tuned_display_name,
)


def inject_styles() -> None:
    inject_global_styles()


def render_sidebar_context(df) -> None:
    render_sidebar_summary(df)


def render_step_grid(steps: Iterable[tuple[str, str]], columns: int = 3) -> None:
    steps = list(steps)
    for start in range(0, len(steps), columns):
        cols = st.columns(columns)
        chunk = steps[start : start + columns]
        for col, (title, body) in zip(cols, chunk):
            with col:
                render_story_card(title, body)


def render_artifact_status(label: str, path: Path) -> None:
    state = "Disponible" if path.exists() else "Pendiente"
    try:
        note = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        note = path.name
    render_metric_card(label, state, note)


def render_placeholder_message(title: str, body: str) -> None:
    render_story_card(title, body)


def confusion_matrix_chart(cm_value, title: str | None = None) -> alt.Chart:
    if isinstance(cm_value, str):
        matrix = pd.DataFrame(ast.literal_eval(cm_value))
        matrix = matrix.values.tolist()
    else:
        matrix = cm_value

    cm_df = pd.DataFrame(
        [
            {"Real": "No pobre", "Prediccion": "No pobre", "Conteo": int(matrix[0][0])},
            {"Real": "No pobre", "Prediccion": "Pobre", "Conteo": int(matrix[0][1])},
            {"Real": "Pobre", "Prediccion": "No pobre", "Conteo": int(matrix[1][0])},
            {"Real": "Pobre", "Prediccion": "Pobre", "Conteo": int(matrix[1][1])},
        ]
    )
    max_value = max(cm_df["Conteo"].max(), 1)

    base = (
        alt.Chart(cm_df)
        .mark_rect(cornerRadius=12)
        .encode(
            x=alt.X("Prediccion:N", title=None),
            y=alt.Y("Real:N", title=None),
            color=alt.Color(
                "Conteo:Q",
                scale=alt.Scale(domain=[0, max_value], range=["#F7F2E9", "#D9EBC3", "#2F6F4F"]),
                legend=None,
            ),
        )
    )
    dark_text = (
        alt.Chart(cm_df)
        .transform_filter(alt.datum.Conteo < (max_value * 0.55))
        .mark_text(fontSize=20, fontWeight="bold", color="#102235")
        .encode(x="Prediccion:N", y="Real:N", text="Conteo:Q")
    )
    light_text = (
        alt.Chart(cm_df)
        .transform_filter(alt.datum.Conteo >= (max_value * 0.55))
        .mark_text(fontSize=20, fontWeight="bold", color="white")
        .encode(x="Prediccion:N", y="Real:N", text="Conteo:Q")
    )
    chart = (base + dark_text + light_text).properties(width=300, height=250)
    if title:
        chart = chart.properties(title=title)
    return chart


def feature_influence_matrix_chart(long_df: pd.DataFrame) -> alt.Chart:
    source = long_df.copy()
    source["score_label"] = source["influencia"].map(lambda x: f"{x:.3f}")
    order = source.groupby("variable_label")["influencia"].mean().sort_values(ascending=False).index.tolist()
    heat = (
        alt.Chart(source)
        .mark_rect(cornerRadius=6)
        .encode(
            x=alt.X("modelo:N", title=None),
            y=alt.Y("variable_label:N", title=None, sort=order),
            color=alt.Color(
                "influencia:Q",
                scale=alt.Scale(domain=[0, 1], range=["#D73027", "#F7F2E9", "#1A9850"]),
                legend=alt.Legend(title="Influencia normalizada", orient="right"),
            ),
            tooltip=[
                alt.Tooltip("variable_label:N", title="Variable"),
                alt.Tooltip("modelo:N", title="Modelo"),
                alt.Tooltip("influencia:Q", title="Influencia", format=".3f"),
            ],
        )
    )
    dark_text = (
        alt.Chart(source)
        .transform_filter(alt.datum.influencia < 0.62)
        .mark_text(fontSize=12, fontWeight="bold", color="#12263A")
        .encode(x="modelo:N", y=alt.Y("variable_label:N", sort=order), text="score_label:N")
    )
    light_text = (
        alt.Chart(source)
        .transform_filter(alt.datum.influencia >= 0.62)
        .mark_text(fontSize=12, fontWeight="bold", color="white")
        .encode(x="modelo:N", y=alt.Y("variable_label:N", sort=order), text="score_label:N")
    )
    return (heat + dark_text + light_text).properties(height=420)


def roc_curve_chart(curves_df: pd.DataFrame, selected_model: str) -> alt.Chart:
    source = curves_df.loc[curves_df["modelo"] == selected_model].copy()
    diagonal = pd.DataFrame({"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]})
    base = (
        alt.Chart(source)
        .mark_line(color="#0E7490", strokeWidth=3)
        .encode(
            x=alt.X("fpr:Q", title="Tasa de falsos positivos", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("tpr:Q", title="Tasa de verdaderos positivos", scale=alt.Scale(domain=[0, 1])),
            tooltip=[
                alt.Tooltip("fpr:Q", format=".3f", title="FPR"),
                alt.Tooltip("tpr:Q", format=".3f", title="TPR"),
            ],
        )
    )
    ref = (
        alt.Chart(diagonal)
        .mark_line(color="#33415C", strokeDash=[6, 4], strokeWidth=1.5)
        .encode(x="fpr:Q", y="tpr:Q")
    )
    return (base + ref).properties(height=360)


def anova_top_features_chart(df: pd.DataFrame, top_n: int = 15) -> alt.Chart:
    source = df.copy().sort_values("f_score_max", ascending=False).head(top_n)
    return (
        alt.Chart(source)
        .mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6)
        .encode(
            x=alt.X("f_score_max:Q", title="F-score maximo (ANOVA)"),
            y=alt.Y("descripcion:N", sort="-x", title=None),
            color=alt.Color(
                "dimension_analitica:N",
                title="Dimension",
                scale=alt.Scale(range=["#0E7490", "#C98B2B", "#D1603D", "#355070", "#6D597A", "#2F6F4F"]),
            ),
            tooltip=[
                alt.Tooltip("descripcion:N", title="Variable"),
                alt.Tooltip("variable:N", title="Codigo"),
                alt.Tooltip("f_score_max:Q", title="F-score max", format=".3f"),
                alt.Tooltip("p_value_min:Q", title="p-value min", format=".2e"),
                alt.Tooltip("expanded_features:Q", title="Dummies evaluadas"),
            ],
        )
        .properties(height=max(360, top_n * 28))
    )


__all__ = [
    "anova_top_features_chart",
    "PALETTE",
    "baseline_display_name",
    "categorical_distribution_chart",
    "confusion_matrix_chart",
    "feature_display_name",
    "format_pct",
    "grouped_metric_chart",
    "inject_styles",
    "feature_influence_matrix_chart",
    "metric_heatmap",
    "numeric_distribution_chart",
    "performance_scatter",
    "poverty_rate_by_category_chart",
    "render_artifact_status",
    "render_feature_input",
    "render_hero",
    "render_metric_card",
    "render_placeholder_message",
    "render_section_intro",
    "render_sidebar_context",
    "render_step_grid",
    "render_story_card",
    "roc_curve_chart",
    "split_features",
    "target_distribution_chart",
    "tuned_display_name",
]
