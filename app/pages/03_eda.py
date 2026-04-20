from __future__ import annotations

import pandas as pd
import streamlit as st

from loaders import (
    load_dataset_safe,
    load_eda_categorical_by_target_safe,
    load_eda_numeric_by_target_safe,
    load_eda_target_correlations_safe,
    load_target_summary_safe,
)
from ui import (
    categorical_distribution_chart,
    feature_display_name,
    numeric_distribution_chart,
    poverty_rate_by_category_chart,
    render_hero,
    render_section_intro,
    render_story_card,
    target_distribution_chart,
)


df = load_dataset_safe()
target_summary = load_target_summary_safe()
numeric_by_target = load_eda_numeric_by_target_safe()
categorical_by_target = load_eda_categorical_by_target_safe()
target_corr = load_eda_target_correlations_safe()

render_hero(
    "EDA y perfil del dato",
    "Qué reveló el análisis exploratorio antes de modelar",
    "El EDA se usó para entender la estructura del dataset, el comportamiento del target y las señales más útiles para diferenciar pobreza y no pobreza.",
)

render_section_intro(
    "El EDA no fue un trámite",
    "Esta etapa permitió ordenar variables, detectar posibles redundancias y justificar por qué ciertas dimensiones del hogar debían entrar al modelado.",
)

col_left, col_right = st.columns([1, 1.1])
with col_left:
    st.markdown("### Distribución del target")
    if df.empty:
        st.warning("No se encontró el dataset procesado para construir esta vista.")
    else:
        st.altair_chart(target_distribution_chart(df), width="stretch")
with col_right:
    st.markdown("### Lectura del target")
    if target_summary.empty:
        render_story_card("Target", "No se encontró el resumen del target en reports/eda.")
    else:
        for row in target_summary.itertuples():
            label = "No pobre" if int(row.target) == 0 else "Pobre"
            render_story_card(label, f"Hogares: {row.n} | Participación: {row.pct:.2f}%")

st.markdown("### Variables numéricas y categóricas")
meta_cols = st.columns(3)
numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist() if not df.empty else []
categorical_cols = [col for col in df.columns if col not in numeric_cols] if not df.empty else []
with meta_cols[0]:
    render_story_card("Variables numéricas", str(len(numeric_cols)))
with meta_cols[1]:
    render_story_card("Variables categóricas", str(len(categorical_cols)))
with meta_cols[2]:
    render_story_card("Insight", "Se observan señales en vivienda, servicios, salud, composición e ingresos.")

if not df.empty:
    st.markdown("### Exploración visual")
    num_options = [col for col in numeric_cols if col != "target_pobreza_monetaria_bin"][:20]
    cat_options = [col for col in categorical_cols if col != "target_pobreza_monetaria_bin"][:20]
    col1, col2 = st.columns(2)
    with col1:
        selected_numeric = st.selectbox("Variable numérica", num_options, index=0 if num_options else None)
        if selected_numeric:
            st.altair_chart(numeric_distribution_chart(df, selected_numeric), width="stretch")
    with col2:
        selected_categorical = st.selectbox("Variable categórica", cat_options, index=0 if cat_options else None)
        if selected_categorical:
            st.altair_chart(categorical_distribution_chart(df, selected_categorical), width="stretch")
            st.altair_chart(poverty_rate_by_category_chart(df, selected_categorical), width="stretch")

st.markdown("### Hallazgos clave")
hallazgos = []
if not target_corr.empty:
    hallazgos.extend(target_corr.head(3)["feature"].astype(str).tolist() if "feature" in target_corr.columns else [])
if numeric_by_target.empty and categorical_by_target.empty:
    render_story_card("Hallazgos", "No se encontraron tablas EDA detalladas en reports/eda.")
else:
    render_story_card(
        "Interpretación",
        "El EDA mostró que la pobreza no depende de una sola variable. La señal aparece combinando condiciones monetarias, de vivienda, servicios y composición del hogar.",
    )
    if hallazgos:
        render_story_card("Variables destacadas", ", ".join(feature_display_name(item) for item in hallazgos[:3]))
