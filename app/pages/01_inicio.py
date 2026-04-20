from __future__ import annotations

import streamlit as st

from config import (
    OFFICIAL_MODEL_FICHA,
    OFFICIAL_MODEL_LABEL,
    OFFICIAL_MODEL_RESULTS,
    REGION_LABEL,
    SOURCE_LABEL,
    TARGET_LABEL,
    UNIT_LABEL,
)
from narrative import EXECUTIVE_FLOW
from loaders import load_dataset_safe
from ui import format_pct, render_hero, render_metric_card, render_section_intro, render_step_grid, render_story_card


df = load_dataset_safe()
poverty_rate = float(df["target_pobreza_monetaria_bin"].mean()) if (not df.empty and "target_pobreza_monetaria_bin" in df.columns) else 0.0

render_hero(
    "Resumen ejecutivo",
    "Predicción de pobreza monetaria en hogares de Arequipa 2023",
    "Esta aplicación explica el proyecto completo: desde la adquisición de datos ENAHO hasta la selección final de modelos y la predicción interactiva de un hogar.",
)

cols = st.columns(5)
with cols[0]:
    render_metric_card("Fuente", SOURCE_LABEL, "Microdatos oficiales del INEI.")
with cols[1]:
    render_metric_card("Región", REGION_LABEL, "Cobertura analítica del proyecto.")
with cols[2]:
    render_metric_card("Unidad", UNIT_LABEL, "La predicción se realiza a nivel hogar.")
with cols[3]:
    render_metric_card("Target", "Pobreza monetaria", TARGET_LABEL)
with cols[4]:
    render_metric_card("Muestra", f"{len(df):,}".replace(",", "."), f"Tasa de pobreza: {format_pct(poverty_rate)}")

render_section_intro(
    "Qué hace y qué no hace el modelo",
    "La solución aproxima la clasificación oficial de pobreza monetaria con microdatos ENAHO. Sirve para análisis, comparación de escenarios y demostración metodológica, pero no reemplaza la medición oficial del INEI.",
)

left, right = st.columns([1.1, 1])
with left:
    render_story_card(
        "Problema social",
        "La pobreza monetaria requiere herramientas que ayuden a identificar patrones de vulnerabilidad con datos públicos, sin perder trazabilidad metodológica.",
    )
    render_story_card(
        "Objetivo del proyecto",
        "Construir una solución de clasificación supervisada que permita comparar modelos, justificar decisiones y terminar en una app presentable para exposición de resultados.",
    )
with right:
    render_story_card(
        "Mensaje central",
        "La app no solo predice. También explica cómo se construyó la base, por qué se eligió top30 full, cómo se trató el desbalance y por qué existen distintos modelos finales según el objetivo.",
    )

st.markdown("### Flujo metodológico")
render_step_grid(EXECUTIVE_FLOW, columns=3)

st.markdown("### Recomendación principal")
render_story_card(
    "Modelo recomendado hoy",
    f"{OFFICIAL_MODEL_LABEL}. Se recomienda para presentación final porque ofrece el mejor equilibrio entre precisión, recall y F1 dentro del set oficial Top 30 Full.",
)

ficha_cols = st.columns(4)
for col, (label, value) in zip(ficha_cols, OFFICIAL_MODEL_FICHA.items()):
    with col:
        render_metric_card(label, value)

result_cols = st.columns(3)
for col, (label, value) in zip(result_cols, OFFICIAL_MODEL_RESULTS.items()):
    with col:
        render_metric_card(label, value, "Resultado holdout")
