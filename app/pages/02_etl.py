from __future__ import annotations

import streamlit as st

from config import DATASET_PATH, ETL_MODULES
from narrative import ETL_STEPS
from loaders import load_dataset_safe
from ui import render_artifact_status, render_hero, render_section_intro, render_step_grid, render_story_card


df = load_dataset_safe()

render_hero(
    "Pipeline ETL",
    "Como los datos crudos se convirtieron en una base lista para modelar",
    "Esta etapa organiza la adquisicion, integracion y validacion de datos ENAHO 2023 hasta llegar a una base final a nivel hogar en data/processed.",
)

render_section_intro(
    "Una etapa clave antes de cualquier modelo",
    "Sin ETL no habia una base consistente para comparar pobreza y no pobreza. El trabajo fue integrar modulos heterogeneos y traducirlos a una sola unidad de analisis: el hogar.",
)

st.markdown("### Flujo Extract - Transform - Load")
render_step_grid(ETL_STEPS, columns=3)

st.markdown("### Modulos integrados")
render_step_grid([(module.split(":")[0], module.split(":")[1].strip()) for module in ETL_MODULES], columns=3)

st.markdown("### Resultado del ETL")
cols = st.columns(3)
with cols[0]:
    render_artifact_status("Dataset final", DATASET_PATH)
with cols[1]:
    render_story_card("Filas finales", f"Se consolidaron {len(df):,} hogares.".replace(",", ".") if not df.empty else "Archivo pendiente.")
with cols[2]:
    render_story_card("Columnas finales", f"La base final contiene {df.shape[1]} columnas." if not df.empty else "Archivo pendiente.")

st.markdown("### Tabla resumen del dataset final")
if df.empty:
    st.warning("No se encontro la base final procesada. Conecta el archivo en data/processed para habilitar esta vista.")
else:
    preview_cols = [col for col in df.columns[:12]]
    st.dataframe(df[preview_cols].head(10), width="stretch")
    render_story_card(
        "Lectura metodologica",
        "El ETL deja lista la materia prima del proyecto: una sola tabla a nivel hogar sobre la cual luego se construyen el EDA, la seleccion de variables y el modelado.",
    )
