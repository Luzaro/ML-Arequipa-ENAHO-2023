from __future__ import annotations

import streamlit as st

from config import OFFICIAL_MODEL_FICHA, OFFICIAL_MODEL_LABEL, OFFICIAL_MODEL_RESULTS
from loaders import load_roc_curve_data_safe, load_top10_feature_influence_safe, load_tuning_safe
from ui import (
    feature_influence_matrix_chart,
    render_hero,
    render_section_intro,
    render_story_card,
    roc_curve_chart,
)


tuning = load_tuning_safe()
roc_payload = load_roc_curve_data_safe()
importance_payload = load_top10_feature_influence_safe()
best_final = tuning.sort_values(["f1", "recall", "precision"], ascending=[False, False, False]).iloc[0] if not tuning.empty else None

render_hero(
    "Resultados y conclusiones",
    "Que deja el proyecto y como debe leerse",
    "Esta pagina resume los hallazgos principales, el rol de las variables monetarias, el trade-off entre precision y recall y las limitaciones que conviene reconocer.",
)

render_section_intro(
    "Cierre del proyecto",
    "La solucion final combina un pipeline reproducible, un set oficial de variables, un modelo recomendado y una app que explica el proceso de forma metodologica.",
)

st.markdown("### Hallazgos principales")
cols = st.columns(3)
with cols[0]:
    render_story_card("Hallazgo 1", "El ETL y la consolidacion a nivel hogar fueron determinantes para construir una base coherente para clasificacion.")
with cols[1]:
    render_story_card("Hallazgo 2", "Top30 full se mantuvo como mejor set practico al comparar top20, top30, top48 y pruebas de feature engineering.")
with cols[2]:
    render_story_card("Hallazgo 3", "Las variables monetarias agregaron poder explicativo, pero el valor del modelo final tambien descansa en senales estructurales de vivienda, servicios y salud.")

st.markdown("### Modelo final recomendado")
if best_final is None:
    render_story_card("Modelo final", OFFICIAL_MODEL_LABEL)
else:
    render_story_card("Recomendacion", OFFICIAL_MODEL_LABEL)
    ficha_cols = st.columns(4)
    for col, (label, value) in zip(ficha_cols, OFFICIAL_MODEL_FICHA.items()):
        with col:
            render_story_card(label, value)
    result_cols = st.columns(3)
    for col, (label, value) in zip(result_cols, OFFICIAL_MODEL_RESULTS.items()):
        with col:
            render_story_card(label, value)

st.markdown("### Top 10 Variables independientes mas influyentes en los modelos de Machine Learning")
importance_matrix = importance_payload.get("matrix")
importance_long = importance_payload.get("long")
importance_sources = importance_payload.get("sources")
if importance_matrix is None or importance_matrix.empty or importance_long is None or importance_long.empty:
    st.warning("No se pudieron construir las importancias de variables a partir de los modelos exportados disponibles.")
else:
    st.altair_chart(feature_influence_matrix_chart(importance_long), width="stretch")
    render_story_card(
        "Como leer esta matriz",
        "Cada columna representa un modelo exportado y cada fila una variable original del Top 30 Full. "
        "Los valores son influencias normalizadas entre 0 y 1 dentro de cada modelo: verde indica mayor peso relativo, tonos claros una influencia intermedia y rojo menor aporte relativo. "
        "Esta matriz resume senales repetidas entre modelos y ayuda a identificar que variables sostienen la capacidad de clasificacion.",
    )
    render_story_card(
        "Lectura metodologica",
        "La comparacion se construyo con modelos exportados comparables sobre Top 30 Full. "
        "No debe interpretarse como causalidad, sino como una evidencia de que variables monetarias, vivienda, conectividad y salud aportan de manera consistente a la separacion entre hogares pobres y no pobres.",
    )
    if importance_sources is not None and not importance_sources.empty:
        with st.expander("Modelos usados para construir la matriz"):
            st.dataframe(importance_sources, width="stretch")

st.markdown("### Curva ROC")
roc_curves = roc_payload.get("curves")
roc_summary = roc_payload.get("summary")
if roc_curves is None or roc_curves.empty or roc_summary is None or roc_summary.empty:
    st.warning("No se pudo construir la curva ROC con los modelos finales exportados.")
else:
    selected_model = st.selectbox("Modelo para curva ROC", roc_summary["modelo"].tolist())
    auc_value = float(roc_summary.loc[roc_summary["modelo"] == selected_model, "auc"].iloc[0])
    st.altair_chart(roc_curve_chart(roc_curves, selected_model), width="stretch")
    render_story_card(
        "Interpretacion ROC",
        f"El area bajo la curva ROC (AUC) para este modelo es {auc_value:.3f}. "
        "Mientras mas se acerque la curva al extremo superior izquierdo, mejor es la capacidad del modelo para discriminar hogares pobres y no pobres a traves de distintos puntos de corte. "
        "La linea diagonal representa un comportamiento aleatorio. Esta curva se presenta como visualizacion referencial a partir de los modelos exportados y debe leerse junto con las metricas holdout reportadas en la fase de tuning.",
    )

st.markdown("### Limitaciones")
limit_cols = st.columns(3)
with limit_cols[0]:
    render_story_card("Trade-off", "Mejorar precision sigue costando recall; no existe un unico modelo mejor para todos los objetivos.")
with limit_cols[1]:
    render_story_card("Alcance", "El target final es pobreza monetaria oficial; la app no mide pobreza multidimensional ni reemplaza medicion institucional.")
with limit_cols[2]:
    render_story_card("Generalizacion", "Los resultados corresponden a Arequipa 2023 y deben revalidarse antes de extrapolarlos a otros contextos.")

st.markdown("### Trabajo futuro")
render_story_card(
    "Siguientes pasos",
    "Profundizar benchmark sin leakage para top30, explorar calibracion de probabilidades, mejorar precision con nuevas variables estructurales y desplegar la app en un entorno publico.",
)
