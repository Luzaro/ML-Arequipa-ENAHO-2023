from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from config import OFFICIAL_MODEL_OPTIONS
from feature_config import FEATURE_CONFIG, TOP30_FULL
from loaders import load_dataset_safe, load_model_safe, load_tuning_safe
from ui import format_pct, render_feature_input, render_hero, render_metric_card, render_section_intro, render_story_card, split_features


df_reference = load_dataset_safe()
tuning = load_tuning_safe()


def tuned_lookup(label: str) -> dict[str, float | str]:
    if tuning.empty or label not in OFFICIAL_MODEL_OPTIONS:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "display_name": label}
    option = OFFICIAL_MODEL_OPTIONS[label]
    threshold = float(option["threshold"])
    algorithm = str(option["algorithm"])
    strategy = "weighted" if "ponder" in str(option["strategy_label"]).lower() else "smote"
    row = tuning.loc[
        (tuning["modelo_familia"] == algorithm)
        & (tuning["estrategia"] == strategy)
        & (tuning["threshold"].round(4) == round(threshold, 4))
    ].iloc[0]
    return {
        "precision": row["precision"],
        "recall": row["recall"],
        "f1": row["f1"],
        "display_name": label,
    }


model_label = st.selectbox("Modelo a utilizar", list(OFFICIAL_MODEL_OPTIONS.keys()))
selected_model = OFFICIAL_MODEL_OPTIONS[model_label]
metrics = tuned_lookup(model_label)

render_hero(
    "Predicción interactiva",
    "Simulador de hogar",
    "Selecciona un modelo, completa las variables del hogar y observa la clasificación estimada junto con su probabilidad y una lectura metodológica del resultado.",
)

render_section_intro(
    "Cómo usar esta página",
    "La predicción se basa en modelos ya entrenados y serializados. No se reentrena nada desde Streamlit. El objetivo es mostrar cómo cambia la decisión según el escenario elegido.",
)

cols = st.columns(4)
with cols[0]:
    render_metric_card("Modelo", model_label, selected_model["technical_name"])
with cols[1]:
    render_metric_card("Precisión esperada", format_pct(metrics["precision"]), "Qué tan selectivo suele ser este escenario.")
with cols[2]:
    render_metric_card("Recall esperado", format_pct(metrics["recall"]), "Qué tanta cobertura logra sobre hogares pobres.")
with cols[3]:
    render_metric_card("Enfoque", selected_model["story"], selected_model["note"])

feature_left, feature_right = split_features(TOP30_FULL)
inputs: dict[str, object] = {}

with st.form("prediction_form"):
    c1, c2 = st.columns(2)
    with c1:
        for feature in feature_left:
            inputs[feature] = render_feature_input(feature, FEATURE_CONFIG[feature], df_reference)
    with c2:
        for feature in feature_right:
            inputs[feature] = render_feature_input(feature, FEATURE_CONFIG[feature], df_reference)
    submitted = st.form_submit_button("Calcular predicción")

model_path: Path = selected_model["path"]
pipeline = load_model_safe(model_path)
if pipeline is None:
    st.warning(f"No se encontró el modelo serializado en {model_path}.")

if submitted and pipeline is not None:
    input_df = pd.DataFrame([inputs])
    prob = float(pipeline.predict_proba(input_df)[0, 1])
    pred = 1 if prob >= float(selected_model["threshold"]) else 0

    result_cols = st.columns(3)
    with result_cols[0]:
        render_metric_card("Clasificación", "Pobre" if pred == 1 else "No pobre", f"Threshold aplicado: {selected_model['threshold']:.2f}")
    with result_cols[1]:
        render_metric_card("Probabilidad", format_pct(prob), "Probabilidad estimada de pertenecer a la clase pobre.")
    with result_cols[2]:
        render_metric_card("Advertencia", "Uso metodológico", "Esta salida aproxima la clasificación oficial, no reemplaza al INEI.")

    if pred == 1:
        st.error("El hogar queda por encima del threshold y se clasifica como pobre en el escenario seleccionado.")
    else:
        st.success("El hogar queda por debajo del threshold y se clasifica como no pobre en el escenario seleccionado.")

    with st.expander("Ver fila de entrada utilizada"):
        st.dataframe(input_df, width="stretch")
