from __future__ import annotations

import pandas as pd
import streamlit as st

from loaders import (
    load_all48_experiment_safe,
    load_anova_ranking_safe,
    load_feature_selection_top30_safe,
    load_leakage_risk_safe,
    load_selectkbest_summary_safe,
    load_top30_experiment_safe,
    load_top30_fe_experiment_safe,
)
from ui import anova_top_features_chart, render_hero, render_section_intro, render_story_card


top30_features = load_feature_selection_top30_safe()
anova_ranking = load_anova_ranking_safe()
selectkbest_summary = load_selectkbest_summary_safe()
leakage = load_leakage_risk_safe()
top30_experiment = load_top30_experiment_safe()
all48_experiment = load_all48_experiment_safe()
top30_fe_experiment = load_top30_fe_experiment_safe()

render_hero(
    "Preparacion metodologica y feature selection",
    "Por que top30 full quedo como set oficial",
    "Esta etapa explica como se definio el target, como se reviso leakage y por que la seleccion de variables no se baso solo en tamaño, sino en desempeno y consistencia metodologica.",
)

render_section_intro(
    "Dos decisiones importaron mucho",
    "Primero, no evaluar solo con accuracy en un problema desbalanceado. Segundo, comparar varios tamanos de set de variables hasta encontrar un punto defendible y util.",
)

st.markdown("### Preparacion metodologica")
prep_cols = st.columns(3)
with prep_cols[0]:
    render_story_card("Target", "Se trabajo con pobreza monetaria oficial del INEI a nivel hogar.")
with prep_cols[1]:
    render_story_card("Leakage", "Se revisaron variables monetarias y otras senales cercanas al target para identificar riesgos metodologicos.")
with prep_cols[2]:
    render_story_card("Metrica guia", "No se eligio por accuracy sola; se priorizaron precision, recall, F1 y matrices de confusion.")

st.markdown("### Resumen corto de como se eligio el Top 30")
summary_cols = st.columns(5)
summary_text = [
    ("Paso 1", "Se usaron 48 variables candidatas provenientes del diccionario del proyecto."),
    ("Paso 2", "Se aplico SelectKBest con f_classif, es decir ANOVA para clasificacion."),
    ("Paso 3", "Las variables categoricas se expandieron a dummies antes de calcular el ranking."),
    ("Paso 4", "Los resultados se reagruparon al nivel de variable original usando F-score maximo y p-value minimo."),
    ("Paso 5", "Top 30 se valido luego contra Top 20, Top 48 y Top 30 + feature engineering."),
]
for col, (title, body) in zip(summary_cols, summary_text):
    with col:
        render_story_card(title, body)

st.markdown("### Evidencia estadistica inicial: ANOVA")
if anova_ranking.empty:
    st.warning("No se encontro variable_ranking_anova.csv para mostrar el ranking estadistico.")
else:
    st.altair_chart(anova_top_features_chart(anova_ranking, top_n=15), width="stretch")
    render_story_card(
        "Como leer este grafico",
        "El grafico muestra las variables originales con mayor F-score maximo segun ANOVA. "
        "Un F-score alto indica mayor capacidad de separar hogares pobres y no pobres. "
        "En variables categoricas, primero se evaluaron sus dummies y luego se agrego el resultado al nivel de la variable original.",
    )
    with st.expander("Ver ranking ANOVA resumido"):
        st.dataframe(
            anova_ranking[
                [
                    "variable",
                    "descripcion",
                    "dimension_analitica",
                    "expanded_features",
                    "f_score_max",
                    "p_value_min",
                    "rank_f_score",
                ]
            ].head(20),
            width="stretch",
        )

st.markdown("### La decision sobre K")
if selectkbest_summary:
    k_cols = st.columns(4)
    k_metrics = [
        ("Variables candidatas", str(selectkbest_summary.get("n_candidate_features", "-"))),
        ("Features expandidas", str(selectkbest_summary.get("n_expanded_features", "-"))),
        ("Significativas al 5%", str(selectkbest_summary.get("n_significant_features_5pct", "-"))),
        ("K recomendado", str(selectkbest_summary.get("recommended_k", "-"))),
    ]
    for col, (title, value) in zip(k_cols, k_metrics):
        with col:
            render_story_card(title, value)

render_story_card(
    "Que K propongo y por que",
    "La propuesta metodologica sigue siendo K = 30. "
    "Es un punto medio defendible: mejora frente a Top 20, no fue superado por Top 48 y las pruebas con feature engineering no mostraron una ganancia neta. "
    "En otras palabras, Top 30 conserva suficiente senal estadistica y al mismo tiempo evita cargar el modelo con variables extra que no mejoraron el desempeno final.",
)

st.markdown("### Comparacion de feature sets")
comparison_rows: list[dict[str, object]] = []
if not top30_experiment.empty:
    top20 = top30_experiment.loc[top30_experiment["feature_set"] == "top20_full"].sort_values("holdout_f1", ascending=False).iloc[0]
    top30 = top30_experiment.loc[top30_experiment["feature_set"] == "top30_full"].sort_values("holdout_f1", ascending=False).iloc[0]
    comparison_rows.extend(
        [
            {"feature_set": "top20_full", "holdout_precision": top20["holdout_precision"], "holdout_recall": top20["holdout_recall"], "holdout_f1": top20["holdout_f1"], "lectura": "Base corta de referencia."},
            {"feature_set": "top30_full", "holdout_precision": top30["holdout_precision"], "holdout_recall": top30["holdout_recall"], "holdout_f1": top30["holdout_f1"], "lectura": "Set oficial final."},
        ]
    )
if not all48_experiment.empty:
    all48 = all48_experiment.loc[all48_experiment["feature_set"] == "all48_full"].sort_values("holdout_f1", ascending=False).iloc[0]
    comparison_rows.append({"feature_set": "top48_full", "holdout_precision": all48["holdout_precision"], "holdout_recall": all48["holdout_recall"], "holdout_f1": all48["holdout_f1"], "lectura": "Mas variables, pero sin superar al top30."})
if not top30_fe_experiment.empty:
    fe = top30_fe_experiment.loc[top30_fe_experiment["feature_set"] == "top30_full_plus_fe"].sort_values("holdout_f1", ascending=False).iloc[0]
    comparison_rows.append({"feature_set": "top30_full + FE", "holdout_precision": fe["holdout_precision"], "holdout_recall": fe["holdout_recall"], "holdout_f1": fe["holdout_f1"], "lectura": "Feature engineering probado, pero no superior."})

if comparison_rows:
    comparison_df = pd.DataFrame(comparison_rows)
    st.dataframe(comparison_df, width="stretch")
    render_story_card(
        "Decision oficial",
        "Top30 full se mantuvo como set oficial porque mejoro frente a top20, no fue superado por top48 y las pruebas de feature engineering no entregaron una mejora neta.",
    )
else:
    st.warning("No se encontraron reportes comparativos de feature selection para poblar esta seccion.")

st.markdown("### Variables oficiales top30 full")
if top30_features.empty:
    st.warning("No se encontro selected_features_top30.csv en reports/feature_selection.")
else:
    st.dataframe(
        top30_features[["variable", "dimension_analitica", "descripcion"]].rename(columns={"variable": "feature"}),
        width="stretch",
    )

st.markdown("### Riesgo de leakage")
if leakage.empty:
    render_story_card("Leakage", "No se encontro la tabla de riesgo de fuga de informacion.")
else:
    st.dataframe(leakage.head(12), width="stretch")
