from __future__ import annotations

import streamlit as st

from loaders import load_baseline_safe
from ui import (
    baseline_display_name,
    confusion_matrix_chart,
    grouped_metric_chart,
    metric_heatmap,
    render_hero,
    render_section_intro,
    render_story_card,
)
from ui_helpers import load_baseline_confusions


baseline = load_baseline_safe()
baseline_confusions = load_baseline_confusions()

render_hero(
    "Modelado base",
    "El punto de partida antes del ajuste fino",
    "El baseline permite entender familias de modelos, detectar senales utiles y justificar por que el tuning era necesario en un problema con clases desbalanceadas.",
)

if baseline.empty:
    st.warning("No se encontro el resumen baseline en reports/scenario_b_modeling.")
else:
    baseline["display_name"] = baseline.apply(baseline_display_name, axis=1)
    best_recall = baseline.sort_values(["recall_pobre", "f1_pobre", "precision_pobre"], ascending=[False, False, False]).iloc[0]
    best_balance = baseline.sort_values(["f1_pobre", "recall_pobre", "precision_pobre"], ascending=[False, False, False]).iloc[0]

    render_section_intro(
        "Por que importo esta etapa",
        "El baseline no buscaba el modelo final. Buscaba mostrar desde donde partiamos y que tipo de modelo capturaba mejor la senal antes de tratar el desbalance.",
    )

    cols = st.columns(2)
    with cols[0]:
        render_story_card("Mejor recall baseline", f"{baseline_display_name(best_recall)}\n\nRecall pobre: {best_recall['recall_pobre']:.3f}")
    with cols[1]:
        render_story_card("Mejor equilibrio baseline", f"{baseline_display_name(best_balance)}\n\nF1 pobre: {best_balance['f1_pobre']:.3f}")

    chart_df = baseline[["display_name", "accuracy", "precision_pobre", "recall_pobre", "f1_pobre"]].copy()
    metric_map = {"accuracy": "Accuracy", "precision_pobre": "Precision", "recall_pobre": "Recall", "f1_pobre": "F1"}

    st.markdown("### Comparacion baseline")
    st.altair_chart(metric_heatmap(chart_df, "display_name", metric_map, sort_by="recall_pobre"), width="stretch")
    st.altair_chart(grouped_metric_chart(chart_df, "display_name", metric_map, sort_by="recall_pobre"), width="stretch")

    st.markdown("### Matrices de confusion")
    available_aliases = [alias for alias in ["LR-30", "RF-30", "XGB-30"] if alias in baseline_confusions]
    if available_aliases:
        alias_to_name = {row["alias"]: row["display_name"] for _, row in baseline.iterrows()}
        selected_alias = st.selectbox("Modelo baseline", available_aliases, format_func=lambda x: alias_to_name.get(x, x))
        st.altair_chart(confusion_matrix_chart(baseline_confusions[selected_alias]["confusion_matrix"], title=alias_to_name.get(selected_alias, selected_alias)), width="stretch")

    with st.expander("Ver tabla baseline completa"):
        st.dataframe(chart_df.rename(columns={"display_name": "modelo"}), width="stretch")
