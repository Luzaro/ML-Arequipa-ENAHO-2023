from __future__ import annotations

import pandas as pd
import streamlit as st

from narrative import TUNING_ACTIONS
from loaders import load_baseline_safe, load_tuning_safe
from ui import grouped_metric_chart, performance_scatter, render_hero, render_section_intro, render_step_grid, render_story_card, tuned_display_name


baseline = load_baseline_safe()
tuning = load_tuning_safe()

render_hero(
    "Tuning y tratamiento del desbalance",
    "Por qué el ajuste fue necesario y qué se hizo",
    "La muestra contiene muchos más hogares no pobres que pobres. Por eso el proyecto dejó de depender solo de accuracy y pasó a trabajar con estrategias de balanceo y ajuste de threshold.",
)

render_section_intro(
    "El problema de fondo",
    "En un dataset desbalanceado, un modelo puede verse aceptable en accuracy y aun así fallar donde más importa: la detección de hogares pobres.",
)

st.markdown("### Acciones aplicadas")
render_step_grid(TUNING_ACTIONS, columns=4)

if baseline.empty or tuning.empty:
    st.warning("No se encontraron los reportes baseline o tuned necesarios para esta página.")
else:
    tuning["display_name"] = tuning.apply(tuned_display_name, axis=1)
    best_baseline = baseline.sort_values(["f1_pobre", "recall_pobre", "precision_pobre"], ascending=[False, False, False]).iloc[0]
    finalists = tuning[["display_name", "precision", "recall", "f1", "pr_auc", "escenario"]].copy()
    finalists = finalists.rename(columns={"escenario": "escenario_label"})

    comparison = pd.DataFrame(
        [
            {
                "modelo": "Mejor baseline",
                "precision": best_baseline["precision_pobre"],
                "recall": best_baseline["recall_pobre"],
                "f1": best_baseline["f1_pobre"],
                "pr_auc": 0.0,
            }
        ]
        + finalists.rename(columns={"display_name": "modelo"}).to_dict(orient="records")
    )

    st.markdown("### Baseline vs tuning")
    metric_map = {"precision": "Precisión", "recall": "Recall", "f1": "F1", "pr_auc": "PR-AUC"}
    st.altair_chart(grouped_metric_chart(comparison, "modelo", metric_map, sort_by="f1"), width="stretch")

    st.markdown("### Trade-off precisión vs recall")
    st.altair_chart(performance_scatter(finalists.rename(columns={"display_name": "label"}), "label", "escenario_label"), width="stretch")
    render_story_card(
        "Lectura metodológica",
        "El tuning permitió construir escenarios diferentes según el objetivo: uno prioriza detectar más hogares pobres, otro balancea mejor precisión y recall, y otro controla más la precisión.",
    )
