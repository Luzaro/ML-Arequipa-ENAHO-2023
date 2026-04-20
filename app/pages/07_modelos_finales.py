from __future__ import annotations

import streamlit as st

from loaders import load_benchmark_holdout_safe, load_tuning_safe
from ui import confusion_matrix_chart, render_hero, render_section_intro, render_story_card, tuned_display_name


tuning = load_tuning_safe()
benchmark = load_benchmark_holdout_safe()


def _friendly_model_name(value: str) -> str:
    return {
        "LogisticRegression": "Regresión logística",
        "RandomForestClassifier": "Bosque aleatorio",
        "XGBoostClassifier": "XGBoost",
        "LightGBMClassifier": "LightGBM",
        "Logistic": "Regresión logística",
        "RandomForest": "Bosque aleatorio",
        "XGBoost": "XGBoost",
        "LightGBM": "LightGBM",
    }.get(str(value), str(value))


def _friendly_strategy(value: str) -> str:
    return {
        "weighted": "Ponderación de clases",
        "smote": "SMOTE",
    }.get(str(value), str(value))


def _friendly_feature_set(value: str) -> str:
    return {
        "top30_full": "Top 30 Full",
        "top20_sin_leakage": "Top 20 sin leakage",
        "top20_full": "Top 20 Full",
    }.get(str(value), str(value))


def _tuning_ficha(row) -> str:
    return (
        f"Algoritmo: {_friendly_model_name(row['modelo_familia'])}\n"
        f"Estrategia: {_friendly_strategy(row['estrategia'])}\n"
        f"Variables: {_friendly_feature_set(row['escenario'])}\n"
        f"Threshold: {float(row['threshold']):.2f}"
    )


def _benchmark_ficha(row) -> str:
    return (
        f"Algoritmo: {_friendly_model_name(row['model'])}\n"
        f"Estrategia: {_friendly_strategy(row['strategy'])}\n"
        f"Variables: {_friendly_feature_set(row['feature_set'])}\n"
        f"Threshold: {float(row['threshold']):.2f}"
    )

render_hero(
    "Modelos finales",
    "No existe un único modelo mejor para todos los objetivos",
    "La selección final depende del uso esperado. Esta página muestra tres escenarios: máxima detección, mejor equilibrio y benchmark complementario sin leakage.",
)

render_section_intro(
    "Cómo leer esta página",
    "Cada escenario responde a una prioridad distinta. La recomendación final del proyecto privilegia equilibrio, pero el benchmarking muestra por qué también importa observar recall máximo y una referencia sin leakage.",
)

cards = []
if not tuning.empty:
    tuning["display_name"] = tuning.apply(tuned_display_name, axis=1)
    recall_row = tuning.sort_values(["recall", "f1", "precision"], ascending=[False, False, False]).iloc[0]
    balance_row = tuning.sort_values(["f1", "recall", "precision"], ascending=[False, False, False]).iloc[0]
    cards.append(
        {
            "title": "Máxima detección",
            "model_name": recall_row["display_name"],
            "technical_ficha": _tuning_ficha(recall_row),
            "precision": float(recall_row["precision"]),
            "recall": float(recall_row["recall"]),
            "f1": float(recall_row["f1"]),
            "cm": [[int(recall_row["tn"]), int(recall_row["fp"])], [int(recall_row["fn"]), int(recall_row["tp"])]],
            "tn": int(recall_row["tn"]),
            "fp": int(recall_row["fp"]),
            "fn": int(recall_row["fn"]),
            "tp": int(recall_row["tp"]),
            "interpretation": "Conviene cuando la prioridad es no dejar hogares pobres fuera.",
        }
    )
    cards.append(
        {
            "title": "Mejor equilibrio",
            "model_name": balance_row["display_name"],
            "technical_ficha": _tuning_ficha(balance_row),
            "precision": float(balance_row["precision"]),
            "recall": float(balance_row["recall"]),
            "f1": float(balance_row["f1"]),
            "cm": [[int(balance_row["tn"]), int(balance_row["fp"])], [int(balance_row["fn"]), int(balance_row["tp"])]],
            "tn": int(balance_row["tn"]),
            "fp": int(balance_row["fp"]),
            "fn": int(balance_row["fn"]),
            "tp": int(balance_row["tp"]),
            "interpretation": "Es la opción principal para exposición final y decisión analítica.",
        }
    )

if not benchmark.empty:
    leakage_row = benchmark.loc[benchmark["scenario_label"] == "sin_leakage"]
    if not leakage_row.empty:
        leakage = leakage_row.iloc[0]
        cards.append(
            {
                "title": "Benchmark sin leakage",
                "model_name": "Modelo de referencia sin leakage",
                "technical_ficha": _benchmark_ficha(leakage),
                "precision": float(leakage["precision"]),
                "recall": float(leakage["recall"]),
                "f1": float(leakage["f1"]),
                "cm": [[int(leakage["tn"]), int(leakage["fp"])], [int(leakage["fn"]), int(leakage["tp"])]],
                "tn": int(leakage["tn"]),
                "fp": int(leakage["fp"]),
                "fn": int(leakage["fn"]),
                "tp": int(leakage["tp"]),
                "interpretation": "Sirve como referencia metodológica para comparar el costo de excluir variables cercanas al target.",
            }
        )

if not cards:
    st.warning("No se encontraron los archivos de modelos finales o benchmark para poblar esta vista.")
else:
    top_cols = st.columns(len(cards))
    for col, card in zip(top_cols, cards):
        with col:
            render_story_card(
                card["title"],
                f"{card['technical_ficha']}\n\n"
                f"Precisión: {card['precision']:.3f}\n"
                f"Recall: {card['recall']:.3f}\n"
                f"F1: {card['f1']:.3f}\n\n"
                f"{card['interpretation']}",
            )

    st.markdown("### Matrices de confusión")
    selector = st.selectbox("Escenario", [item["title"] for item in cards])
    selected = next(item for item in cards if item["title"] == selector)
    st.altair_chart(confusion_matrix_chart(selected["cm"], title=selected["model_name"]), width="stretch")
    render_story_card(
        "Cómo interpretar esta matriz",
        f"{selected['technical_ficha']}\n\n"
        f"Verdaderos negativos: {selected['tn']} hogares no pobres fueron clasificados correctamente como no pobres. "
        f"Falsos positivos: {selected['fp']} hogares no pobres fueron marcados como pobres. "
        f"Falsos negativos: {selected['fn']} hogares pobres quedaron sin detectar. "
        f"Verdaderos positivos: {selected['tp']} hogares pobres fueron identificados correctamente. "
        f"En este escenario, la lectura de negocio es: {selected['interpretation']}",
    )

    st.markdown("### Cuándo elegir cada modelo")
    for card in cards:
        render_story_card(card["title"], card["interpretation"])
