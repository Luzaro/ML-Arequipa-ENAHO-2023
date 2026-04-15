from __future__ import annotations

import streamlit as st

from loaders import load_dataset_safe
from ui import inject_styles, render_sidebar_context


def run() -> None:
    st.set_page_config(
        page_title="Pobreza monetaria Arequipa 2023",
        page_icon=":material/insights:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_styles()
    df = load_dataset_safe()
    render_sidebar_context(df)

    navigation = st.navigation(
        [
            st.Page("pages/01_inicio.py", title="RESUMEN EJECUTIVO", icon=":material/home:"),
            st.Page("pages/02_etl.py", title="PIPELINE ETL", icon=":material/account_tree:"),
            st.Page("pages/03_eda.py", title="EDA Y PERFIL DEL DATO", icon=":material/monitoring:"),
            st.Page("pages/04_feature_selection.py", title="PREPARACION Y FEATURE SELECTION", icon=":material/filter_alt:"),
            st.Page("pages/05_baseline.py", title="MODELADO BASE", icon=":material/stacked_bar_chart:"),
            st.Page("pages/06_tuning.py", title="TUNING Y DESBALANCE", icon=":material/tune:"),
            st.Page("pages/07_modelos_finales.py", title="MODELOS FINALES", icon=":material/analytics:"),
            st.Page("pages/08_prediccion.py", title="PREDICCION INTERACTIVA", icon=":material/query_stats:"),
            st.Page("pages/09_conclusiones.py", title="RESULTADOS Y CONCLUSIONES", icon=":material/fact_check:"),
        ],
        position="sidebar",
    )

    navigation.run()


if __name__ == "__main__":
    run()
