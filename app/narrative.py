from __future__ import annotations


EXECUTIVE_FLOW = [
    ("1. Datos crudos", "Se descargaron modulos ENAHO 2023 y se tomaron como materia prima del proyecto."),
    ("2. ETL", "Se integraron, limpiaron y homogeneizaron variables hasta llegar a una base a nivel hogar."),
    ("3. EDA", "Se reviso estructura, distribucion del target y patrones descriptivos de pobreza y no pobreza."),
    ("4. Seleccion de variables", "Se compararon top20, top30, top40 y top48; top30 full quedo como set oficial."),
    ("5. Modelado", "Se compararon modelos baseline y luego se aplico tuning con tratamiento de desbalance."),
    ("6. Decision final", "Se eligieron escenarios finales segun el objetivo: recall, equilibrio o rigor metodologico."),
]

ETL_STEPS = [
    ("Extract", "Descarga de modulos ENAHO 2023 mediante API y organizacion de datos crudos."),
    ("Transform", "Integracion, limpieza, homogeneizacion y construccion de variables a nivel hogar."),
    ("Load", "Validacion de salidas, consolidacion de artefactos y escritura final en data/processed."),
]

TUNING_ACTIONS = [
    ("Class weight", "Se ajustaron pesos de clase para evitar sesgos hacia hogares no pobres."),
    ("Scale pos weight", "Se reforzo la clase positiva en modelos tipo boosting."),
    ("SMOTE", "Se evaluo sobremuestreo sintetico para ampliar cobertura de la clase pobre."),
    ("Threshold tuning", "Se ajustaron puntos de corte para construir escenarios con prioridades distintas."),
]
