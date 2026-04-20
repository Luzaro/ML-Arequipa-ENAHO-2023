from __future__ import annotations


EXECUTIVE_FLOW = [
    ("1. Datos crudos", "Se descargaron módulos ENAHO 2023 y se tomaron como materia prima del proyecto."),
    ("2. ETL", "Se integraron, limpiaron y homogeneizaron variables hasta llegar a una base a nivel hogar."),
    ("3. EDA", "Se revisó estructura, distribución del target y patrones descriptivos de pobreza y no pobreza."),
    ("4. Selección de variables", "Se compararon top20, top30, top40 y top48; top30 full quedó como set oficial."),
    ("5. Modelado", "Se compararon modelos baseline y luego se aplicó tuning con tratamiento de desbalance."),
    ("6. Decisión final", "Se eligieron escenarios finales según el objetivo: recall, equilibrio o rigor metodológico."),
]

ETL_STEPS = [
    ("Extract", "Descarga de módulos ENAHO 2023 mediante API y organización de datos crudos."),
    ("Transform", "Integración, limpieza, homogeneización y construcción de variables a nivel hogar."),
    ("Load", "Validación de salidas, consolidación de artefactos y escritura final en data/processed."),
]

TUNING_ACTIONS = [
    ("Class weight", "Se ajustaron pesos de clase para evitar sesgos hacia hogares no pobres."),
    ("Scale pos weight", "Se reforzó la clase positiva en modelos tipo boosting."),
    ("SMOTE", "Se evaluó sobremuestreo sintético para ampliar cobertura de la clase pobre."),
    ("Threshold tuning", "Se ajustaron puntos de corte para construir escenarios con prioridades distintas."),
]
