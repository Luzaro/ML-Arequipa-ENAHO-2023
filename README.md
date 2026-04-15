# ENAHO Arequipa Pobreza 2023 ML

Proyecto de clasificacion supervisada para predecir pobreza monetaria a nivel hogar en el departamento de Arequipa usando microdatos de ENAHO 2023 Anual del INEI.

## Descripcion general

El proyecto construye un flujo reproducible de extraccion, transformacion y carga (ETL), prepara un dataset final a nivel hogar, evalua modelos baseline y modelos ajustados con foco en `recall`, y presenta resultados mediante notebooks y una aplicacion local en Streamlit.

La variable objetivo es `target_pobreza_monetaria_bin`, derivada de la variable oficial `pobreza` del modulo `Sumaria`.

La presentacion final del proyecto tiene un enfoque hibrido: mantiene la trazabilidad tecnica del pipeline, pero traduce los resultados a una narrativa mas clara para exposicion, sustentacion y demostracion local.

## Objetivos

- Construir un ETL reproducible para ENAHO 2023 Anual en Arequipa.
- Integrar modulos de vivienda, hogar, educacion, salud, empleo e ingresos a nivel hogar.
- Preparar un dataset final limpio para modelado supervisado.
- Comparar desempeno antes del tuning y despues del tuning.
- Presentar los resultados en notebooks y en una app local de Streamlit.

## Fuente de datos

- Encuesta Nacional de Hogares (ENAHO) 2023 Anual
- Instituto Nacional de Estadistica e Informatica (INEI)
- Acceso mediante `inei-microdatos`

### Modulos utilizados

- `Modulo01`: Vivienda y Hogar
- `Modulo02`: Miembros del Hogar
- `Modulo03`: Educacion
- `Modulo04`: Salud
- `Modulo05`: Empleo e Ingresos
- `Modulo34`: Sumaria

## Variable objetivo

- `target_pobreza_monetaria_bin`
- `0 = no pobre`
- `1 = pobre`

## Estructura oficial del proyecto

```text
enaho-arequipa-pobreza-2023-ml/
  app/
    app.py
    streamlit_app.py
    config.py
    loaders.py
    ui.py
    narrative.py
    feature_config.py
    ui_helpers.py
    pages/
      01_inicio.py
      02_etl.py
      03_eda.py
      04_feature_selection.py
      05_baseline.py
      06_tuning.py
      07_modelos_finales.py
      08_prediccion.py
      09_conclusiones.py
  data/
    raw/
    interim/
    processed/
      enaho_arequipa_escenario_b_clean.csv
      enaho_arequipa_escenario_b_clean.parquet
      enaho_arequipa_escenario_b_top30.csv
      enaho_arequipa_escenario_b_top30.parquet
  logs/
  models/
    baseline/
    tuned/
    model_inventory.csv
    model_inventory.json
    model_export_validation.json
  notebooks/
    01_eda_enaho_arequipa.ipynb
    02_modelado_baseline.ipynb
    03_modelado_recall_tuning.ipynb
  reports/
    eda/
    feature_selection/
    scenario_b_modeling/
    scenario_b_modeling_recall/
    scenario_b_modeling_imbalance/
    scenario_b_modeling_top30_experiment/
    scenario_b_modeling_all48_experiment/
    scenario_b_modeling_top30_fe_experiment/
  src/
    etl/
      extract.py
      transform.py
      load.py
    modeling/
      feature_selection.py
      baseline.py
      recall_tuning.py
      export_models.py
      imbalance_benchmark.py
    utils/
      paths.py
      logging_utils.py
      io_utils.py
      runtime.py
  archive/
    processed_versions/
    legacy_reports/
    legacy_src/
  .gitignore
  README.md
  requirements.txt
```

## Convencion de carpetas

- `src/etl/`: capa oficial del ETL en tres etapas visibles.
- `src/etl/`: implementacion oficial del ETL. Ya no depende del codigo historico archivado para `extract`, `transform` y `load`.
- `src/modeling/`: capa oficial de seleccion de variables, baseline, tuning y exportacion. Ya no depende de `legacy`.
- `src/utils/`: funciones compartidas de rutas, logging, IO y arranque.
- `archive/legacy_src/`: scripts historicos del proyecto conservados por trazabilidad. Ya no forman parte de la ruta activa.
- `reports/`: reportes vigentes del flujo final.
- `archive/`: datasets y reportes historicos que ya no forman parte de la cara principal del repo.

## Nombre sugerido para GitHub

- Repositorio / carpeta sugerida: `enaho-arequipa-pobreza-2023-ml`
- Motivo: resume el tema, la fuente territorial, el anio y el enfoque de machine learning en un nombre limpio y facil de versionar.

## Flujo del proyecto

### 1. ETL

Puntos de entrada oficiales:

- `python src/etl/extract.py`
- `python src/etl/transform.py`
- `python src/etl/load.py`

### 2. EDA

Notebook principal:

- `notebooks/01_eda_enaho_arequipa.ipynb`

### 3. Modelado

Notebooks principales:

- `notebooks/02_modelado_baseline.ipynb`
- `notebooks/03_modelado_recall_tuning.ipynb`

Scripts oficiales:

- `python src/modeling/feature_selection.py`
- `python src/modeling/baseline.py`
- `python src/modeling/recall_tuning.py`
- `python src/modeling/export_models.py`

### 4. Presentacion local

Aplicacion Streamlit:

- `streamlit run app/streamlit_app.py`

Paginas principales de la app:

- `RESUMEN EJECUTIVO`
- `PIPELINE ETL`
- `EDA Y PERFIL DEL DATO`
- `PREPARACION Y FEATURE SELECTION`
- `MODELADO BASE`
- `TUNING Y DESBALANCE`
- `MODELOS FINALES`
- `PREDICCION INTERACTIVA`
- `RESULTADOS Y CONCLUSIONES`

## Datasets principales

El proyecto prioriza `CSV` como formato principal.

Archivo principal para modelado:

- `data/processed/enaho_arequipa_escenario_b_clean.csv`

Archivo opcional de apoyo:

- `data/processed/enaho_arequipa_escenario_b_clean.parquet`

## Seleccion de variables

La ruta oficial del proyecto se basa en un unico conjunto principal:

- `top30_full`: conjunto oficial de 30 variables seleccionadas, incluyendo indicadores monetarios y de vivienda.

## Modelos evaluados

### Baseline

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- SVC

### Ajustados

Se realizaron pruebas con:

- `class_weight="balanced"`
- `scale_pos_weight` para XGBoost
- `SMOTE`
- thresholds de decision `0.50`, `0.45`, `0.40`, `0.35`, `0.30`, `0.25`, `0.20`

## Resultados principales

### Baseline

- Baseline oficial: comparacion de `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `XGBoostClassifier` y `SVC` sobre `top30_full`

### Tuning oficial

- `Modelo de maxima deteccion`: LogisticRegression + SMOTE | Top 30 Full | threshold 0.20
- `Modelo balanceado recomendado`: XGBoostClassifier + Class Weight / Scale Pos Weight | Top 30 Full | threshold 0.40
- `Modelo de mayor precision relativa`: LightGBMClassifier + SMOTE | Top 30 Full | threshold 0.35

### Modelo final recomendado

- Nombre ejecutivo: `Modelo balanceado recomendado`
- Algoritmo: `XGBoostClassifier`
- Estrategia de balanceo: `ponderacion de clases`
- Variables: `Top 30 Full`
- Threshold de decision: `0.40`
- Resultados holdout: `precision = 0.416`, `recall = 0.597`, `F1 = 0.490`
- Lectura principal: mejor equilibrio entre `precision`, `recall` y `F1` para una presentacion final defendible

## Modelos exportados para la app

### Baseline

- `models/baseline/logistic_regression_top30_full_pipeline.pkl`
- `models/baseline/random_forest_top30_full_pipeline.pkl`
- `models/baseline/xgboost_top30_full_pipeline.pkl`

### Ajustados

- `models/tuned/logistic_smote_deteccion_top30_full_pipeline.pkl`
- `models/tuned/xgboost_weighted_equilibrio_top30_full_pipeline.pkl`
- `models/tuned/lightgbm_smote_precision_top30_full_pipeline.pkl`

## Instalacion

### 1. Crear entorno virtual

En Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

### 2. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Ejecucion

### ETL

```bash
python src/etl/extract.py
python src/etl/transform.py
python src/etl/load.py
```

### Modelado

```bash
python src/modeling/feature_selection.py
python src/modeling/baseline.py
python src/modeling/recall_tuning.py
python src/modeling/export_models.py
```

### Notebooks

Abrir y ejecutar:

- `notebooks/01_eda_enaho_arequipa.ipynb`
- `notebooks/02_modelado_baseline.ipynb`
- `notebooks/03_modelado_recall_tuning.ipynb`

### Streamlit local

```bash
streamlit run app/streamlit_app.py
```

## Consideraciones metodologicas

- El target corresponde a pobreza monetaria oficial del INEI.
- No se calcula MPI como resultado final.
- El proyecto prioriza `CSV` como formato principal de trabajo y revision.
- El proyecto adopta oficialmente `top30_full`, incorporando de manera explicita indicadores monetarios junto con variables de vivienda, servicios, salud y composicion del hogar.
- La etapa de tuning incorpora tratamiento del desbalance mediante ponderacion de clases, `scale_pos_weight`, `SMOTE` y ajuste de `threshold`.
- La app final no se plantea solo como demo tecnica: organiza los hallazgos en clave descriptiva, de resultados, conclusiones y pipeline operativo.
- La estructura nueva de `src/` es la oficial; el codigo historico fue movido a `archive/legacy_src/`.

## Autor

Proyecto: **ENAHO Arequipa Pobreza 2023 ML**

Autor: _[Luis Zavalaga Rodrigo]_
