# ENAHO Arequipa Pobreza 2023 ML

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Community%20Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-ETL%20%26%20EDA-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Modeling-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-FF6600?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Classifier-00A651?style=for-the-badge)](https://lightgbm.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-Deployed-success?style=for-the-badge)](https://ml-arequipa-enaho-2023.streamlit.app)

[![Ver app en Streamlit](https://img.shields.io/badge/Ver%20app-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://ml-arequipa-enaho-2023.streamlit.app)

Proyecto de machine learning para predecir pobreza monetaria a nivel hogar en el departamento de Arequipa usando microdatos de la ENAHO 2023 Anual del INEI.

El repositorio integra:

- un pipeline ETL reproducible
- exploracion y perfilamiento del dato
- seleccion de variables
- comparacion de modelos baseline y ajustados
- una aplicacion local en Streamlit para exposicion, demostracion y analisis interactivo

## Resumen ejecutivo

La meta del proyecto es construir un flujo de datos y modelado que permita identificar hogares con mayor probabilidad de encontrarse en situacion de pobreza monetaria oficial, tomando como unidad de analisis el hogar y usando variables provenientes de vivienda, composicion del hogar, salud, educacion, empleo e ingresos.

La solucion final no se limita a entrenar modelos. Tambien organiza evidencia tecnica y narrativa de resultados en una app multipagina de Streamlit pensada para:

- sustentar el pipeline de extremo a extremo
- mostrar hallazgos del EDA y feature selection
- comparar escenarios de modelado
- explorar una prediccion interactiva sobre variables del hogar

## Objetivo

Construir un pipeline reproducible para:

1. extraer e integrar microdatos ENAHO 2023
2. transformar la informacion a nivel hogar
3. preparar un dataset limpio para modelado supervisado
4. comparar modelos baseline y estrategias de ajuste para desbalance
5. presentar resultados en notebooks y en una app local de Streamlit

## Fuente de datos

- Encuesta Nacional de Hogares (ENAHO) 2023 Anual
- Instituto Nacional de Estadistica e Informatica (INEI)
- Acceso programatico mediante `inei-microdatos`

### Modulos utilizados

- `Modulo01`: vivienda y hogar
- `Modulo02`: miembros del hogar
- `Modulo03`: educacion
- `Modulo04`: salud
- `Modulo05`: empleo e ingresos
- `Modulo34`: sumaria

## Variable objetivo

La variable objetivo oficial del proyecto es:

- `target_pobreza_monetaria_bin`

Interpretacion:

- `0`: hogar no pobre
- `1`: hogar pobre

Esta variable se deriva de la variable oficial `pobreza` presente en el modulo `Sumaria`.

## Alcance del proyecto

El flujo del proyecto esta dividido en cuatro capas principales:

### 1. ETL

Pipeline para descarga, estandarizacion, filtrado geografico, integracion y consolidacion del dataset a nivel hogar.

Entrypoints oficiales:

```bash
python src/etl/extract.py
python src/etl/transform.py
python src/etl/load.py
```

### 2. EDA

Analisis exploratorio para estudiar:

- distribucion del target
- faltantes y calidad del dato
- relaciones por variable numerica y categorica
- correlaciones y riesgos de redundancia

Notebook principal:

```bash
notebooks/01_eda_enaho_arequipa.ipynb
```

### 3. Modelado

Incluye:

- seleccion de variables
- modelos baseline
- tuning con foco en `recall`
- tratamiento del desbalance con `SMOTE`, `class_weight` y `scale_pos_weight`
- exportacion de pipelines listos para uso en la app

Scripts oficiales:

```bash
python src/modeling/feature_selection.py
python src/modeling/baseline.py
python src/modeling/recall_tuning.py
python src/modeling/export_models.py
```

Notebooks principales:

```bash
notebooks/02_modelado_baseline.ipynb
notebooks/03_modelado_recall_tuning.ipynb
```

### 4. Presentacion local

La aplicacion Streamlit organiza los resultados en una experiencia navegable y visual para demo local.

Ejecucion:

```bash
streamlit run app/streamlit_app.py
```

## App de Streamlit

La app esta compuesta por las siguientes secciones:

- `RESUMEN EJECUTIVO`
- `PIPELINE ETL`
- `EDA Y PERFIL DEL DATO`
- `PREPARACION Y FEATURE SELECTION`
- `MODELADO BASE`
- `TUNING Y DESBALANCE`
- `MODELOS FINALES`
- `PREDICCION INTERACTIVA`
- `RESULTADOS Y CONCLUSIONES`

Entrypoint principal:

- [app/streamlit_app.py](app/streamlit_app.py)

## Dataset principal

El proyecto prioriza `CSV` como formato principal de trabajo.

Archivos principales:

- `data/processed/enaho_arequipa_escenario_b_clean.csv`
- `data/processed/enaho_arequipa_escenario_b_clean.parquet`

Dataset de apoyo para escenario Top 30:

- `data/processed/enaho_arequipa_escenario_b_top30.csv`
- `data/processed/enaho_arequipa_escenario_b_top30.parquet`

## Seleccion de variables

La ruta oficial del proyecto adopta un conjunto principal:

- `top30_full`

Este conjunto prioriza una combinacion de variables de:

- ingresos y gasto
- condiciones de vivienda
- acceso a servicios
- estructura del hogar
- salud y capital humano

## Modelos evaluados

### Baseline

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- SVC

### Estrategias de ajuste

- `class_weight="balanced"`
- `scale_pos_weight` para XGBoost
- `SMOTE`
- ajuste de `threshold`

Thresholds probados:

- `0.50`
- `0.45`
- `0.40`
- `0.35`
- `0.30`
- `0.25`
- `0.20`

## Resultados principales

### Modelo recomendado

El modelo final recomendado para presentacion ejecutiva es:

| Campo | Valor |
| --- | --- |
| Nombre ejecutivo | Modelo balanceado recomendado |
| Algoritmo | XGBoostClassifier |
| Estrategia de balanceo | Ponderacion de clases |
| Variables | Top 30 Full |
| Threshold | 0.40 |
| Precision | 0.416 |
| Recall | 0.597 |
| F1 | 0.490 |

### Lectura del resultado

- `Modelo de maxima deteccion`: prioriza cobertura de hogares pobres
- `Modelo balanceado recomendado`: mejor equilibrio entre precision, recall y F1
- `Modelo de mayor precision relativa`: escenario mas selectivo para reducir sobre-alertas

## Modelos exportados para la app

### Baseline

- `models/baseline/logistic_regression_top30_full_pipeline.pkl`
- `models/baseline/random_forest_top30_full_pipeline.pkl`
- `models/baseline/xgboost_top30_full_pipeline.pkl`

### Ajustados

- `models/tuned/logistic_smote_deteccion_top30_full_pipeline.pkl`
- `models/tuned/xgboost_weighted_equilibrio_top30_full_pipeline.pkl`
- `models/tuned/lightgbm_smote_precision_top30_full_pipeline.pkl`

## Estructura del repositorio

```text
ML_AREQUIPA_ENAHO_2023/
  app/
    app.py
    streamlit_app.py
    config.py
    loaders.py
    ui.py
    ui_helpers.py
    pages/
  data/
    processed/
  models/
    baseline/
    tuned/
  notebooks/
  reports/
  src/
    etl/
    modeling/
    utils/
  archive/
  requirements.txt
  README.md
```

## Instalacion

### 1. Clonar el repositorio

```bash
git clone https://github.com/Luzaro/ML_AREQUIPA_ENAHO_2023.git
cd ML_AREQUIPA_ENAHO_2023
```

### 2. Crear entorno virtual

En Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
```

En PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Ejecucion rapida

### Levantar la app local

```bash
source .venv/Scripts/activate
streamlit run app/streamlit_app.py
```

Si prefieres usar Python directamente:

```bash
source .venv/Scripts/activate
python -m streamlit run app/streamlit_app.py
```

### Ejecutar ETL

```bash
python src/etl/extract.py
python src/etl/transform.py
python src/etl/load.py
```

### Ejecutar modelado

```bash
python src/modeling/feature_selection.py
python src/modeling/baseline.py
python src/modeling/recall_tuning.py
python src/modeling/export_models.py
```

## Artefactos incluidos y artefactos ignorados

Este repositorio conserva:

- dataset procesado principal
- reportes finales
- modelos exportados
- notebooks de analisis
- codigo fuente oficial

No se versionan por defecto:

- `.venv/`
- `logs/`
- `data/raw/`
- `data/interim/`

Esto mantiene el repositorio mas limpio y evita subir artefactos pesados o temporales.

## Consideraciones metodologicas

- El target corresponde a pobreza monetaria oficial del INEI.
- El proyecto no busca estimar un MPI como salida final.
- La ruta oficial privilegia `top30_full` como conjunto de variables principal.
- La app no es solo una demo tecnica: tambien funciona como narrativa de sustentacion.
- La carpeta `archive/` conserva trazabilidad historica del proyecto, pero no representa la ruta activa principal.

## Autor

**Luis Zavalaga Rodrigo**

## Licencia

No se ha definido una licencia explicita en este repositorio. Si vas a reutilizar el proyecto o publicarlo formalmente, conviene agregar una licencia segun el uso esperado.
