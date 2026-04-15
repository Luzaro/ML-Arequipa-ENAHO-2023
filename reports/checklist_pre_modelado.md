# Checklist pre-modelado

- Total features candidatas: 177
- Numéricas: 111
- Categóricas: 43
- Binarias: 23
- Variables descartadas o con leakage: 0
- Revisar imputación en columnas con nulos antes de entrenar.
- Revisar encoding para las categóricas de alta cardinalidad.
- Mantener fuera del entrenamiento las variables monetarias directas descartadas por riesgo de fuga.
- Confirmar tratamiento de ponderadores en evaluación y no solo en entrenamiento.
- Próximo paso: modelado supervisado.