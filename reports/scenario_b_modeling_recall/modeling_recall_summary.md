# Tuning oficial con SMOTE

El tuning oficial del proyecto quedo concentrado en modelos con variables monetarias (`top30_full`) y con tratamiento explicito del desbalance.

## Finalistas oficiales

### Maxima deteccion | RADAR MAX
- Modelo: LogisticRegression
- Estrategia: smote
- Threshold: 0.20
- Accuracy holdout: 0.5877
- Precision holdout: 0.2460
- Recall holdout: 0.9839
- F1 holdout: 0.3935
- ROC AUC holdout: 0.8672
- PR AUC CV: 0.5469
- Matriz de confusion: TN=207 | FP=187 | FN=1 | TP=61

### Mejor equilibrio | XGB BALANCE
- Modelo: XGBoostClassifier
- Estrategia: weighted
- Threshold: 0.40
- Accuracy holdout: 0.8311
- Precision holdout: 0.4157
- Recall holdout: 0.5968
- F1 holdout: 0.4901
- ROC AUC holdout: 0.8662
- PR AUC CV: 0.4820
- Matriz de confusion: TN=342 | FP=52 | FN=25 | TP=37

### Mayor precision | LGBM PRECISO
- Modelo: LightGBMClassifier
- Estrategia: smote
- Threshold: 0.35
- Accuracy holdout: 0.8158
- Precision holdout: 0.3625
- Recall holdout: 0.4677
- F1 holdout: 0.4085
- ROC AUC holdout: 0.8485
- PR AUC CV: 0.4861
- Matriz de confusion: TN=343 | FP=51 | FN=33 | TP=29
