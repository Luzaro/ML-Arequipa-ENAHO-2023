# Benchmark de desbalance

## Diagnostico
- Hogares totales: 1517
- Hogares pobres: 207
- Tasa positiva: 0.1365
- Se compararon class_weight/scale_pos_weight y SMOTE dentro de validacion cruzada estratificada.
- Las metricas principales fueron recall, precision, f1, roc_auc y pr_auc.

## Ganadores por escenario (CV)
### maxima_deteccion
- Modelo: Logistic
- Estrategia: smote
- Feature set: top20_full
- Threshold: 0.2
- Recall CV: 0.9370
- Precision CV: 0.2697
- F1 CV: 0.4187
- ROC AUC CV: 0.8661
- PR AUC CV: 0.5472

### mejor_equilibrio
- Modelo: XGBoost
- Estrategia: smote
- Feature set: top20_full
- Threshold: 0.3
- Recall CV: 0.5898
- Precision CV: 0.5084
- F1 CV: 0.5436
- ROC AUC CV: 0.8650
- PR AUC CV: 0.5030

### sin_leakage
- Modelo: Logistic
- Estrategia: weighted
- Feature set: top20_sin_leakage
- Threshold: 0.5
- Recall CV: 0.7539
- Precision CV: 0.3366
- F1 CV: 0.4647
- ROC AUC CV: 0.8400
- PR AUC CV: 0.4842

## Validacion holdout
### maxima_deteccion
- Modelo final: Logistic (smote)
- Recall: 0.9839
- Precision: 0.2450
- F1: 0.3923
- ROC AUC: 0.8687
- PR AUC: 0.5000
- Matriz de confusion: TN=206 | FP=188 | FN=1 | TP=61

### mejor_equilibrio
- Modelo final: XGBoost (smote)
- Recall: 0.5000
- Precision: 0.3605
- F1: 0.4189
- ROC AUC: 0.8555
- PR AUC: 0.4482
- Matriz de confusion: TN=339 | FP=55 | FN=31 | TP=31

### sin_leakage
- Modelo final: Logistic (weighted)
- Recall: 0.7581
- Precision: 0.3092
- F1: 0.4393
- ROC AUC: 0.8452
- PR AUC: 0.4701
- Matriz de confusion: TN=289 | FP=105 | FN=15 | TP=47
