# Reporte EDA ENAHO Arequipa 2023

- Filas/hogares: 1517
- Columnas: 176
- Hogares con algun nulo: 541
- Hogares sin nulos: 976
- Target 0 no pobre: 1310
- Target 1 pobre: 207

## Hallazgos
- El target esta desbalanceado hacia no pobres, por lo que en modelado se deben revisar metricas como recall, F1 y ROC-AUC.
- Existen nulos concentrados en pocas variables de servicios/vivienda, no en el target.
- Se recomienda iniciar con un set conservador que excluya variables monetarias cercanas a la regla oficial de pobreza.

## Proximo paso
- Definir set final de features e iniciar modelado supervisado.