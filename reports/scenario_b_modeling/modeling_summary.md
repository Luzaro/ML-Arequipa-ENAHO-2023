# Modelado inicial escenario B

- Dataset detectado: enaho_arequipa_escenario_b_clean.parquet
- Target detectado: target_pobreza_monetaria_bin
- Total top30_full: 30

## top30_full

- usa_combustible_limpio
- usa_combustible_precario
- dispositivo_servicio_digital_2
- material_techo
- nivel_educativo_jefe_ord
- dispositivo_servicio_digital_3
- piso_precario
- ingreso_neto_total_hogar
- num_habitaciones
- acceso_internet_hogar_derivado
- dispositivo_servicio_digital_1
- material_pared
- tam_hogar
- num_ninos_0_5
- num_ninos_6_16
- num_miembros_afiliados_essalud
- estrato_geografico
- monto_juntos
- acceso_desague_red
- monto_bono_gas
- monto_pension_65
- num_miembros_sin_atencion_salud
- acceso_agua_red
- num_miembros_con_enfermedad_4_sem
- monto_programa_contigo
- agua_potable_reportada
- tipo_vivienda
- monto_otras_transferencias_publicas
- monto_bono_electricidad
- num_miembros_con_sintoma_4_sem

## Resumen de modelos

- top30_full | XGBoostClassifier: accuracy=0.8596, precision=0.4800, recall=0.3871, f1=0.4286, recall_pobre=0.3871
- top30_full | DecisionTreeClassifier: accuracy=0.8180, precision=0.3478, recall=0.3871, f1=0.3664, recall_pobre=0.3871
- top30_full | LogisticRegression: accuracy=0.8662, precision=0.5122, recall=0.3387, f1=0.4078, recall_pobre=0.3387
- top30_full | RandomForestClassifier: accuracy=0.8662, precision=0.5333, recall=0.1290, f1=0.2078, recall_pobre=0.1290
- top30_full | SVC: accuracy=0.8618, precision=0.3333, recall=0.0161, f1=0.0308, recall_pobre=0.0161

## Variables por escenario

- top30_full: 30 variables (23 numericas, 7 categoricas)

## Mejor modelo

- Mejor desempeno priorizando recall de la clase pobre: top30_full | XGBoostClassifier
- recall_pobre=0.3871, f1_pobre=0.4286, accuracy=0.8596