# Resumen ejecutivo tecnico

## Modulos usados
- 906-Modulo01 Caracteristicas de la Vivienda y del Hogar
- 906-Modulo02 Caracteristicas de los Miembros del Hogar
- 906-Modulo03 Educacion
- 906-Modulo04 Salud
- 906-Modulo05 Empleo e Ingresos
- 906-Modulo34 Sumarias (Variables Calculadas)

## Filtros aplicados
- Encuesta: ENAHO 2023 Anual no panel
- Ambito geografico: departamento Arequipa usando `ubigeo[:2] == "04"`
- Catalogo empleado en extraccion: bundled_load_catalog
- Modulos validados en descarga: 6
- Hogares finales en Arequipa: 1517

## Unidad de analisis final
- Unidad final: hogar
- Tablas a nivel hogar: Modulo01 y Sumaria
- Tablas a nivel persona agregadas a hogar: Modulo02, Modulo03, Modulo04, Modulo05

## Target elegido
- Variable oficial: `pobreza` en Sumaria
- Definicion binaria para modelado: pobre extremo + pobre no extremo = 1; no pobre = 0
- Frecuencia binaria: {'0': 1310, '1': 207}

## Features construidas
- Total features finales candidatas: 177
- Total columnas del dataset model-ready: 178
- Features registradas en inventario: 177
- Variables descartadas por reglas tecnicas: 91
- Variables con riesgo de fuga registradas: 0
- Bloques incluidos: demografia del jefe, composicion del hogar, educacion, salud, vivienda, servicios, geografia y variables economicas de soporte aun permitidas.

## Problemas encontrados
- El acceso al portal INEI tuvo intentos inestables al inicio del proceso.
- En Modulo02 `p203` llego etiquetada como texto en parte de los registros y hubo que normalizar la deteccion del jefe.
- Se detectaron diferencias entre `mieperho` y el conteo simple de personas desde Modulo02 (`tam_hogar_mod2`).
- Varias columnas monetarias de Sumaria fueron marcadas como riesgo de fuga y quedaron fuera del model-ready.

## Decisiones tomadas
- Se trabajo sobre raw STATA para preservar la estructura original del INEI.
- Se exporto una vista analitica y otra model-ready en Parquet y CSV.
- No se calculo MPI; el objetivo fue construir un dataset para clasificacion supervisada de pobreza.

## Pendientes para modelado
- Definir estrategia de imputacion para variables con nulos residuales.
- Revisar encoding para variables categoricas.
- Evaluar tratamiento de ponderadores en entrenamiento y evaluacion.
- Confirmar si se desea excluir mas variables economicas por criterio estricto de leakage.

## Proximo paso: modelado supervisado