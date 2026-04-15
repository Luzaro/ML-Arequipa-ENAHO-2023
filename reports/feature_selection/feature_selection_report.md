# Seleccion de variables para clasificacion

## Set conservador recomendado
- educ_jefe_ord
- combustible_limpio
- p104
- miembros_afiliados_essalud
- p103a
- n_ninos_0_5
- tam_hogar
- edad_jefe
- estrato
- piso_precario
- p105a
- n_ninos_6_16
- internet_hogar
- miembros_sin_atencion_salud
- p101
- p102
- acceso_desague_red
- n_6_16_no_matriculados

## Set amplio recomendado
- educ_jefe_ord
- combustible_limpio
- p104
- miembros_afiliados_essalud
- p103a
- n_ninos_0_5
- tam_hogar
- edad_jefe
- estrato
- piso_precario
- p105a
- n_ninos_6_16
- internet_hogar
- miembros_sin_atencion_salud
- p101
- p102
- acceso_desague_red
- n_6_16_no_matriculados
- agua_segura_proxy
- num_perceptores
- acceso_agua_red
- sexo_jefe_bin
- agua_potable_reportada
- agua_disponible_diaria
- n_6_16_no_asisten
- n_adultos_65_mas
- saneamiento_inadecuado
- acceso_electricidad

## Criterio
- Se excluyeron variables monetarias o muy cercanas a la regla oficial de pobreza monetaria.
- Se priorizaron variables sociodemograficas, educativas, de salud, vivienda y servicios.
- El ranking combina mutual information y random forest importance.

## Uso sugerido
- Empezar modelado con el set conservador.
- Evaluar el set amplio como analisis de sensibilidad.