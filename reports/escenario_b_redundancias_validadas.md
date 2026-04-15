# Escenario B limpio

- La depuración se realizó con validación semántica basada en el diccionario de variables.
- No se descartó ninguna variable solo por correlación alta.
- Variables eliminadas por redundancia validada: 19
- Columnas finales del escenario B limpio: 55

## Variables descartadas

- internet_hogar_enaho -> se conserva acceso_internet_hogar_derivado: Ambas representan acceso a internet en el hogar; se conserva la derivada binaria por ser más directa para modelado.
- hogar_con_ninos_no_asisten -> se conserva al_menos_un_6_16_no_asiste: Son dos nombres para el mismo indicador binario de no asistencia.
- al_menos_un_6_16_no_asiste -> se conserva num_6_16_no_asisten: La binaria se deriva directamente del conteo y pierde intensidad del fenómeno.
- hogar_con_ninos_no_matriculados -> se conserva al_menos_un_6_16_no_matriculado: Son dos nombres para el mismo indicador binario de no matrícula.
- al_menos_un_6_16_no_matriculado -> se conserva num_6_16_no_matriculados: La binaria se deriva directamente del conteo y pierde intensidad del fenómeno.
- nivel_educativo_jefe_cat -> se conserva nivel_educativo_jefe_ord: Ambas representan el nivel educativo del jefe; se conserva la versión ordinal más estable para selección univariada.
- al_menos_un_ingreso_laboral_hogar -> se conserva num_miembros_con_ingreso_laboral: La binaria se deriva del conteo de miembros con ingreso laboral.
- combustible_principal_cocina -> se conserva usa_combustible_limpio: La categoría cruda es la fuente de las variables binarias de combustible; se conserva la recodificación más interpretable.
- es_area_rural -> se conserva estrato_geografico: es_area_rural es una simplificación derivada de estrato_geografico.
- al_menos_un_enfermedad_4_sem -> se conserva num_miembros_con_enfermedad_4_sem: La binaria se deriva directamente del conteo de miembros con enfermedad reciente.
- al_menos_un_sintoma_4_sem -> se conserva num_miembros_con_sintoma_4_sem: La binaria se deriva directamente del conteo de miembros con síntomas recientes.
- health_no_attention -> se conserva num_miembros_sin_atencion_salud: La binaria resume el conteo de miembros sin atención en salud.
- agua_segura_proxy -> se conserva acceso_agua_red: El proxy combina acceso_agua_red y agua_potable_reportada; se prefieren las componentes originales explícitas.
- material_piso_cat -> se conserva piso_precario: piso_precario resume la condición crítica del material de piso y es más interpretable para clasificación.
- monto_total_transferencias_publicas -> se conserva monto_total_programas_sociales_no_alimentarios: El total de transferencias públicas se solapa fuertemente con el agregado de programas no alimentarios en esta muestra.
- monto_total_programas_sociales_alimentarios -> se conserva monto_bono_alimentario: El agregado alimentario resume variables específicas; se conservan los componentes para distinguir tipos de apoyo.
- monto_total_programas_sociales_no_alimentarios -> se conserva monto_juntos: El agregado no alimentario resume programas específicos; se conservan los componentes para mantener detalle de política pública.
- recibe_programa_social_alimentario -> se conserva monto_bono_alimentario: La recepción binaria se deriva del monto positivo en programas alimentarios.
- recibe_programa_social_no_alimentario -> se conserva monto_juntos: La recepción binaria se deriva de montos positivos en programas no alimentarios.