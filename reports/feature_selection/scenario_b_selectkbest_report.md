# SelectKBest sobre escenario B limpio

- MÃ©todo: SelectKBest con f_classif (ANOVA).
- Las variables categÃ³ricas se expandieron a dummies para calcular F-score y luego se agregaron al nivel de variable original usando el mÃ¡ximo F-score y el mÃ­nimo p-value.
- Variables candidatas evaluadas: 48
- Variables significativas al 5%: 35
- K recomendado de trabajo: 30

## Top por F-score

- usa_combustible_limpio: F_max=122.8115, p_min=1.69183e-27, dimensiÃ³n=Energia
- usa_combustible_precario: F_max=105.9853, p_min=4.51696e-24, dimensiÃ³n=Energia
- dispositivo_servicio_digital_2: F_max=104.3935, p_min=9.57371e-24, dimensiÃ³n=Conectividad
- material_techo: F_max=97.4441, p_min=2.56818e-22, dimensiÃ³n=Vivienda y entorno
- nivel_educativo_jefe_ord: F_max=81.9602, p_min=4.15595e-19, dimensiÃ³n=Educacion
- dispositivo_servicio_digital_3: F_max=80.0142, p_min=1.05833e-18, dimensiÃ³n=Conectividad
- piso_precario: F_max=62.0518, p_min=6.33483e-15, dimensiÃ³n=Vivienda y entorno
- ingreso_neto_total_hogar: F_max=61.6397, p_min=7.74561e-15, dimensiÃ³n=programas_sociales
- num_habitaciones: F_max=56.0707, p_min=1.18108e-13, dimensiÃ³n=Vivienda y entorno
- acceso_internet_hogar_derivado: F_max=50.2985, p_min=2.01916e-12, dimensiÃ³n=Conectividad
- dispositivo_servicio_digital_1: F_max=50.2985, p_min=2.01916e-12, dimensiÃ³n=Conectividad
- material_pared: F_max=47.9105, p_min=6.56557e-12, dimensiÃ³n=Vivienda y entorno
- tam_hogar: F_max=46.5005, p_min=1.31901e-11, dimensiÃ³n=Demografia y composicion del hogar
- num_ninos_0_5: F_max=45.1323, p_min=2.59811e-11, dimensiÃ³n=Demografia y composicion del hogar
- num_ninos_6_16: F_max=44.7110, p_min=3.20184e-11, dimensiÃ³n=Demografia y composicion del hogar
- num_miembros_afiliados_essalud: F_max=43.6880, p_min=5.32025e-11, dimensiÃ³n=Salud
- estrato_geografico: F_max=39.8094, p_min=3.66731e-10, dimensiÃ³n=Geografia y contexto territorial
- monto_juntos: F_max=27.2793, p_min=2.00557e-07, dimensiÃ³n=programas_sociales
- acceso_desague_red: F_max=25.8617, p_min=4.12482e-07, dimensiÃ³n=Servicios basicos
- monto_bono_gas: F_max=25.1871, p_min=5.8175e-07, dimensiÃ³n=programas_sociales
- monto_pension_65: F_max=23.5414, p_min=1.34858e-06, dimensiÃ³n=programas_sociales
- num_miembros_sin_atencion_salud: F_max=20.7103, p_min=5.76925e-06, dimensiÃ³n=Salud
- acceso_agua_red: F_max=20.6916, p_min=5.82506e-06, dimensiÃ³n=Servicios basicos
- num_miembros_con_enfermedad_4_sem: F_max=17.8238, p_min=2.56734e-05, dimensiÃ³n=Salud
- monto_programa_contigo: F_max=14.5069, p_min=0.000145223, dimensiÃ³n=programas_sociales
- agua_potable_reportada: F_max=14.4572, p_min=0.000149074, dimensiÃ³n=Servicios basicos
- tipo_vivienda: F_max=12.7636, p_min=0.000364488, dimensiÃ³n=Vivienda y entorno
- monto_otras_transferencias_publicas: F_max=12.7044, p_min=0.000376124, dimensiÃ³n=programas_sociales
- monto_bono_electricidad: F_max=9.4152, p_min=0.00219009, dimensiÃ³n=programas_sociales
- num_miembros_con_sintoma_4_sem: F_max=8.8705, p_min=0.00294424, dimensiÃ³n=Salud