# SelectKBest con ANOVA (f_classif)

## Supuestos usados
- Se incluyeron variables dimensionales previamente mapeadas.
- Se añadieron programas sociales alimentarios y no alimentarios solicitados por el usuario.
- Se añadió ingreso neto total del hogar (`inghog2d`) por solicitud expresa.
- Los programas alimentarios se aproximaron con `ingtpu16`, `sg27` y `sig28`.
- Los programas no alimentarios se aproximaron con `ingtpu01`, `ingtpu02`, `ingtpu03`, `ingtpu04`, `ingtpu05`, `ingtpu10`, `ingtpu12`, `ingtpu13` e `ingtpu14`.

## Nota metodológica
- `inghog2d` y varias transferencias públicas están más cerca del target monetario que las variables estructurales; interpretar su selección con cautela por riesgo de leakage o proximidad conceptual.

## Top 20 por F-score
- combustible_limpio: F=122.8115, p=1.69183e-27
- p114b3: F=107.0345, p=2.75431e-24
- combustible_precario: F=105.9853, p=4.51696e-24
- p103a: F=95.4335, p=6.67206e-22
- educ_jefe_ord: F=81.9602, p=4.15595e-19
- piso_precario: F=62.0518, p=6.33483e-15
- inghog2d: F=61.6397, p=7.74561e-15
- p104: F=56.0707, p=1.18108e-13
- internet_hogar: F=50.2985, p=2.01916e-12
- p1144: F=50.2985, p=2.01916e-12
- tam_hogar: F=46.5005, p=1.31901e-11
- n_ninos_0_5: F=45.1323, p=2.59811e-11
- n_ninos_6_16: F=44.7110, p=3.20184e-11
- miembros_afiliados_essalud: F=43.6880, p=5.32025e-11
- ingtpu01: F=27.2793, p=2.00557e-07
- acceso_desague_red: F=25.8617, p=4.12482e-07
- ingtpu05: F=25.1871, p=5.8175e-07
- agua_segura_proxy: F=24.5570, p=8.02407e-07
- ingtpu03: F=23.5414, p=1.34858e-06
- miembros_sin_atencion_salud: F=20.7103, p=5.76925e-06