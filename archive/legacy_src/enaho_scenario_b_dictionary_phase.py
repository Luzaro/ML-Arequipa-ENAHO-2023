from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .enaho_etl_phase01 import build_paths, configure_logging, ensure_directories, save_json


SCENARIO_B_DATASET = "enaho_arequipa_escenario_b_renamed.parquet"
SCENARIO_B_MAP = "escenario_b_renombrado_map.csv"
DIMENSIONS_FILE = "variables_mapeadas_dimensiones.csv"
PHASE6_DICT = "phase6_variable_dictionary.csv"
SUMARIA_LABELS = "sumaria_variable_labels.csv"


REVERSE_RENAME_SOURCE_HINTS = {
    "acceso_internet_hogar_derivado": "internet_hogar",
    "internet_hogar_enaho": "p1144",
    "dispositivo_servicio_digital_1": "p114b1",
    "dispositivo_servicio_digital_2": "p114b2",
    "dispositivo_servicio_digital_3": "p114b3",
    "num_adultos_65_mas": "n_adultos_65_mas",
    "num_ninos_0_5": "n_ninos_0_5",
    "num_ninos_6_16": "n_ninos_6_16",
    "al_menos_un_6_16_no_asiste": "any_6_16_no_asisten",
    "al_menos_un_6_16_no_matriculado": "any_6_16_no_matriculados",
    "hogar_con_ninos_no_asisten": "children_not_attending",
    "hogar_con_ninos_no_matriculados": "children_not_enrolled",
    "nivel_educativo_jefe_cat": "educ_jefe",
    "nivel_educativo_jefe_ord": "educ_jefe_ord",
    "num_6_16_no_asisten": "n_6_16_no_asisten",
    "num_6_16_no_matriculados": "n_6_16_no_matriculados",
    "al_menos_un_ingreso_laboral_hogar": "any_ingreso_laboral_hogar",
    "num_miembros_con_ingreso_laboral": "miembros_con_ingreso_laboral",
    "num_perceptores_hogar": "num_perceptores",
    "usa_combustible_limpio": "combustible_limpio",
    "usa_combustible_precario": "combustible_precario",
    "combustible_principal_cocina": "combustible_principal",
    "es_area_rural": "area_rural_bin",
    "estrato_geografico": "estrato",
    "al_menos_un_enfermedad_4_sem": "any_enfermedad_4w",
    "al_menos_un_sintoma_4_sem": "any_sintoma_4w",
    "num_miembros_afiliados_essalud": "miembros_afiliados_essalud",
    "num_miembros_con_enfermedad_4_sem": "miembros_con_enfermedad_4w",
    "num_miembros_con_sintoma_4_sem": "miembros_con_sintoma_4w",
    "num_miembros_sin_atencion_salud": "miembros_sin_atencion_salud",
    "material_piso_cat": "material_piso",
    "tipo_vivienda": "p101",
    "material_pared": "p102",
    "material_techo": "p103a",
    "num_habitaciones": "p104",
    "tenencia_vivienda": "p105a",
    "ingreso_neto_total_hogar": "inghog2d",
    "monto_total_transferencias_publicas": "ingtpuhd",
    "monto_total_programas_sociales_alimentarios": "monto_programas_sociales_alimentarios",
    "monto_total_programas_sociales_no_alimentarios": "monto_programas_sociales_no_alimentarios",
    "recibe_programa_social_alimentario": "recibe_programa_social_alimentario",
    "recibe_programa_social_no_alimentario": "recibe_programa_social_no_alimentario",
    "monto_bono_alimentario": "ingtpu16",
    "monto_apoyo_ollas_comunes": "sg27",
    "monto_ingresos_gastos_ollas_comunes": "sig28",
    "monto_juntos": "ingtpu01",
    "monto_otras_transferencias_publicas": "ingtpu02",
    "monto_pension_65": "ingtpu03",
    "monto_beca_18": "ingtpu04",
    "monto_bono_gas": "ingtpu05",
    "monto_bono_electricidad": "ingtpu10",
    "monto_bono_onp_jubilados": "ingtpu12",
    "monto_programa_contigo": "ingtpu13",
    "monto_bono_yanapay": "ingtpu14",
    "factor_expansion_anual": "factor07",
    "target_pobreza_monetaria_bin": "target_pobreza_bin",
}


ROLE_HINTS = {
    "hogar_id": "identificador",
    "conglome": "identificador",
    "vivienda": "identificador",
    "hogar": "identificador",
    "ubigeo": "identificador",
    "factor_expansion_anual": "ponderador",
    "target_pobreza_monetaria_bin": "target",
}


def load_inputs(paths: Any) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset = pd.read_parquet(paths.data_processed / SCENARIO_B_DATASET)
    rename_map = pd.read_csv(paths.reports / SCENARIO_B_MAP)
    dimensions = pd.read_csv(paths.reports / DIMENSIONS_FILE)
    phase6 = pd.read_csv(paths.reports / PHASE6_DICT)
    sumaria = pd.read_csv(paths.reports / SUMARIA_LABELS)
    return dataset, rename_map, dimensions, phase6, sumaria


def get_source_variable(col: str, rename_map_df: pd.DataFrame) -> str:
    match = rename_map_df.loc[rename_map_df["variable_renombrada"] == col, "variable_original"]
    if not match.empty:
        return match.iloc[0]
    return REVERSE_RENAME_SOURCE_HINTS.get(col, col)


def get_description(col: str, source: str, rename_map_df: pd.DataFrame, dim_df: pd.DataFrame, phase6_df: pd.DataFrame, sumaria_df: pd.DataFrame) -> str:
    match = rename_map_df.loc[rename_map_df["variable_renombrada"] == col, "descripcion"]
    if not match.empty:
        return match.iloc[0]

    match = dim_df.loc[dim_df["variable"] == source, "descripcion"]
    if not match.empty:
        return match.iloc[0]

    match = phase6_df.loc[phase6_df["variable_final"] == source, "descripcion"]
    if not match.empty:
        return match.iloc[0]

    match = sumaria_df.loc[sumaria_df["variable"] == source, "label"]
    if not match.empty:
        return match.iloc[0]

    return col


def get_dimension(col: str, source: str, rename_map_df: pd.DataFrame, dim_df: pd.DataFrame) -> str:
    match = rename_map_df.loc[rename_map_df["variable_renombrada"] == col, "dimension_analitica"]
    if not match.empty:
        return match.iloc[0]

    match = dim_df.loc[dim_df["variable"] == source, "dimension_analitica"]
    if not match.empty:
        return match.iloc[0]

    if col in ROLE_HINTS:
        return "No aplica"
    return "Pendiente"


def get_source_module(source: str, dim_df: pd.DataFrame, phase6_df: pd.DataFrame) -> str:
    match = dim_df.loc[dim_df["variable"] == source, "modulo_fuente"]
    if not match.empty:
        return match.iloc[0]

    match = phase6_df.loc[phase6_df["variable_final"] == source, "tablas_fuente"]
    if not match.empty:
        return match.iloc[0]

    if source in {"conglome", "vivienda", "hogar"}:
        return "Identificacion ENAHO"
    return "No determinado"


def get_type(source: str, col: str, dim_df: pd.DataFrame, phase6_df: pd.DataFrame, dtype_str: str) -> str:
    match = dim_df.loc[dim_df["variable"] == source, "tipo_feature"]
    if not match.empty:
        return match.iloc[0]

    match = phase6_df.loc[phase6_df["variable_final"] == source, "tipo"]
    if not match.empty:
        return match.iloc[0]

    if col in {"hogar_id", "conglome", "vivienda", "hogar", "ubigeo"}:
        return "identificador"
    if col == "factor_expansion_anual":
        return "ponderador"
    if col == "target_pobreza_monetaria_bin":
        return "target_binario"
    if "int" in dtype_str or "float" in dtype_str:
        return "numerica"
    return "categorica"


def get_role(col: str, dimension: str) -> str:
    if col in ROLE_HINTS:
        return ROLE_HINTS[col]
    if dimension == "programas_sociales":
        return "feature_candidata"
    return "feature_candidata"


def get_observation(col: str, source: str, dimension: str) -> str:
    observations = {
        "acceso_internet_hogar_derivado": "Derivada desde la variable original de internet; posible redundancia con internet_hogar_enaho.",
        "internet_hogar_enaho": "Variable original del cuestionario; posible redundancia con acceso_internet_hogar_derivado.",
        "al_menos_un_6_16_no_asiste": "Indicador binario del mismo fenómeno medido también por hogar_con_ninos_no_asisten y num_6_16_no_asisten.",
        "hogar_con_ninos_no_asisten": "Duplicado semántico del indicador al_menos_un_6_16_no_asiste; revisar antes de modelar.",
        "al_menos_un_6_16_no_matriculado": "Indicador binario del mismo fenómeno medido también por hogar_con_ninos_no_matriculados y num_6_16_no_matriculados.",
        "hogar_con_ninos_no_matriculados": "Duplicado semántico del indicador al_menos_un_6_16_no_matriculado; revisar antes de modelar.",
        "nivel_educativo_jefe_cat": "Versión categórica del nivel educativo; evaluar frente a la versión ordinal.",
        "nivel_educativo_jefe_ord": "Versión ordinal del nivel educativo; evaluar frente a la versión categórica.",
        "material_piso_cat": "Variable categórica original; posible redundancia parcial con piso_precario.",
        "piso_precario": "Proxy binario derivado desde material_piso_cat.",
        "monto_total_transferencias_publicas": "Agregado de transferencias públicas; puede solaparse con montos específicos y con monto_total_programas_sociales_no_alimentarios.",
        "monto_total_programas_sociales_no_alimentarios": "Agregado derivado de programas públicos no alimentarios; puede solaparse con monto_total_transferencias_publicas.",
        "monto_total_programas_sociales_alimentarios": "Agregado derivado de apoyos alimentarios; revisar frente a sus componentes.",
        "recibe_programa_social_alimentario": "Indicador binario derivado de monto_total_programas_sociales_alimentarios.",
        "recibe_programa_social_no_alimentario": "Indicador binario derivado de monto_total_programas_sociales_no_alimentarios.",
        "ingreso_neto_total_hogar": "Variable monetaria potente pero cercana al target monetario; interpretar con cautela.",
        "factor_expansion_anual": "No ingresa a la matriz X; usar para análisis ponderado.",
        "target_pobreza_monetaria_bin": "Variable dependiente; no usar como feature.",
    }
    if col in observations:
        return observations[col]
    if dimension == "programas_sociales":
        return "Variable monetaria de apoyo/programa social; evaluar por riesgo de proximidad conceptual al target."
    return ""


def build_dictionary(
    dataset: pd.DataFrame,
    rename_map_df: pd.DataFrame,
    dim_df: pd.DataFrame,
    phase6_df: pd.DataFrame,
    sumaria_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for col in dataset.columns:
        source = get_source_variable(col, rename_map_df)
        dtype_str = str(dataset[col].dtype)
        dimension = get_dimension(col, source, rename_map_df, dim_df)
        rows.append(
            {
                "variable_final": col,
                "variable_origen": source,
                "descripcion": get_description(col, source, rename_map_df, dim_df, phase6_df, sumaria_df),
                "dimension_analitica": dimension,
                "modulo_fuente": get_source_module(source, dim_df, phase6_df),
                "tipo_logico": get_type(source, col, dim_df, phase6_df, dtype_str),
                "dtype_dataset": dtype_str,
                "nulos_pct": round(dataset[col].isna().mean() * 100, 4),
                "rol": get_role(col, dimension),
                "observaciones": get_observation(col, source, dimension),
            }
        )
    return pd.DataFrame(rows)


def write_summary(path: Path, dictionary_df: pd.DataFrame) -> None:
    features = dictionary_df.loc[dictionary_df["rol"] == "feature_candidata"]
    text = "\n".join(
        [
            "# Diccionario del escenario B",
            "",
            f"- Variables totales en la base renombrada: {dictionary_df.shape[0]}",
            f"- Features candidatas: {features.shape[0]}",
            "",
            "## Roles",
            "",
            *[
                f"- {rol}: {count}"
                for rol, count in dictionary_df.groupby("rol").size().to_dict().items()
            ],
        ]
    )
    path.write_text(text, encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    paths = build_paths(root)
    ensure_directories(paths)
    logger = configure_logging(paths.logs / f"scenario_b_dictionary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    dataset, rename_map_df, dim_df, phase6_df, sumaria_df = load_inputs(paths)
    dictionary_df = build_dictionary(dataset, rename_map_df, dim_df, phase6_df, sumaria_df)

    dictionary_df.to_csv(paths.reports / "escenario_b_diccionario_variables.csv", index=False, encoding="utf-8-sig")
    write_summary(paths.reports / "escenario_b_diccionario_variables.md", dictionary_df)

    validation = {
        "phase": "SCENARIO_B_DICTIONARY",
        "passed": bool(dictionary_df.shape[0] == dataset.shape[1]),
        "n_columns_dataset": int(dataset.shape[1]),
        "n_rows_dictionary": int(dictionary_df.shape[0]),
        "n_features": int((dictionary_df["rol"] == "feature_candidata").sum()),
    }
    save_json(validation, paths.reports / "escenario_b_diccionario_variables_validation.json")
    logger.info("SCENARIO_B_DICTIONARY validacion: %s", validation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
