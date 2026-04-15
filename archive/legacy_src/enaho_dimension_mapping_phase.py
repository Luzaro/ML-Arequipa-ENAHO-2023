from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .enaho_etl_phase01 import build_paths, configure_logging, ensure_directories, save_json


FEATURES_FILE = "features_finales.csv"
SUMARIA_LABELS_FILE = "sumaria_variable_labels.csv"
DERIVED_DICT_FILE = "phase6_variable_dictionary.csv"


MANUAL_LABELS = {
    "acceso_agua_red": "Acceso a agua por red pública",
    "acceso_desague_red": "Acceso a desagüe por red pública",
    "acceso_electricidad": "Acceso a electricidad",
    "agua_disponible_diaria": "Disponibilidad diaria de agua",
    "agua_potable_reportada": "Agua reportada como potable",
    "agua_segura_proxy": "Indicador proxy de agua segura",
    "any_6_16_no_asisten": "Existe al menos un menor de 6 a 16 años que no asiste",
    "any_6_16_no_matriculados": "Existe al menos un menor de 6 a 16 años no matriculado",
    "any_enfermedad_4w": "Existe al menos un miembro con enfermedad en últimas 4 semanas",
    "any_ingreso_laboral_hogar": "Existe al menos un miembro con ingreso laboral",
    "any_sintoma_4w": "Existe al menos un miembro con síntoma o malestar en últimas 4 semanas",
    "area_rural_bin": "Indicador de área rural",
    "area_urbana_rural": "Clasificación urbana o rural",
    "children_not_attending": "Hogar con niños en edad escolar que no asisten",
    "children_not_enrolled": "Hogar con niños en edad escolar no matriculados",
    "combustible_limpio": "Uso de combustible limpio para cocinar",
    "combustible_precario": "Uso de combustible precario para cocinar",
    "combustible_principal": "Combustible principal para cocinar",
    "distrito_cod": "Código de distrito",
    "dominio": "Dominio geográfico",
    "educ_jefe": "Nivel educativo del jefe del hogar",
    "educ_jefe_ord": "Nivel educativo ordinal del jefe del hogar",
    "edad_jefe": "Edad del jefe del hogar",
    "estrato": "Estrato geográfico",
    "estrsocial": "Estrato socioeconómico",
    "factor07": "Factor de expansión anual del hogar",
    "fuente_agua": "Fuente principal de abastecimiento de agua",
    "gasto_hogar": "Gasto total del hogar",
    "internet_hogar": "Acceso a internet en el hogar",
    "material_piso": "Material predominante del piso",
    "miembros_afiliados_essalud": "Miembros afiliados a EsSalud",
    "miembros_con_enfermedad_4w": "Miembros con enfermedad en últimas 4 semanas",
    "miembros_con_ingreso_laboral": "Miembros con ingreso laboral",
    "miembros_con_sintoma_4w": "Miembros con síntoma o malestar en últimas 4 semanas",
    "miembros_sin_atencion_salud": "Miembros sin atención en salud",
    "n_6_16_matriculados": "Número de personas de 6 a 16 matriculadas",
    "n_6_16_no_asisten": "Número de personas de 6 a 16 que no asisten",
    "n_6_16_no_matriculados": "Número de personas de 6 a 16 no matriculadas",
    "n_6_16_total": "Número total de personas de 6 a 16 años",
    "n_adultos_65_mas": "Número de adultos de 65 años o más",
    "n_ninos_0_5": "Número de niños de 0 a 5 años",
    "n_ninos_6_16": "Número de niños de 6 a 16 años",
    "nconglome": "Número de conglomerado en el marco",
    "num_perceptores": "Número de perceptores del hogar",
    "p101": "Tipo de vivienda",
    "p102": "Material predominante de la pared",
    "p103": "Material predominante del piso",
    "p103a": "Material predominante del techo",
    "p104": "Cantidad de habitaciones",
    "p105a": "Tenencia de la vivienda",
    "p106": "Condición o estado asociado a la vivienda",
    "p110": "Fuente de abastecimiento de agua",
    "p110a1": "Agua reportada como potable",
    "p110c": "Frecuencia de disponibilidad de agua",
    "p111a": "Tipo de servicio higiénico o desagüe",
    "p1121": "Disponibilidad de alumbrado por electricidad",
    "p112a": "Característica asociada al servicio eléctrico del hogar",
    "p1131": "Equipamiento/uso energético 1 del hogar",
    "p1132": "Equipamiento/uso energético 2 del hogar",
    "p1133": "Equipamiento/uso energético 3 del hogar",
    "p1135": "Equipamiento/uso energético 5 del hogar",
    "p1136": "Equipamiento/uso energético 6 del hogar",
    "p1137": "Equipamiento/uso energético 7 del hogar",
    "p1139": "Equipamiento/uso energético 9 del hogar",
    "p113a": "Combustible principal para cocinar",
    "p1141": "Acceso a tecnología/comunicación 1 en el hogar",
    "p1142": "Acceso a tecnología/comunicación 2 en el hogar",
    "p1143": "Acceso a tecnología/comunicación 3 en el hogar",
    "p1144": "Acceso a internet en el hogar",
    "p1145": "Acceso a tecnología/comunicación 5 en el hogar",
    "p114b1": "Tenencia de dispositivo o servicio digital 1",
    "p114b2": "Tenencia de dispositivo o servicio digital 2",
    "p114b3": "Tenencia de dispositivo o servicio digital 3",
    "piso_precario": "Indicador de piso precario",
    "pobrezav": "Pobreza y vulnerable con línea corriente",
    "provincia_cod": "Código de provincia",
    "saneamiento_inadecuado": "Indicador de saneamiento inadecuado",
    "sexo_jefe_bin": "Sexo del jefe del hogar",
    "sub_conglome": "Número de subconglomerado",
    "tam_hogar": "Tamaño del hogar",
    "tipo_desague": "Tipo de desagüe",
    "totmieho": "Total de personas en el hogar",
    "ubigeo": "Ubicación geográfica",
    "jefe_con_ingreso_laboral": "Jefe del hogar con ingreso laboral",
}


MODULE_OVERRIDES = {
    "acceso_agua_red": "Modulo01",
    "acceso_desague_red": "Modulo01",
    "acceso_electricidad": "Modulo01",
    "agua_disponible_diaria": "Modulo01",
    "agua_potable_reportada": "Modulo01",
    "agua_segura_proxy": "Modulo01",
    "any_6_16_no_asisten": "Modulo03",
    "any_6_16_no_matriculados": "Modulo03",
    "any_enfermedad_4w": "Modulo04",
    "any_ingreso_laboral_hogar": "Modulo05",
    "any_sintoma_4w": "Modulo04",
    "area_rural_bin": "Derivada",
    "area_urbana_rural": "Derivada",
    "children_not_attending": "Modulo03",
    "children_not_enrolled": "Modulo03",
    "combustible_limpio": "Modulo01",
    "combustible_precario": "Modulo01",
    "combustible_principal": "Modulo01",
    "distrito_cod": "Derivada",
    "dominio": "Sumaria",
    "educ_jefe": "Modulo02/Modulo03",
    "educ_jefe_ord": "Modulo02/Modulo03",
    "edad_jefe": "Modulo02",
    "estrato": "Sumaria",
    "estrsocial": "Sumaria",
    "factor07": "Sumaria",
    "fuente_agua": "Modulo01",
    "gasto_hogar": "Sumaria",
    "internet_hogar": "Modulo01",
    "material_piso": "Modulo01",
    "miembros_afiliados_essalud": "Modulo04",
    "miembros_con_enfermedad_4w": "Modulo04",
    "miembros_con_ingreso_laboral": "Modulo05",
    "miembros_con_sintoma_4w": "Modulo04",
    "miembros_sin_atencion_salud": "Modulo04",
    "n_6_16_matriculados": "Modulo03",
    "n_6_16_no_asisten": "Modulo03",
    "n_6_16_no_matriculados": "Modulo03",
    "n_6_16_total": "Modulo03",
    "n_adultos_65_mas": "Modulo02",
    "n_ninos_0_5": "Modulo02",
    "n_ninos_6_16": "Modulo02",
    "nconglome": "Sumaria",
    "num_perceptores": "Sumaria",
    "p101": "Modulo01",
    "p102": "Modulo01",
    "p103": "Modulo01",
    "p103a": "Modulo01",
    "p104": "Modulo01",
    "p105a": "Modulo01",
    "p106": "Modulo01",
    "p110": "Modulo01",
    "p110a1": "Modulo01",
    "p110c": "Modulo01",
    "p111a": "Modulo01",
    "p1121": "Modulo01",
    "p112a": "Modulo01",
    "p1131": "Modulo01",
    "p1132": "Modulo01",
    "p1133": "Modulo01",
    "p1135": "Modulo01",
    "p1136": "Modulo01",
    "p1137": "Modulo01",
    "p1139": "Modulo01",
    "p113a": "Modulo01",
    "p1141": "Modulo01",
    "p1142": "Modulo01",
    "p1143": "Modulo01",
    "p1144": "Modulo01",
    "p1145": "Modulo01",
    "p114b1": "Modulo01",
    "p114b2": "Modulo01",
    "p114b3": "Modulo01",
    "piso_precario": "Modulo01",
    "pobrezav": "Sumaria",
    "provincia_cod": "Derivada",
    "saneamiento_inadecuado": "Modulo01",
    "sexo_jefe_bin": "Modulo02",
    "sub_conglome": "Sumaria",
    "tam_hogar": "Sumaria",
    "tipo_desague": "Modulo01",
    "totmieho": "Sumaria",
    "ubigeo": "Sumaria",
    "jefe_con_ingreso_laboral": "Modulo05",
}


def load_inputs(paths: Any) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(paths.reports / FEATURES_FILE)
    sumaria_labels = pd.read_csv(paths.reports / SUMARIA_LABELS_FILE)
    if (paths.reports / DERIVED_DICT_FILE).exists():
        derived_dict = pd.read_csv(paths.reports / DERIVED_DICT_FILE)
    else:
        derived_dict = pd.DataFrame(columns=["variable_final", "descripcion"])
    return features, sumaria_labels, derived_dict


def norm(text: Any) -> str:
    return str(text or "").strip().lower()


def description_for(variable: str, sumaria_map: dict[str, str], derived_map: dict[str, str]) -> str:
    if variable in MANUAL_LABELS:
        return MANUAL_LABELS[variable]
    if variable in derived_map:
        return derived_map[variable]
    return sumaria_map.get(variable, variable)


def source_for(variable: str) -> str:
    if variable in MODULE_OVERRIDES:
        return MODULE_OVERRIDES[variable]
    if variable.startswith("p"):
        return "Modulo01"
    return "Sumaria"


def classify_variable(variable: str) -> dict[str, Any]:
    if variable == "factor07":
        return {
            "dimension_inei": "No aplica",
            "dimension_analitica": "Ponderacion muestral",
            "subdimension": "Expansion",
            "apta_selectkbest": False,
            "motivo_selectkbest": "Ponderador; conservar fuera de la matriz X.",
        }

    if variable in {"ubigeo", "dominio", "estrato", "estrsocial", "provincia_cod", "distrito_cod", "area_urbana_rural", "area_rural_bin", "mes", "nconglome", "sub_conglome"}:
        return {
            "dimension_inei": "No aplica",
            "dimension_analitica": "Geografia y contexto territorial",
            "subdimension": "Contexto espacial/muestral",
            "apta_selectkbest": variable in {"estrato", "area_rural_bin"},
            "motivo_selectkbest": "Contexto útil; evaluar según codificación y riesgo de redundancia.",
        }

    if variable in {"edad_jefe", "sexo_jefe_bin", "tam_hogar", "totmieho", "n_ninos_0_5", "n_ninos_6_16", "n_adultos_65_mas"}:
        return {
            "dimension_inei": "No aplica",
            "dimension_analitica": "Demografia y composicion del hogar",
            "subdimension": "Estructura demografica",
            "apta_selectkbest": variable != "totmieho",
            "motivo_selectkbest": "Variable estructural del hogar; totmieho es redundante con tam_hogar.",
        }

    if variable in {"educ_jefe", "educ_jefe_ord", "n_6_16_total", "n_6_16_matriculados", "n_6_16_no_matriculados", "n_6_16_no_asisten", "children_not_enrolled", "children_not_attending", "any_6_16_no_matriculados", "any_6_16_no_asisten", "personas_con_mod_educ"}:
        return {
            "dimension_inei": "Educacion",
            "dimension_analitica": "Educacion",
            "subdimension": "Capital educativo y asistencia escolar",
            "apta_selectkbest": variable not in {"n_6_16_total", "n_6_16_matriculados", "personas_con_mod_educ"},
            "motivo_selectkbest": "Mantener variables con señal de privación; evitar conteos auxiliares redundantes si fuera necesario.",
        }

    if variable in {"miembros_afiliados_essalud", "miembros_con_enfermedad_4w", "miembros_con_sintoma_4w", "miembros_sin_atencion_salud", "any_enfermedad_4w", "any_sintoma_4w", "health_no_attention", "personas_con_mod_salud"}:
        return {
            "dimension_inei": "Salud",
            "dimension_analitica": "Salud",
            "subdimension": "Morbilidad, atencion y aseguramiento",
            "apta_selectkbest": variable not in {"personas_con_mod_salud"},
            "motivo_selectkbest": "Adecuada para modelado; evitar auxiliares de cobertura del módulo.",
        }

    if variable in {"p101", "p102", "p103", "p103a", "p104", "p105a", "p106", "material_piso", "piso_precario"}:
        return {
            "dimension_inei": "Vivienda y Entorno",
            "dimension_analitica": "Vivienda y entorno",
            "subdimension": "Calidad física y tenencia",
            "apta_selectkbest": variable not in {"p103", "p106"},
            "motivo_selectkbest": "Usar preferentemente variables interpretables y evitar duplicidad con derivadas más limpias.",
        }

    if variable in {"p110", "p110a1", "p110c", "p111a", "fuente_agua", "tipo_desague", "acceso_agua_red", "agua_potable_reportada", "agua_disponible_diaria", "agua_segura_proxy", "acceso_desague_red", "saneamiento_inadecuado"}:
        return {
            "dimension_inei": "Servicios Básicos",
            "dimension_analitica": "Servicios basicos",
            "subdimension": "Agua y saneamiento",
            "apta_selectkbest": variable not in {"p110", "p110a1", "p110c", "p111a", "fuente_agua", "tipo_desague"},
            "motivo_selectkbest": "Preferir derivadas claras y binarias sobre códigos crudos cuando exista redundancia.",
        }

    if variable in {"p1121", "p112a", "p1131", "p1132", "p1133", "p1135", "p1136", "p1137", "p1139", "p113a", "acceso_electricidad", "combustible_principal", "combustible_limpio", "combustible_precario"}:
        return {
            "dimension_inei": "Energia",
            "dimension_analitica": "Energia",
            "subdimension": "Electricidad y combustible",
            "apta_selectkbest": variable in {"acceso_electricidad", "combustible_limpio", "combustible_precario", "combustible_principal"},
            "motivo_selectkbest": "Priorizar acceso a electricidad y calidad del combustible; otros p113* son auxiliares de equipamiento/uso.",
        }

    if variable in {"p1141", "p1142", "p1143", "p1144", "p1145", "p114b1", "p114b2", "p114b3", "internet_hogar"}:
        return {
            "dimension_inei": "Conectividad",
            "dimension_analitica": "Conectividad",
            "subdimension": "Acceso digital y equipamiento TIC",
            "apta_selectkbest": variable in {"internet_hogar", "p1144", "p114b1", "p114b2", "p114b3"},
            "motivo_selectkbest": "Mantener variables de conectividad final o de equipamiento digital con interpretación clara.",
        }

    if variable in {"miembros_con_ingreso_laboral", "any_ingreso_laboral_hogar", "jefe_con_ingreso_laboral", "num_perceptores", "personas_con_mod_empleo"}:
        return {
            "dimension_inei": "Empleo y Prevision Social",
            "dimension_analitica": "Empleo y prevision social",
            "subdimension": "Participación laboral e ingresos del trabajo",
            "apta_selectkbest": variable not in {"personas_con_mod_empleo"},
            "motivo_selectkbest": "Variables adecuadas para ML cuando no son montos monetarios directos.",
        }

    if variable == "pobrezav":
        return {
            "dimension_inei": "No aplica",
            "dimension_analitica": "Variables relacionadas al objetivo",
            "subdimension": "Pobreza/vulnerabilidad calculada",
            "apta_selectkbest": False,
            "motivo_selectkbest": "Muy cercana al objetivo; excluir por riesgo de leakage.",
        }

    if (
        variable.startswith("ing")
        or variable.startswith("g0")
        or variable.startswith("ga")
        or variable.startswith("ia")
        or variable.startswith("ig")
        or variable.startswith("gru")
        or variable.startswith("sg")
        or variable.startswith("sig")
        or variable.startswith("insed")
        or variable.startswith("isec")
        or variable in {"gashog1d", "gasto_hogar", "ld", "lineav", "paesechd", "pagesphd"}
    ):
        return {
            "dimension_inei": "No aplica",
            "dimension_analitica": "Ingresos, gastos y transferencias monetarias",
            "subdimension": "Agregados monetarios de Sumaria",
            "apta_selectkbest": False,
            "motivo_selectkbest": "Excluir en el enfoque dimensional por cercanía conceptual al target monetario o baja interpretabilidad.",
        }

    return {
        "dimension_inei": "No aplica",
        "dimension_analitica": "Revision manual",
        "subdimension": "Pendiente de clasificación fina",
        "apta_selectkbest": False,
        "motivo_selectkbest": "Requiere revisión manual antes del modelado.",
    }


def build_mapping(features: pd.DataFrame, sumaria_labels: pd.DataFrame, derived_dict: pd.DataFrame) -> pd.DataFrame:
    sumaria_map = {row["variable"]: row["label"] for _, row in sumaria_labels.iterrows() if pd.notna(row.get("label"))}
    derived_map = {row["variable_final"]: row["descripcion"] for _, row in derived_dict.iterrows() if pd.notna(row.get("descripcion"))}

    rows: list[dict[str, Any]] = []
    for record in features.to_dict(orient="records"):
        variable = record["variable"]
        meta = classify_variable(variable)
        rows.append(
            {
                "variable": variable,
                "descripcion": description_for(variable, sumaria_map, derived_map),
                "modulo_fuente": source_for(variable),
                "tipo_feature": record["tipo_feature"],
                "dtype": record["dtype"],
                "nulos_pct": record["nulos_pct"],
                "requiere_imputacion": record["requiere_imputacion"],
                "requiere_encoding": record["requiere_encoding"],
                "riesgo_leakage_reportado": record["riesgo_leakage"],
                "dimension_inei": meta["dimension_inei"],
                "dimension_analitica": meta["dimension_analitica"],
                "subdimension": meta["subdimension"],
                "apta_selectkbest": meta["apta_selectkbest"],
                "motivo_selectkbest": meta["motivo_selectkbest"],
            }
        )
    return pd.DataFrame(rows).sort_values(["dimension_analitica", "subdimension", "variable"]).reset_index(drop=True)


def write_notes(path: Path, mapped: pd.DataFrame) -> None:
    select_df = mapped.loc[mapped["apta_selectkbest"]].copy()
    text = "\n".join(
        [
            "# Mapa dimensional de variables",
            "",
            f"- Variables candidatas analizadas: {mapped.shape[0]}",
            f"- Variables marcadas como aptas para SelectKBest: {select_df.shape[0]}",
            "",
            "## Criterio",
            "",
            "- Se usaron las dimensiones del enfoque multidimensional del INEI cuando la variable encaja de forma sustantiva.",
            "- Las variables geográficas, de ponderación, identificación y agregados monetarios de Sumaria se clasificaron en familias analíticas separadas.",
            "- Para el enfoque de tesis, SelectKBest con f_classif debería aplicarse sobre las variables marcadas como aptas_selectkbest = True, luego de limpieza, imputación y encoding.",
            "",
            "## Resumen por dimensión analítica",
            "",
            *[
                f"- {row['dimension_analitica']}: {int(row['n_variables'])} variables"
                for _, row in mapped.groupby("dimension_analitica").size().reset_index(name="n_variables").iterrows()
            ],
        ]
    )
    path.write_text(text, encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    paths = build_paths(root)
    ensure_directories(paths)
    logger = configure_logging(paths.logs / f"dimension_mapping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    features, sumaria_labels, derived_dict = load_inputs(paths)
    mapped = build_mapping(features, sumaria_labels, derived_dict)

    summary = (
        mapped.groupby(["dimension_analitica", "dimension_inei", "apta_selectkbest"])
        .size()
        .reset_index(name="n_variables")
        .sort_values(["dimension_analitica", "apta_selectkbest"], ascending=[True, False])
    )

    selected = mapped.loc[mapped["apta_selectkbest"]].copy()

    mapped.to_csv(paths.reports / "variables_mapeadas_dimensiones.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(paths.reports / "variables_mapeadas_dimensiones_resumen.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(paths.reports / "variables_candidatas_selectkbest.csv", index=False, encoding="utf-8-sig")
    write_notes(paths.reports / "variables_mapeadas_dimensiones.md", mapped)

    validation = {
        "phase": "VARIABLE_DIMENSION_MAPPING",
        "passed": bool(mapped.shape[0] == features.shape[0]),
        "n_features_input": int(features.shape[0]),
        "n_features_mapped": int(mapped.shape[0]),
        "n_candidates_selectkbest": int(selected.shape[0]),
        "dimensiones_analiticas": sorted(mapped["dimension_analitica"].dropna().unique().tolist()),
    }
    save_json(validation, paths.reports / "variables_mapeadas_dimensiones_validation.json")
    logger.info("VARIABLE_DIMENSION_MAPPING validacion: %s", validation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
