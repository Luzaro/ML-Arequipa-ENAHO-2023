from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .enaho_etl_phase01 import build_paths, configure_logging, ensure_directories, save_json


BASE_FILE = "enaho_arequipa_hogar_base_phase5.pkl"
TARGET_RAW = "pobreza"

EDUC_LEVEL_MAP = {
    1: "sin nivel",
    2: "educacion inicial",
    3: "primaria incompleta",
    4: "primaria completa",
    5: "secundaria incompleta",
    6: "secundaria completa",
    7: "superior no universitaria incompleta",
    8: "superior no universitaria completa",
    9: "superior universitaria incompleta",
    10: "superior universitaria completa",
    11: "maestria_doctorado",
    12: "basica especial",
}


def derive_phase6_variables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()

    work["ubigeo"] = work["ubigeo"].astype(str).str.zfill(6)
    work["departamento_cod"] = work["ubigeo"].str[:2]
    work["provincia_cod"] = work["ubigeo"].str[:4]
    work["distrito_cod"] = work["ubigeo"].str[:6]

    work["sexo_jefe_bin"] = work["sexo_jefe"].astype(str).str.strip().str.lower().map({"hombre": 1, "mujer": 0})
    work["educ_jefe_ord"] = pd.to_numeric(work["educ_jefe_nivel"], errors="coerce")
    work["educ_jefe"] = work["educ_jefe_ord"].map(EDUC_LEVEL_MAP)

    work["tam_hogar"] = pd.to_numeric(work["mieperho"], errors="coerce")
    work["num_perceptores"] = pd.to_numeric(work["percepho"], errors="coerce")
    work["ingreso_hogar"] = pd.to_numeric(work["inghog1d"], errors="coerce")
    work["gasto_hogar"] = pd.to_numeric(work["gashog2d"], errors="coerce")
    work["ingreso_percapita"] = work["ingreso_hogar"] / work["tam_hogar"].replace({0: np.nan})

    work["material_piso"] = work["p103"].astype(str).str.strip().str.lower()
    work["piso_precario"] = work["material_piso"].isin(["tierra", "otro material"]).astype(int)

    fuente_agua = work["p110"].astype(str).str.strip().str.lower()
    work["fuente_agua"] = fuente_agua
    work["acceso_agua_red"] = fuente_agua.str.contains("red publica", na=False).astype(int)
    work["agua_potable_reportada"] = work["p110a1"].astype(str).str.strip().str.lower().eq("si").astype(int)
    work["agua_disponible_diaria"] = work["p110c"].astype(str).str.strip().str.lower().eq("si").astype(int)
    work["agua_segura_proxy"] = (
        (work["acceso_agua_red"] == 1) & (work["agua_potable_reportada"] == 1)
    ).astype(int)

    desague = work["p111a"].astype(str).str.strip().str.lower()
    work["tipo_desague"] = desague
    work["acceso_desague_red"] = desague.str.contains("red publica", na=False).astype(int)
    work["saneamiento_inadecuado"] = desague.isin(["campo abierto o al aire libre", "rio, acequia, canal o similar", "otra"]).astype(int)

    work["acceso_electricidad"] = work["p1121"].astype(str).str.strip().str.lower().eq("electricidad").astype(int)

    combustible = work["p113a"].astype(str).str.strip().str.lower()
    work["combustible_principal"] = combustible
    work["combustible_limpio"] = combustible.isin(
        ["electricidad", "gas (balon glp)", "gas natural (sistema de tuberias)"]
    ).astype(int)
    work["combustible_precario"] = combustible.isin(
        ["leña", "carbon", "bosta, estiercol", "otro"]
    ).astype(int)

    work["internet_hogar"] = work["p1144"].astype(str).str.strip().str.lower().eq("conexion a internet").astype(int)
    estrato = work["estrato"].astype(str).str.strip().str.lower()
    work["area_urbana_rural"] = np.where(estrato.str.contains("rural", na=False), "rural", "urbana")
    work["area_rural_bin"] = (work["area_urbana_rural"] == "rural").astype(int)

    work["health_no_attention"] = work["any_sin_atencion_salud"].fillna(0).astype(int)
    work["children_not_enrolled"] = work["any_6_16_no_matriculados"].fillna(0).astype(int)
    work["children_not_attending"] = work["any_6_16_no_asisten"].fillna(0).astype(int)

    work["tam_hogar_consistente"] = (work["tam_hogar"] == work["tam_hogar_mod2"]).astype(int)
    work["perceptores_leq_tam_hogar"] = (work["num_perceptores"] <= work["tam_hogar"]).astype(int)

    variable_dict = pd.DataFrame(
        [
            {"variable_final": "edad_jefe", "descripcion": "Edad del jefe del hogar", "tablas_fuente": "Modulo02", "columnas_fuente": "p203,p208a", "tipo": "numerica", "regla_transformacion": "p208a del miembro con p203=jefe/jefa", "porcentaje_nulos": round(work["edad_jefe"].isna().mean() * 100, 4), "observaciones": "Validado contra identificacion del jefe desde Modulo02"},
            {"variable_final": "sexo_jefe_bin", "descripcion": "Sexo del jefe del hogar", "tablas_fuente": "Modulo02", "columnas_fuente": "p203,p207", "tipo": "binaria", "regla_transformacion": "hombre=1, mujer=0 para jefe", "porcentaje_nulos": round(work["sexo_jefe_bin"].isna().mean() * 100, 4), "observaciones": "Conserva sexo_jefe textual en dataset base"},
            {"variable_final": "educ_jefe", "descripcion": "Nivel educativo del jefe", "tablas_fuente": "Modulo02,Modulo03", "columnas_fuente": "p203,p301a", "tipo": "categorica", "regla_transformacion": "Mapeo de p301a del jefe usando etiquetas oficiales ENAHO", "porcentaje_nulos": round(work["educ_jefe"].isna().mean() * 100, 4), "observaciones": "No se construyo anios_educ_jefe por ambiguedad en homologacion exacta de anos/grados"},
            {"variable_final": "tam_hogar", "descripcion": "Tamano del hogar", "tablas_fuente": "Sumaria", "columnas_fuente": "mieperho", "tipo": "numerica", "regla_transformacion": "Copia de mieperho", "porcentaje_nulos": round(work["tam_hogar"].isna().mean() * 100, 4), "observaciones": "Se valida contra tam_hogar_mod2"},
            {"variable_final": "num_perceptores", "descripcion": "Numero de perceptores del hogar", "tablas_fuente": "Sumaria", "columnas_fuente": "percepho", "tipo": "numerica", "regla_transformacion": "Copia de percepho", "porcentaje_nulos": round(work["num_perceptores"].isna().mean() * 100, 4), "observaciones": "Variable oficial de Sumaria"},
            {"variable_final": "material_piso", "descripcion": "Material predominante del piso", "tablas_fuente": "Modulo01", "columnas_fuente": "p103", "tipo": "categorica", "regla_transformacion": "Normalizacion textual a minusculas", "porcentaje_nulos": round(work["material_piso"].isna().mean() * 100, 4), "observaciones": "Se complementa con piso_precario"},
            {"variable_final": "piso_precario", "descripcion": "Indicador proxy de piso precario", "tablas_fuente": "Modulo01", "columnas_fuente": "p103", "tipo": "binaria", "regla_transformacion": "1 si piso es tierra u otro material", "porcentaje_nulos": round(work["piso_precario"].isna().mean() * 100, 4), "observaciones": "Proxy simple y explicita"},
            {"variable_final": "agua_segura_proxy", "descripcion": "Indicador proxy de agua mas segura", "tablas_fuente": "Modulo01", "columnas_fuente": "p110,p110a1", "tipo": "binaria", "regla_transformacion": "1 si fuente es red publica y p110a1 reporta agua potable", "porcentaje_nulos": round(work["agua_segura_proxy"].isna().mean() * 100, 4), "observaciones": "Proxy, no equivalente exacto a definicion normativa externa"},
            {"variable_final": "acceso_desague_red", "descripcion": "Acceso a desague por red publica", "tablas_fuente": "Modulo01", "columnas_fuente": "p111a", "tipo": "binaria", "regla_transformacion": "1 si p111a contiene red publica", "porcentaje_nulos": round(work["acceso_desague_red"].isna().mean() * 100, 4), "observaciones": "Mantiene tambien tipo_desague"},
            {"variable_final": "acceso_electricidad", "descripcion": "Acceso a electricidad", "tablas_fuente": "Modulo01", "columnas_fuente": "p1121", "tipo": "binaria", "regla_transformacion": "1 si p1121=electricidad", "porcentaje_nulos": round(work["acceso_electricidad"].isna().mean() * 100, 4), "observaciones": "Basado en alumbrado del hogar"},
            {"variable_final": "combustible_limpio", "descripcion": "Combustible principal limpio para cocinar", "tablas_fuente": "Modulo01", "columnas_fuente": "p113a", "tipo": "binaria", "regla_transformacion": "1 si combustible principal es electricidad, GLP o gas natural", "porcentaje_nulos": round(work["combustible_limpio"].isna().mean() * 100, 4), "observaciones": "Se complementa con combustible_precario"},
            {"variable_final": "internet_hogar", "descripcion": "Hogar con conexion a internet", "tablas_fuente": "Modulo01", "columnas_fuente": "p1144", "tipo": "binaria", "regla_transformacion": "1 si p1144 indica conexion a internet", "porcentaje_nulos": round(work["internet_hogar"].isna().mean() * 100, 4), "observaciones": "Indicador directo del hogar"},
            {"variable_final": "area_urbana_rural", "descripcion": "Condicion urbana o rural", "tablas_fuente": "Sumaria", "columnas_fuente": "estrato", "tipo": "categorica", "regla_transformacion": "rural si estrato contiene rural; en otro caso urbana", "porcentaje_nulos": round(work["area_urbana_rural"].isna().mean() * 100, 4), "observaciones": "Derivado textual del estrato"},
            {"variable_final": "health_no_attention", "descripcion": "Hogar con al menos un miembro sin atencion de salud", "tablas_fuente": "Modulo04", "columnas_fuente": "p4091..p40911", "tipo": "binaria", "regla_transformacion": "1 si algun miembro reporta motivo de no acudir al centro de salud", "porcentaje_nulos": round(work["health_no_attention"].isna().mean() * 100, 4), "observaciones": "Agregado persona->hogar"},
            {"variable_final": "children_not_enrolled", "descripcion": "Hogar con menores 6-16 no matriculados", "tablas_fuente": "Modulo02,Modulo03", "columnas_fuente": "p208a,p306", "tipo": "binaria", "regla_transformacion": "1 si existe al menos una persona 6-16 con p306=2", "porcentaje_nulos": round(work["children_not_enrolled"].isna().mean() * 100, 4), "observaciones": "Agregado persona->hogar"},
            {"variable_final": "ingreso_hogar", "descripcion": "Ingreso bruto total del hogar", "tablas_fuente": "Sumaria", "columnas_fuente": "inghog1d", "tipo": "numerica", "regla_transformacion": "Copia de inghog1d", "porcentaje_nulos": round(work["ingreso_hogar"].isna().mean() * 100, 4), "observaciones": "Riesgo potencial de leakage a evaluar en FASE 9"},
        ]
    )
    return work, variable_dict


def distribution_report(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            rows.append(
                {
                    "variable": col,
                    "tipo": "numerica",
                    "nulos_pct": round(series.isna().mean() * 100, 4),
                    "min": series.min(),
                    "p25": series.quantile(0.25),
                    "p50": series.quantile(0.50),
                    "p75": series.quantile(0.75),
                    "max": series.max(),
                    "rare_categories": "",
                }
            )
        else:
            counts = series.astype(str).value_counts(dropna=False)
            rare = counts[counts < 10].index.tolist()[:10]
            rows.append(
                {
                    "variable": col,
                    "tipo": "categorica",
                    "nulos_pct": round(series.isna().mean() * 100, 4),
                    "min": None,
                    "p25": None,
                    "p50": None,
                    "p75": None,
                    "max": None,
                    "rare_categories": json.dumps(rare, ensure_ascii=False),
                }
            )
    return pd.DataFrame(rows)


def binary_validation(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        values = sorted(pd.Series(df[col].dropna().unique()).tolist())
        rows.append({"variable": col, "unique_values": json.dumps(values, ensure_ascii=False), "is_binary_01": set(values).issubset({0, 1})})
    return pd.DataFrame(rows)


def target_phase7(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = df.copy()
    raw = work[TARGET_RAW].astype(str).str.strip().str.lower()
    mapping = {"pobre extremo": 1, "pobre no extremo": 1, "no pobre": 0}
    work["target_pobreza_bin"] = raw.map(mapping)

    validation = {
        "phase": "FASE 7",
        "target_source_table": "Sumaria",
        "target_source_variable": TARGET_RAW,
        "target_source_label": "pobreza",
        "target_raw_frequencies": raw.value_counts(dropna=False).to_dict(),
        "target_binary_frequencies": work["target_pobreza_bin"].value_counts(dropna=False).to_dict(),
        "target_nulls": int(work["target_pobreza_bin"].isna().sum()),
        "target_rows_before_dropna": int(work.shape[0]),
    }
    work = work[work["target_pobreza_bin"].notna()].copy()
    validation["target_rows_after_dropna"] = int(work.shape[0])
    validation["target_is_binary"] = set(work["target_pobreza_bin"].dropna().unique()).issubset({0, 1})
    validation["target_passed"] = validation["target_nulls"] == 0 and validation["target_is_binary"]
    return work, validation


def logical_checks(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "tam_hogar_ge_num_perceptores_violations": int((df["tam_hogar"] < df["num_perceptores"]).sum()),
        "edad_jefe_outside_12_100": int(((df["edad_jefe"] < 12) | (df["edad_jefe"] > 100)).sum()),
        "tam_hogar_vs_mod2_mismatches": int((df["tam_hogar"] != df["tam_hogar_mod2"]).sum()),
    }


def outlier_report(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        s = pd.to_numeric(df[col], errors="coerce")
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        lower = q1 - 1.5 * iqr
        rows.append(
            {
                "variable": col,
                "lower_bound": lower,
                "upper_bound": upper,
                "outlier_count": int(((s < lower) | (s > upper)).sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    paths = build_paths(root)
    ensure_directories(paths)
    logger = configure_logging(paths.logs / f"etl_phase06_07_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    base_df = pd.read_pickle(paths.data_interim / BASE_FILE)
    phase6_df, variable_dict = derive_phase6_variables(base_df)

    selected_cols = [
        "edad_jefe",
        "sexo_jefe_bin",
        "educ_jefe",
        "educ_jefe_ord",
        "tam_hogar",
        "num_perceptores",
        "material_piso",
        "piso_precario",
        "acceso_agua_red",
        "agua_potable_reportada",
        "agua_segura_proxy",
        "acceso_desague_red",
        "acceso_electricidad",
        "combustible_limpio",
        "combustible_precario",
        "internet_hogar",
        "area_urbana_rural",
        "health_no_attention",
        "children_not_enrolled",
        "children_not_attending",
        "ingreso_hogar",
        "ingreso_percapita",
    ]
    binary_cols = [
        "sexo_jefe_bin",
        "piso_precario",
        "acceso_agua_red",
        "agua_potable_reportada",
        "agua_segura_proxy",
        "acceso_desague_red",
        "acceso_electricidad",
        "combustible_limpio",
        "combustible_precario",
        "internet_hogar",
        "health_no_attention",
        "children_not_enrolled",
        "children_not_attending",
        "area_rural_bin",
    ]

    phase6_df.to_pickle(paths.data_interim / "enaho_arequipa_hogar_phase6.pkl")
    variable_dict.to_csv(paths.reports / "phase6_variable_dictionary.csv", index=False, encoding="utf-8-sig")
    distribution_report(phase6_df, selected_cols).to_csv(paths.reports / "phase6_distributions.csv", index=False, encoding="utf-8-sig")
    binary_validation(phase6_df, binary_cols).to_csv(paths.reports / "phase6_binary_validation.csv", index=False, encoding="utf-8-sig")
    outlier_report(phase6_df, ["edad_jefe", "tam_hogar", "num_perceptores", "ingreso_hogar", "ingreso_percapita"]).to_csv(paths.reports / "phase6_outliers.csv", index=False, encoding="utf-8-sig")
    checks = logical_checks(phase6_df)
    save_json(checks, paths.reports / "phase6_logical_checks.json")

    phase7_df, target_validation = target_phase7(phase6_df)
    phase7_df.to_pickle(paths.data_interim / "enaho_arequipa_hogar_phase7.pkl")
    save_json(target_validation, paths.reports / "phase7_target_validation.json")

    validation = {
        "phase": "FASE 6-7",
        "passed": target_validation["target_passed"] and checks["tam_hogar_ge_num_perceptores_violations"] == 0,
        "phase6_rows": int(phase6_df.shape[0]),
        "phase7_rows": int(phase7_df.shape[0]),
        "target_variable": "target_pobreza_bin",
        "target_source_variable": TARGET_RAW,
    }
    save_json(validation, paths.reports / "phase6_7_validation.json")
    logger.info("FASE 6-7 validacion: %s", validation)
    return 0 if validation["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
