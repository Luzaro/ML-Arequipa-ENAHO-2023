from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .enaho_etl_phase01 import build_paths, configure_logging, ensure_directories, save_json


PHASE7_FILE = "enaho_arequipa_hogar_phase7.pkl"
SCENARIO_B_CANDIDATES_FILE = "candidate_variables_selectkbest_social.csv"

ID_COLS = ["conglome", "vivienda", "hogar", "ubigeo"]
EXTRA_COLS = ["factor07", "target_pobreza_bin"]

ALIMENTARY_PROGRAM_VARS = ["ingtpu16", "sg27", "sig28"]
NON_ALIMENTARY_PROGRAM_VARS = ["ingtpu01", "ingtpu02", "ingtpu03", "ingtpu04", "ingtpu05", "ingtpu10", "ingtpu12", "ingtpu13", "ingtpu14"]

RENAME_MAP = {
    "target_pobreza_bin": "target_pobreza_monetaria_bin",
    "factor07": "factor_expansion_anual",
    "internet_hogar": "acceso_internet_hogar_derivado",
    "p1144": "internet_hogar_enaho",
    "p114b1": "dispositivo_servicio_digital_1",
    "p114b2": "dispositivo_servicio_digital_2",
    "p114b3": "dispositivo_servicio_digital_3",
    "edad_jefe": "edad_jefe",
    "n_adultos_65_mas": "num_adultos_65_mas",
    "n_ninos_0_5": "num_ninos_0_5",
    "n_ninos_6_16": "num_ninos_6_16",
    "sexo_jefe_bin": "sexo_jefe_bin",
    "tam_hogar": "tam_hogar",
    "any_6_16_no_asisten": "al_menos_un_6_16_no_asiste",
    "any_6_16_no_matriculados": "al_menos_un_6_16_no_matriculado",
    "children_not_attending": "hogar_con_ninos_no_asisten",
    "children_not_enrolled": "hogar_con_ninos_no_matriculados",
    "educ_jefe": "nivel_educativo_jefe_cat",
    "educ_jefe_ord": "nivel_educativo_jefe_ord",
    "n_6_16_no_asisten": "num_6_16_no_asisten",
    "n_6_16_no_matriculados": "num_6_16_no_matriculados",
    "any_ingreso_laboral_hogar": "al_menos_un_ingreso_laboral_hogar",
    "jefe_con_ingreso_laboral": "jefe_con_ingreso_laboral",
    "miembros_con_ingreso_laboral": "num_miembros_con_ingreso_laboral",
    "num_perceptores": "num_perceptores_hogar",
    "acceso_electricidad": "acceso_electricidad",
    "combustible_limpio": "usa_combustible_limpio",
    "combustible_precario": "usa_combustible_precario",
    "combustible_principal": "combustible_principal_cocina",
    "area_rural_bin": "es_area_rural",
    "estrato": "estrato_geografico",
    "any_enfermedad_4w": "al_menos_un_enfermedad_4_sem",
    "any_sintoma_4w": "al_menos_un_sintoma_4_sem",
    "miembros_afiliados_essalud": "num_miembros_afiliados_essalud",
    "miembros_con_enfermedad_4w": "num_miembros_con_enfermedad_4_sem",
    "miembros_con_sintoma_4w": "num_miembros_con_sintoma_4_sem",
    "miembros_sin_atencion_salud": "num_miembros_sin_atencion_salud",
    "acceso_agua_red": "acceso_agua_red",
    "acceso_desague_red": "acceso_desague_red",
    "agua_disponible_diaria": "agua_disponible_diaria",
    "agua_potable_reportada": "agua_potable_reportada",
    "agua_segura_proxy": "agua_segura_proxy",
    "saneamiento_inadecuado": "saneamiento_inadecuado",
    "material_piso": "material_piso_cat",
    "p101": "tipo_vivienda",
    "p102": "material_pared",
    "p103a": "material_techo",
    "p104": "num_habitaciones",
    "p105a": "tenencia_vivienda",
    "piso_precario": "piso_precario",
    "inghog2d": "ingreso_neto_total_hogar",
    "ingtpu01": "monto_juntos",
    "ingtpu02": "monto_otras_transferencias_publicas",
    "ingtpu03": "monto_pension_65",
    "ingtpu04": "monto_beca_18",
    "ingtpu05": "monto_bono_gas",
    "ingtpu10": "monto_bono_electricidad",
    "ingtpu12": "monto_bono_onp_jubilados",
    "ingtpu13": "monto_programa_contigo",
    "ingtpu14": "monto_bono_yanapay",
    "ingtpu16": "monto_bono_alimentario",
    "ingtpuhd": "monto_total_transferencias_publicas",
    "monto_programas_sociales_alimentarios": "monto_total_programas_sociales_alimentarios",
    "monto_programas_sociales_no_alimentarios": "monto_total_programas_sociales_no_alimentarios",
    "recibe_programa_social_alimentario": "recibe_programa_social_alimentario",
    "recibe_programa_social_no_alimentario": "recibe_programa_social_no_alimentario",
    "sg27": "monto_apoyo_ollas_comunes",
    "sig28": "monto_ingresos_gastos_ollas_comunes",
}


def parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    safe = df.copy()
    for col in safe.columns:
        if str(safe[col].dtype) in {"category", "object"}:
            safe[col] = safe[col].astype("string")
    return safe


def clean_id_part(value: Any) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def build_hogar_id(df: pd.DataFrame) -> pd.Series:
    return df.apply(
        lambda row: "_".join(clean_id_part(row[col]) for col in ["ubigeo", "conglome", "vivienda", "hogar"]),
        axis=1,
    )


def load_inputs(paths: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    phase7 = pd.read_pickle(paths.data_interim / PHASE7_FILE)
    candidates = pd.read_csv(paths.reports / "selectkbest_social" / SCENARIO_B_CANDIDATES_FILE)
    return phase7, candidates


def add_social_derived_variables(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in set(ALIMENTARY_PROGRAM_VARS + NON_ALIMENTARY_PROGRAM_VARS + ["ingtpuhd", "inghog2d"]):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    out["monto_programas_sociales_alimentarios"] = out[[c for c in ALIMENTARY_PROGRAM_VARS if c in out.columns]].sum(axis=1)
    out["monto_programas_sociales_no_alimentarios"] = out[[c for c in NON_ALIMENTARY_PROGRAM_VARS if c in out.columns]].sum(axis=1)
    out["recibe_programa_social_alimentario"] = (out["monto_programas_sociales_alimentarios"] > 0).astype(int)
    out["recibe_programa_social_no_alimentario"] = (out["monto_programas_sociales_no_alimentarios"] > 0).astype(int)
    return out


def build_mapping(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "variable_original": "hogar_id",
            "variable_renombrada": "hogar_id",
            "dimension_analitica": "Identificacion",
            "descripcion": "Identificador unico del hogar: ubigeo_conglome_vivienda_hogar",
            "observacion": "Creada para trazabilidad; no entra a modelado.",
        },
        {
            "variable_original": "factor07",
            "variable_renombrada": "factor_expansion_anual",
            "dimension_analitica": "Ponderacion",
            "descripcion": "Factor de expansión anual del hogar",
            "observacion": "Se conserva para análisis ponderado; no entra a modelado.",
        },
        {
            "variable_original": "target_pobreza_bin",
            "variable_renombrada": "target_pobreza_monetaria_bin",
            "dimension_analitica": "Objetivo",
            "descripcion": "Condición oficial de pobreza monetaria binaria",
            "observacion": "Variable dependiente.",
        },
    ]

    for _, row in candidates.iterrows():
        original = row["variable"]
        renamed = RENAME_MAP.get(original, original)
        rows.append(
            {
                "variable_original": original,
                "variable_renombrada": renamed,
                "dimension_analitica": row.get("dimension_analitica", ""),
                "descripcion": row.get("descripcion", ""),
                "observacion": "Variable candidata del escenario B.",
            }
        )

    mapping = pd.DataFrame(rows)
    if mapping["variable_renombrada"].duplicated().any():
        duplicated = mapping.loc[mapping["variable_renombrada"].duplicated(keep=False), "variable_renombrada"].tolist()
        raise ValueError(f"Existen nombres renombrados duplicados: {duplicated}")
    return mapping


def build_dataset(df: pd.DataFrame, candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_cols = candidates["variable"].tolist()
    keep_cols = ["conglome", "vivienda", "hogar", "ubigeo", "factor07", "target_pobreza_bin", *candidate_cols]
    keep_cols = list(dict.fromkeys([col for col in keep_cols if col in df.columns]))

    out = df[keep_cols].copy()
    if out.duplicated(subset=["conglome", "vivienda", "hogar"]).any():
        raise ValueError("La base del escenario B no es unica por hogar.")

    out.insert(0, "hogar_id", build_hogar_id(out))
    if out["hogar_id"].duplicated().any():
        raise ValueError("hogar_id no es unico en el escenario B.")

    rename_map = {"factor07": "factor_expansion_anual", "target_pobreza_bin": "target_pobreza_monetaria_bin"}
    rename_map.update({row["variable_original"]: row["variable_renombrada"] for _, row in build_mapping(candidates).iterrows()})
    rename_map = {k: v for k, v in rename_map.items() if k in out.columns}

    out = out.rename(columns=rename_map)
    ordered_cols = [
        "hogar_id",
        "conglome",
        "vivienda",
        "hogar",
        "ubigeo",
        "factor_expansion_anual",
        "target_pobreza_monetaria_bin",
        *[RENAME_MAP.get(col, col) for col in candidate_cols if RENAME_MAP.get(col, col) in out.columns],
    ]
    ordered_cols = list(dict.fromkeys([col for col in ordered_cols if col in out.columns]))
    return out[ordered_cols].copy(), build_mapping(candidates)


def write_report(path: Path, mapping: pd.DataFrame) -> None:
    dims = mapping.loc[~mapping["dimension_analitica"].isin(["Identificacion", "Ponderacion", "Objetivo"])].groupby("dimension_analitica").size()
    lines = [
        "# Renombrado del escenario B",
        "",
        "- Se renombraron todas las variables candidatas del escenario B a nombres analíticos en snake_case.",
        "- La salida conserva hogar_id, llaves originales, ponderador y target.",
        "",
        "## Variables por dimensión",
        "",
    ]
    lines.extend([f"- {dim}: {count} variables" for dim, count in dims.items()])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    paths = build_paths(root)
    ensure_directories(paths)
    logger = configure_logging(paths.logs / f"scenario_b_rename_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    phase7, candidates = load_inputs(paths)
    phase7 = add_social_derived_variables(phase7)
    renamed_df, mapping = build_dataset(phase7, candidates)

    parquet_path = paths.data_processed / "enaho_arequipa_escenario_b_renamed.parquet"
    csv_path = paths.data_processed / "enaho_arequipa_escenario_b_renamed.csv"

    parquet_safe(renamed_df).to_parquet(parquet_path, index=False)
    renamed_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    mapping.to_csv(paths.reports / "escenario_b_renombrado_map.csv", index=False, encoding="utf-8-sig")
    write_report(paths.reports / "escenario_b_renombrado.md", mapping)

    validation = {
        "phase": "SCENARIO_B_RENAME",
        "passed": True,
        "rows": int(renamed_df.shape[0]),
        "cols": int(renamed_df.shape[1]),
        "hogar_id_unique": bool(not renamed_df["hogar_id"].duplicated().any()),
        "variables_escenario_b": int(candidates.shape[0]),
        "target_present": "target_pobreza_monetaria_bin" in renamed_df.columns,
        "factor_present": "factor_expansion_anual" in renamed_df.columns,
    }
    save_json(validation, paths.reports / "escenario_b_renombrado_validation.json")
    logger.info("SCENARIO_B_RENAME validacion: %s", validation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
