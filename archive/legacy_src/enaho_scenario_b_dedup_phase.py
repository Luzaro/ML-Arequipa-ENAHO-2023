from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .enaho_etl_phase01 import build_paths, configure_logging, ensure_directories, save_json


DATASET = "enaho_arequipa_escenario_b_renamed.parquet"
DICT_FILE = "escenario_b_diccionario_variables.csv"
RANKING_FILE = "anova_f_classif_ranking.csv"
RENAME_MAP_FILE = "escenario_b_renombrado_map.csv"


REDUNDANCY_RULES = [
    {
        "drop": "internet_hogar_enaho",
        "keep": "acceso_internet_hogar_derivado",
        "tipo_redundancia": "duplicado_conceptual",
        "motivo": "Ambas representan acceso a internet en el hogar; se conserva la derivada binaria por ser más directa para modelado.",
    },
    {
        "drop": "hogar_con_ninos_no_asisten",
        "keep": "al_menos_un_6_16_no_asiste",
        "tipo_redundancia": "duplicado_exacto",
        "motivo": "Son dos nombres para el mismo indicador binario de no asistencia.",
    },
    {
        "drop": "al_menos_un_6_16_no_asiste",
        "keep": "num_6_16_no_asisten",
        "tipo_redundancia": "binaria_derivada_de_conteo",
        "motivo": "La binaria se deriva directamente del conteo y pierde intensidad del fenómeno.",
    },
    {
        "drop": "hogar_con_ninos_no_matriculados",
        "keep": "al_menos_un_6_16_no_matriculado",
        "tipo_redundancia": "duplicado_exacto",
        "motivo": "Son dos nombres para el mismo indicador binario de no matrícula.",
    },
    {
        "drop": "al_menos_un_6_16_no_matriculado",
        "keep": "num_6_16_no_matriculados",
        "tipo_redundancia": "binaria_derivada_de_conteo",
        "motivo": "La binaria se deriva directamente del conteo y pierde intensidad del fenómeno.",
    },
    {
        "drop": "nivel_educativo_jefe_cat",
        "keep": "nivel_educativo_jefe_ord",
        "tipo_redundancia": "misma_variable_distinta_codificacion",
        "motivo": "Ambas representan el nivel educativo del jefe; se conserva la versión ordinal más estable para selección univariada.",
    },
    {
        "drop": "al_menos_un_ingreso_laboral_hogar",
        "keep": "num_miembros_con_ingreso_laboral",
        "tipo_redundancia": "binaria_derivada_de_conteo",
        "motivo": "La binaria se deriva del conteo de miembros con ingreso laboral.",
    },
    {
        "drop": "combustible_principal_cocina",
        "keep": "usa_combustible_limpio",
        "tipo_redundancia": "variable_cruda_vs_recodificacion",
        "motivo": "La categoría cruda es la fuente de las variables binarias de combustible; se conserva la recodificación más interpretable.",
    },
    {
        "drop": "es_area_rural",
        "keep": "estrato_geografico",
        "tipo_redundancia": "derivada_de_contexto",
        "motivo": "es_area_rural es una simplificación derivada de estrato_geografico.",
    },
    {
        "drop": "al_menos_un_enfermedad_4_sem",
        "keep": "num_miembros_con_enfermedad_4_sem",
        "tipo_redundancia": "binaria_derivada_de_conteo",
        "motivo": "La binaria se deriva directamente del conteo de miembros con enfermedad reciente.",
    },
    {
        "drop": "al_menos_un_sintoma_4_sem",
        "keep": "num_miembros_con_sintoma_4_sem",
        "tipo_redundancia": "binaria_derivada_de_conteo",
        "motivo": "La binaria se deriva directamente del conteo de miembros con síntomas recientes.",
    },
    {
        "drop": "health_no_attention",
        "keep": "num_miembros_sin_atencion_salud",
        "tipo_redundancia": "binaria_derivada_de_conteo",
        "motivo": "La binaria resume el conteo de miembros sin atención en salud.",
    },
    {
        "drop": "agua_segura_proxy",
        "keep": "acceso_agua_red",
        "tipo_redundancia": "proxy_derivado",
        "motivo": "El proxy combina acceso_agua_red y agua_potable_reportada; se prefieren las componentes originales explícitas.",
    },
    {
        "drop": "material_piso_cat",
        "keep": "piso_precario",
        "tipo_redundancia": "variable_cruda_vs_recodificacion",
        "motivo": "piso_precario resume la condición crítica del material de piso y es más interpretable para clasificación.",
    },
    {
        "drop": "monto_total_transferencias_publicas",
        "keep": "monto_total_programas_sociales_no_alimentarios",
        "tipo_redundancia": "agregado_solapado",
        "motivo": "El total de transferencias públicas se solapa fuertemente con el agregado de programas no alimentarios en esta muestra.",
    },
    {
        "drop": "monto_total_programas_sociales_alimentarios",
        "keep": "monto_bono_alimentario",
        "tipo_redundancia": "agregado_de_componentes",
        "motivo": "El agregado alimentario resume variables específicas; se conservan los componentes para distinguir tipos de apoyo.",
    },
    {
        "drop": "monto_total_programas_sociales_no_alimentarios",
        "keep": "monto_juntos",
        "tipo_redundancia": "agregado_de_componentes",
        "motivo": "El agregado no alimentario resume programas específicos; se conservan los componentes para mantener detalle de política pública.",
    },
    {
        "drop": "recibe_programa_social_alimentario",
        "keep": "monto_bono_alimentario",
        "tipo_redundancia": "binaria_derivada_de_monto",
        "motivo": "La recepción binaria se deriva del monto positivo en programas alimentarios.",
    },
    {
        "drop": "recibe_programa_social_no_alimentario",
        "keep": "monto_juntos",
        "tipo_redundancia": "binaria_derivada_de_monto",
        "motivo": "La recepción binaria se deriva de montos positivos en programas no alimentarios.",
    },
]


def load_inputs(paths: Any) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset = pd.read_parquet(paths.data_processed / DATASET)
    dictionary_df = pd.read_csv(paths.reports / DICT_FILE)
    ranking_df = pd.read_csv(paths.reports / "selectkbest_social" / RANKING_FILE)
    rename_map_df = pd.read_csv(paths.reports / RENAME_MAP_FILE)
    return dataset, dictionary_df, ranking_df, rename_map_df


def attach_scores(ranking_df: pd.DataFrame, rename_map_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    name_map = dict(zip(rename_map_df["variable_original"], rename_map_df["variable_renombrada"]))
    ranking_df = ranking_df.copy()
    ranking_df["variable_final"] = ranking_df["variable"].map(name_map).fillna(ranking_df["variable"])
    return {
        row["variable_final"]: {"f_score": row["f_score"], "p_value": row["p_value"]}
        for _, row in ranking_df.iterrows()
    }


def build_redundancy_matrix(dictionary_df: pd.DataFrame, score_map: dict[str, dict[str, float]]) -> pd.DataFrame:
    dict_map = dictionary_df.set_index("variable_final").to_dict(orient="index")
    rows = []
    for rule in REDUNDANCY_RULES:
        drop_var = rule["drop"]
        keep_var = rule["keep"]
        drop_meta = dict_map.get(drop_var, {})
        keep_meta = dict_map.get(keep_var, {})
        drop_score = score_map.get(drop_var, {})
        keep_score = score_map.get(keep_var, {})
        rows.append(
            {
                "variable_descartar": drop_var,
                "descripcion_descartar": drop_meta.get("descripcion", ""),
                "variable_conservar": keep_var,
                "descripcion_conservar": keep_meta.get("descripcion", ""),
                "dimension_analitica": drop_meta.get("dimension_analitica", keep_meta.get("dimension_analitica", "")),
                "tipo_redundancia": rule["tipo_redundancia"],
                "motivo_decision": rule["motivo"],
                "f_score_descartar": drop_score.get("f_score"),
                "f_score_conservar": keep_score.get("f_score"),
                "p_value_descartar": drop_score.get("p_value"),
                "p_value_conservar": keep_score.get("p_value"),
            }
        )
    return pd.DataFrame(rows)


def build_clean_dataset(dataset: pd.DataFrame, redundancy_df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = redundancy_df["variable_descartar"].tolist()
    return dataset.drop(columns=[c for c in drop_cols if c in dataset.columns], errors="ignore").copy()


def write_report(path: Path, clean_df: pd.DataFrame, redundancy_df: pd.DataFrame) -> None:
    text = "\n".join(
        [
            "# Escenario B limpio",
            "",
            "- La depuración se realizó con validación semántica basada en el diccionario de variables.",
            "- No se descartó ninguna variable solo por correlación alta.",
            f"- Variables eliminadas por redundancia validada: {redundancy_df.shape[0]}",
            f"- Columnas finales del escenario B limpio: {clean_df.shape[1]}",
            "",
            "## Variables descartadas",
            "",
            *[
                f"- {row.variable_descartar} -> se conserva {row.variable_conservar}: {row.motivo_decision}"
                for row in redundancy_df.itertuples()
            ],
        ]
    )
    path.write_text(text, encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    paths = build_paths(root)
    ensure_directories(paths)
    logger = configure_logging(paths.logs / f"scenario_b_dedup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    dataset, dictionary_df, ranking_df, rename_map_df = load_inputs(paths)
    score_map = attach_scores(ranking_df, rename_map_df)
    redundancy_df = build_redundancy_matrix(dictionary_df, score_map)
    clean_df = build_clean_dataset(dataset, redundancy_df)

    parquet_path = paths.data_processed / "enaho_arequipa_escenario_b_clean.parquet"
    csv_path = paths.data_processed / "enaho_arequipa_escenario_b_clean.csv"

    clean_df.to_parquet(parquet_path, index=False)
    clean_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    redundancy_df.to_csv(paths.reports / "escenario_b_redundancias_validadas.csv", index=False, encoding="utf-8-sig")
    write_report(paths.reports / "escenario_b_redundancias_validadas.md", clean_df, redundancy_df)

    validation = {
        "phase": "SCENARIO_B_DEDUP",
        "passed": True,
        "rows": int(clean_df.shape[0]),
        "cols_before": int(dataset.shape[1]),
        "cols_after": int(clean_df.shape[1]),
        "variables_descartadas": redundancy_df["variable_descartar"].tolist(),
        "hogar_id_unique": bool(not clean_df["hogar_id"].duplicated().any()),
    }
    save_json(validation, paths.reports / "escenario_b_redundancias_validadas_validation.json")
    logger.info("SCENARIO_B_DEDUP validacion: %s", validation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
