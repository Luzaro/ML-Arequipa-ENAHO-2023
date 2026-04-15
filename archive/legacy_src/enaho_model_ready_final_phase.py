from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .enaho_etl_phase01 import build_paths, configure_logging, ensure_directories, save_json


PHASE7_FILE = "enaho_arequipa_hogar_phase7.pkl"
CONSERVATIVE_FEATURES_FILE = "selected_features_conservative.csv"

KEY_COLS = ["conglome", "vivienda", "hogar"]
GEO_COLS = [
    "ubigeo",
    "departamento_cod",
    "provincia_cod",
    "distrito_cod",
    "dominio",
    "estrato",
    "area_urbana_rural",
    "area_rural_bin",
]
TARGET_COL = "target_pobreza_bin"
WEIGHT_COL = "factor07"

FEATURE_RENAME_MAP = {
    "educ_jefe_ord": "nivel_educativo_jefe_ord",
    "p104": "num_habitaciones",
    "combustible_limpio": "usa_combustible_limpio",
    "p103a": "material_techo",
    "miembros_afiliados_essalud": "num_miembros_afiliados_essalud",
    "miembros_sin_atencion_salud": "num_miembros_sin_atencion_salud",
    "n_ninos_0_5": "num_ninos_0_5",
    "tam_hogar": "tam_hogar",
    "edad_jefe": "edad_jefe",
    "estrato": "estrato_geografico",
    "piso_precario": "piso_precario",
    "p105a": "tenencia_vivienda",
    "internet_hogar": "acceso_internet_hogar",
    "n_ninos_6_16": "num_ninos_6_16",
    "p101": "tipo_vivienda",
    "p102": "material_pared",
    "acceso_desague_red": "acceso_desague_red",
    "n_6_16_no_matriculados": "num_6_16_no_matriculados",
}

CONTEXT_RENAME_MAP = {
    "departamento_cod": "departamento_codigo",
    "provincia_cod": "provincia_codigo",
    "distrito_cod": "distrito_codigo",
    "dominio": "dominio_geografico",
    "estrato": "estrato_geografico",
    "factor07": "factor_expansion_anual",
    "target_pobreza_bin": "target_pobreza_monetaria_bin",
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


def load_conservative_features(paths: Any) -> pd.DataFrame:
    ranking_path = paths.reports / "feature_selection" / CONSERVATIVE_FEATURES_FILE
    df = pd.read_csv(ranking_path)
    if "variable" not in df.columns:
        raise ValueError("No se encontro la columna 'variable' en selected_features_conservative.csv.")
    return df


def build_variable_map(
    selected_features: pd.DataFrame,
    final_df: pd.DataFrame,
    renamed_columns: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "variable_original": "hogar_id",
            "variable_final": "hogar_id",
            "rol": "identificador",
            "incluir_entrenamiento": False,
            "grupo": "trazabilidad",
            "descripcion": "Identificador unico del hogar: ubigeo_conglome_vivienda_hogar",
        }
    ]

    for col in ["conglome", "vivienda", "hogar", "ubigeo", "departamento_cod", "provincia_cod", "distrito_cod", "dominio", "estrato", "area_urbana_rural", "area_rural_bin", "factor07", "target_pobreza_bin"]:
        if col not in final_df.columns and col not in renamed_columns:
            continue
        final_name = renamed_columns.get(col, col)
        role = "feature"
        include = True
        group = "metadata"
        description = ""

        if col in KEY_COLS or col == "ubigeo":
            role = "identificador"
            include = False
            group = "trazabilidad"
        elif col in {"departamento_cod", "provincia_cod", "distrito_cod", "dominio", "area_urbana_rural", "area_rural_bin"}:
            role = "contexto"
            include = False
            group = "geografia"
        elif col == "estrato":
            role = "feature"
            include = True
            group = "geografia"
            description = "Estrato geografico"
        elif col == WEIGHT_COL:
            role = "ponderador"
            include = False
            group = "ponderacion"
        elif col == TARGET_COL:
            role = "target"
            include = False
            group = "objetivo"

        rows.append(
            {
                "variable_original": col,
                "variable_final": final_name,
                "rol": role,
                "incluir_entrenamiento": include,
                "grupo": group,
                "descripcion": description,
            }
        )

    for record in selected_features.to_dict(orient="records"):
        original = record["variable"]
        final_name = renamed_columns.get(original, original)
        rows.append(
            {
                "variable_original": original,
                "variable_final": final_name,
                "rol": "feature",
                "incluir_entrenamiento": True,
                "grupo": record.get("grupo", ""),
                "descripcion": record.get("descripcion", ""),
            }
        )

    mapped = pd.DataFrame(rows).drop_duplicates(subset=["variable_original", "variable_final", "rol"])
    return mapped.sort_values(["rol", "grupo", "variable_final"]).reset_index(drop=True)


def build_final_dataset(phase7_df: pd.DataFrame, selected_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, str]]:
    required_cols = ["ubigeo", *KEY_COLS, *GEO_COLS, WEIGHT_COL, TARGET_COL]
    feature_cols = selected_features["variable"].tolist()
    required_cols.extend(feature_cols)
    required_cols = list(dict.fromkeys(col for col in required_cols if col in phase7_df.columns))

    final_df = phase7_df[required_cols].copy()
    if final_df.duplicated(subset=KEY_COLS).any():
        raise ValueError("La base fuente no es unica por hogar al construir model-ready-final.")

    final_df.insert(0, "hogar_id", build_hogar_id(final_df))
    if final_df["hogar_id"].duplicated().any():
        raise ValueError("hogar_id no es unico. Revisar llaves del hogar.")

    rename_map = dict(CONTEXT_RENAME_MAP)
    rename_map.update(FEATURE_RENAME_MAP)
    rename_map = {old: new for old, new in rename_map.items() if old in final_df.columns}

    final_df = final_df.rename(columns=rename_map)
    final_df = final_df.drop(columns=["estrato"], errors="ignore") if "estrato" in final_df.columns and "estrato_geografico" in final_df.columns else final_df

    feature_cols_final = [
        rename_map.get(col, col)
        for col in feature_cols
        if rename_map.get(col, col) in final_df.columns
    ]

    ordered_cols = [
        "hogar_id",
        "conglome",
        "vivienda",
        "hogar",
        "ubigeo",
        "departamento_codigo",
        "provincia_codigo",
        "distrito_codigo",
        "dominio_geografico",
        "estrato_geografico",
        "area_urbana_rural",
        "area_rural_bin",
        "factor_expansion_anual",
        "target_pobreza_monetaria_bin",
        *feature_cols_final,
    ]
    ordered_cols = list(dict.fromkeys(col for col in ordered_cols if col in final_df.columns))
    final_df = final_df[ordered_cols].copy()
    return final_df, selected_features, feature_cols_final, rename_map


def write_summary(path: Path, final_df: pd.DataFrame, feature_cols_final: list[str]) -> None:
    text = "\n".join(
        [
            "# Model-ready final",
            "",
            f"- Filas: {final_df.shape[0]}",
            f"- Columnas: {final_df.shape[1]}",
            f"- Features finales para entrenamiento: {len(feature_cols_final)}",
            "- Esta salida conserva trazabilidad del hogar con hogar_id y llaves originales.",
            "- El target final es target_pobreza_monetaria_bin.",
            "- El ponderador se conserva como factor_expansion_anual.",
            "",
            "## Features finales",
            "",
            *[f"- {col}" for col in feature_cols_final],
        ]
    )
    path.write_text(text, encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    paths = build_paths(root)
    ensure_directories(paths)
    logger = configure_logging(paths.logs / f"model_ready_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    phase7_path = paths.data_interim / PHASE7_FILE
    phase7_df = pd.read_pickle(phase7_path)
    selected_features = load_conservative_features(paths)
    final_df, selected_features, feature_cols_final, rename_map = build_final_dataset(phase7_df, selected_features)

    variable_map = build_variable_map(selected_features, final_df, rename_map)
    final_features_report = variable_map.loc[variable_map["incluir_entrenamiento"]].copy()

    parquet_path = paths.data_processed / "enaho_arequipa_model_ready_final.parquet"
    csv_path = paths.data_processed / "enaho_arequipa_model_ready_final.csv"

    parquet_safe(final_df).to_parquet(parquet_path, index=False)
    final_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    variable_map.to_csv(paths.reports / "model_ready_final_variable_map.csv", index=False, encoding="utf-8-sig")
    final_features_report.to_csv(paths.reports / "model_ready_final_features.csv", index=False, encoding="utf-8-sig")
    write_summary(paths.reports / "model_ready_final_summary.md", final_df, feature_cols_final)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "source_phase7": str(phase7_path),
        "output_parquet": str(parquet_path),
        "output_csv": str(csv_path),
        "target_column": "target_pobreza_monetaria_bin",
        "weight_column": "factor_expansion_anual",
        "id_columns": ["hogar_id", "conglome", "vivienda", "hogar", "ubigeo"],
        "feature_columns": feature_cols_final,
        "n_rows": int(final_df.shape[0]),
        "n_cols": int(final_df.shape[1]),
    }
    save_json(metadata, paths.reports / "model_ready_final_metadata.json")

    validation = {
        "phase": "MODEL_READY_FINAL",
        "passed": True,
        "rows": int(final_df.shape[0]),
        "cols": int(final_df.shape[1]),
        "hogar_id_unique": bool(not final_df["hogar_id"].duplicated().any()),
        "target_present": "target_pobreza_monetaria_bin" in final_df.columns,
        "weight_present": "factor_expansion_anual" in final_df.columns,
        "n_final_features": len(feature_cols_final),
    }
    save_json(validation, paths.reports / "model_ready_final_validation.json")
    logger.info("MODEL_READY_FINAL validacion: %s", json.dumps(validation, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
