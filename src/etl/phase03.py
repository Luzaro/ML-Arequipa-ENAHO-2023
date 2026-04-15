from __future__ import annotations

import json
import logging
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .phase01 import (
    ProjectPaths,
    build_paths,
    configure_logging,
    ensure_directories,
    save_json,
)


RAW_DIRNAME = "enaho_2023_anual"
STANDARDIZED_DIRNAME = "phase3_standardized"
EXPECTED_MERGE_COLUMNS = ["conglome", "vivienda", "hogar", "ubigeo"]


def standardize_column_name(name: Any) -> str:
    text = str(name).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def build_table_id(zip_stem: str, table_name: str) -> str:
    return f"{zip_stem}__{standardize_column_name(table_name)}"


def candidate_key_tests(columns: list[str]) -> list[list[str]]:
    combos = [
        ["conglome", "vivienda", "hogar", "codperso"],
        ["conglome", "vivienda", "hogar"],
        ["conglome", "vivienda"],
        ["ubigeo", "conglome", "vivienda", "hogar", "codperso"],
        ["ubigeo", "conglome", "vivienda", "hogar"],
        ["codocupa"],
        ["codigo_cno_2015"],
        ["codrev3"],
        ["codrev4"],
        ["grangrup", "grupo", "subgrupo", "variedad", "codocupa"],
    ]
    return [combo for combo in combos if all(col in columns for col in combo)]


def profile_table(
    table_id: str,
    zip_name: str,
    table_name: str,
    df: pd.DataFrame,
    output_dir: Path,
    standardized_dir: Path,
    logger: logging.Logger,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    original_columns = list(df.columns)
    standardized_columns = [standardize_column_name(col) for col in original_columns]

    renamed_df = df.copy()
    renamed_df.columns = standardized_columns

    column_mapping = pd.DataFrame(
        {
            "table_id": table_id,
            "zip_name": zip_name,
            "table_name": table_name,
            "original_column": original_columns,
            "standardized_column": standardized_columns,
        }
    )

    profile_df = pd.DataFrame(
        {
            "table_id": table_id,
            "zip_name": zip_name,
            "table_name": table_name,
            "column_name": renamed_df.columns,
            "original_column_name": original_columns,
            "dtype": renamed_df.dtypes.astype(str).tolist(),
            "null_count": renamed_df.isna().sum().astype(int).tolist(),
            "null_pct": (renamed_df.isna().mean().mul(100).round(4)).tolist(),
            "n_unique": renamed_df.nunique(dropna=True).astype(int).tolist(),
        }
    )
    profile_df.to_csv(output_dir / f"profile_{table_id}.csv", index=False, encoding="utf-8-sig")

    standardized_path = standardized_dir / f"{table_id}.pkl"
    renamed_df.to_pickle(standardized_path)
    logger.info("Tabla estandarizada guardada: %s | shape=%s", standardized_path.name, renamed_df.shape)

    tested_keys = candidate_key_tests(list(renamed_df.columns))
    key_results: list[dict[str, Any]] = []
    for combo in tested_keys:
        duplicate_count = int(renamed_df.duplicated(subset=combo).sum())
        key_results.append(
            {
                "table_id": table_id,
                "table_name": table_name,
                "candidate_key": "|".join(combo),
                "duplicate_count": duplicate_count,
                "is_unique": duplicate_count == 0,
            }
        )

    duplicate_audit_rows: list[dict[str, Any]] = []
    for combo in tested_keys:
        duplicate_mask = renamed_df.duplicated(subset=combo, keep=False)
        if duplicate_mask.any():
            duplicated_subset = renamed_df.loc[duplicate_mask, combo].copy()
            duplicated_subset = duplicated_subset.value_counts().reset_index(name="duplicate_rows")
            duplicated_subset.insert(0, "candidate_key", "|".join(combo))
            duplicated_subset.insert(0, "table_id", table_id)
            duplicate_audit_rows.extend(duplicated_subset.head(20).to_dict(orient="records"))

    merge_presence = {col: (col in renamed_df.columns) for col in EXPECTED_MERGE_COLUMNS}
    if {"conglome", "vivienda", "hogar", "codperso"}.issubset(renamed_df.columns):
        merge_strategy = "agregar_persona_a_hogar"
        analysis_level = "persona"
    elif all(merge_presence.values()):
        merge_strategy = "merge_hogar_directo"
        analysis_level = "hogar"
    elif table_name.startswith("enaho_tabla_"):
        merge_strategy = "tabla_clasificacion_apoyo_no_merge_directo"
        analysis_level = "catalogo_clasificacion"
    else:
        merge_strategy = "requiere_revision_manual"
        analysis_level = "desconocido"

    table_summary = {
        "table_id": table_id,
        "zip_name": zip_name,
        "table_name": table_name,
        "rows": int(renamed_df.shape[0]),
        "columns": int(renamed_df.shape[1]),
        "analysis_level": analysis_level,
        "merge_strategy": merge_strategy,
        "merge_columns_present_json": json.dumps(merge_presence, ensure_ascii=False),
        "candidate_keys_json": json.dumps(key_results, ensure_ascii=False),
        "unique_candidate_keys": json.dumps(
            [item["candidate_key"] for item in key_results if item["is_unique"]],
            ensure_ascii=False,
        ),
        "duplicate_rows_all_columns": int(renamed_df.duplicated().sum()),
    }

    key_audit_df = pd.DataFrame(key_results)
    duplicate_audit_df = pd.DataFrame(duplicate_audit_rows)
    return table_summary, column_mapping, pd.concat([key_audit_df, duplicate_audit_df], axis=0, ignore_index=True)


def run_phase3(paths: ProjectPaths, logger: logging.Logger) -> dict[str, Any]:
    import inei_microdatos

    raw_dir = paths.data_raw / RAW_DIRNAME
    standardized_dir = paths.data_interim / STANDARDIZED_DIRNAME
    standardized_dir.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(raw_dir.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No se encontraron ZIPs en {raw_dir}")

    schema_rows: list[dict[str, Any]] = []
    mapping_frames: list[pd.DataFrame] = []
    key_audit_frames: list[pd.DataFrame] = []

    for zip_path in zip_files:
        logger.info("Leyendo modulo: %s", zip_path.name)
        tables = inei_microdatos.read_module(zip_path)
        for table_name, df in tables.items():
            table_id = build_table_id(zip_path.stem, table_name)
            summary, mapping_df, key_df = profile_table(
                table_id=table_id,
                zip_name=zip_path.name,
                table_name=table_name,
                df=df,
                output_dir=paths.reports,
                standardized_dir=standardized_dir,
                logger=logger,
            )
            schema_rows.append(summary)
            mapping_frames.append(mapping_df)
            key_audit_frames.append(key_df)

    schema_df = pd.DataFrame(schema_rows).sort_values(["zip_name", "table_name"]).reset_index(drop=True)
    mapping_df = pd.concat(mapping_frames, ignore_index=True)
    key_audit_df = pd.concat(key_audit_frames, ignore_index=True)

    schema_df.to_csv(paths.reports / "schema_summary.csv", index=False, encoding="utf-8-sig")
    mapping_df.to_csv(paths.reports / "phase3_column_mapping.csv", index=False, encoding="utf-8-sig")
    key_audit_df.to_csv(paths.reports / "phase3_key_audit.csv", index=False, encoding="utf-8-sig")

    missing_merge_tables = schema_df[
        ~schema_df["merge_strategy"].isin(
            [
                "merge_hogar_directo",
                "agregar_persona_a_hogar",
                "tabla_clasificacion_apoyo_no_merge_directo",
            ]
        )
    ]["table_id"].tolist()

    validation = {
        "phase": "FASE 3",
        "passed": len(schema_df) > 0 and len(missing_merge_tables) == 0,
        "tables_profiled": int(schema_df.shape[0]),
        "profiles_generated": int(len(list(paths.reports.glob("profile_*.csv")))),
        "missing_merge_tables": missing_merge_tables,
        "standardized_tables_dir": str(standardized_dir),
    }
    save_json(validation, paths.reports / "phase3_validation.json")
    return validation


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    paths = build_paths(root)
    ensure_directories(paths)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = configure_logging(paths.logs / f"etl_phase03_{timestamp}.log")

    validation = run_phase3(paths, logger)
    if validation["passed"]:
        logger.info("FASE 3 completada y validada")
        return 0

    logger.error("FASE 3 con incidencias: %s", validation)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

