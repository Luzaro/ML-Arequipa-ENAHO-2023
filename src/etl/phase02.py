from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .phase01 import (
    MINIMAL_ANNUAL_MODULE_CODES,
    ProjectPaths,
    build_paths,
    configure_logging,
    ensure_directories,
    load_catalog_with_fallback,
    normalize_text,
    save_json,
)


RAW_SUBDIR = "enaho_2023_anual"


def ensure_reports_subdirs(paths: ProjectPaths) -> Path:
    samples_dir = paths.reports / "samples_phase2"
    samples_dir.mkdir(parents=True, exist_ok=True)
    return samples_dir


def build_phase2_catalog(catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    subset: list[dict[str, Any]] = []

    for entry in catalog:
        if "panel" in normalize_text(entry.get("label")):
            continue
        if "enaho" not in normalize_text(entry.get("label")):
            continue
        if "2023" not in entry.get("years", {}):
            continue

        year_data = entry["years"]["2023"]
        filtered_periods: dict[str, Any] = {}
        for period_label, period_data in year_data.items():
            if "anual" not in normalize_text(period_label):
                continue

            filtered_modules = [
                mod
                for mod in period_data.get("modules", [])
                if str(mod.get("module_code")) in MINIMAL_ANNUAL_MODULE_CODES
            ]
            if filtered_modules:
                filtered_periods[period_label] = {
                    "period_value": period_data.get("period_value"),
                    "modules": filtered_modules,
                    "docs": period_data.get("docs", []),
                }

        if filtered_periods:
            subset.append(
                {
                    "category": entry.get("category"),
                    "value": entry.get("value"),
                    "label": entry.get("label"),
                    "years": {"2023": filtered_periods},
                }
            )

    return subset


def expected_download_inventory(catalog_subset: list[dict[str, Any]], raw_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for entry in catalog_subset:
        for year, year_data in entry["years"].items():
            for period_label, period_data in year_data.items():
                for mod in period_data["modules"]:
                    actual_format = None
                    actual_code = None
                    for fmt_name, key in (("STATA", "stata_code"), ("CSV", "csv_code"), ("SPSS", "spss_code")):
                        if mod.get(key):
                            actual_format = fmt_name
                            actual_code = mod.get(key)
                            break
                    rows.append(
                        {
                            "survey_label": entry["label"],
                            "category": entry["category"],
                            "year": year,
                            "period_label": period_label,
                            "module_code": str(mod.get("module_code")),
                            "module_name": mod.get("module_name"),
                            "preferred_request_format": "STATA",
                            "actual_download_format": actual_format,
                            "download_code": actual_code,
                            "zip_path": str(raw_dir / f"{actual_code}.zip") if actual_code else None,
                        }
                    )
    return pd.DataFrame(rows)


def download_minimal_modules(catalog_subset: list[dict[str, Any]], raw_dir: Path, logger: logging.Logger) -> dict[str, int]:
    import inei_microdatos

    logger.info("Descargando modulos minimos a %s", raw_dir)
    stats = inei_microdatos.download_modules(
        catalog=catalog_subset,
        dest=raw_dir,
        fmt="STATA",
        fallback=True,
        layout="{code}.zip",
        workers=2,
        progress=False,
        dry_run=False,
    )
    logger.info("Resultado descarga: %s", stats)
    return stats


def inspect_downloaded_modules(
    inventory_df: pd.DataFrame,
    reports_dir: Path,
    samples_dir: Path,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import inei_microdatos

    file_rows: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []

    for item in inventory_df.to_dict(orient="records"):
        zip_path = Path(item["zip_path"]) if item.get("zip_path") else None
        exists_ok = bool(zip_path and zip_path.exists() and zip_path.is_file())

        row = dict(item)
        row["download_exists"] = exists_ok
        row["file_size_bytes"] = zip_path.stat().st_size if exists_ok else None

        if not exists_ok:
            row["status"] = "missing"
            row["table_count"] = 0
            row["tables_json"] = json.dumps([], ensure_ascii=False)
            failed_rows.append(
                {
                    "download_code": item.get("download_code"),
                    "module_code": item.get("module_code"),
                    "module_name": item.get("module_name"),
                    "zip_path": item.get("zip_path"),
                    "failure_reason": "zip_no_encontrado",
                }
            )
            file_rows.append(row)
            continue

        try:
            tables = inei_microdatos.list_tables(zip_path)
            row["status"] = "ok"
            row["table_count"] = len(tables)
            row["tables_json"] = json.dumps([table["name"] for table in tables], ensure_ascii=False)
            file_rows.append(row)

            for table in tables:
                table_rows.append(
                    {
                        "download_code": item.get("download_code"),
                        "module_code": item.get("module_code"),
                        "module_name": item.get("module_name"),
                        "zip_path": str(zip_path),
                        "table_name": table["name"],
                        "table_format": table["format"],
                        "table_size_bytes": table["size_bytes"],
                        "table_full_path": table["full_path"],
                    }
                )

            dfs = inei_microdatos.read_module(zip_path)
            for table_name, df in dfs.items():
                sample_path = samples_dir / f"{item['download_code']}__{table_name}.csv"
                df.head(5).to_csv(sample_path, index=False, encoding="utf-8-sig")
                logger.info(
                    "Muestra guardada: %s | shape=%s",
                    sample_path.name,
                    df.shape,
                )
        except Exception as exc:  # pragma: no cover - runtime I/O
            row["status"] = "inspection_failed"
            row["table_count"] = 0
            row["tables_json"] = json.dumps([], ensure_ascii=False)
            file_rows.append(row)
            failed_rows.append(
                {
                    "download_code": item.get("download_code"),
                    "module_code": item.get("module_code"),
                    "module_name": item.get("module_name"),
                    "zip_path": item.get("zip_path"),
                    "failure_reason": str(exc),
                }
            )

    files_df = pd.DataFrame(file_rows).drop_duplicates()
    tables_df = pd.DataFrame(table_rows).drop_duplicates()
    failed_df = pd.DataFrame(failed_rows).drop_duplicates()

    files_df.to_csv(reports_dir / "phase2_download_inventory.csv", index=False, encoding="utf-8-sig")
    tables_df.to_csv(reports_dir / "phase2_tables_inventory.csv", index=False, encoding="utf-8-sig")
    failed_df.to_csv(reports_dir / "phase2_failed_modules.csv", index=False, encoding="utf-8-sig")
    return files_df, tables_df, failed_df


def validate_phase2(
    expected_count: int,
    download_stats: dict[str, int],
    files_df: pd.DataFrame,
    tables_df: pd.DataFrame,
    failed_df: pd.DataFrame,
) -> dict[str, Any]:
    ok_files = int((files_df["status"] == "ok").sum()) if not files_df.empty else 0
    missing_files = int((files_df["status"] == "missing").sum()) if not files_df.empty else expected_count
    validation = {
        "phase": "FASE 2",
        "passed": expected_count > 0 and ok_files == expected_count and failed_df.empty,
        "expected_modules": expected_count,
        "download_stats": download_stats,
        "ok_files": ok_files,
        "missing_files": missing_files,
        "failed_modules": int(failed_df.shape[0]),
        "tables_detected": int(tables_df.shape[0]),
    }
    return validation


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    paths = build_paths(root)
    ensure_directories(paths)
    samples_dir = ensure_reports_subdirs(paths)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = configure_logging(paths.logs / f"etl_phase02_{timestamp}.log")

    catalog, catalog_source, catalog_age = load_catalog_with_fallback(paths, logger)
    catalog_subset = build_phase2_catalog(catalog)
    raw_dir = paths.data_raw / RAW_SUBDIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    inventory_df = expected_download_inventory(catalog_subset, raw_dir)
    inventory_df.to_csv(paths.reports / "phase2_expected_modules.csv", index=False, encoding="utf-8-sig")

    if inventory_df.empty:
        validation = {
            "phase": "FASE 2",
            "passed": False,
            "error": "No se identificaron modulos minimos ENAHO 2023 Anual para descargar.",
            "catalog_source": catalog_source,
            "catalog_age": catalog_age,
        }
        save_json(validation, paths.reports / "phase2_validation.json")
        logger.error(validation["error"])
        return 1

    download_stats = download_minimal_modules(catalog_subset, raw_dir, logger)
    files_df, tables_df, failed_df = inspect_downloaded_modules(
        inventory_df=inventory_df,
        reports_dir=paths.reports,
        samples_dir=samples_dir,
        logger=logger,
    )

    validation = validate_phase2(
        expected_count=int(inventory_df.shape[0]),
        download_stats=download_stats,
        files_df=files_df,
        tables_df=tables_df,
        failed_df=failed_df,
    )
    validation["catalog_source"] = catalog_source
    validation["catalog_age"] = catalog_age
    save_json(validation, paths.reports / "phase2_validation.json")

    if validation["passed"]:
        logger.info("FASE 2 completada y validada")
        return 0

    logger.error("FASE 2 con incidencias: %s", validation)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

