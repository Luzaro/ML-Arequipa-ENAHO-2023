from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .phase01 import build_paths, configure_logging, ensure_directories, save_json


STANDARDIZED_DIRNAME = "phase3_standardized"
FILTERED_DIRNAME = "phase4_arequipa"
AREQUIPA_DEPT_CODE = "04"


def filter_arequipa(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if "ubigeo" not in df.columns:
        return df.copy(), {
            "has_ubigeo": False,
            "before_rows": int(df.shape[0]),
            "after_rows": int(df.shape[0]),
            "dept_codes_found_after": [],
        }

    ubigeo_str = df["ubigeo"].astype(str).str.zfill(6)
    dept_code = ubigeo_str.str[:2]
    mask = dept_code.eq(AREQUIPA_DEPT_CODE)
    filtered = df.loc[mask].copy()
    filtered["ubigeo"] = filtered["ubigeo"].astype(str).str.zfill(6)

    return filtered, {
        "has_ubigeo": True,
        "before_rows": int(df.shape[0]),
        "after_rows": int(filtered.shape[0]),
        "dept_codes_found_after": sorted(filtered["ubigeo"].astype(str).str[:2].dropna().unique().tolist()),
    }


def province_district_summary(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["ubigeo"] = work["ubigeo"].astype(str).str.zfill(6)
    work["departamento_cod"] = work["ubigeo"].str[:2]
    work["provincia_cod"] = work["ubigeo"].str[:4]
    work["distrito_cod"] = work["ubigeo"].str[:6]

    key_cols = [col for col in ["conglome", "vivienda", "hogar"] if col in work.columns]
    if key_cols:
        hogar_counts = (
            work[key_cols + ["provincia_cod", "distrito_cod"]]
            .drop_duplicates()
            .groupby(["provincia_cod", "distrito_cod"])
            .size()
            .reset_index(name="hogares_unicos")
        )
    else:
        hogar_counts = pd.DataFrame(columns=["provincia_cod", "distrito_cod", "hogares_unicos"])

    row_counts = (
        work.groupby(["provincia_cod", "distrito_cod"])
        .size()
        .reset_index(name="filas")
    )
    summary = row_counts.merge(hogar_counts, on=["provincia_cod", "distrito_cod"], how="left")
    summary["departamento_cod"] = AREQUIPA_DEPT_CODE
    return summary.sort_values(["provincia_cod", "distrito_cod"]).reset_index(drop=True)


def run_phase4(root: Path) -> dict[str, Any]:
    paths = build_paths(root)
    ensure_directories(paths)
    logger = configure_logging(paths.logs / f"etl_phase04_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    source_dir = paths.data_interim / STANDARDIZED_DIRNAME
    target_dir = paths.data_interim / FILTERED_DIRNAME
    target_dir.mkdir(parents=True, exist_ok=True)

    table_rows: list[dict[str, Any]] = []
    geotables_valid = True
    sumaria_summary_exported = False

    for pkl_path in sorted(source_dir.glob("*.pkl")):
        table_id = pkl_path.stem
        df = pd.read_pickle(pkl_path)
        filtered_df, stats = filter_arequipa(df)
        filtered_path = target_dir / pkl_path.name
        filtered_df.to_pickle(filtered_path)

        row = {"table_id": table_id, **stats}
        row["filter_rule"] = 'ubigeo.str[:2] == "04"' if stats["has_ubigeo"] else "sin_ubigeo_no_aplica"
        row["filtered_path"] = str(filtered_path)
        table_rows.append(row)

        if stats["has_ubigeo"]:
            if stats["dept_codes_found_after"] != [AREQUIPA_DEPT_CODE]:
                geotables_valid = False
            logger.info("%s | before=%s after=%s", table_id, stats["before_rows"], stats["after_rows"])

            if table_id == "906-Modulo34__sumaria_2023":
                summary_df = province_district_summary(filtered_df)
                summary_df.to_csv(
                    paths.reports / "phase4_arequipa_provincia_distrito_summary.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
                sumaria_summary_exported = True
        else:
            logger.info("%s | tabla sin ubigeo, no se filtra", table_id)

    filter_audit_df = pd.DataFrame(table_rows)
    filter_audit_df.to_csv(paths.reports / "phase4_filter_audit.csv", index=False, encoding="utf-8-sig")

    validation = {
        "phase": "FASE 4",
        "passed": bool(not filter_audit_df.empty and geotables_valid),
        "filter_rule": 'departamento = ubigeo[:2] == "04"',
        "tables_processed": int(filter_audit_df.shape[0]),
        "tables_with_ubigeo": int(filter_audit_df["has_ubigeo"].sum()) if not filter_audit_df.empty else 0,
        "sumaria_summary_exported": sumaria_summary_exported,
        "all_filtered_rows_are_arequipa": geotables_valid,
        "target_dir": str(target_dir),
    }
    save_json(validation, paths.reports / "phase4_validation.json")
    return validation


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    validation = run_phase4(root)
    return 0 if validation["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

