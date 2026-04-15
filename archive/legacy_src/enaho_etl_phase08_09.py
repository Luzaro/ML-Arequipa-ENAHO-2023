from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .enaho_etl_phase01 import build_paths, configure_logging, ensure_directories, save_json


PHASE7_FILE = "enaho_arequipa_hogar_phase7.pkl"
KEY_COLS = ["conglome", "vivienda", "hogar"]
GEO_COLS = ["ubigeo", "departamento_cod", "provincia_cod", "distrito_cod", "dominio", "estrato", "area_urbana_rural", "area_rural_bin"]
TARGET_COL = "target_pobreza_bin"
WEIGHT_COLS = ["factor07", "facpob07"]


def safe_corr(a: pd.Series, b: pd.Series) -> float | None:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    if x.nunique(dropna=True) <= 1 or y.nunique(dropna=True) <= 1:
        return None
    return float(x.corr(y))


def parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    safe = df.copy()
    for col in safe.columns:
        if str(safe[col].dtype) in {"category", "object"}:
            safe[col] = safe[col].astype("string")
    return safe


def build_modeling_views(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    analytical = df.copy()
    analytical = analytical.dropna(axis=1, how="all")

    leakage_rows: list[dict[str, Any]] = []
    dropped_rows: list[dict[str, Any]] = []

    manual_drop = {
        "pobreza": "target_raw_source",
        "linea": "derivada_directa_regla_oficial_pobreza",
        "linpe": "linea_extrema_relacionada_regla_oficial",
        "gashog2d": "variable_monetaria_muy_cercana_a_regla_oficial",
        "inghog1d": "variable_monetaria_muy_cercana_a_regla_oficial",
        "ingreso_hogar": "variable_derivada_monetaria_potencial_leakage",
        "ingreso_percapita": "transformacion_directa_monetaria_potencial_leakage",
        "gru11hd": "componente_gasto_sumaria",
        "gru12hd1": "componente_gasto_sumaria",
        "gru12hd2": "componente_gasto_sumaria",
        "gru13hd1": "componente_gasto_sumaria",
        "gru13hd2": "componente_gasto_sumaria",
        "gru13hd3": "componente_gasto_sumaria",
        "gru13hd4": "componente_gasto_sumaria",
        "gru13hd5": "componente_gasto_sumaria",
        "gru13hd6": "componente_gasto_sumaria",
        "gru14hd": "componente_gasto_sumaria",
        "gru21hd": "componente_gasto_sumaria",
        "gru22hd1": "componente_gasto_sumaria",
        "gru22hd2": "componente_gasto_sumaria",
        "gru23hd1": "componente_gasto_sumaria",
        "gru23hd2": "componente_gasto_sumaria",
        "gru23hd3": "componente_gasto_sumaria",
        "gru23hd4": "componente_gasto_sumaria",
        "gru23hd5": "componente_gasto_sumaria",
        "gru24hd": "componente_gasto_sumaria",
        "gru31hd": "componente_gasto_sumaria",
        "gru32hd1": "componente_gasto_sumaria",
        "gru32hd2": "componente_gasto_sumaria",
        "gru33hd1": "componente_gasto_sumaria",
        "gru33hd2": "componente_gasto_sumaria",
        "gru33hd3": "componente_gasto_sumaria",
        "gru33hd4": "componente_gasto_sumaria",
        "gru33hd5": "componente_gasto_sumaria",
        "gru34hd": "componente_gasto_sumaria",
        "gru41hd": "componente_gasto_sumaria",
        "gru42hd": "componente_gasto_sumaria",
        "gru43hd": "componente_gasto_sumaria",
        "gru44hd": "componente_gasto_sumaria",
        "gru45hd": "componente_gasto_sumaria",
        "gru46hd": "componente_gasto_sumaria",
        "gru47hd": "componente_gasto_sumaria",
        "gru51hd": "componente_gasto_sumaria",
        "gru52hd1": "componente_gasto_sumaria",
        "gru52hd2": "componente_gasto_sumaria",
        "gru53hd1": "componente_gasto_sumaria",
        "gru53hd2": "componente_gasto_sumaria",
        "gru53hd3": "componente_gasto_sumaria",
        "gru53hd4": "componente_gasto_sumaria",
        "gru53hd5": "componente_gasto_sumaria",
        "gru54hd": "componente_gasto_sumaria",
        "gru61hd": "componente_gasto_sumaria",
        "gru62hd1": "componente_gasto_sumaria",
        "gru62hd2": "componente_gasto_sumaria",
        "gru63hd1": "componente_gasto_sumaria",
        "gru63hd2": "componente_gasto_sumaria",
        "gru63hd3": "componente_gasto_sumaria",
        "gru63hd4": "componente_gasto_sumaria",
        "gru63hd5": "componente_gasto_sumaria",
        "gru64hd": "componente_gasto_sumaria",
    }

    for col, reason in manual_drop.items():
        if col in analytical.columns:
            leakage_rows.append({"variable": col, "riesgo": "alto", "motivo": reason})

    constant_cols = [col for col in analytical.columns if analytical[col].nunique(dropna=False) <= 1]
    for col in constant_cols:
        dropped_rows.append({"variable": col, "motivo": "columna_constante"})

    redundant_rows = []
    candidate_cols = ["tam_hogar", "tam_hogar_mod2", "mieperho", "num_perceptores", "percepho", "sexo_jefe", "sexo_jefe_bin"]
    for col in candidate_cols:
        if col in analytical.columns and col != TARGET_COL:
            if col in {"tam_hogar_mod2", "mieperho"}:
                redundant_rows.append({"variable": col, "motivo": "redundante_con_tam_hogar"})
            if col == "percepho":
                redundant_rows.append({"variable": col, "motivo": "redundante_con_num_perceptores"})
            if col == "sexo_jefe":
                redundant_rows.append({"variable": col, "motivo": "redundante_con_sexo_jefe_bin"})

    for row in redundant_rows:
        dropped_rows.append(row)

    identifier_drop = [col for col in KEY_COLS if col in analytical.columns]
    for col in identifier_drop:
        dropped_rows.append({"variable": col, "motivo": "identificador_irrelevante_modelado"})

    raw_coded_drop = [
        "educ_jefe_nivel",
        "educ_jefe_anio",
        "educ_jefe_grado",
        "codigo_parentesco_jefe",
        "tam_hogar_consistente",
        "perceptores_leq_tam_hogar",
    ]
    for col in raw_coded_drop:
        if col in analytical.columns:
            dropped_rows.append({"variable": col, "motivo": "auxiliar_o_control"})

    drop_vars = sorted({row["variable"] for row in dropped_rows} | {row["variable"] for row in leakage_rows})
    model_ready = analytical.drop(columns=[col for col in drop_vars if col in analytical.columns], errors="ignore").copy()

    null_report = pd.DataFrame(
        {
            "column_name": model_ready.columns,
            "null_count": model_ready.isna().sum().astype(int).tolist(),
            "null_pct": model_ready.isna().mean().mul(100).round(4).tolist(),
            "dtype": model_ready.dtypes.astype(str).tolist(),
        }
    )
    constant_report = pd.DataFrame({"column_name": constant_cols})

    redundancy_report = pd.DataFrame(redundant_rows).drop_duplicates()
    leakage_report = pd.DataFrame(leakage_rows).drop_duplicates()
    dropped_report = pd.DataFrame(dropped_rows).drop_duplicates()

    return analytical, model_ready, null_report, constant_report, pd.concat([redundancy_report, leakage_report, dropped_report], ignore_index=True)


def classify_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    feature_rows: list[dict[str, Any]] = []
    discard_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []

    for col in df.columns:
        if col == TARGET_COL:
            continue

        series = df[col]
        unique_values = set(pd.Series(series.dropna().unique()).tolist())
        missing_pct = round(series.isna().mean() * 100, 4)
        dtype = str(series.dtype)

        if col in KEY_COLS:
            discard_rows.append({"variable": col, "motivo": "identificador"})
            continue

        if col in {"pobreza", "linea", "linpe", "inghog1d", "gashog2d", "ingreso_hogar", "ingreso_percapita"}:
            leakage_rows.append({"variable": col, "motivo": "target_o_transformacion_monetaria_muy_cercana", "riesgo": "alto"})
            continue

        if unique_values.issubset({0, 1}):
            col_type = "binaria"
            needs_encoding = False
        elif pd.api.types.is_numeric_dtype(series):
            col_type = "numerica"
            needs_encoding = False
        else:
            col_type = "categorica"
            needs_encoding = True

        feature_rows.append(
            {
                "variable": col,
                "tipo_feature": col_type,
                "dtype": dtype,
                "nulos_pct": missing_pct,
                "requiere_imputacion": missing_pct > 0,
                "requiere_encoding": needs_encoding,
                "riesgo_leakage": "bajo",
            }
        )

    features_df = pd.DataFrame(feature_rows)
    if not features_df.empty:
        features_df = features_df.sort_values(["tipo_feature", "variable"]).reset_index(drop=True)

    discarded_df = pd.concat([pd.DataFrame(discard_rows), pd.DataFrame(leakage_rows)], ignore_index=True).drop_duplicates()
    if "variable" not in discarded_df.columns:
        discarded_df = pd.DataFrame(columns=["variable", "motivo", "riesgo"])

    metadata = {
        "target": TARGET_COL,
        "n_features": int(features_df.shape[0]),
        "numericas": features_df.loc[features_df["tipo_feature"] == "numerica", "variable"].tolist() if not features_df.empty else [],
        "categoricas": features_df.loc[features_df["tipo_feature"] == "categorica", "variable"].tolist() if not features_df.empty else [],
        "binarias": features_df.loc[features_df["tipo_feature"] == "binaria", "variable"].tolist() if not features_df.empty else [],
        "descartadas": discarded_df["variable"].dropna().tolist(),
    }
    return features_df, discarded_df, metadata


def highly_redundant_columns(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    rows = []
    checked = set()
    for i, col_a in enumerate(numeric_cols):
        for col_b in numeric_cols[i + 1:]:
            pair = tuple(sorted((col_a, col_b)))
            if pair in checked:
                continue
            checked.add(pair)
            corr = safe_corr(df[col_a], df[col_b])
            if corr is not None and abs(corr) >= 0.98:
                rows.append({"col_a": col_a, "col_b": col_b, "abs_corr": round(abs(corr), 6)})
    return pd.DataFrame(rows)


def write_checklist(path: Path, features_df: pd.DataFrame, discarded_df: pd.DataFrame) -> None:
    text = "\n".join(
        [
            "# Checklist pre-modelado",
            "",
            f"- Total features candidatas: {features_df.shape[0]}",
            f"- Numéricas: {(features_df['tipo_feature'] == 'numerica').sum()}",
            f"- Categóricas: {(features_df['tipo_feature'] == 'categorica').sum()}",
            f"- Binarias: {(features_df['tipo_feature'] == 'binaria').sum()}",
            f"- Variables descartadas o con leakage: {discarded_df.shape[0]}",
            "- Revisar imputación en columnas con nulos antes de entrenar.",
            "- Revisar encoding para las categóricas de alta cardinalidad.",
            "- Mantener fuera del entrenamiento las variables monetarias directas descartadas por riesgo de fuga.",
            "- Confirmar tratamiento de ponderadores en evaluación y no solo en entrenamiento.",
            "- Próximo paso: modelado supervisado.",
        ]
    )
    path.write_text(text, encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    paths = build_paths(root)
    ensure_directories(paths)
    logger = configure_logging(paths.logs / f"etl_phase08_09_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    phase7_df = pd.read_pickle(paths.data_interim / PHASE7_FILE)
    analytical, model_ready, null_report, constant_report, dropped_and_redundant = build_modeling_views(phase7_df)

    if analytical.duplicated(subset=KEY_COLS).sum() != 0:
        raise ValueError("El dataset final no es unico por hogar.")

    redundant_corr = highly_redundant_columns(model_ready.drop(columns=[TARGET_COL], errors="ignore"))
    features_df, discarded_df, metadata = classify_features(model_ready)

    analytical_path = paths.data_interim / "enaho_arequipa_hogar_interim.parquet"
    processed_parquet_path = paths.data_processed / "enaho_arequipa_model_ready.parquet"
    processed_csv_path = paths.data_processed / "enaho_arequipa_model_ready.csv"

    analytical_safe = parquet_safe(analytical)
    model_ready_safe = parquet_safe(model_ready)

    analytical_safe.to_parquet(analytical_path, index=False)
    model_ready_safe.to_parquet(processed_parquet_path, index=False)
    model_ready.to_csv(processed_csv_path, index=False, encoding="utf-8-sig")

    null_report.to_csv(paths.reports / "phase8_null_report.csv", index=False, encoding="utf-8-sig")
    constant_report.to_csv(paths.reports / "phase8_constant_columns.csv", index=False, encoding="utf-8-sig")
    redundant_corr.to_csv(paths.reports / "phase8_high_redundancy.csv", index=False, encoding="utf-8-sig")
    dropped_and_redundant.to_csv(paths.reports / "columnas_descartadas.csv", index=False, encoding="utf-8-sig")
    features_df.to_csv(paths.reports / "features_finales.csv", index=False, encoding="utf-8-sig")
    discarded_df.to_csv(paths.reports / "riesgo_fuga_informacion.csv", index=False, encoding="utf-8-sig")
    write_checklist(paths.reports / "checklist_pre_modelado.md", features_df, discarded_df)

    metadata["analytical_path"] = str(analytical_path)
    metadata["model_ready_parquet_path"] = str(processed_parquet_path)
    metadata["model_ready_csv_path"] = str(processed_csv_path)
    (paths.reports / "metadata_modelado.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    validation = {
        "phase": "FASE 8-9",
        "passed": True,
        "analytical_rows": int(analytical.shape[0]),
        "analytical_cols": int(analytical.shape[1]),
        "model_ready_rows": int(model_ready.shape[0]),
        "model_ready_cols": int(model_ready.shape[1]),
        "target_present": TARGET_COL in model_ready.columns,
        "constant_cols": int(constant_report.shape[0]),
        "high_redundancy_pairs": int(redundant_corr.shape[0]),
        "features_finales": int(features_df.shape[0]),
    }
    save_json(validation, paths.reports / "phase8_9_validation.json")
    logger.info("FASE 8-9 validacion: %s", validation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
