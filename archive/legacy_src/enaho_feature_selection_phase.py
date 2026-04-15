from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

from .enaho_etl_phase01 import build_paths, configure_logging, ensure_directories, save_json


DATASET = "enaho_arequipa_model_ready.parquet"
TARGET = "target_pobreza_bin"


FEATURE_SPECS = [
    ("edad_jefe", "demografia_jefe", "numerica", "Edad del jefe del hogar"),
    ("sexo_jefe_bin", "demografia_jefe", "binaria", "Sexo del jefe del hogar"),
    ("educ_jefe_ord", "educacion_jefe", "ordinal", "Nivel educativo del jefe"),
    ("tam_hogar", "hogar", "numerica", "Tamano del hogar"),
    ("num_perceptores", "hogar", "numerica", "Numero de perceptores"),
    ("n_ninos_0_5", "hogar", "numerica", "Ninos de 0 a 5 anios"),
    ("n_ninos_6_16", "hogar", "numerica", "Ninos de 6 a 16 anios"),
    ("n_adultos_65_mas", "hogar", "numerica", "Adultos mayores de 65"),
    ("n_6_16_no_matriculados", "educacion", "numerica", "Conteo de menores no matriculados"),
    ("n_6_16_no_asisten", "educacion", "numerica", "Conteo de menores que no asisten"),
    ("children_not_enrolled", "educacion", "binaria", "Hogar con menores no matriculados"),
    ("children_not_attending", "educacion", "binaria", "Hogar con menores que no asisten"),
    ("miembros_sin_atencion_salud", "salud", "numerica", "Miembros sin atencion en salud"),
    ("health_no_attention", "salud", "binaria", "Hogar con al menos un miembro sin atencion"),
    ("miembros_afiliados_essalud", "salud", "numerica", "Miembros afiliados a EsSalud"),
    ("material_piso", "vivienda", "categorica", "Material del piso"),
    ("piso_precario", "vivienda", "binaria", "Indicador de piso precario"),
    ("p101", "vivienda", "categorica", "Tipo de vivienda"),
    ("p102", "vivienda", "categorica", "Material de paredes"),
    ("p103a", "vivienda", "categorica", "Material de techo"),
    ("p104", "vivienda", "numerica", "Cantidad de habitaciones"),
    ("p105a", "vivienda", "categorica", "Tenencia de vivienda"),
    ("acceso_agua_red", "servicios", "binaria", "Acceso a agua por red"),
    ("agua_potable_reportada", "servicios", "binaria", "Agua potable reportada"),
    ("agua_disponible_diaria", "servicios", "binaria", "Agua disponible todos los dias"),
    ("agua_segura_proxy", "servicios", "binaria", "Proxy de agua segura"),
    ("acceso_desague_red", "servicios", "binaria", "Acceso a desague por red"),
    ("saneamiento_inadecuado", "servicios", "binaria", "Indicador de saneamiento inadecuado"),
    ("acceso_electricidad", "servicios", "binaria", "Acceso a electricidad"),
    ("combustible_limpio", "servicios", "binaria", "Uso de combustible limpio"),
    ("combustible_precario", "servicios", "binaria", "Uso de combustible precario"),
    ("internet_hogar", "servicios", "binaria", "Conexion a internet en el hogar"),
    ("area_rural_bin", "geografia", "binaria", "Hogar en area rural"),
    ("dominio", "geografia", "categorica", "Dominio geografico"),
    ("estrato", "geografia", "categorica", "Estrato geografico"),
]


REDUNDANT_GROUPS = [
    ["n_6_16_no_matriculados", "children_not_enrolled"],
    ["n_6_16_no_asisten", "children_not_attending"],
    ["miembros_sin_atencion_salud", "health_no_attention"],
    ["material_piso", "piso_precario"],
    ["combustible_limpio", "combustible_precario"],
    ["dominio", "estrato", "area_rural_bin"],
]


def load_dataset(root: Path) -> pd.DataFrame:
    return pd.read_parquet(build_paths(root).data_processed / DATASET)


def available_specs(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variable, group, var_type, description in FEATURE_SPECS:
        if variable in df.columns:
            rows.append(
                {
                    "variable": variable,
                    "grupo": group,
                    "tipo": var_type,
                    "descripcion": description,
                    "nulos_pct": round(df[variable].isna().mean() * 100, 4),
                }
            )
    return pd.DataFrame(rows)


def encode_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = df[features].copy()
    for col in X.columns:
        if str(X[col].dtype) in {"object", "string", "category"}:
            X[col] = X[col].astype("string").fillna("<NA>").astype("category").cat.codes
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            median = X[col].median()
            if pd.isna(median):
                median = 0
            X[col] = X[col].fillna(median)
        if X[col].isna().any():
            X[col] = X[col].fillna(-1)
    return X


def compute_rankings(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = encode_features(df, features)
    y = df[TARGET].astype(int)

    discrete_mask = [pd.Series(df[col].dropna().unique()).isin([0, 1]).all() if df[col].dropna().size else False for col in features]
    mi = mutual_info_classif(X, y, discrete_features=discrete_mask, random_state=42)

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=3,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=1,
    )
    rf.fit(X, y)
    rf_imp = rf.feature_importances_

    rank_df = pd.DataFrame(
        {
            "variable": features,
            "mutual_info": mi,
            "rf_importance": rf_imp,
        }
    )
    rank_df["rank_mi"] = rank_df["mutual_info"].rank(ascending=False, method="min")
    rank_df["rank_rf"] = rank_df["rf_importance"].rank(ascending=False, method="min")
    rank_df["rank_promedio"] = (rank_df["rank_mi"] + rank_df["rank_rf"]) / 2
    rank_df = rank_df.sort_values(["rank_promedio", "mutual_info", "rf_importance"], ascending=[True, False, False]).reset_index(drop=True)
    return rank_df


def select_conservative(rank_df: pd.DataFrame) -> pd.DataFrame:
    selected = []
    used = set()
    for _, row in rank_df.iterrows():
        var = row["variable"]
        skip = False
        for group in REDUNDANT_GROUPS:
            if var in group and any(g in used for g in group if g != var):
                skip = True
                break
        if skip:
            continue
        selected.append(row.to_dict())
        used.add(var)
        if len(selected) >= 18:
            break
    return pd.DataFrame(selected)


def select_extended(rank_df: pd.DataFrame) -> pd.DataFrame:
    selected = []
    used = set()
    for _, row in rank_df.iterrows():
        var = row["variable"]
        skip = False
        for group in REDUNDANT_GROUPS:
            if var in group and any(g in used for g in group if g != var):
                skip = True
                break
        if skip:
            continue
        selected.append(row.to_dict())
        used.add(var)
        if len(selected) >= 28:
            break
    return pd.DataFrame(selected)


def join_metadata(rank_df: pd.DataFrame, specs_df: pd.DataFrame) -> pd.DataFrame:
    return specs_df.merge(rank_df, on="variable", how="left").sort_values("rank_promedio").reset_index(drop=True)


def write_report(path: Path, conservative: pd.DataFrame, extended: pd.DataFrame) -> None:
    text = "\n".join(
        [
            "# Seleccion de variables para clasificacion",
            "",
            "## Set conservador recomendado",
            *[f"- {v}" for v in conservative["variable"].tolist()],
            "",
            "## Set amplio recomendado",
            *[f"- {v}" for v in extended["variable"].tolist()],
            "",
            "## Criterio",
            "- Se excluyeron variables monetarias o muy cercanas a la regla oficial de pobreza monetaria.",
            "- Se priorizaron variables sociodemograficas, educativas, de salud, vivienda y servicios.",
            "- El ranking combina mutual information y random forest importance.",
            "",
            "## Uso sugerido",
            "- Empezar modelado con el set conservador.",
            "- Evaluar el set amplio como analisis de sensibilidad.",
        ]
    )
    path.write_text(text, encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    paths = build_paths(root)
    ensure_directories(paths)
    logger = configure_logging(paths.logs / f"feature_selection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    df = load_dataset(root)
    specs_df = available_specs(df)
    feature_list = specs_df["variable"].tolist()
    ranking_df = compute_rankings(df, feature_list)
    ranked_features = join_metadata(ranking_df, specs_df)

    conservative = select_conservative(ranking_df).merge(specs_df, on="variable", how="left")
    extended = select_extended(ranking_df).merge(specs_df, on="variable", how="left")

    out_dir = paths.reports / "feature_selection"
    out_dir.mkdir(parents=True, exist_ok=True)

    specs_df.to_csv(out_dir / "candidate_features_theory.csv", index=False, encoding="utf-8-sig")
    ranked_features.to_csv(out_dir / "feature_ranking.csv", index=False, encoding="utf-8-sig")
    conservative.to_csv(out_dir / "selected_features_conservative.csv", index=False, encoding="utf-8-sig")
    extended.to_csv(out_dir / "selected_features_extended.csv", index=False, encoding="utf-8-sig")
    write_report(out_dir / "feature_selection_report.md", conservative, extended)

    save_json(
        {
            "candidate_features": int(specs_df.shape[0]),
            "selected_conservative": conservative["variable"].tolist(),
            "selected_extended": extended["variable"].tolist(),
        },
        out_dir / "feature_selection_summary.json",
    )
    logger.info("Seleccion de variables completada en %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
