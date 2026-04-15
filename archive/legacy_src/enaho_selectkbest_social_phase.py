from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

from .enaho_etl_phase01 import build_paths, configure_logging, ensure_directories, save_json


DATASET = "enaho_arequipa_model_ready.parquet"
TARGET = "target_pobreza_bin"

ALIMENTARY_PROGRAM_VARS = ["ingtpu16", "sg27", "sig28"]
NON_ALIMENTARY_PROGRAM_VARS = ["ingtpu01", "ingtpu02", "ingtpu03", "ingtpu04", "ingtpu05", "ingtpu10", "ingtpu12", "ingtpu13", "ingtpu14"]
REQUESTED_RAW_VARS = ["inghog2d", "ingtpuhd", *ALIMENTARY_PROGRAM_VARS, *NON_ALIMENTARY_PROGRAM_VARS]

DERIVED_SOCIAL_VARS = {
    "inghog2d": ("programas_sociales", "Ingreso neto total del hogar", "cuantitativa"),
    "ingtpuhd": ("programas_sociales", "Ingreso total por transferencias públicas", "cuantitativa"),
    "monto_programas_sociales_alimentarios": ("programas_sociales", "Monto total de apoyos sociales alimentarios", "cuantitativa"),
    "monto_programas_sociales_no_alimentarios": ("programas_sociales", "Monto total de programas sociales no alimentarios", "cuantitativa"),
    "recibe_programa_social_alimentario": ("programas_sociales", "Indicador de recepción de programa social alimentario", "binaria"),
    "recibe_programa_social_no_alimentario": ("programas_sociales", "Indicador de recepción de programa social no alimentario", "binaria"),
    "ingtpu16": ("programas_sociales", "Ingreso por bono alimentario", "cuantitativa"),
    "sg27": ("programas_sociales", "Apoyo alimentario por ollas comunes", "cuantitativa"),
    "sig28": ("programas_sociales", "Ingreso/gasto por alimentos consumidos por ollas comunes", "cuantitativa"),
    "ingtpu01": ("programas_sociales", "Ingreso por Juntos", "cuantitativa"),
    "ingtpu02": ("programas_sociales", "Ingreso por otras transferencias públicas", "cuantitativa"),
    "ingtpu03": ("programas_sociales", "Ingreso por Pensión 65", "cuantitativa"),
    "ingtpu04": ("programas_sociales", "Ingreso por Beca 18", "cuantitativa"),
    "ingtpu05": ("programas_sociales", "Ingreso por Bono del Gas", "cuantitativa"),
    "ingtpu10": ("programas_sociales", "Ingreso por Bono Electricidad", "cuantitativa"),
    "ingtpu12": ("programas_sociales", "Ingreso por bono ONP para jubilados", "cuantitativa"),
    "ingtpu13": ("programas_sociales", "Ingreso por programa Contigo", "cuantitativa"),
    "ingtpu14": ("programas_sociales", "Ingreso por bono Yanapay", "cuantitativa"),
}


def load_dataset(paths: any) -> pd.DataFrame:
    return pd.read_parquet(paths.data_processed / DATASET)


def load_base_candidates(paths: any) -> pd.DataFrame:
    return pd.read_csv(paths.reports / "variables_candidatas_selectkbest.csv")


def add_requested_variables(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in REQUESTED_RAW_VARS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    out["monto_programas_sociales_alimentarios"] = out[[c for c in ALIMENTARY_PROGRAM_VARS if c in out.columns]].sum(axis=1)
    out["monto_programas_sociales_no_alimentarios"] = out[[c for c in NON_ALIMENTARY_PROGRAM_VARS if c in out.columns]].sum(axis=1)
    out["recibe_programa_social_alimentario"] = (out["monto_programas_sociales_alimentarios"] > 0).astype(int)
    out["recibe_programa_social_no_alimentario"] = (out["monto_programas_sociales_no_alimentarios"] > 0).astype(int)
    return out


def build_candidate_table(base_candidates: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    rows = base_candidates.to_dict(orient="records")
    existing = {row["variable"] for row in rows}

    for variable, (dimension, descripcion, tipo) in DERIVED_SOCIAL_VARS.items():
        if variable not in df.columns or variable in existing:
            continue
        rows.append(
            {
                "variable": variable,
                "descripcion": descripcion,
                "modulo_fuente": "Sumaria",
                "tipo_feature": tipo,
                "dtype": str(df[variable].dtype),
                "nulos_pct": round(df[variable].isna().mean() * 100, 4),
                "requiere_imputacion": bool(df[variable].isna().any()),
                "requiere_encoding": False,
                "riesgo_leakage_reportado": "medio" if variable != "inghog2d" else "alto",
                "dimension_inei": "No aplica",
                "dimension_analitica": dimension,
                "subdimension": "Programas sociales e ingreso neto",
                "apta_selectkbest": True,
                "motivo_selectkbest": "Variable solicitada explícitamente para evaluación ANOVA.",
            }
        )

    candidate_df = pd.DataFrame(rows)
    candidate_df = candidate_df.loc[candidate_df["variable"].isin(df.columns)].copy()
    return candidate_df.drop_duplicates(subset=["variable"]).reset_index(drop=True)


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


def compute_anova(df: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    feature_list = candidates["variable"].tolist()
    X = encode_features(df, feature_list)
    y = df[TARGET].astype(int)

    selector = SelectKBest(score_func=f_classif, k="all")
    selector.fit(X, y)

    result = candidates.copy()
    result["f_score"] = selector.scores_
    result["p_value"] = selector.pvalues_
    result["f_score"] = pd.to_numeric(result["f_score"], errors="coerce").fillna(0)
    result["p_value"] = pd.to_numeric(result["p_value"], errors="coerce")
    result["rank_f"] = result["f_score"].rank(ascending=False, method="min")
    result["significativa_5pct"] = result["p_value"] < 0.05
    return result.sort_values(["f_score", "p_value"], ascending=[False, True]).reset_index(drop=True)


def write_report(path: Path, ranking: pd.DataFrame) -> None:
    top20 = ranking.head(20)
    text = "\n".join(
        [
            "# SelectKBest con ANOVA (f_classif)",
            "",
            "## Supuestos usados",
            "- Se incluyeron variables dimensionales previamente mapeadas.",
            "- Se añadieron programas sociales alimentarios y no alimentarios solicitados por el usuario.",
            "- Se añadió ingreso neto total del hogar (`inghog2d`) por solicitud expresa.",
            "- Los programas alimentarios se aproximaron con `ingtpu16`, `sg27` y `sig28`.",
            "- Los programas no alimentarios se aproximaron con `ingtpu01`, `ingtpu02`, `ingtpu03`, `ingtpu04`, `ingtpu05`, `ingtpu10`, `ingtpu12`, `ingtpu13` e `ingtpu14`.",
            "",
            "## Nota metodológica",
            "- `inghog2d` y varias transferencias públicas están más cerca del target monetario que las variables estructurales; interpretar su selección con cautela por riesgo de leakage o proximidad conceptual.",
            "",
            "## Top 20 por F-score",
            *[f"- {row.variable}: F={row.f_score:.4f}, p={row.p_value:.6g}" for row in top20.itertuples()],
        ]
    )
    path.write_text(text, encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    paths = build_paths(root)
    ensure_directories(paths)
    logger = configure_logging(paths.logs / f"selectkbest_social_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    df = load_dataset(paths)
    df = add_requested_variables(df)
    base_candidates = load_base_candidates(paths)
    candidates = build_candidate_table(base_candidates, df)
    ranking = compute_anova(df, candidates)

    out_dir = paths.reports / "selectkbest_social"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates.to_csv(out_dir / "candidate_variables_selectkbest_social.csv", index=False, encoding="utf-8-sig")
    ranking.to_csv(out_dir / "anova_f_classif_ranking.csv", index=False, encoding="utf-8-sig")
    ranking.head(20).to_csv(out_dir / "anova_top20.csv", index=False, encoding="utf-8-sig")
    ranking.loc[ranking["significativa_5pct"]].to_csv(out_dir / "anova_significativas_5pct.csv", index=False, encoding="utf-8-sig")
    write_report(out_dir / "selectkbest_social_report.md", ranking)

    save_json(
        {
            "n_candidates": int(candidates.shape[0]),
            "n_significativas_5pct": int(ranking["significativa_5pct"].sum()),
            "top10": ranking.head(10)["variable"].tolist(),
        },
        out_dir / "selectkbest_social_summary.json",
    )
    logger.info("SelectKBest social completado en %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
