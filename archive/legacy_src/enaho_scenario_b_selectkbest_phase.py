from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

from .enaho_etl_phase01 import build_paths, configure_logging, ensure_directories, save_json


DATASET = "enaho_arequipa_escenario_b_clean.parquet"
DICT_FILE = "escenario_b_diccionario_variables.csv"
RECOMMENDED_K = 20

ID_COLS = ["hogar_id", "conglome", "vivienda", "hogar", "ubigeo"]
WEIGHT_COL = "factor_expansion_anual"
TARGET_COL = "target_pobreza_monetaria_bin"


def load_inputs(paths: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = pd.read_parquet(paths.data_processed / DATASET)
    dictionary_df = pd.read_csv(paths.reports / DICT_FILE)
    return dataset, dictionary_df


def feature_candidates(dataset: pd.DataFrame, dictionary_df: pd.DataFrame) -> list[str]:
    candidate_mask = dictionary_df["rol"].eq("feature_candidata")
    cols = dictionary_df.loc[candidate_mask, "variable_final"].tolist()
    return [col for col in cols if col in dataset.columns]


def build_expanded_matrix(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    expanded_parts: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []

    for col in features:
        series = df[col]
        dtype_str = str(series.dtype)

        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            fill_value = numeric.median()
            if pd.isna(fill_value):
                fill_value = 0
            part = pd.DataFrame({col: numeric.fillna(fill_value)})
            expanded_parts.append(part)
            metadata_rows.append(
                {
                    "expanded_feature": col,
                    "variable_original": col,
                    "tipo_expansion": "directa_numerica",
                }
            )
            continue

        cat = series.astype("string").fillna("<NA>")
        dummies = pd.get_dummies(cat, prefix=col, prefix_sep="__", dummy_na=False, dtype=int)
        if dummies.empty:
            dummies = pd.DataFrame({f"{col}__<NA>": np.ones(len(cat), dtype=int)})

        expanded_parts.append(dummies)
        metadata_rows.extend(
            {
                "expanded_feature": dummy_col,
                "variable_original": col,
                "tipo_expansion": f"dummy_desde_{dtype_str}",
            }
            for dummy_col in dummies.columns
        )

    expanded = pd.concat(expanded_parts, axis=1)
    expanded_meta = pd.DataFrame(metadata_rows)
    return expanded, expanded_meta


def compute_anova_scores(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, list[str]]:
    nunique = X.nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()
    X_used = X.drop(columns=constant_cols, errors="ignore")

    selector = SelectKBest(score_func=f_classif, k="all")
    selector.fit(X_used, y)

    scores_df = pd.DataFrame(
        {
            "expanded_feature": X_used.columns.tolist(),
            "f_score": selector.scores_,
            "p_value": selector.pvalues_,
        }
    )

    if constant_cols:
        scores_df = pd.concat(
            [
                scores_df,
                pd.DataFrame(
                    {
                        "expanded_feature": constant_cols,
                        "f_score": np.zeros(len(constant_cols), dtype=float),
                        "p_value": np.full(len(constant_cols), np.nan),
                    }
                ),
            ],
            ignore_index=True,
        )

    scores_df["f_score"] = pd.to_numeric(scores_df["f_score"], errors="coerce").fillna(0)
    scores_df["p_value"] = pd.to_numeric(scores_df["p_value"], errors="coerce")
    return scores_df, constant_cols


def aggregate_scores(expanded_meta: pd.DataFrame, scores_df: pd.DataFrame, dictionary_df: pd.DataFrame) -> pd.DataFrame:
    merged = expanded_meta.merge(scores_df, on="expanded_feature", how="left")
    merged["significativa_5pct"] = merged["p_value"] < 0.05

    feature_dict = dictionary_df.set_index("variable_final").to_dict(orient="index")
    rows = []
    for variable, group in merged.groupby("variable_original", sort=False):
        meta = feature_dict.get(variable, {})
        rows.append(
            {
                "variable": variable,
                "descripcion": meta.get("descripcion", variable),
                "dimension_analitica": meta.get("dimension_analitica", ""),
                "modulo_fuente": meta.get("modulo_fuente", ""),
                "tipo_logico": meta.get("tipo_logico", ""),
                "dtype_dataset": meta.get("dtype_dataset", ""),
                "expanded_features": int(group.shape[0]),
                "f_score_max": float(group["f_score"].max()),
                "f_score_mean": float(group["f_score"].mean()),
                "p_value_min": float(group["p_value"].min()) if group["p_value"].notna().any() else np.nan,
                "significativa_5pct": bool(group["significativa_5pct"].fillna(False).any()),
                "constant_expanded_features": int(group["f_score"].eq(0).sum()),
                "rol": meta.get("rol", ""),
                "observaciones": meta.get("observaciones", ""),
            }
        )

    ranking_df = pd.DataFrame(rows)
    ranking_df["rank_f_score"] = ranking_df["f_score_max"].rank(ascending=False, method="min")
    ranking_df = ranking_df.sort_values(["f_score_max", "p_value_min"], ascending=[False, True]).reset_index(drop=True)
    return ranking_df, merged


def build_selected_dataset(df: pd.DataFrame, selected_features: list[str]) -> pd.DataFrame:
    keep_cols = [*ID_COLS, WEIGHT_COL, TARGET_COL, *selected_features]
    keep_cols = [col for col in keep_cols if col in df.columns]
    return df[keep_cols].copy()


def write_report(path: Path, ranking_df: pd.DataFrame, top_k: pd.DataFrame) -> None:
    lines = [
        "# SelectKBest sobre escenario B limpio",
        "",
        "- Método: SelectKBest con f_classif (ANOVA).",
        "- Las variables categóricas se expandieron a dummies para calcular F-score y luego se agregaron al nivel de variable original usando el máximo F-score y el mínimo p-value.",
        f"- Variables candidatas evaluadas: {ranking_df.shape[0]}",
        f"- Variables significativas al 5%: {int(ranking_df['significativa_5pct'].sum())}",
        f"- K recomendado de trabajo: {RECOMMENDED_K}",
        "",
        "## Top por F-score",
        "",
    ]
    lines.extend(
        f"- {row.variable}: F_max={row.f_score_max:.4f}, p_min={row.p_value_min:.6g}, dimensión={row.dimension_analitica}"
        for row in top_k.itertuples()
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    paths = build_paths(root)
    ensure_directories(paths)
    logger = configure_logging(paths.logs / f"scenario_b_selectkbest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    dataset, dictionary_df = load_inputs(paths)
    features = feature_candidates(dataset, dictionary_df)
    X_expanded, expanded_meta = build_expanded_matrix(dataset, features)
    scores_df, constant_cols = compute_anova_scores(X_expanded, dataset[TARGET_COL].astype(int))
    ranking_df, expanded_scores_df = aggregate_scores(expanded_meta, scores_df, dictionary_df)

    top_k = ranking_df.head(RECOMMENDED_K).copy()
    significant_df = ranking_df.loc[ranking_df["significativa_5pct"]].copy()
    selected_dataset = build_selected_dataset(dataset, top_k["variable"].tolist())

    out_dir = paths.reports / "scenario_b_selectkbest"
    out_dir.mkdir(parents=True, exist_ok=True)

    expanded_scores_df.to_csv(out_dir / "expanded_feature_scores.csv", index=False, encoding="utf-8-sig")
    ranking_df.to_csv(out_dir / "variable_ranking_anova.csv", index=False, encoding="utf-8-sig")
    top_k.to_csv(out_dir / f"selected_features_top{RECOMMENDED_K}.csv", index=False, encoding="utf-8-sig")
    significant_df.to_csv(out_dir / "selected_features_significativas_5pct.csv", index=False, encoding="utf-8-sig")
    write_report(out_dir / "scenario_b_selectkbest_report.md", ranking_df, top_k)

    selected_parquet = paths.data_processed / f"enaho_arequipa_escenario_b_top{RECOMMENDED_K}.parquet"
    selected_csv = paths.data_processed / f"enaho_arequipa_escenario_b_top{RECOMMENDED_K}.csv"
    selected_dataset.to_parquet(selected_parquet, index=False)
    selected_dataset.to_csv(selected_csv, index=False, encoding="utf-8-sig")

    save_json(
        {
            "n_candidate_features": len(features),
            "n_expanded_features": int(X_expanded.shape[1]),
            "n_constant_expanded_features": len(constant_cols),
            "n_significant_features_5pct": int(significant_df.shape[0]),
            "recommended_k": RECOMMENDED_K,
            "top_features": top_k["variable"].tolist(),
            "selected_dataset_parquet": str(selected_parquet),
            "selected_dataset_csv": str(selected_csv),
        },
        out_dir / "scenario_b_selectkbest_summary.json",
    )

    validation = {
        "phase": "SCENARIO_B_SELECTKBEST",
        "passed": True,
        "candidate_features": len(features),
        "expanded_features": int(X_expanded.shape[1]),
        "constant_expanded_features": len(constant_cols),
        "recommended_k": RECOMMENDED_K,
        "selected_rows": int(selected_dataset.shape[0]),
        "selected_cols": int(selected_dataset.shape[1]),
    }
    save_json(validation, out_dir / "scenario_b_selectkbest_validation.json")
    logger.info("SCENARIO_B_SELECTKBEST validacion: %s", validation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
