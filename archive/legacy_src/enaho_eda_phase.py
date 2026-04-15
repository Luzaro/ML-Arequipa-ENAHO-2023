from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .enaho_etl_phase01 import build_paths, configure_logging, ensure_directories, save_json


TARGET = "target_pobreza_bin"
DATASET = "enaho_arequipa_model_ready.parquet"


def prepare_dirs(root: Path) -> tuple[Path, Path, Path]:
    paths = build_paths(root)
    ensure_directories(paths)
    eda_dir = paths.reports / "eda"
    figures_dir = eda_dir / "figures"
    eda_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return eda_dir, figures_dir, paths.data_processed / DATASET


def classify_columns(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        if col == TARGET:
            role = "target"
        elif set(pd.Series(df[col].dropna().unique()).tolist()).issubset({0, 1}):
            role = "binaria"
        elif pd.api.types.is_numeric_dtype(df[col]):
            role = "numerica"
        else:
            role = "categorica"
        rows.append(
            {
                "variable": col,
                "rol_eda": role,
                "dtype": str(df[col].dtype),
                "nulos": int(df[col].isna().sum()),
                "nulos_pct": round(df[col].isna().mean() * 100, 4),
                "n_unicos": int(df[col].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def target_summary(df: pd.DataFrame) -> pd.DataFrame:
    counts = df[TARGET].value_counts(dropna=False).rename_axis("target").reset_index(name="n")
    counts["pct"] = (counts["n"] / counts["n"].sum() * 100).round(4)
    return counts


def numeric_summary(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        rows.append(
            {
                "variable": col,
                "count": int(s.notna().sum()),
                "missing_pct": round(s.isna().mean() * 100, 4),
                "mean": s.mean(),
                "std": s.std(),
                "min": s.min(),
                "p25": s.quantile(0.25),
                "median": s.quantile(0.50),
                "p75": s.quantile(0.75),
                "max": s.max(),
            }
        )
    return pd.DataFrame(rows)


def categorical_summary(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cat_cols:
        counts = df[col].astype("string").fillna("<NA>").value_counts(dropna=False)
        rows.append(
            {
                "variable": col,
                "n_categorias": int(counts.shape[0]),
                "missing_pct": round(df[col].isna().mean() * 100, 4),
                "top_categoria": str(counts.index[0]) if not counts.empty else None,
                "top_n": int(counts.iloc[0]) if not counts.empty else 0,
                "rare_categorias_lt_10": int((counts < 10).sum()),
            }
        )
    return pd.DataFrame(rows)


def grouped_numeric_by_target(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in numeric_cols:
        grouped = df.groupby(TARGET)[col].agg(["count", "mean", "median", "std", "min", "max"]).reset_index()
        grouped.insert(0, "variable", col)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def categorical_by_target(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cat_cols:
        tab = pd.crosstab(df[col].astype("string").fillna("<NA>"), df[TARGET], normalize="columns").reset_index()
        tab.insert(0, "variable", col)
        tab = tab.rename(columns={0: "pct_target_0", 1: "pct_target_1", col: "categoria"})
        rows.append(tab)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def missing_by_household(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum(axis=1)
    return pd.DataFrame(
        {
            "missing_cols": missing,
            "missing_pct": (missing / df.shape[1] * 100).round(4),
            TARGET: df[TARGET],
        }
    )


def outlier_iqr(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        rows.append(
            {
                "variable": col,
                "lower_iqr": lower,
                "upper_iqr": upper,
                "outlier_count": int(((s < lower) | (s > upper)).sum()),
                "outlier_pct": round(((s < lower) | (s > upper)).mean() * 100, 4),
            }
        )
    return pd.DataFrame(rows)


def correlation_report(df: pd.DataFrame, numeric_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr(numeric_only=True)
    target_corr = corr[TARGET].drop(TARGET, errors="ignore").abs().sort_values(ascending=False).reset_index()
    target_corr.columns = ["variable", "abs_corr_target"]

    pairs = []
    for i, col_a in enumerate(corr.columns):
        for col_b in corr.columns[i + 1:]:
            value = corr.loc[col_a, col_b]
            if pd.notna(value) and abs(value) >= 0.90:
                pairs.append({"col_a": col_a, "col_b": col_b, "corr": value, "abs_corr": abs(value)})
    return target_corr, pd.DataFrame(pairs).sort_values("abs_corr", ascending=False) if pairs else pd.DataFrame()


def feature_recommendation(df: pd.DataFrame, column_profile: pd.DataFrame) -> pd.DataFrame:
    leakage_patterns = [
        "ing", "gasto", "gashog", "gru", "linea", "linpe", "pobreza", "factor07", "pobrezav",
        "ingbruhd", "ingnethd", "inghog", "ingtpu", "ingtra",
    ]
    rows = []
    for item in column_profile.to_dict(orient="records"):
        var = item["variable"]
        if var == TARGET:
            decision = "target"
            reason = "variable objetivo"
        elif any(pattern in var.lower() for pattern in leakage_patterns):
            decision = "excluir_conservador"
            reason = "riesgo de fuga o cercania monetaria a pobreza oficial"
        elif item["nulos_pct"] > 30:
            decision = "revisar"
            reason = "alto porcentaje de nulos"
        elif item["n_unicos"] <= 1:
            decision = "excluir"
            reason = "columna constante"
        else:
            decision = "incluir_conservador"
            reason = "feature defendible para EDA/modelado inicial"
        rows.append({"variable": var, "decision_eda": decision, "motivo": reason})
    return pd.DataFrame(rows)


def make_plots(df: pd.DataFrame, figures_dir: Path, numeric_focus: list[str], cat_focus: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=TARGET, hue=TARGET, palette="Blues", legend=False)
    plt.title("Distribucion del target de pobreza")
    plt.tight_layout()
    plt.savefig(figures_dir / "target_distribution.png", dpi=160)
    plt.close()

    for col in numeric_focus:
        if col not in df.columns:
            continue
        plt.figure(figsize=(7, 4))
        sns.boxplot(data=df, x=TARGET, y=col, hue=TARGET, palette="Blues", legend=False)
        plt.title(f"{col} por target")
        plt.tight_layout()
        plt.savefig(figures_dir / f"box_{col}_by_target.png", dpi=160)
        plt.close()

    for col in cat_focus:
        if col not in df.columns:
            continue
        tab = pd.crosstab(df[col].astype("string").fillna("<NA>"), df[TARGET], normalize="index")
        tab.plot(kind="bar", stacked=True, figsize=(9, 4), color=["#7aa6c2", "#d97852"])
        plt.title(f"Distribucion target por {col}")
        plt.ylabel("proporcion")
        plt.tight_layout()
        plt.savefig(figures_dir / f"cat_{col}_target_share.png", dpi=160)
        plt.close()

    missing = df.isna().mean().mul(100).sort_values(ascending=False).head(25)
    plt.figure(figsize=(9, 5))
    sns.barplot(x=missing.values, y=missing.index, color="#6c8ea4")
    plt.title("Top variables con nulos")
    plt.xlabel("% nulos")
    plt.tight_layout()
    plt.savefig(figures_dir / "missing_top25.png", dpi=160)
    plt.close()


def write_eda_report(path: Path, summary: dict) -> None:
    text = "\n".join(
        [
            "# Reporte EDA ENAHO Arequipa 2023",
            "",
            f"- Filas/hogares: {summary['rows']}",
            f"- Columnas: {summary['cols']}",
            f"- Hogares con algun nulo: {summary['households_with_any_null']}",
            f"- Hogares sin nulos: {summary['households_without_nulls']}",
            f"- Target 0 no pobre: {summary['target_counts'].get('0', summary['target_counts'].get(0, 0))}",
            f"- Target 1 pobre: {summary['target_counts'].get('1', summary['target_counts'].get(1, 0))}",
            "",
            "## Hallazgos",
            "- El target esta desbalanceado hacia no pobres, por lo que en modelado se deben revisar metricas como recall, F1 y ROC-AUC.",
            "- Existen nulos concentrados en pocas variables de servicios/vivienda, no en el target.",
            "- Se recomienda iniciar con un set conservador que excluya variables monetarias cercanas a la regla oficial de pobreza.",
            "",
            "## Proximo paso",
            "- Definir set final de features e iniciar modelado supervisado.",
        ]
    )
    path.write_text(text, encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    paths = build_paths(root)
    ensure_directories(paths)
    logger = configure_logging(paths.logs / f"eda_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    eda_dir, figures_dir, dataset_path = prepare_dirs(root)

    df = pd.read_parquet(dataset_path)
    profile = classify_columns(df)
    numeric_cols = profile.loc[profile["rol_eda"].isin(["numerica", "binaria", "target"]), "variable"].tolist()
    numeric_feature_cols = [col for col in numeric_cols if col != TARGET]
    cat_cols = profile.loc[profile["rol_eda"] == "categorica", "variable"].tolist()

    profile.to_csv(eda_dir / "eda_column_profile.csv", index=False, encoding="utf-8-sig")
    target_summary(df).to_csv(eda_dir / "eda_target_summary.csv", index=False, encoding="utf-8-sig")
    numeric_summary(df, numeric_feature_cols).to_csv(eda_dir / "eda_numeric_summary.csv", index=False, encoding="utf-8-sig")
    categorical_summary(df, cat_cols).to_csv(eda_dir / "eda_categorical_summary.csv", index=False, encoding="utf-8-sig")
    grouped_numeric_by_target(df, numeric_feature_cols).to_csv(eda_dir / "eda_numeric_by_target.csv", index=False, encoding="utf-8-sig")
    categorical_by_target(df, cat_cols).to_csv(eda_dir / "eda_categorical_by_target.csv", index=False, encoding="utf-8-sig")
    missing_by_household(df).to_csv(eda_dir / "eda_missing_by_household.csv", index=False, encoding="utf-8-sig")
    outlier_iqr(df, numeric_feature_cols).to_csv(eda_dir / "eda_outliers_iqr.csv", index=False, encoding="utf-8-sig")
    target_corr, high_corr_pairs = correlation_report(df, numeric_cols)
    target_corr.to_csv(eda_dir / "eda_target_correlations.csv", index=False, encoding="utf-8-sig")
    high_corr_pairs.to_csv(eda_dir / "eda_high_correlation_pairs.csv", index=False, encoding="utf-8-sig")
    feature_recommendation(df, profile).to_csv(eda_dir / "eda_feature_recommendation.csv", index=False, encoding="utf-8-sig")

    numeric_focus = ["edad_jefe", "tam_hogar", "num_perceptores", "n_ninos_6_16", "miembros_sin_atencion_salud"]
    cat_focus = ["educ_jefe", "area_urbana_rural", "material_piso", "combustible_principal"]
    make_plots(df, figures_dir, numeric_focus, cat_focus)

    summary = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "households_with_any_null": int(df.isna().any(axis=1).sum()),
        "households_without_nulls": int((~df.isna().any(axis=1)).sum()),
        "target_counts": {str(k): int(v) for k, v in df[TARGET].value_counts(dropna=False).to_dict().items()},
        "eda_dir": str(eda_dir),
        "figures_dir": str(figures_dir),
    }
    save_json(summary, eda_dir / "eda_summary.json")
    write_eda_report(eda_dir / "eda_report.md", summary)
    logger.info("EDA completado: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
