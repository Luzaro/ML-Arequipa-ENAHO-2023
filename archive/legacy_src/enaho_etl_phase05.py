from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .enaho_etl_phase01 import build_paths, configure_logging, ensure_directories, save_json


FILTERED_DIRNAME = "phase4_arequipa"
HOUSEHOLD_KEY = ["conglome", "vivienda", "hogar"]
PERSON_KEY = HOUSEHOLD_KEY + ["codperso"]
MOD01_KEEP = [
    "ubigeo",
    "dominio",
    "estrato",
    "p101",
    "p102",
    "p103",
    "p103a",
    "p104",
    "p105a",
    "p106",
    "p110",
    "p110a1",
    "p110c",
    "p111a",
    "p1121",
    "p112a",
    "p1131",
    "p1132",
    "p1133",
    "p1135",
    "p1136",
    "p1137",
    "p1139",
    "p113a",
    "p1141",
    "p1142",
    "p1143",
    "p1144",
    "p1145",
    "p1146",
    "p1147",
    "p1148",
    "p114a",
    "p114b1",
    "p114b2",
    "p114b3",
    "p114b4",
    "p114b5",
    "p114b6",
    "p114b7",
    "p114d",
]


def load_table(base_dir: Path, filename: str) -> pd.DataFrame:
    return pd.read_pickle(base_dir / filename)


def ensure_household_uniqueness(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    duplicates = df[df.duplicated(subset=HOUSEHOLD_KEY, keep=False)].copy()
    if not duplicates.empty:
        raise ValueError(f"La tabla {table_name} no es unica por hogar.")
    return df


def aggregate_roster(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    roster = df.copy()
    p203_text = roster["p203"].astype(str).str.strip().str.lower()
    roster["is_jefe"] = roster["p203"].eq(1) | p203_text.str.contains("jefe", na=False)
    roster["age_num"] = pd.to_numeric(roster["p208a"], errors="coerce")
    roster["sexo_num"] = pd.to_numeric(roster["p207"], errors="coerce")

    chief_rows = (
        roster.loc[roster["is_jefe"], PERSON_KEY + ["p203", "p207", "p208a"]]
        .sort_values(PERSON_KEY)
        .drop_duplicates(subset=HOUSEHOLD_KEY, keep="first")
        .rename(columns={"p207": "sexo_jefe", "p208a": "edad_jefe", "p203": "codigo_parentesco_jefe"})
    )

    chief_count_audit = (
        roster.groupby(HOUSEHOLD_KEY)["is_jefe"]
        .sum()
        .reset_index(name="num_jefes_detectados")
    )

    hh = (
        roster.groupby(HOUSEHOLD_KEY)
        .agg(
            tam_hogar_mod2=("codperso", "nunique"),
            n_ninos_0_5=("age_num", lambda s: int(((s >= 0) & (s <= 5)).sum())),
            n_ninos_6_16=("age_num", lambda s: int(((s >= 6) & (s <= 16)).sum())),
            n_adultos_65_mas=("age_num", lambda s: int((s >= 65).sum())),
        )
        .reset_index()
    )
    hh["jefe_hogar"] = 1
    hh = hh.merge(chief_rows[HOUSEHOLD_KEY + ["sexo_jefe", "edad_jefe", "codigo_parentesco_jefe"]], on=HOUSEHOLD_KEY, how="left")
    hh = hh.merge(chief_count_audit, on=HOUSEHOLD_KEY, how="left")

    chief_audit = chief_count_audit[chief_count_audit["num_jefes_detectados"] != 1].copy()
    return hh, chief_rows[PERSON_KEY].copy(), chief_audit


def aggregate_education(df: pd.DataFrame, chief_keys: pd.DataFrame, roster: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    edu = df.copy()
    roster_age = roster[PERSON_KEY + ["p208a"]].copy()
    roster_age["age_num"] = pd.to_numeric(roster_age["p208a"], errors="coerce")

    edu = edu.merge(roster_age[PERSON_KEY + ["age_num"]], on=PERSON_KEY, how="left")
    edu = edu.merge(chief_keys.assign(is_jefe=1), on=PERSON_KEY, how="left")
    edu["is_jefe"] = edu["is_jefe"].fillna(0).astype(int)

    chief_edu = (
        edu.loc[edu["is_jefe"].eq(1), PERSON_KEY + ["p301a", "p301b", "p301c"]]
        .sort_values(PERSON_KEY)
        .drop_duplicates(subset=HOUSEHOLD_KEY, keep="first")
        .rename(
            columns={
                "p301a": "educ_jefe_nivel",
                "p301b": "educ_jefe_anio",
                "p301c": "educ_jefe_grado",
            }
        )
    )

    hh = (
        edu.groupby(HOUSEHOLD_KEY)
        .agg(personas_con_mod_educ=("codperso", "nunique"))
        .reset_index()
    )

    school_age = edu.loc[edu["age_num"].between(6, 16, inclusive="both")].copy()
    school_age_counts = school_age.groupby(HOUSEHOLD_KEY).size().reset_index(name="n_6_16_total")
    school_age_matricula = (
        school_age.groupby(HOUSEHOLD_KEY)
        .agg(
            n_6_16_matriculados=("p306", lambda s: int((pd.to_numeric(s, errors="coerce") == 1).sum())),
            n_6_16_no_matriculados=("p306", lambda s: int((pd.to_numeric(s, errors="coerce") == 2).sum())),
            n_6_16_no_asisten=("p307", lambda s: int((pd.to_numeric(s, errors="coerce") == 2).sum())),
        )
        .reset_index()
    )

    hh = hh.merge(school_age_counts, on=HOUSEHOLD_KEY, how="left")
    hh = hh.merge(school_age_matricula, on=HOUSEHOLD_KEY, how="left")
    hh["n_6_16_total"] = hh["n_6_16_total"].fillna(0).astype(int)
    hh["n_6_16_matriculados"] = hh["n_6_16_matriculados"].fillna(0).astype(int)
    hh["n_6_16_no_matriculados"] = hh["n_6_16_no_matriculados"].fillna(0).astype(int)
    hh["n_6_16_no_asisten"] = hh["n_6_16_no_asisten"].fillna(0).astype(int)
    hh["any_6_16_no_matriculados"] = (hh["n_6_16_no_matriculados"] > 0).astype(int)
    hh["any_6_16_no_asisten"] = (hh["n_6_16_no_asisten"] > 0).astype(int)
    hh = hh.merge(chief_edu[HOUSEHOLD_KEY + ["educ_jefe_nivel", "educ_jefe_anio", "educ_jefe_grado"]], on=HOUSEHOLD_KEY, how="left")

    chief_edu_audit = hh[hh["educ_jefe_nivel"].isna()].copy()
    return hh, chief_edu_audit


def any_not_pase(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
    present = [col for col in cols if col in frame.columns]
    if not present:
        return pd.Series(np.zeros(len(frame), dtype=int), index=frame.index)
    cleaned = frame[present].copy()
    for col in present:
        cleaned[col] = cleaned[col].astype("string").str.strip().str.lower()
    mask = cleaned.notna() & ~cleaned.isin(["pase", "none", "nan", ""])
    return mask.any(axis=1).astype(int)


def aggregate_health(df: pd.DataFrame) -> pd.DataFrame:
    health = df.copy()
    no_attention_cols = [f"p409{i}" for i in range(1, 12)]
    health["salud_sintoma_4w"] = (health["p4021"].notna() & health["p4021"].astype(str).str.strip().str.lower().ne("pase")).astype(int)
    health["salud_enfermedad_4w"] = (health["p4022"].notna() & health["p4022"].astype(str).str.strip().str.lower().ne("pase")).astype(int)
    health["salud_no_atencion"] = any_not_pase(health, no_attention_cols)
    health["salud_afiliado_essalud"] = (health["p4191"].notna() & health["p4191"].astype(str).str.strip().str.lower().eq("essalud")).astype(int)

    hh = (
        health.groupby(HOUSEHOLD_KEY)
        .agg(
            personas_con_mod_salud=("codperso", "nunique"),
            miembros_con_sintoma_4w=("salud_sintoma_4w", "sum"),
            miembros_con_enfermedad_4w=("salud_enfermedad_4w", "sum"),
            miembros_sin_atencion_salud=("salud_no_atencion", "sum"),
            miembros_afiliados_essalud=("salud_afiliado_essalud", "sum"),
        )
        .reset_index()
    )
    hh["any_sintoma_4w"] = (hh["miembros_con_sintoma_4w"] > 0).astype(int)
    hh["any_enfermedad_4w"] = (hh["miembros_con_enfermedad_4w"] > 0).astype(int)
    hh["any_sin_atencion_salud"] = (hh["miembros_sin_atencion_salud"] > 0).astype(int)
    return hh


def aggregate_employment(df: pd.DataFrame, chief_keys: pd.DataFrame) -> pd.DataFrame:
    emp = df.copy()
    income_cols = [col for col in ["p524a1", "p530a", "p538a1", "p541a"] if col in emp.columns]
    positive_income = sum(pd.to_numeric(emp[col], errors="coerce").fillna(0).gt(0).astype(int) for col in income_cols)
    emp["any_ingreso_laboral_declarado"] = positive_income.gt(0).astype(int)

    emp = emp.merge(chief_keys.assign(is_jefe=1), on=PERSON_KEY, how="left")
    emp["is_jefe"] = emp["is_jefe"].fillna(0).astype(int)
    emp["chief_income_declared"] = ((emp["is_jefe"] == 1) & (emp["any_ingreso_laboral_declarado"] == 1)).astype(int)

    hh = (
        emp.groupby(HOUSEHOLD_KEY)
        .agg(
            personas_con_mod_empleo=("codperso", "nunique"),
            miembros_con_ingreso_laboral=("any_ingreso_laboral_declarado", "sum"),
            jefe_con_ingreso_laboral=("chief_income_declared", "max"),
        )
        .reset_index()
    )
    hh["any_ingreso_laboral_hogar"] = (hh["miembros_con_ingreso_laboral"] > 0).astype(int)
    return hh


def merge_with_audit(base: pd.DataFrame, addon: pd.DataFrame, name: str, audit_rows: list[dict[str, Any]]) -> pd.DataFrame:
    before_rows = int(base.shape[0])
    addon_cols = HOUSEHOLD_KEY + [col for col in addon.columns if col not in HOUSEHOLD_KEY and col not in base.columns]
    merged = base.merge(addon[addon_cols], on=HOUSEHOLD_KEY, how="left", validate="one_to_one")
    after_rows = int(merged.shape[0])
    dup_count = int(merged.duplicated(subset=HOUSEHOLD_KEY).sum())
    audit_rows.append(
        {
            "merge_name": name,
            "base_rows_before": before_rows,
            "addon_rows": int(addon.shape[0]),
            "rows_after_merge": after_rows,
            "duplicated_households_after_merge": dup_count,
            "new_columns_added": "|".join([col for col in addon_cols if col not in HOUSEHOLD_KEY]),
        }
    )
    if dup_count != 0:
        raise ValueError(f"Merge {name} genero duplicados por hogar.")
    return merged


def run_phase5(root: Path) -> dict[str, Any]:
    paths = build_paths(root)
    ensure_directories(paths)
    logger = configure_logging(paths.logs / f"etl_phase05_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    base_dir = paths.data_interim / FILTERED_DIRNAME

    mod01 = load_table(base_dir, "906-Modulo01__enaho01_2023_100.pkl")
    mod02 = load_table(base_dir, "906-Modulo02__enaho01_2023_200.pkl")
    mod03 = load_table(base_dir, "906-Modulo03__enaho01a_2023_300.pkl")
    mod04 = load_table(base_dir, "906-Modulo04__enaho01a_2023_400.pkl")
    mod05 = load_table(base_dir, "906-Modulo05__enaho01a_2023_500.pkl")
    sumaria = load_table(base_dir, "906-Modulo34__sumaria_2023.pkl")

    base_hogar = ensure_household_uniqueness(sumaria.copy(), "sumaria_2023")
    mod01_cols = [col for col in HOUSEHOLD_KEY + MOD01_KEEP if col in mod01.columns]
    mod01_unique = ensure_household_uniqueness(mod01[mod01_cols].drop_duplicates(), "modulo01")

    roster_hh, chief_keys, chief_audit = aggregate_roster(mod02)
    edu_hh, chief_edu_audit = aggregate_education(mod03, chief_keys, mod02[PERSON_KEY + ["p208a"]].copy())
    health_hh = aggregate_health(mod04)
    employment_hh = aggregate_employment(mod05, chief_keys)

    merge_audit_rows: list[dict[str, Any]] = []
    hogar_df = base_hogar.copy()
    hogar_df = merge_with_audit(hogar_df, mod01_unique, "modulo01_context", merge_audit_rows)
    hogar_df = merge_with_audit(hogar_df, roster_hh, "modulo02_roster", merge_audit_rows)
    hogar_df = merge_with_audit(hogar_df, edu_hh, "modulo03_education", merge_audit_rows)
    hogar_df = merge_with_audit(hogar_df, health_hh, "modulo04_health", merge_audit_rows)
    hogar_df = merge_with_audit(hogar_df, employment_hh, "modulo05_employment", merge_audit_rows)

    duplicate_households = hogar_df[hogar_df.duplicated(subset=HOUSEHOLD_KEY, keep=False)].copy()
    duplicate_households.to_csv(paths.reports / "phase5_duplicate_households.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(merge_audit_rows).to_csv(paths.reports / "phase5_merge_audit.csv", index=False, encoding="utf-8-sig")
    chief_audit.to_csv(paths.reports / "phase5_chief_audit.csv", index=False, encoding="utf-8-sig")
    chief_edu_audit.to_csv(paths.reports / "phase5_chief_education_missing.csv", index=False, encoding="utf-8-sig")

    aggregation_rules = pd.DataFrame(
        [
            {"variable_agregada": "tam_hogar_mod2", "fuente": "Modulo02", "nivel_origen": "persona", "regla": "nunique(codperso) por hogar"},
            {"variable_agregada": "edad_jefe", "fuente": "Modulo02", "nivel_origen": "persona", "regla": "p208a del miembro con p203 == 1"},
            {"variable_agregada": "sexo_jefe", "fuente": "Modulo02", "nivel_origen": "persona", "regla": "p207 del miembro con p203 == 1"},
            {"variable_agregada": "n_ninos_6_16", "fuente": "Modulo02", "nivel_origen": "persona", "regla": "conteo de p208a entre 6 y 16"},
            {"variable_agregada": "educ_jefe_nivel", "fuente": "Modulo03", "nivel_origen": "persona", "regla": "p301a del jefe de hogar identificado en Modulo02"},
            {"variable_agregada": "n_6_16_no_matriculados", "fuente": "Modulo03", "nivel_origen": "persona", "regla": "conteo de p306 == 2 para edades 6-16"},
            {"variable_agregada": "any_6_16_no_asisten", "fuente": "Modulo03", "nivel_origen": "persona", "regla": "indicador de algun p307 == 2 para edades 6-16"},
            {"variable_agregada": "miembros_sin_atencion_salud", "fuente": "Modulo04", "nivel_origen": "persona", "regla": "conteo si alguna razon p4091..p40911 es distinta de pase/nulo"},
            {"variable_agregada": "any_sin_atencion_salud", "fuente": "Modulo04", "nivel_origen": "persona", "regla": "1 si miembros_sin_atencion_salud > 0"},
            {"variable_agregada": "miembros_con_ingreso_laboral", "fuente": "Modulo05", "nivel_origen": "persona", "regla": "conteo si algun ingreso positivo en p524a1/p530a/p538a1/p541a"},
        ]
    )
    aggregation_rules.to_csv(paths.reports / "phase5_aggregation_rules.csv", index=False, encoding="utf-8-sig")

    hogar_df.to_pickle(paths.data_interim / "enaho_arequipa_hogar_base_phase5.pkl")

    validation = {
        "phase": "FASE 5",
        "passed": bool(hogar_df.shape[0] > 0 and duplicate_households.empty),
        "hogares_finales": int(hogar_df.shape[0]),
        "duplicated_households": int(duplicate_households.shape[0]),
        "chief_households_with_issues": int(chief_audit.shape[0]),
        "chief_education_missing": int(chief_edu_audit.shape[0]),
        "output_path": str(paths.data_interim / "enaho_arequipa_hogar_base_phase5.pkl"),
    }
    save_json(validation, paths.reports / "phase5_validation.json")
    logger.info("FASE 5 validacion: %s", validation)
    return validation


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    validation = run_phase5(root)
    return 0 if validation["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
