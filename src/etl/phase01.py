from __future__ import annotations

import importlib
import json
import logging
import platform
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SEED = 42
MINIMAL_ANNUAL_MODULE_CODES = {"1", "2", "3", "4", "5", "34"}


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_raw: Path
    data_interim: Path
    data_processed: Path
    reports: Path
    logs: Path
    src: Path
    notebooks: Path


def build_paths(root: Path) -> ProjectPaths:
    return ProjectPaths(
        root=root,
        data_raw=root / "data" / "raw",
        data_interim=root / "data" / "interim",
        data_processed=root / "data" / "processed",
        reports=root / "reports",
        logs=root / "logs",
        src=root / "src",
        notebooks=root / "notebooks",
    )


def ensure_directories(paths: ProjectPaths) -> None:
    for path in (
        paths.data_raw,
        paths.data_interim,
        paths.data_processed,
        paths.reports,
        paths.logs,
        paths.src,
        paths.notebooks,
    ):
        path.mkdir(parents=True, exist_ok=True)


def configure_logging(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("enaho_etl")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    try:
        numpy = importlib.import_module("numpy")
        numpy.random.seed(seed)
    except ModuleNotFoundError:
        pass


def module_version(module_name: str) -> str | None:
    try:
        module = importlib.import_module(module_name)
        return getattr(module, "__version__", None)
    except ModuleNotFoundError:
        return None


def capture_environment() -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "libraries": {
            "inei_microdatos": module_version("inei_microdatos"),
            "pandas": module_version("pandas"),
            "numpy": module_version("numpy"),
            "pyreadstat": module_version("pyreadstat"),
        },
    }


def validate_phase0(env_info: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not env_info["libraries"]["inei_microdatos"]:
        errors.append("La libreria 'inei_microdatos' no esta instalada o no es importable.")
    return errors


def save_json(data: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_dataframe_csv(df: Any, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def normalize_columns(df: Any) -> Any:
    df = df.copy()
    df.columns = [str(col).strip().lower() for col in df.columns]
    return df


def serialize_nested_objects(df: Any) -> Any:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].map(
                lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list))
                else value
            )
    return df


def normalize_text(value: Any) -> str:
    text = str(value or "")
    text = text.replace("\x96", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def load_catalog_with_fallback(paths: ProjectPaths, logger: logging.Logger) -> tuple[list[dict[str, Any]], str, str | None]:
    inei_microdatos = importlib.import_module("inei_microdatos")
    catalog_age = getattr(inei_microdatos, "catalog_age", lambda path=None: None)

    try:
        logger.info("Intentando reconstruir catalogo ENAHO 2023 desde el portal INEI")
        catalog = inei_microdatos.build_catalog(
            surveys=["ENAHO"],
            years=(2023, 2023),
            progress=False,
        )
        source = "portal_build_catalog"
        save_json(
            {"catalog": catalog, "source": source},
            paths.reports / "catalog_enaho_2023_built.json",
        )
        age = datetime.now(timezone.utc).isoformat()
        return catalog, source, age
    except Exception as exc:
        logger.warning("Fallo build_catalog(); se usa catalogo embebido. Error: %s", exc)
        catalog = inei_microdatos.load_catalog()
        source = "bundled_load_catalog"
        age = catalog_age()
        save_json(
            {"catalog": catalog, "source": source, "catalog_age": age},
            paths.reports / "catalog_enaho_fallback.json",
        )
        return catalog, source, age


def flatten_catalog(catalog: list[dict[str, Any]]) -> Any:
    pandas = importlib.import_module("pandas")
    rows: list[dict[str, Any]] = []

    for entry in catalog:
        for year, year_data in entry.get("years", {}).items():
            for period_label, period_data in year_data.items():
                for mod in period_data.get("modules", []):
                    preferred_format = None
                    preferred_code = None
                    for fmt_name, code_key in (
                        ("STATA", "stata_code"),
                        ("CSV", "csv_code"),
                        ("SPSS", "spss_code"),
                    ):
                        if mod.get(code_key):
                            preferred_format = fmt_name
                            preferred_code = mod.get(code_key)
                            break

                    rows.append(
                        {
                            "category": entry.get("category"),
                            "survey_value": entry.get("value"),
                            "survey_label": entry.get("label"),
                            "year": str(year),
                            "period_label": period_label,
                            "period_value": period_data.get("period_value"),
                            "module_code": str(mod.get("module_code")),
                            "module_name": mod.get("module_name"),
                            "csv_code": mod.get("csv_code"),
                            "stata_code": mod.get("stata_code"),
                            "spss_code": mod.get("spss_code"),
                            "preferred_format": preferred_format,
                            "preferred_code": preferred_code,
                            "is_downloadable": bool(preferred_code),
                        }
                    )

    return normalize_columns(pandas.DataFrame(rows))


def filter_enaho_2023_anual_modules(catalog_df: Any) -> tuple[Any, dict[str, Any]]:
    df = normalize_columns(catalog_df)
    annual_mask = (
        df["survey_label"].map(normalize_text).str.contains("enaho", na=False)
        & ~df["survey_label"].map(normalize_text).str.contains("panel", na=False)
        & df["year"].astype(str).eq("2023")
        & df["period_label"].map(normalize_text).str.contains("anual", na=False)
    )
    filtered = df.loc[annual_mask].copy()

    filtered["module_code"] = filtered["module_code"].astype(str)
    filtered = filtered[filtered["module_code"].isin(MINIMAL_ANNUAL_MODULE_CODES)].copy()

    summary = {
        "survey_labels": sorted(filtered["survey_label"].dropna().unique().tolist()),
        "period_labels": sorted(filtered["period_label"].dropna().unique().tolist()),
        "module_count": int(filtered.shape[0]),
        "downloadable_count": int(filtered["is_downloadable"].sum()) if not filtered.empty else 0,
    }
    return filtered, summary


def search_concepts(
    annual_modules_df: Any,
    logger: logging.Logger,
    concept_queries: dict[str, list[str]],
) -> tuple[Any, Any, Any]:
    inei_microdatos = importlib.import_module("inei_microdatos")
    pandas = importlib.import_module("pandas")

    allowed_codes = set(annual_modules_df["preferred_code"].dropna().astype(str).tolist())
    allowed_survey_labels = {normalize_text(value) for value in annual_modules_df["survey_label"].dropna().unique().tolist()}
    search_rows: list[Any] = []
    concept_summary_rows: list[dict[str, Any]] = []
    equivalence_rows: list[dict[str, Any]] = []

    for concept, queries in concept_queries.items():
        logger.info("Buscando concepto '%s' con consultas: %s", concept, queries)
        concept_hits: list[Any] = []
        used_query = None
        matched_variable_names: list[str] = []

        for query in queries:
            results = inei_microdatos.search_variables(query, survey="ENAHO", year="2023")
            result_df = pandas.DataFrame(results)
            if result_df.empty:
                concept_summary_rows.append(
                    {
                        "concept": concept,
                        "query": query,
                        "result_count_raw": 0,
                        "result_count_filtered": 0,
                        "status": "sin_resultados",
                    }
                )
                continue

            result_df = normalize_columns(result_df)
            if "module_code" in result_df.columns:
                result_df = result_df[result_df["module_code"].astype(str).isin(allowed_codes)]
            if "survey" in result_df.columns:
                result_df = result_df[result_df["survey"].map(normalize_text).isin(allowed_survey_labels)]
            if "period" in result_df.columns:
                result_df = result_df[result_df["period"].map(normalize_text).str.contains("anual", na=False)]

            result_df["concept"] = concept
            result_df["query"] = query
            result_df["matched_annual_enaho_2023"] = not result_df.empty

            concept_summary_rows.append(
                {
                    "concept": concept,
                    "query": query,
                    "result_count_raw": len(results),
                    "result_count_filtered": int(result_df.shape[0]),
                    "status": "ok" if not result_df.empty else "fuera_universo_o_sin_hit",
                }
            )

            if not result_df.empty:
                concept_hits.append(serialize_nested_objects(result_df))
                used_query = query
                matched_variable_names.extend(
                    result_df["variable"].dropna().astype(str).drop_duplicates().tolist()
                )

        if concept_hits:
            concept_df = pandas.concat(concept_hits, ignore_index=True).drop_duplicates()
            search_rows.append(concept_df)

            top_variables = list(dict.fromkeys(matched_variable_names))[:2]
            for variable_name in top_variables:
                across = inei_microdatos.search_across_years(variable_name, survey="ENAHO")
                for year, year_hits in across.items():
                    for hit in year_hits:
                        equivalence_rows.append(
                            {
                                "concept": concept,
                                "query_used": used_query,
                                "candidate_variable": variable_name,
                                "year": year,
                                "survey": hit.get("survey"),
                                "period": hit.get("period"),
                                "module_code": hit.get("module_code"),
                                "module_name": hit.get("module_name"),
                                "label": hit.get("label"),
                            }
                        )
        else:
            concept_summary_rows.append(
                {
                    "concept": concept,
                    "query": "|".join(queries),
                    "result_count_raw": 0,
                    "result_count_filtered": 0,
                    "status": "sin_equivalencia_directa",
                }
            )

    search_df = (
        pandas.concat(search_rows, ignore_index=True).drop_duplicates()
        if search_rows
        else pandas.DataFrame()
    )
    concept_summary_df = pandas.DataFrame(concept_summary_rows).drop_duplicates()
    equivalence_df = pandas.DataFrame(equivalence_rows).drop_duplicates()
    return search_df, concept_summary_df, equivalence_df


def run_phase0(paths: ProjectPaths, logger: logging.Logger) -> dict[str, Any]:
    logger.info("Iniciando FASE 0 - configuracion del proyecto")
    set_seed(SEED)
    env_info = capture_environment()
    save_json(env_info, paths.reports / "phase0_environment.json")

    errors = validate_phase0(env_info)
    validation_payload = {
        "phase": "FASE 0",
        "passed": not errors,
        "errors": errors,
    }
    save_json(validation_payload, paths.reports / "phase0_validation.json")

    logger.info("Python version: %s", env_info["python_version"])
    logger.info("Versiones detectadas: %s", env_info["libraries"])
    if errors:
        logger.error("Validacion FASE 0 fallida: %s", errors)
    else:
        logger.info("FASE 0 validada correctamente")
    return validation_payload


def run_phase1(paths: ProjectPaths, logger: logging.Logger) -> dict[str, Any]:
    logger.info("Iniciando FASE 1 - catalogo y busqueda de variables")
    pandas = importlib.import_module("pandas")

    catalog, catalog_source, catalog_age = load_catalog_with_fallback(paths, logger)
    catalog_df = flatten_catalog(catalog)
    save_dataframe_csv(catalog_df, paths.reports / "catalog_flat_inventory.csv")

    annual_modules_df, annual_summary = filter_enaho_2023_anual_modules(catalog_df)
    if annual_modules_df.empty:
        raise ValueError("El catalogo filtrado para ENAHO 2023 Anual no panel quedo vacio.")

    summary_df = annual_modules_df[
        [
            "survey_label",
            "category",
            "year",
            "period_label",
            "module_code",
            "module_name",
            "preferred_format",
            "preferred_code",
            "csv_code",
            "stata_code",
            "spss_code",
            "is_downloadable",
        ]
    ].drop_duplicates()
    save_dataframe_csv(summary_df, paths.reports / "phase1_modules_summary.csv")

    concept_queries = {
        "ubigeo": ["ubigeo"],
        "conglome": ["conglome"],
        "vivienda": ["vivienda"],
        "hogar": ["hogar"],
        "factor_expansion": ["factor de expansion", "factor07", "facpob07"],
        "pobreza_monetaria": ["pobreza monetaria", "pobreza"],
        "jefe_hogar": ["jefe de hogar", "parentesco jefe", "p203"],
        "educacion_jefe": ["nivel educativo", "p301a"],
        "matricula_asistencia": ["matricula", "asistencia escolar"],
        "salud_acceso": ["salud", "atencion"],
        "material_piso": ["piso", "material piso"],
        "agua": ["agua"],
        "desague": ["desague", "servicio higienico"],
        "electricidad": ["electricidad", "alumbrado"],
        "combustible_cocina": ["combustible cocinar", "combustible"],
        "tamano_hogar": ["tamano del hogar", "mieperho"],
        "num_perceptores": ["perceptores", "ingreso"],
        "ingreso_hogar": ["ingreso del hogar", "inghog1d"],
        "edad_jefe": ["edad del jefe", "p208a"],
        "sexo_jefe": ["sexo del jefe", "p207"],
        "area_urbana_rural": ["urbana rural", "estrato", "dominio"],
        "ponderadores": ["ponderador", "factor07"],
    }
    save_json(concept_queries, paths.reports / "phase1_concept_queries.json")

    search_df, concept_summary_df, equivalence_df = search_concepts(
        annual_modules_df=summary_df,
        logger=logger,
        concept_queries=concept_queries,
    )
    save_dataframe_csv(search_df, paths.reports / "phase1_variable_search_audit.csv")
    save_dataframe_csv(concept_summary_df, paths.reports / "phase1_variable_search_summary.csv")
    save_dataframe_csv(equivalence_df, paths.reports / "phase1_variable_equivalences.csv")

    missing_concepts = (
        concept_summary_df.groupby("concept")["result_count_filtered"].sum().reset_index()
        .query("result_count_filtered == 0")["concept"]
        .tolist()
        if not concept_summary_df.empty
        else list(concept_queries)
    )

    validation_payload = {
        "phase": "FASE 1",
        "passed": not annual_modules_df.empty,
        "catalog_source": catalog_source,
        "catalog_age": catalog_age,
        "catalog_rows_flat": int(catalog_df.shape[0]),
        "enaho_2023_anual_module_rows": int(annual_modules_df.shape[0]),
        "module_count": int(summary_df.shape[0]),
        "available_survey_labels": annual_summary["survey_labels"],
        "available_period_labels": annual_summary["period_labels"],
        "missing_concepts": missing_concepts,
    }
    save_json(validation_payload, paths.reports / "phase1_validation.json")

    logger.info("Catalogo fuente: %s", catalog_source)
    logger.info("Catalogo total filas planas: %s", catalog_df.shape[0])
    logger.info("ENAHO 2023 Anual no panel modulos: %s", annual_modules_df.shape[0])
    logger.info("Resumen de modulos exportado con %s filas", summary_df.shape[0])
    if missing_concepts:
        logger.warning("Conceptos sin hallazgo filtrado directo: %s", missing_concepts)
    logger.info("FASE 1 completada")
    return validation_payload


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    paths = build_paths(root)
    ensure_directories(paths)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = configure_logging(paths.logs / f"etl_phase01_{run_timestamp}.log")

    phase0 = run_phase0(paths, logger)
    if not phase0["passed"]:
        logger.error("No se puede continuar con FASE 1 hasta resolver FASE 0.")
        return 1

    try:
        run_phase1(paths, logger)
    except Exception as exc:
        logger.exception("FASE 1 fallo: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

