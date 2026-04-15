from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .phase01 import build_paths, configure_logging, ensure_directories


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_report_path(paths, filename: str) -> Path:
    candidates = [
        paths.reports / filename,
        paths.root / "archive" / "legacy_reports" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No se encontro el archivo requerido: {filename}")


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    paths = build_paths(root)
    ensure_directories(paths)
    logger = configure_logging(paths.logs / f"etl_phase10_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    phase1 = load_json(resolve_report_path(paths, "phase1_validation.json"))
    phase2 = load_json(resolve_report_path(paths, "phase2_validation.json"))
    phase3 = load_json(resolve_report_path(paths, "phase3_validation.json"))
    phase4 = load_json(resolve_report_path(paths, "phase4_validation.json"))
    phase5 = load_json(resolve_report_path(paths, "phase5_validation.json"))
    phase7 = load_json(resolve_report_path(paths, "phase7_target_validation.json"))
    phase89 = load_json(resolve_report_path(paths, "phase8_9_validation.json"))

    features = pd.read_csv(resolve_report_path(paths, "features_finales.csv"))
    discarded = pd.read_csv(resolve_report_path(paths, "columnas_descartadas.csv"))
    leakage = pd.read_csv(resolve_report_path(paths, "riesgo_fuga_informacion.csv"))

    summary = "\n".join(
        [
            "# Resumen ejecutivo tecnico",
            "",
            "## Modulos usados",
            "- 906-Modulo01 Caracteristicas de la Vivienda y del Hogar",
            "- 906-Modulo02 Caracteristicas de los Miembros del Hogar",
            "- 906-Modulo03 Educacion",
            "- 906-Modulo04 Salud",
            "- 906-Modulo05 Empleo e Ingresos",
            "- 906-Modulo34 Sumarias (Variables Calculadas)",
            "",
            "## Filtros aplicados",
            "- Encuesta: ENAHO 2023 Anual no panel",
            "- Ambito geografico: departamento Arequipa usando `ubigeo[:2] == \"04\"`",
            f"- Catalogo empleado en extraccion: {phase1.get('catalog_source', 'no disponible')}",
            f"- Modulos validados en descarga: {phase2.get('ok_files', 'no disponible')}",
            f"- Hogares finales en Arequipa: {phase5['hogares_finales']}",
            "",
            "## Unidad de analisis final",
            "- Unidad final: hogar",
            "- Tablas a nivel hogar: Modulo01 y Sumaria",
            "- Tablas a nivel persona agregadas a hogar: Modulo02, Modulo03, Modulo04, Modulo05",
            "",
            "## Target elegido",
            f"- Variable oficial: `{phase7['target_source_variable']}` en Sumaria",
            "- Definicion binaria para modelado: pobre extremo + pobre no extremo = 1; no pobre = 0",
            f"- Frecuencia binaria: {phase7['target_binary_frequencies']}",
            "",
            "## Features construidas",
            f"- Total features finales candidatas: {phase89['features_finales']}",
            f"- Total columnas del dataset model-ready: {phase89['model_ready_cols']}",
            f"- Features registradas en inventario: {features.shape[0]}",
            f"- Variables descartadas por reglas tecnicas: {discarded.shape[0]}",
            f"- Variables con riesgo de fuga registradas: {leakage.shape[0]}",
            "- Bloques incluidos: demografia del jefe, composicion del hogar, educacion, salud, vivienda, servicios, geografia y variables economicas de soporte aun permitidas.",
            "",
            "## Problemas encontrados",
            "- El acceso al portal INEI tuvo intentos inestables al inicio del proceso.",
            "- En Modulo02 `p203` llego etiquetada como texto en parte de los registros y hubo que normalizar la deteccion del jefe.",
            "- Se detectaron diferencias entre `mieperho` y el conteo simple de personas desde Modulo02 (`tam_hogar_mod2`).",
            "- Varias columnas monetarias de Sumaria fueron marcadas como riesgo de fuga y quedaron fuera del model-ready.",
            "",
            "## Decisiones tomadas",
            "- Se trabajo sobre raw STATA para preservar la estructura original del INEI.",
            "- Se exporto una vista analitica y otra model-ready en Parquet y CSV.",
            "- No se calculo MPI; el objetivo fue construir un dataset para clasificacion supervisada de pobreza.",
            "",
            "## Pendientes para modelado",
            "- Definir estrategia de imputacion para variables con nulos residuales.",
            "- Revisar encoding para variables categoricas.",
            "- Evaluar tratamiento de ponderadores en entrenamiento y evaluacion.",
            "- Confirmar si se desea excluir mas variables economicas por criterio estricto de leakage.",
            "",
            "## Proximo paso: modelado supervisado",
        ]
    )

    output_path = paths.reports / "reporte_final_validacion.md"
    output_path.write_text(summary, encoding="utf-8")
    logger.info("Reporte final escrito en %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
