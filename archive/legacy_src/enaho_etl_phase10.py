from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .enaho_etl_phase01 import build_paths, configure_logging, ensure_directories


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    paths = build_paths(root)
    ensure_directories(paths)
    logger = configure_logging(paths.logs / f"etl_phase10_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    phase1 = load_json(paths.reports / "phase1_validation.json")
    phase2 = load_json(paths.reports / "phase2_validation.json")
    phase3 = load_json(paths.reports / "phase3_validation.json")
    phase4 = load_json(paths.reports / "phase4_validation.json")
    phase5 = load_json(paths.reports / "phase5_validation.json")
    phase7 = load_json(paths.reports / "phase7_target_validation.json")
    phase89 = load_json(paths.reports / "phase8_9_validation.json")

    features = pd.read_csv(paths.reports / "features_finales.csv")
    discarded = pd.read_csv(paths.reports / "columnas_descartadas.csv")
    leakage = pd.read_csv(paths.reports / "riesgo_fuga_informacion.csv")

    summary = "\n".join(
        [
            "# Resumen ejecutivo técnico",
            "",
            "## Módulos usados",
            "- 906-Modulo01 Características de la Vivienda y del Hogar",
            "- 906-Modulo02 Características de los Miembros del Hogar",
            "- 906-Modulo03 Educación",
            "- 906-Modulo04 Salud",
            "- 906-Modulo05 Empleo e Ingresos",
            "- 906-Modulo34 Sumarias (Variables Calculadas)",
            "",
            "## Filtros aplicados",
            "- Encuesta: ENAHO 2023 Anual no panel",
            "- Ámbito geográfico: departamento Arequipa usando `ubigeo[:2] == \"04\"`",
            f"- Hogares finales en Arequipa: {phase5['hogares_finales']}",
            "",
            "## Unidad de análisis final",
            "- Unidad final: hogar",
            "- Tablas a nivel hogar: Modulo01 y Sumaria",
            "- Tablas a nivel persona agregadas a hogar: Modulo02, Modulo03, Modulo04, Modulo05",
            "",
            "## Target elegido",
            f"- Variable oficial: `{phase7['target_source_variable']}` en Sumaria",
            "- Definición binaria para modelado: pobre extremo + pobre no extremo = 1; no pobre = 0",
            f"- Frecuencia binaria: {phase7['target_binary_frequencies']}",
            "",
            "## Features construidas",
            f"- Total features finales candidatas: {phase89['features_finales']}",
            f"- Total columnas del dataset model-ready: {phase89['model_ready_cols']}",
            "- Bloques incluidos: demografía del jefe, composición del hogar, educación, salud, vivienda, servicios, geografía y algunas variables económicas de soporte no descartadas",
            "",
            "## Problemas encontrados",
            "- El acceso al portal INEI estuvo bloqueado en algunos intentos iniciales; luego FASE 2 sí logró catálogo y descargas en vivo.",
            "- En Modulo02 `p203` llegó etiquetada como texto (`jefe/jefa`, etc.), no siempre como código numérico; se corrigió la detección del jefe.",
            "- Se detectaron 196 diferencias entre `mieperho` (Sumaria) y el conteo simple de personas desde Modulo02 (`tam_hogar_mod2`).",
            "- Varias columnas monetarias de Sumaria fueron marcadas como riesgo de fuga y descartadas del model-ready.",
            "",
            "## Decisiones tomadas",
            "- Se trabajó sobre raw STATA para preservar estructura original del INEI.",
            "- Se exportó una vista analítica y otra model-ready en Parquet y CSV.",
            "- No se calculó MPI; solo se construyó un dataset para clasificación supervisada de pobreza.",
            "",
            "## Pendientes para modelado",
            "- Definir estrategia de imputación para variables con nulos residuales.",
            "- Revisar encoding de variables categóricas.",
            "- Evaluar tratamiento de ponderadores en entrenamiento y evaluación.",
            "- Confirmar si se desea excluir más variables económicas por criterio estricto de leakage.",
            "",
            "## Próximo paso: modelado supervisado",
        ]
    )

    output_path = paths.reports / "reporte_final_validacion.md"
    output_path.write_text(summary, encoding="utf-8")
    logger.info("Reporte final escrito en %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
