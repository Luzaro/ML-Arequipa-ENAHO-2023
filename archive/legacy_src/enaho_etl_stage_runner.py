from __future__ import annotations

import importlib
from typing import Iterable


ETL_STAGES: dict[str, list[str]] = {
    "extract": [
        ".enaho_etl_phase01",
        ".enaho_etl_phase02",
    ],
    "transform": [
        ".enaho_etl_phase03",
        ".enaho_etl_phase04",
        ".enaho_etl_phase05",
        ".enaho_etl_phase06_07",
        ".enaho_etl_phase08_09",
    ],
    "load": [
        ".enaho_etl_phase10",
    ],
}


def run_modules(module_names: Iterable[str]) -> int:
    for module_name in module_names:
        module = importlib.import_module(module_name, package=__package__)
        if not hasattr(module, "main"):
            raise AttributeError(f"El modulo {module_name} no expone una funcion main().")

        status = int(module.main())
        if status != 0:
            return status
    return 0


def run_stage(stage_name: str) -> int:
    if stage_name not in ETL_STAGES:
        valid = ", ".join(sorted(ETL_STAGES))
        raise ValueError(f"Etapa invalida: {stage_name}. Opciones validas: {valid}.")
    return run_modules(ETL_STAGES[stage_name])
