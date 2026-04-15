from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .logging_utils import configure_logging
from .paths import build_paths, ensure_directories


def bootstrap_runtime(script_name: str) -> tuple[Any, Any]:
    root = Path(__file__).resolve().parents[2]
    paths = build_paths(root)
    ensure_directories(paths)
    logger = configure_logging(paths.logs / f"{script_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    return paths, logger


__all__ = ["bootstrap_runtime"]
