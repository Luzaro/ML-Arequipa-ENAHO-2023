from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(data: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_dataframe_csv(df: Any, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


__all__ = ["save_json", "save_dataframe_csv"]
