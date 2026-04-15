from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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


__all__ = ["ProjectPaths", "build_paths", "ensure_directories"]
