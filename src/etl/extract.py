from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from etl.phase01 import main as phase01_main  # noqa: E402
from etl.phase02 import main as phase02_main  # noqa: E402


def run_steps(steps: list[Callable[[], int]]) -> int:
    for step in steps:
        status = int(step())
        if status != 0:
            return status
    return 0


def main() -> int:
    return run_steps([phase01_main, phase02_main])


if __name__ == "__main__":
    raise SystemExit(main())
