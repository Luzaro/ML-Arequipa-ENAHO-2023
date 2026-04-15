from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from etl.phase03 import main as phase03_main  # noqa: E402
from etl.phase04 import main as phase04_main  # noqa: E402
from etl.phase05 import main as phase05_main  # noqa: E402
from etl.phase06_07 import main as phase06_07_main  # noqa: E402
from etl.phase08_09 import main as phase08_09_main  # noqa: E402


def run_steps(steps: list[Callable[[], int]]) -> int:
    for step in steps:
        status = int(step())
        if status != 0:
            return status
    return 0


def main() -> int:
    return run_steps(
        [
            phase03_main,
            phase04_main,
            phase05_main,
            phase06_07_main,
            phase08_09_main,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
