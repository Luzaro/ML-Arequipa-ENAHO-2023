from __future__ import annotations

from .enaho_etl_stage_runner import run_stage


def main() -> int:
    return run_stage("extract")


if __name__ == "__main__":
    raise SystemExit(main())
