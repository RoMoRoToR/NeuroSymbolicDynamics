#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ses.core import ExperimentConfig, run_experiment


def main() -> None:
    print(
        "Deprecated entrypoint: use scripts/run_experiment.py for canonical publication-ready runs. "
        "The original monolithic implementation is preserved in legacy/."
    )
    run_experiment(ExperimentConfig())


if __name__ == "__main__":
    main()
