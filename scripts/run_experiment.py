#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ses.cli import parse_experiment_config
from ses.core import run_experiment


def main() -> None:
    cfg = parse_experiment_config("Run SES experiments from the canonical publication-ready pipeline.")
    try:
        outputs = run_experiment(cfg)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
