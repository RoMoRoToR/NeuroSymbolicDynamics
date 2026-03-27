#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ses.reporting import aggregate_result_files


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate per-run SES result CSV files.")
    ap.add_argument("--inputs", nargs="+", required=True, help="Directories or per_run.csv files.")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--oracle_eps", type=float, default=0.01)
    args = ap.parse_args()
    outputs = aggregate_result_files(args.inputs, args.out_root, oracle_eps=args.oracle_eps)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
