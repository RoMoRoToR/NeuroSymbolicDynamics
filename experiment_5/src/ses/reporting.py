from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

MPL_CACHE = Path(".matplotlib_cache").resolve()
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .core import ensure_dir, make_summary_tables


def _resolve_result_files(inputs: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            files.extend(sorted(path.rglob("per_run.csv")))
        elif path.name.endswith(".csv"):
            files.append(path)
    uniq = []
    seen = set()
    for path in files:
        rp = str(path.resolve())
        if rp not in seen:
            seen.add(rp)
            uniq.append(path.resolve())
    if not uniq:
        raise FileNotFoundError("No result CSV files found.")
    return uniq


def aggregate_result_files(inputs: Iterable[str], out_root: str, oracle_eps: float = 0.01) -> dict:
    out_dir = Path(out_root)
    ensure_dir(out_dir)
    frames = []
    source_rows = []
    for path in _resolve_result_files(inputs):
        df = pd.read_csv(path)
        df["source_file"] = str(path)
        frames.append(df)
        source_rows.append({"source_file": str(path), "rows": int(len(df))})
    df_all = pd.concat(frames, ignore_index=True)
    per_run = out_dir / "aggregated_per_run.csv"
    df_all.to_csv(per_run, index=False)
    pd.DataFrame(source_rows).to_csv(out_dir / "sources.csv", index=False)
    summary, compact = make_summary_tables(df_all, oracle_eps=oracle_eps)
    summary_path = out_dir / "aggregated_summary_by_noise.csv"
    compact_path = out_dir / "aggregated_compact_table.csv"
    summary.to_csv(summary_path, index=False)
    compact.to_csv(compact_path, index=False)
    return {"per_run": str(per_run), "summary": str(summary_path), "compact": str(compact_path)}


def build_figures_and_tables(input_csv: str, out_root: str, oracle_eps: float = 0.01) -> dict:
    out_dir = Path(out_root)
    fig_dir = out_dir / "figures"
    table_dir = out_dir / "tables"
    ensure_dir(fig_dir)
    ensure_dir(table_dir)
    df = pd.read_csv(input_csv)
    if "ses_epoch" not in df.columns:
        raise ValueError("Expected a per-run CSV with method columns such as ses_epoch.")
    summary, compact = make_summary_tables(df, oracle_eps=oracle_eps)
    summary.to_csv(table_dir / "summary_by_noise.csv", index=False)
    compact.to_csv(table_dir / "compact_table.csv", index=False)
    summary.to_latex(table_dir / "summary_by_noise.tex", index=False)
    compact.to_latex(table_dir / "compact_table.tex", index=False)

    methods = ["ses", "pat", "slope", "svcca", "cdsc"]
    labels = {"ses": "SES", "pat": "PAT", "slope": "SLOPE", "svcca": "SVCCA", "cdsc": "CDSC"}
    mean_delta = [float(df[f"{m}_delta_best"].mean()) for m in methods]
    mean_saved = [float(df[f"{m}_saved_epochs"].mean()) for m in methods]

    plt.figure(figsize=(7, 4))
    plt.bar([labels[m] for m in methods], mean_delta)
    plt.ylabel("Mean delta-to-best validation loss")
    plt.tight_layout()
    plt.savefig(fig_dir / "mean_delta_best.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar([labels[m] for m in methods], mean_saved)
    plt.ylabel("Mean epochs saved")
    plt.tight_layout()
    plt.savefig(fig_dir / "mean_epochs_saved.png", dpi=200)
    plt.close()

    if "noise_sigma" in df.columns and df["noise_sigma"].nunique() > 1:
        grouped = df.groupby("noise_sigma")
        plt.figure(figsize=(8, 4))
        for method in methods:
            xs = sorted(grouped.groups.keys())
            ys = [float(grouped.get_group(x)[f"{method}_delta_best"].median()) for x in xs]
            plt.plot(xs, ys, marker="o", label=labels[method])
        plt.xlabel("Noise sigma")
        plt.ylabel("Median delta-to-best")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "delta_best_by_noise.png", dpi=200)
        plt.close()

    return {
        "summary_csv": str(table_dir / "summary_by_noise.csv"),
        "compact_csv": str(table_dir / "compact_table.csv"),
        "summary_tex": str(table_dir / "summary_by_noise.tex"),
        "compact_tex": str(table_dir / "compact_table.tex"),
        "figures_dir": str(fig_dir),
    }
