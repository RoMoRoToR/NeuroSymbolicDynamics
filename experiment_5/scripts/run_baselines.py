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

from ses.core import ExperimentConfig, run_experiment


def parse_args() -> ExperimentConfig:
    ap = argparse.ArgumentParser(
        description=(
            "Run the canonical training pipeline and report baseline early-stopping criteria "
            "(PAT, SLOPE, SVCCA, CDSC) alongside SES for direct comparison."
        )
    )
    ap.add_argument("--config", type=str, default=None, help="Optional JSON config file. CLI flags override config values.")
    ap.add_argument("--data_root", type=str, default=None)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--out_root", type=str, default=None)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--n_runs", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seq_len", type=int, default=None)
    ap.add_argument("--pred_horizon", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--hidden_dim", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--train_frac", type=float, default=None)
    ap.add_argument("--val_frac", type=float, default=None)
    ap.add_argument("--test_frac", type=float, default=None)
    ap.add_argument("--split_gap", type=int, default=None)
    ap.add_argument("--noise_sigma", type=float, default=None)
    ap.add_argument("--noise_min", type=float, default=None)
    ap.add_argument("--noise_max", type=float, default=None)
    ap.add_argument("--noise_step", type=float, default=None)
    ap.add_argument("--noise_mode", type=str, default=None, choices=["all", "target"])
    ap.add_argument("--repr_batches", type=int, default=None)
    ap.add_argument("--pe_m", type=int, default=None)
    ap.add_argument("--pe_tau", type=int, default=None)
    ap.add_argument("--d2_emb", type=int, default=None)
    ap.add_argument("--d2_rbins", type=int, default=None)
    ap.add_argument("--d2_cap", type=int, default=None)
    ap.add_argument("--df_emb", type=int, default=None)
    ap.add_argument("--df_scales", type=int, default=None)
    ap.add_argument("--ses_include_df", action="store_true")
    ap.add_argument("--ses_agg", type=str, default=None, choices=["median", "topq", "min"])
    ap.add_argument("--ses_rank_top", type=float, default=None)
    ap.add_argument("--ses_ema_alpha", type=float, default=None)
    ap.add_argument("--ses_liveness_win", type=int, default=None)
    ap.add_argument("--ses_liveness_abs", type=float, default=None)
    ap.add_argument("--ses_liveness_rel", type=float, default=None)
    ap.add_argument("--ses_patience", type=int, default=None)
    ap.add_argument("--ses_min_delta_sym", type=float, default=None)
    ap.add_argument("--ses_min_epoch", type=int, default=None)
    ap.add_argument("--ses_slope_win", type=int, default=None)
    ap.add_argument("--ses_slope_eps", type=float, default=None)
    ap.add_argument("--val_guard_win", type=int, default=None)
    ap.add_argument("--val_guard_abs", type=float, default=None)
    ap.add_argument("--val_guard_rel", type=float, default=None)
    ap.add_argument("--mapper_d_embed", type=int, default=None)
    ap.add_argument("--mapper_bins", type=int, default=None)
    ap.add_argument("--mapper_overlap", type=float, default=None)
    ap.add_argument("--mapper_local_k", type=int, default=None)
    ap.add_argument("--mapper_merge_eps", type=float, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--pat_min_epoch", type=int, default=None)
    ap.add_argument("--pat_min_delta", type=float, default=None)
    ap.add_argument("--slope_win", type=int, default=None)
    ap.add_argument("--slope_eps", type=float, default=None)
    ap.add_argument("--slope_patience", type=int, default=None)
    ap.add_argument("--slope_min_epoch", type=int, default=None)
    ap.add_argument("--svcca_dim", type=int, default=None)
    ap.add_argument("--svcca_sim_thr", type=float, default=None)
    ap.add_argument("--svcca_patience", type=int, default=None)
    ap.add_argument("--svcca_min_epoch", type=int, default=None)
    ap.add_argument("--cdsc_win", type=int, default=None)
    ap.add_argument("--cdsc_ratio", type=float, default=None)
    ap.add_argument("--cdsc_min_epoch", type=int, default=None)
    ap.add_argument("--oracle_eps", type=float, default=None)
    ap.add_argument("--log_every", type=int, default=None)
    ap.add_argument("--base_seed", type=int, default=None)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--save_epoch_metrics", action="store_true")
    ns = ap.parse_args()
    cfg = ExperimentConfig()
    if ns.config:
        with open(ns.config, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        cfg = ExperimentConfig(**payload)
    for key, value in vars(ns).items():
        if key == "config":
            continue
        if key == "deterministic":
            if value:
                cfg.deterministic = True
            continue
        if key == "ses_include_df":
            if value:
                cfg.ses_include_df = True
            continue
        if key == "save_epoch_metrics":
            if value:
                cfg.save_epoch_metrics = True
            continue
        if value is not None:
            setattr(cfg, key, value)
    return cfg


def main() -> None:
    cfg = parse_args()
    outputs = run_experiment(cfg)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
