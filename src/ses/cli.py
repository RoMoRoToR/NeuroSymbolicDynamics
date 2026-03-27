from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import ExperimentConfig


def build_experiment_parser(description: str, default_config: str | None = None) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    config_help = "Optional JSON config file. CLI flags override config values."
    if default_config:
        config_help += f" Defaults to {default_config}."
    ap.add_argument("--config", type=str, default=default_config, help=config_help)
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
    return ap


def config_from_namespace(ns: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig()
    if ns.config:
        config_path = Path(ns.config)
        with config_path.open("r", encoding="utf-8") as handle:
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


def parse_experiment_config(description: str, default_config: str | None = None) -> ExperimentConfig:
    parser = build_experiment_parser(description=description, default_config=default_config)
    return config_from_namespace(parser.parse_args())
