# Symbolic Early Stopping for Neural Sequence Models

This repository contains the publication-oriented implementation of **Symbolic Early Stopping (SES)** for neural sequence models from the manuscript *Symbolic Early Stopping in Neural Sequence Models via Mapper-Induced Symbolic Dynamics*.

SES monitors **validation hidden-state dynamics** during training, constructs an epoch-specific **Mapper-induced symbolic representation**, computes symbolic descriptors such as **Lempel-Ziv complexity**, **Markov entropy rate**, **permutation entropy**, and **correlation dimension**, smooths and filters these signals, aggregates their ranks, and triggers stopping when symbolic improvement stalls under a conservative validation-loss guard.

The canonical model families are:

- `rnn`
- `birnn`
- `transformer`

The article-level dataset suite comprises:

- quasi-periodic: `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `AirPassengers`
- intermediate / regime-switching: `BTC_15m`, EEG-derived tabular input `train_with_header`
- near-chaotic: synthetic `lorenz`

## Canonical Publication Path

The publication path in this repository is intentionally narrow:

- core SES implementation: `src/ses/core.py`
- aggregation and compact reporting: `src/ses/reporting.py`
- shared CLI parsing: `src/ses/cli.py`
- reproducible entry points: `scripts/run_experiment.py`, `scripts/run_baselines.py`, `scripts/aggregate_results.py`, `scripts/build_figures_tables.py`, `scripts/generate_lorenz_data.py`
- versioned experiment configs: `configs/article_main.json`, `configs/article_baselines.json`, `configs/article_full_external.json`, `configs/smoke_lorenz.json`

Historical development code remains in the repository for traceability, but it is **not** part of the recommended publication workflow. See `docs/publication_path.md`.

## What Matches The Article

The released canonical pipeline follows the main SES path described in the manuscript:

- hidden representations are collected from the **validation split** after each epoch
- recurrent models use the last hidden state
- bidirectional recurrent models use the concatenated final forward/backward hidden states
- Transformer models use the final encoder representation followed by mean pooling
- Mapper is fit on the epoch-specific validation embedding cloud
- symbol sequences are evaluated with `LZ`, `hM`, `PermEn`, and `D2`
- SES uses EMA smoothing, liveness filtering, rank aggregation, and a hybrid symbolic stop with a validation-loss guard
- the same run also reports baseline stopping rules: `PAT`, `SLOPE`, `SVCCA`, `CDSC`

## Repository Structure

```text
.
├── AGENTS.md
├── CITATION.cff
├── LICENSE
├── configs/
├── data/
├── docs/
├── figures/
├── legacy/
├── results/
├── scripts/
├── src/ses/
├── tables/
├── environment.yml
├── pyproject.toml
└── requirements.txt
```

Only `src/`, `scripts/`, `configs/`, `data/README.md`, `docs/`, and the citation/license metadata should be treated as the canonical publication surface.

## Installation

Python `3.10` is the recommended baseline.

Using `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

Using Conda:

```bash
conda env create -f environment.yml
conda activate ses-research
python3 -m pip install -e .
```

## Data

Dataset-specific instructions are in `data/README.md`.

Current repository snapshot:

- included and ready in canonical layout: `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`
- included/generated in canonical layout: `lorenz`
- expected but **not redistributed in this snapshot**: `AirPassengers`, `BTC_15m`, `train_with_header`

The experiment code resolves dataset names recursively under `data/`. For example:

- `ETTh1` -> `data/ETT/ETTh1.csv`
- `lorenz` -> `data/lorenz.csv`
- `train_with_header` -> `data/EEG/train_with_header.csv`

If you want to regenerate the synthetic Lorenz dataset:

```bash
python3 scripts/generate_lorenz_data.py --out_csv data/lorenz.csv
```

## Entry Points

Run the canonical SES experiment path on the bundled publication-core datasets:

```bash
python3 scripts/run_experiment.py --config configs/article_main.json
```

Run the canonical baseline-comparison path on the same bundled publication-core datasets:

```bash
python3 scripts/run_baselines.py --config configs/article_baselines.json
```

Run the full article dataset template after preparing the external datasets described in `data/README.md`:

```bash
python3 scripts/run_experiment.py --config configs/article_full_external.json
```

Aggregate experiment directories:

```bash
python3 scripts/aggregate_results.py \
  --inputs results/article_main results/article_baselines \
  --out_root results/aggregate_article
```

Build compact tables and figures from an aggregated per-run CSV:

```bash
python3 scripts/build_figures_tables.py \
  --input_csv results/aggregate_article/aggregated_per_run.csv \
  --out_root results/publication_assets
```

## Minimal Smoke Run

The minimal bundled smoke path uses the included Lorenz sample:

```bash
python3 scripts/run_experiment.py --config configs/smoke_lorenz.json
```

Equivalent explicit command:

```bash
python3 scripts/run_experiment.py \
  --data_root data \
  --datasets data/synthetic/lorenz_smoke.csv \
  --out_root results/smoke_lorenz \
  --models rnn \
  --n_runs 1 \
  --epochs 4 \
  --seq_len 24 \
  --batch_size 32 \
  --hidden_dim 16 \
  --device cpu \
  --log_every 1
```

## Reproducing The Main Experiment

`configs/article_main.json` is the runnable publication-core configuration shipped with this repository snapshot. It covers:

- models: `rnn`, `birnn`, `transformer`
- datasets: `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `lorenz`
- runs per setting: `10`
- epochs per run: `100`
- noise schedule: `0.0:0.1:0.5`

For the full manuscript dataset suite, prepare the missing external datasets and use `configs/article_full_external.json`. That config includes dataset-specific overrides for small-series cases such as `AirPassengers` and `lorenz`, so the run does not fail on incompatible window sizes.

## Reproducing Tables And Figures

The canonical table/figure pipeline is:

```bash
python3 scripts/aggregate_results.py \
  --inputs results/article_main results/article_baselines \
  --out_root results/aggregate_article

python3 scripts/build_figures_tables.py \
  --input_csv results/aggregate_article/aggregated_per_run.csv \
  --out_root results/publication_assets
```

Generated outputs include:

- aggregated run-level CSVs
- compact summary CSVs
- LaTeX tables
- overview PNG figures

Manuscript-specific layout polishing, if needed, should build on these generated assets rather than on notebooks.

## Reproducibility

- runtime parameters are saved in `run_config.json`
- dataset preflight checks fail fast on missing or empty files
- dataset-specific overrides are versioned in JSON configs
- seeds are deterministic via `base_seed + run_id`
- the code path for the article is confined to `src/ses/` and `scripts/`
- historical notebooks and exploratory pipelines are preserved for traceability but excluded from the canonical publication workflow

See also:

- `docs/publication_path.md`
- `docs/article_repo_mapping.md`
- `docs/reproducibility_statements.md`

## Data Availability

Dataset provenance and acquisition instructions are documented in `data/README.md`. Ready-to-paste manuscript wording is provided in `docs/reproducibility_statements.md`.

## Code Availability

This repository provides the code needed for the canonical SES experiment path, baseline comparison, aggregation, and compact publication-asset generation. Citation metadata is provided in `CITATION.cff`; Zenodo metadata is in `.zenodo.json`.
