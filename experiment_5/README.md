# Symbolic Early Stopping for Neural Sequence Models

This repository contains the publication-oriented research implementation of **Symbolic Early Stopping (SES)** for neural sequence models corresponding to the article *Symbolic Early Stopping in Neural Sequence Models via Mapper-Induced Symbolic Dynamics*. SES monitors validation hidden-state dynamics during training, builds a Mapper-induced symbolic representation, computes symbolic complexity metrics, smooths them over epochs, filters inactive metrics, aggregates their ranks, and triggers early stopping when symbolic improvement stalls under a validation-loss guard.

The canonical models in this repository are:

- `rnn`
- `birnn`
- `transformer`

The canonical datasets and cases are:

- quasi-periodic: `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `AirPassengers`
- intermediate / regime-switching: `BTC_15m`, EEG/DEAP-derived tabular input `train_with_header`
- near-chaotic: synthetic `lorenz`

## What In The Repository Matches The Article

The publication path is intentionally narrow:

- Core implementation: `src/ses/core.py`
- Aggregation and figure/table utilities: `src/ses/reporting.py`
- Canonical CLIs: `scripts/run_experiment.py`, `scripts/run_baselines.py`, `scripts/aggregate_results.py`, `scripts/build_figures_tables.py`
- Reproducible configs: `configs/smoke_airpassengers.json`, `configs/smoke_lorenz.json`, `configs/article_main.json`, `configs/article_baselines.json`

Historical or exploratory code is preserved under `legacy/` and via deprecated root-level wrappers, but those files are not the recommended path for reproduction.

The released canonical path matches the **main SES pipeline described in the article**:

- validation-obtained hidden states are monitored after each epoch
- the monitored representation in the released pipeline is the final hidden representation used in the main SES experiments
- recurrent models use the last hidden state
- bidirectional recurrent models use the concatenated final forward/backward hidden states
- Transformer models use the final encoder representation followed by mean pooling
- SES is compared against Patience, Slope, SVCCA, and CDSC on the same logged runs

## Repository Structure

```text
experiment_5/
├── AGENTS.md                    Short repo rules for code agents
├── configs/                     Reproducible JSON configurations
├── data/                        Local data directory plus data instructions
├── docs/                        Publication, reproducibility, and release notes
├── legacy/                      Archived monolithic SES scripts
├── results/                     Default output location for generated results
├── scripts/                     Canonical CLI entry points
├── src/ses/                     Core SES and reporting implementation
├── CITATION.cff                 Citation metadata for GitHub
├── .zenodo.json                 Citation/archive metadata for Zenodo
├── LICENSE                      Repository license
├── pyproject.toml               Package metadata
└── requirements.txt             Python dependencies
```

## Method Summary

The canonical SES implementation operates on validation representations collected at each epoch and includes the following components:

- hidden-state extraction from the validation split
- Mapper-based symbolization of validation embeddings
- symbolic metrics: Lempel-Ziv complexity (`LZ`), Markov entropy rate (`hM`), permutation entropy (`PermEn`), correlation dimension (`D2`), and optional box-counting dimension (`DF`)
- exponential moving average smoothing
- aliveness filtering
- rank-based aggregation of symbolic signals
- early stopping by symbolic no-improve with plateau/slope confirmation and a validation-loss guard

The same canonical run also reports the baseline stopping criteria used for comparison:

- patience-based stopping (`PAT`)
- validation-loss slope stopping (`SLOPE`)
- SVCCA-based stopping (`SVCCA`)
- CDSC-based stopping (`CDSC`)

These names and roles match the terminology used in the article.

## Installation

Python `3.10+` is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Data Availability And Preparation

See `data/README.md` for dataset-by-dataset instructions and article-aligned data notes.

High-level distinction:

- External public datasets: ETT, AirPassengers, Bitcoin, EEG/DEAP-derived inputs
- Synthetic/generated data: Lorenz trajectories via `scripts/generate_lorenz_data.py`

The canonical repository layout expects data under `data/`. Dataset names are resolved recursively, so `ETTh1` maps to `data/ETT/ETTh1.csv`, `train_with_header` maps to `data/EEG/train_with_header.csv`, and `lorenz` maps to `data/lorenz.csv`.

If you regenerate Lorenz data:

```bash
python scripts/generate_lorenz_data.py --out_csv data/lorenz.csv
```

## Canonical Entry Points

Run the full SES pipeline with baselines:

```bash
python scripts/run_experiment.py --config configs/article_main.json
```

Run the baseline-comparison entry point:

```bash
python scripts/run_baselines.py --config configs/article_baselines.json
```

Aggregate one or more experiment directories:

```bash
python scripts/aggregate_results.py \
  --inputs results/article_main results/article_baselines \
  --out_root results/aggregate_article
```

Build compact publication tables and figures:

```bash
python scripts/build_figures_tables.py \
  --input_csv results/aggregate_article/aggregated_per_run.csv \
  --out_root results/publication_assets
```

## Smoke Run

Minimal smoke configuration:

```bash
python scripts/run_experiment.py --config configs/smoke_airpassengers.json
```

Equivalent explicit command:

```bash
python scripts/run_experiment.py \
  --data_root data \
  --datasets AirPassengers \
  --out_root results/smoke_airpassengers \
  --models rnn \
  --n_runs 1 \
  --epochs 4 \
  --seq_len 12 \
  --batch_size 16 \
  --hidden_dim 16 \
  --device cpu \
  --log_every 1
```

## Main Experiment

The article-scale configuration is stored in `configs/article_main.json`. The equivalent CLI form is:

```bash
python scripts/run_experiment.py \
  --data_root data \
  --datasets ETTh1 ETTh2 ETTm1 ETTm2 AirPassengers BTC_15m train_with_header lorenz \
  --out_root results/article_main \
  --models rnn birnn transformer \
  --n_runs 10 \
  --epochs 100 \
  --seq_len 96 \
  --pred_horizon 1 \
  --batch_size 256 \
  --hidden_dim 64 \
  --noise_min 0.0 \
  --noise_max 0.5 \
  --noise_step 0.1 \
  --ses_agg median \
  --device auto
```

This command corresponds to the shared article protocol in which SES and the baseline stopping rules are evaluated on the same runs and compared by stop epoch, validation loss at stop, oracle regret `ΔBest`, and saved epochs.

## Baseline Experiment Path

`scripts/run_baselines.py` uses the same canonical training loop and writes the same output schema as `scripts/run_experiment.py`. This is intentional and matches the article protocol: SES and the baseline stopping rules are evaluated on identical runs so that stopping decisions remain directly comparable within each dataset/model/seed/noise setting.

## Aggregation, Tables, And Figures

Each experiment directory contains:

- `per_run.csv`
- `summary_by_noise.csv`
- `compact_table.csv`
- `run_config.json`
- `epoch_metrics/*.csv` if `save_epoch_metrics` is enabled

For multi-run or multi-directory aggregation:

```bash
python scripts/aggregate_results.py \
  --inputs results/article_main/per_run.csv results/article_baselines/per_run.csv \
  --out_root results/aggregate_article
```

For figures and tables:

```bash
python scripts/build_figures_tables.py \
  --input_csv results/aggregate_article/aggregated_per_run.csv \
  --out_root results/publication_assets
```

The current figure/table builder produces compact CSV, LaTeX, and PNG artifacts from the aggregated `per_run.csv` schema. If manuscript-specific panels require additional styling, those edits should build on the same aggregated outputs rather than on notebooks or ad hoc scripts.

## Reproducibility

- The canonical publication path is documented in `docs/publication_path.md`.
- A section-by-section article mapping is documented in `docs/article_repo_mapping.md`.
- Runtime parameters are saved automatically in `run_config.json`.
- The pipeline uses deterministic seeds via `base_seed + run_id`.
- Train/validation/test splits are deterministic for a given input CSV and configuration.
- CLI entry points provide `--help` and accept JSON configs for versioned experiment control.
- Historical code is preserved for traceability but is excluded from the canonical publication path.
- The canonical release corresponds to the main final-layer SES pipeline described in the manuscript. Article-side analyses such as layer-wise monitoring, Mapper hyperparameter transfer, and runtime profiling should be treated as extensions built around the same core outputs rather than as separate primary entry points.

## Data Availability

Public and synthetic data notes are documented in `data/README.md`. Ready-to-paste academic wording for the manuscript is provided in `docs/reproducibility_statements.md`.

## Code Availability

This repository contains the code required to reproduce the canonical SES experiments, baseline comparisons, result aggregation, and compact publication assets. For citation metadata, use `CITATION.cff` on GitHub and `.zenodo.json` for Zenodo archiving. The GitHub-to-Zenodo release flow is documented in `docs/zenodo_release_checklist.md`.
