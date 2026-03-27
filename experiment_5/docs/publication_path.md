# Publication Path

This repository contains historical SES development artifacts as well as a canonical, publication-oriented implementation.

## Canonical Path

- Core SES implementation: `src/ses/core.py`
- Result aggregation and figure/table generation: `src/ses/reporting.py`
- Reproducible entry points:
  - `scripts/run_experiment.py`
  - `scripts/run_baselines.py`
  - `scripts/aggregate_results.py`
  - `scripts/build_figures_tables.py`
  - `scripts/generate_lorenz_data.py`
- Reproducible configurations: `configs/*.json`

This canonical path corresponds to the article's main SES implementation in which:

- validation-obtained hidden states are monitored after every epoch
- the monitored representation is the final hidden representation used in the main SES pipeline
- SES is compared with Patience, Slope, SVCCA, and CDSC on the same logged runs
- article-level outputs are summarized by stop epoch, validation loss at stop, oracle regret, and saved epochs

## Archival / Legacy Path

- `legacy/` preserves older monolithic SES scripts that predate the canonical package layout.
- Root-level files named `online_es_timeseries_*` are compatibility wrappers or older entry-point remnants and are not the recommended publication path.

## Minimal Working Pipeline

1. Install dependencies from `requirements.txt`.
2. Ensure required datasets are available under `data/`.
3. Run a smoke configuration from `configs/smoke_airpassengers.json` or `configs/smoke_lorenz.json`.
4. Run the article-scale configuration from `configs/article_main.json`.
5. Aggregate one or more `per_run.csv` files with `scripts/aggregate_results.py`.
6. Build compact figures/tables with `scripts/build_figures_tables.py`.

## Relation To Additional Article Experiments

The manuscript also discusses experiment groups beyond the minimal core pipeline, including layer-wise monitoring analysis, Mapper hyperparameter transfer, and runtime profiling. The current canonical release is centered on the reproducible run-and-report path above. Additional article analyses should be interpreted as extensions built from the same core outputs and method definitions rather than as separate primary entry points for general users.
