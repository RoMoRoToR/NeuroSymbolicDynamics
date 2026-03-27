# Publication Path

This repository contains both the canonical publication workflow and a substantial amount of historical research code. Only the path below should be cited as the reproducible publication workflow.

## Canonical Directories

- `src/ses/`
  - `core.py`: training loop, hidden-state extraction, Mapper symbolization, SES score construction, and baseline stopping rules
  - `reporting.py`: result aggregation plus compact table/figure generation
  - `cli.py`: shared CLI/config parsing
- `scripts/`
  - `run_experiment.py`: main SES experiment entry point
  - `run_baselines.py`: baseline-comparison entry point using the same canonical loop
  - `aggregate_results.py`: multi-directory result aggregation
  - `build_figures_tables.py`: compact CSV/LaTeX/PNG asset generation
  - `generate_lorenz_data.py`: synthetic data generation
- `configs/`
  - `article_main.json`: bundled publication-core run
  - `article_baselines.json`: bundled baseline-comparison run
  - `article_full_external.json`: full manuscript template requiring additional external datasets
  - `smoke_lorenz.json`: lightweight sanity run
- `data/README.md`
- `docs/reproducibility_statements.md`
- `CITATION.cff`, `.zenodo.json`, `LICENSE`

## Historical Or Exploratory Material

The following directories remain for traceability but are not part of the canonical reproduction path:

- `experiment/`, `experiment_1/`, `experiment_2/`, `experiment_3/`, `experiment_4/`, `experiment_6/`, `experiment_SES/`
- `metric/`
- `online_check/`
- `train models/`
- `Karman/`
- `full_early_stopping_metod/`
- notebooks and manuscript-drafting helpers outside `src/` and `scripts/`

These directories may contain:

- earlier SES variants
- ablations and diagnostics
- layer-wise or profiling experiments
- plotting notebooks
- intermediate or legacy artifact formats

They should not be used as the primary entry point for publication reproduction unless a manuscript section explicitly states that a historical analysis is being revisited.
