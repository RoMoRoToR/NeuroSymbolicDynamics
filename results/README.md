# Results

This directory is the default output root for generated experiment artifacts.

Canonical scripts write:

- `per_run.csv`
- `summary_by_noise.csv`
- `compact_table.csv`
- `run_config.json`
- `epoch_metrics/*.csv` when enabled

Generated contents under `results/` are not part of the source distribution unless a release explicitly includes them.
