# Results Directory

This directory is the default destination for generated experiment outputs.

## Expected Contents

- `per_run.csv`: one row per dataset, model, seed, and noise setting.
- `summary_by_noise.csv`: grouped publication summary table.
- `compact_table.csv`: compact cross-method table.
- `run_config.json`: exact configuration used for the run.
- `epoch_metrics/`: optional per-epoch trajectories for losses and symbolic metrics.

## Version-Control Policy

Generated outputs are ignored by default. Keep only small, intentional reference artifacts under version control when they are needed for the manuscript or for smoke-validation examples.
