# Figures

This directory is reserved for generated figure assets produced by the canonical reporting pipeline.

The publication-oriented path is:

1. run experiments with `scripts/run_experiment.py` and/or `scripts/run_baselines.py`
2. aggregate outputs with `scripts/aggregate_results.py`
3. build compact figure assets with `scripts/build_figures_tables.py`

Manuscript-specific polishing should derive from those generated assets rather than from ad hoc notebooks.
