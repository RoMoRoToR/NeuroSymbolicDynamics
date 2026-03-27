# AGENTS.md

This repository is a publication-oriented research codebase for Symbolic Early Stopping (SES).

## Repository Rules

- Preserve the canonical publication path: `src/ses/` plus CLI entry points in `scripts/`.
- Treat root-level historical scripts and files under `legacy/` as archival unless a task explicitly targets them.
- Do not change the scientific SES logic without a concrete reproducibility or correctness reason.
- Prefer adding reproducible CLI scripts and JSON configs over notebooks or ad hoc shell snippets.
- Keep generated outputs out of version control unless they are small reference artifacts already intentionally tracked.
- When modifying documentation, make sure commands match the actual repository layout and existing CLI arguments.
- When modifying data instructions, distinguish clearly between external public datasets and synthetic/generated data.
- Never invent results, DOIs, data sources, or manuscript claims.

## Canonical Entry Points

- `scripts/run_experiment.py`: canonical SES plus baseline comparison pipeline.
- `scripts/run_baselines.py`: baseline-focused entry point using the same training/evaluation core.
- `scripts/aggregate_results.py`: merge multiple `per_run.csv` files.
- `scripts/build_figures_tables.py`: build compact publication tables and figures.
- `scripts/generate_lorenz_data.py`: generate the synthetic Lorenz dataset.
