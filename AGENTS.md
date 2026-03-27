# AGENTS.md

Short rules for code agents working in this repository:

- Treat `src/ses/`, `scripts/`, `configs/`, `data/README.md`, and `docs/` as the canonical publication path.
- Do not introduce new primary experiment paths outside `src/ses` and `scripts`.
- Keep scientific behavior conservative: do not change SES logic unless the task explicitly requires it.
- Prefer JSON config changes over hard-coded experiment parameters.
- If a dataset is external or missing, document the preparation step instead of inventing data.
- Historical directories such as `experiment*`, `metric/`, `online_check/`, `train models/`, and notebooks are legacy unless the task explicitly targets them.
- When changing CLI behavior, update `README.md`, `data/README.md`, and `docs/reproducibility_statements.md` in the same patch.
