# Article To Repository Mapping

This file maps the main article claims and workflow components to the canonical code path shipped in this repository.

## Main SES Method

- validation hidden-state monitoring: `src/ses/core.py`
- monitored representations:
  - `RNNRegressor`: final recurrent hidden state
  - `BiRNNRegressor`: concatenated final forward/backward hidden states
  - `TransformerRegressor`: final encoder output with mean pooling
- Mapper-based symbolization: `MapperSymbolizer` in `src/ses/core.py`
- symbolic metrics:
  - `LZ`: `lz_normalized`
  - `hM`: `markov_entropy_rate`
  - `PermEn`: `permutation_entropy`
  - `D2`: `correlation_dimension_d2`
- smoothing, liveness filtering, and rank aggregation:
  - `ema`
  - `build_symbolic_score`
- hybrid stopping rule:
  - `validation_guard_ok`
  - `find_stop_symbolic_hybrid`

## Baseline Criteria

The article baselines are implemented in the same canonical module:

- patience: `patience_es`
- slope: `slope_es`
- SVCCA: `svcca_score`, `svcca_stop`
- CDSC: `cdsc_stop`

This shared implementation is intentional. SES and baselines are computed on the same logged runs so that stop epochs and validation losses are directly comparable within each dataset/model/seed/noise setting.

## Canonical Reproduction Commands

- main bundled reproduction: `python3 scripts/run_experiment.py --config configs/article_main.json`
- baseline comparison: `python3 scripts/run_baselines.py --config configs/article_baselines.json`
- full manuscript template after adding external data: `python3 scripts/run_experiment.py --config configs/article_full_external.json`
- aggregation: `python3 scripts/aggregate_results.py ...`
- compact figure/table build: `python3 scripts/build_figures_tables.py ...`

## What Corresponds To The Article In This Snapshot

- bundled and directly runnable:
  - `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `lorenz`
- referenced in the manuscript but requiring user-side preparation in this snapshot:
  - `AirPassengers`
  - `BTC_15m`
  - `train_with_header`

## Extensions Preserved Outside The Canonical Path

Some manuscript-adjacent explorations are preserved only as historical code:

- layer-wise SES experiments: mainly under `experiment_6/`
- older Mapper robustness sweeps and ablations: `experiment_SES/`, `metric/`
- online symbolic stopping prototypes: `online_check/`

These paths are retained for traceability and method history, but the repository should be cited through the canonical `src/ses` + `scripts` workflow rather than through those legacy experiments.
