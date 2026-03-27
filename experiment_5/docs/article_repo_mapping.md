# Article-to-Repository Mapping

This document maps the manuscript
`/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/data/Статья_Замотаев_42.docx`
to the canonical implementation and reproducibility path in `experiment_5`.

It is intended for editorial responses, supplementary materials, and internal release checks.

## Coverage Labels

- `Direct`: the article component maps directly to a canonical file or CLI in `experiment_5`.
- `Partial`: the article component is supported by the core codebase, but not exposed as a dedicated publication CLI or not fully packaged as a separate workflow.
- `Archival`: relevant historical material exists, but it is not part of the canonical publication path.

## High-Level Mapping

| Article section | Manuscript content | Repository mapping | Coverage |
| --- | --- | --- | --- |
| Title / Abstract | SES for neural sequence models via Mapper-induced symbolic dynamics | [README.md](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/README.md), [src/ses/core.py](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/src/ses/core.py) | Direct |
| Section 1 | Problem framing, SES motivation, monitored hidden representations, baselines, contributions | [README.md](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/README.md), [docs/publication_path.md](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/docs/publication_path.md), [src/ses/core.py](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/src/ses/core.py) | Direct |
| Section 2 | Related work context | Article-only narrative; not expected as executable repo content | Direct |
| Section 3 | SES methodology, metrics, datasets, protocol | [src/ses/core.py](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/src/ses/core.py), [configs/article_main.json](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/configs/article_main.json), [data/README.md](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/data/README.md) | Direct |
| Section 4 | Results, robustness, comparisons, supplementary experiment groups | [scripts/run_experiment.py](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/scripts/run_experiment.py), [scripts/run_baselines.py](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/scripts/run_baselines.py), [scripts/aggregate_results.py](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/scripts/aggregate_results.py), [scripts/build_figures_tables.py](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/scripts/build_figures_tables.py) | Partial |
| Section 4.7 / Section 6 | Reproducibility and availability statements | [docs/reproducibility_statements.md](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/docs/reproducibility_statements.md), [README.md](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/README.md), [CITATION.cff](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/CITATION.cff), [.zenodo.json](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/.zenodo.json) | Direct |

## Method-Level Mapping

| Article component | Repository file(s) | Notes |
| --- | --- | --- |
| Validation hidden-state monitoring | [src/ses/core.py#L869](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/src/ses/core.py#L869) | `run_one()` collects validation representations epoch by epoch. |
| Final hidden representation for RNN | [src/ses/core.py#L119](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/src/ses/core.py#L119) | `RNNRegressor.forward()` uses the final hidden state. |
| Final hidden representation for BiRNN | [src/ses/core.py#L131](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/src/ses/core.py#L131) | Concatenates final forward/backward hidden states. |
| Final hidden representation for Transformer | [src/ses/core.py#L153](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/src/ses/core.py#L153) | Mean pooling over final encoder outputs. |
| Mapper-based symbolization | [src/ses/core.py#L242](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/src/ses/core.py#L242) | `MapperSymbolizer` implements the coarse symbolic partition. |
| Symbol sequence extraction over validation trajectories | [src/ses/core.py#L327](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/src/ses/core.py#L327), [src/ses/core.py#L503](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/src/ses/core.py#L503) | Symbol sequences are split by validation-batch lengths and summarized per epoch. |
| Lempel-Ziv complexity | [src/ses/core.py#L334](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/src/ses/core.py#L334) | `lz76_phrase_count()` and `lz_normalized()`. |
| Markov entropy rate | [src/ses/core.py#L383](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/src/ses/core.py#L383) | `markov_entropy_rate()`. |
| Permutation entropy | [src/ses/core.py#L407](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/src/ses/core.py#L407) | `permutation_entropy()`. |
| Correlation dimension D2 | [src/ses/core.py#L421](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/src/ses/core.py#L421) | `correlation_dimension_d2()`. |
| Optional fractal / box-counting dimension DF | [src/ses/core.py#L445](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/src/ses/core.py#L445) | `box_counting_dimension_df()`, controlled by `ses_include_df`. |
| EMA smoothing | [src/ses/core.py#L469](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/src/ses/core.py#L469) | `ema()`. |
| Rank aggregation and liveness filtering | [src/ses/core.py#L550](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/src/ses/core.py#L550) | `build_symbolic_score()`. |
| Hybrid SES stopping decision | [src/ses/core.py#L598](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/src/ses/core.py#L598) | `find_stop_symbolic_hybrid()`. |

## Baseline Mapping

| Article baseline | Repository file(s) | Notes |
| --- | --- | --- |
| Patience | [src/ses/core.py#L683](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/src/ses/core.py#L683) | `patience_es()`. |
| Slope | [src/ses/core.py#L701](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/src/ses/core.py#L701) | `slope_es()`. |
| SVCCA | [src/ses/core.py#L655](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/src/ses/core.py#L655) | `svcca_stop()`. |
| CDSC | [src/ses/core.py#L712](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/src/ses/core.py#L712) | `cdsc_stop()`. |
| Shared run protocol for SES + baselines | [src/ses/core.py#L869](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/src/ses/core.py#L869), [scripts/run_experiment.py](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/scripts/run_experiment.py), [scripts/run_baselines.py](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/scripts/run_baselines.py) | SES and baselines are evaluated on the same logged runs, matching the article protocol. |

## Dataset Mapping

| Article dataset group | Dataset(s) in article | Repository mapping | Coverage |
| --- | --- | --- | --- |
| Quasi-periodic | `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `AirPassengers` | [data/README.md](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/data/README.md), [configs/article_main.json](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/configs/article_main.json) | Direct |
| Intermediate / regime-switching | `BTC_15m`, DEAP-derived EEG | [data/README.md](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/data/README.md), [configs/article_main.json](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/configs/article_main.json) | Direct |
| Near-chaotic | `lorenz` | [scripts/generate_lorenz_data.py](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/scripts/generate_lorenz_data.py), [data/README.md](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/data/README.md) | Direct |

## Experiment-Group Mapping

| Article experiment group | Manuscript intent | Repository mapping | Coverage | Notes |
| --- | --- | --- | --- | --- |
| E1 / Section 4.1 | Variability of individual symbolic indicators vs SES ensemble | [src/ses/core.py#L503](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/src/ses/core.py#L503), [src/ses/core.py#L550](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/src/ses/core.py#L550), epoch CSVs written by [src/ses/core.py#L980](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/src/ses/core.py#L980) | Direct | Per-epoch symbolic metrics are produced in canonical runs. |
| E2 / Section 4.2 | SES vs Patience, Slope, SVCCA, CDSC | [scripts/run_experiment.py](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/scripts/run_experiment.py), [scripts/run_baselines.py](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/scripts/run_baselines.py), [src/ses/core.py#L869](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/src/ses/core.py#L869) | Direct | Core comparison path in the release. |
| E3 / Section 4.3 | Robustness under additive Gaussian noise | [configs/article_main.json](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/configs/article_main.json), [src/ses/core.py#L862](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/src/ses/core.py#L862), [src/ses/core.py#L980](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/src/ses/core.py#L980) | Direct | Noise sweeps are part of the canonical config and core run loop. |
| E4 / Section 4.4 | Layer-wise analysis | Core architecture code in [src/ses/core.py](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/src/ses/core.py); supplementary interpretation in [docs/publication_path.md](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/docs/publication_path.md) | Partial | No dedicated canonical layer-sweep CLI is currently packaged in `experiment_5`. |
| E5 / Section 4.5 | Mapper hyperparameter transfer across architectures | Mapper controls in [src/ses/core.py#L774](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/src/ses/core.py#L774), article configs in [configs/article_main.json](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/configs/article_main.json) | Partial | Hyperparameters are configurable, but no dedicated sweep/report script is exposed as a canonical CLI. |
| E6 / Section 4.6 | Runtime profiling | Core run loop in [src/ses/core.py#L869](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/src/ses/core.py#L869), article-path note in [docs/publication_path.md](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/docs/publication_path.md) | Partial | No dedicated runtime-profiling CLI/report generator is currently packaged in `experiment_5`. |
| Section 4.7 | Reproducibility and limitations | [README.md](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/README.md), [docs/reproducibility_statements.md](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/docs/reproducibility_statements.md), [docs/publication_path.md](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/docs/publication_path.md) | Direct | Explicitly documented in the repository. |

## Output-Metric Mapping

| Article metric / output | Repository mapping | Notes |
| --- | --- | --- |
| Stop epoch `e_stop` | [src/ses/core.py#L934](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/src/ses/core.py#L934) | Stored per method in `per_run.csv`. |
| Validation loss at stop | [src/ses/core.py#L934](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/src/ses/core.py#L934) | Stored as `*_val_at_stop`. |
| Oracle epoch / oracle validation loss | [src/ses/core.py#L930](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/src/ses/core.py#L930) | Stored as `oracle_epoch` and `oracle_val`. |
| Regret `ΔBest` | [src/ses/core.py#L934](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/src/ses/core.py#L934) | Stored as `*_delta_best`. |
| Saved epochs | [src/ses/core.py#L934](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/src/ses/core.py#L934) | Stored as `*_saved_epochs`. |
| Summary tables by noise | [src/ses/core.py#L739](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/src/ses/core.py#L739), [src/ses/reporting.py](/Users/taniyashuba/PycharmProjects/NeuroSymbolicDynamics/experiment_5/src/ses/reporting.py) | Written as `summary_by_noise.csv`, `compact_table.csv`, and aggregated variants. |

## CLI Mapping For Supplementary Materials

| Use case | Canonical command |
| --- | --- |
| Smoke validation | `python scripts/run_experiment.py --config configs/smoke_airpassengers.json` |
| Main article-scale run | `python scripts/run_experiment.py --config configs/article_main.json` |
| Baseline-comparison entry point | `python scripts/run_baselines.py --config configs/article_baselines.json` |
| Aggregate results | `python scripts/aggregate_results.py --inputs results/article_main results/article_baselines --out_root results/aggregate_article` |
| Build compact tables/figures | `python scripts/build_figures_tables.py --input_csv results/aggregate_article/aggregated_per_run.csv --out_root results/publication_assets` |
| Generate synthetic Lorenz data | `python scripts/generate_lorenz_data.py --out_csv data/lorenz.csv` |

## Editorial Notes

- The canonical publication path in `experiment_5` directly covers the main SES training-and-evaluation pipeline described in the article.
- The repository directly supports the article's primary method comparison protocol: SES vs Patience, Slope, SVCCA, and CDSC on shared logged runs.
- Some article analyses are only partially packaged as standalone public workflows in `experiment_5`, most notably:
  - layer-wise analysis
  - Mapper hyperparameter sweeps / transfer studies
  - dedicated runtime profiling reports
- These components are supported conceptually by the same core codebase and method definitions, but they are not yet exposed as separate canonical publication CLIs in the current repository state.
