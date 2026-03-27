# Reproducibility Statements

## Data Availability Statement

The study uses a combination of public benchmark data, prepared derivative data, and synthetic data. The repository snapshot includes the ETT benchmark files used by the canonical bundled experiments (`ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`) together with synthetic Lorenz trajectories generated within the repository. Additional datasets referenced in the manuscript, namely `AirPassengers`, the Bitcoin time series (`BTC_15m`), and the prepared EEG-derived tabular input (`train_with_header`), are expected under the documented canonical filenames but are not redistributed in this snapshot. Dataset-specific provenance, expected filenames, and preparation notes are provided in `data/README.md`. Users preparing a public release should verify the redistribution terms of each third-party or derivative dataset and, where necessary, obtain the data from the original public source.

## Code Availability Statement

All code required for the canonical Symbolic Early Stopping (SES) experiment pipeline is available in this repository. The publication-oriented implementation is organized under `src/ses/`, and the recommended command-line entry points are provided in `scripts/` for experiment execution, baseline comparison, result aggregation, compact table/figure generation, and synthetic Lorenz data generation. Runtime parameters are saved automatically in `run_config.json` for each executed experiment. Historical development code is retained in the repository for traceability, but the publication workflow is explicitly confined to the documented canonical path described in `docs/publication_path.md`.

## Suggested Editorial Response Paragraph

To support reproducibility, we provide a structured research-code repository containing the canonical implementation of Symbolic Early Stopping, the baseline stopping criteria used for comparison, and documented command-line workflows for experiment execution, aggregation, and compact publication-asset generation. The repository distinguishes the publication path from historical exploratory code, documents the required dataset layout, and records exact runtime configurations for executed runs. Public ETT benchmark files and synthetic Lorenz data are included in the current snapshot, while additional external or derivative datasets referenced in the manuscript are documented through stable filenames and preparation instructions in `data/README.md`.
