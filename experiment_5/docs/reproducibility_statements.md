# Reproducibility Statements

## Data Availability Statement

The experiments in this study use a combination of public benchmark datasets, prepared derivative data, and synthetic data. The benchmark suite comprises the ETT family (`ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`), the `AirPassengers` series, a Bitcoin price time series, a prepared EEG/DEAP-derived tabular export (`EEG/train_with_header.csv`), and synthetic Lorenz trajectories generated from the governing system within the repository. Source-specific acquisition and preparation notes, including the expected filenames used by the experiment scripts, are documented in `data/README.md`. Because redistribution rights may differ across third-party datasets and prepared derivatives, users preparing a public archive should verify the applicable licenses and, where necessary, re-download the original data from the cited public sources.

## Code Availability Statement

All code required to reproduce the canonical Symbolic Early Stopping (SES) training-and-evaluation pipeline reported in the article is provided in this repository. The publication-oriented implementation is organized under `src/ses/`, and the recommended command-line entry points are provided in `scripts/`. These scripts cover experiment execution, baseline comparison against Patience, Slope, SVCCA, and CDSC, result aggregation, compact publication-table generation, and synthetic Lorenz data generation. Historical development scripts are retained under `legacy/` for traceability, but they are not required for reproducing the canonical article pipeline. Exact runtime parameters for each executed experiment are saved automatically in `run_config.json` within the corresponding results directory.

## Suggested Editorial Response Paragraph

To support reproducibility, we provide a structured research-code repository containing the canonical implementation of Symbolic Early Stopping, the baseline stopping criteria used for comparison in the article, and command-line scripts for experiment execution, aggregation, and compact figure/table generation. The repository documents the expected dataset layout, the preparation of external data sources, and the generation of synthetic Lorenz data, while preserving historical exploratory code only for traceability. The publication-ready workflow is confined to the package under `src/ses/`, the reproducible scripts under `scripts/`, and the accompanying JSON configurations under `configs/`.
