# Data README

This file documents the datasets referenced by the canonical SES pipeline, whether they are external or generated, and which files are expected under `data/`.

## Canonical Layout

```text
data/
├── ETT/
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
├── EEG/
│   └── train_with_header.csv
├── synthetic/
│   └── lorenz_smoke.csv
├── AirPassengers.csv
├── BTC_15m.csv
└── lorenz.csv
```

Dataset names are resolved recursively. For example:

- `ETTh1` resolves to `data/ETT/ETTh1.csv`
- `train_with_header` resolves to `data/EEG/train_with_header.csv`
- `lorenz` resolves to `data/lorenz.csv`

## Current Snapshot Status

- bundled and ready: `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `lorenz`
- bundled smoke-only helper: `data/synthetic/lorenz_smoke.csv`
- expected but not redistributed in this snapshot: `AirPassengers`, `BTC_15m`, `train_with_header`

If a dataset file is missing or empty, the canonical scripts now fail fast with a preflight error instead of silently starting a broken run.

## Dataset Inventory

### ETT family

- Status: external public benchmark, bundled here in canonical layout for reproducibility
- Files:
  - `data/ETT/ETTh1.csv`
  - `data/ETT/ETTh2.csv`
  - `data/ETT/ETTm1.csv`
  - `data/ETT/ETTm2.csv`
- Article group: quasi-periodic
- Typical target column: `OT`
- Public source: <https://github.com/zhouhaoyi/ETDataset>
- Notes:
  - keep the CSV header row
  - preserve the original `OT` column name
  - these files are the canonical ETT inputs used by `configs/article_main.json`

### AirPassengers

- Status: external public dataset, not redistributed in this snapshot
- Expected file: `data/AirPassengers.csv`
- Article group: quasi-periodic
- Public reference: <https://stat.ethz.ch/R-manual/R-devel/RHOME/library/datasets/html/AirPassengers.html>
- Notes:
  - export the series as CSV with a header row
  - if there is only one numeric column, the loader will add a synthetic time index automatically
  - the full external-data config `configs/article_full_external.json` applies `seq_len=12` for this dataset

### Bitcoin time series

- Status: external public or externally exported dataset, not redistributed in this snapshot
- Expected file: `data/BTC_15m.csv`
- Article group: intermediate / regime-switching
- Recommended target column: `Close`
- Example public source: <https://zenodo.org/records/4292991>
- Notes:
  - provide a CSV with numeric OHLCV-style columns where possible
  - keep `Close` as the target column if available, or ensure the target is the last meaningful numeric column

### EEG / prepared derivative tabular input

- Status: derived external dataset, not redistributed in this snapshot
- Expected canonical file: `data/EEG/train_with_header.csv`
- Article group: intermediate / regime-switching
- Raw-source benchmark context: <https://www.eecs.qmul.ac.uk/mmv/datasets/deap/>
- Notes:
  - the canonical code expects a prepared tabular CSV, not raw EEG signals
  - any preprocessing, feature extraction, or windowing should be documented outside the repository if the prepared derivative cannot be redistributed
  - if you regenerate this file locally, keep the canonical filename unchanged so configs continue to work

### Lorenz trajectories

- Status: generated synthetic dataset
- Canonical file: `data/lorenz.csv`
- Smoke helper file: `data/synthetic/lorenz_smoke.csv`
- Article group: near-chaotic
- Generator script: `scripts/generate_lorenz_data.py`
- Example command:

```bash
python3 scripts/generate_lorenz_data.py --out_csv data/lorenz.csv
```

- Output columns: `t`, `x`, `y`, `z`
- Notes:
  - `data/lorenz.csv` is the canonical Lorenz input for the main bundled configs
  - `data/synthetic/lorenz_smoke.csv` is a small included file used by the smoke config

## Public Versus Generated Data

- external public datasets: ETT, AirPassengers, Bitcoin, raw EEG/DEAP sources
- prepared derivative datasets: `train_with_header`
- generated inside this repository: `lorenz`

## Repository Policy

- do not assume that every third-party dataset can be redistributed in a public code archive
- verify redistribution rights before releasing GitHub or Zenodo snapshots
- if a dataset cannot be redistributed, keep the expected filename and the preparation instructions stable

## Minimal Checklist Before Running

- every CSV must contain a header row
- the target column should resolve as `OT`, `Close`, `y`, or the last non-constant numeric column
- non-numeric timestamp columns should be placed first or removed during preparation
- file names must match the canonical names used in the configs
