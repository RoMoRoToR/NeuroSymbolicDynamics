# Data README

This file documents the datasets used by the canonical SES pipeline and aligned with the article, their status as external or generated data, and the expected filenames under `data/`.

## Expected Layout

```text
data/
├── ETT/
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
├── AirPassengers.csv
├── BTC_15m.csv
├── EEG/
│   └── train_with_header.csv
└── lorenz.csv
```

Dataset names are resolved recursively by the CLI. For example, `--data_root data --datasets ETTh1` resolves to `data/ETT/ETTh1.csv`, and `--datasets train_with_header` resolves to `data/EEG/train_with_header.csv`.

## Dataset Inventory

### ETT family

- Status: external public benchmark
- Article group: quasi-periodic
- Expected files:
  - `data/ETT/ETTh1.csv`
  - `data/ETT/ETTh2.csv`
  - `data/ETT/ETTm1.csv`
  - `data/ETT/ETTm2.csv`
- Intended use: multivariate time-series forecasting
- Typical target column: `OT`
- Source: ETT benchmark distributed with the Informer line of work
- Public source: <https://ieee-dataport.org/documents/merged-ett-dataset-time-series-forecasting>
- Preparation notes:
  - Keep the CSV header row.
  - Preserve original column names, especially `OT`.
  - Place the files under `data/ETT/`.

### AirPassengers

- Status: external public dataset
- Expected file: `data/AirPassengers.csv`
- Article group: quasi-periodic
- Intended use: monthly airline-passenger series used as a compact quasi-periodic benchmark and smoke-test dataset
- Public reference: <https://stat.ethz.ch/R-manual/R-devel/RHOME/library/datasets/html/AirPassengers.html>
- Preparation notes:
  - Export the series as a CSV with a header row.
  - If only one numeric column is present, the SES loader will automatically add a synthetic time index.

### Bitcoin time series

- Status: external public or externally exported dataset
- Expected file: `data/BTC_15m.csv`
- Article group: intermediate / regime-switching
- Intended use: non-stationary price-forecasting stress test within the article pipeline
- Recommended target column: `Close`
- Example archival source: <https://zenodo.org/records/4292991>
- Preparation notes:
  - Provide a CSV with numeric OHLCV-style columns where possible.
  - Ensure the target column is named `Close` or `y`, or keep it as the last meaningful numeric column.

### EEG / DEAP-derived tabular input

- Status: derived external dataset
- Expected canonical file: `data/EEG/train_with_header.csv`
- Additional local helper files may include:
  - `data/EEG/train.csv`
  - `data/EEG/features_raw.csv`
  - `data/EEG/labels_0.dat`
- Article group: intermediate / regime-switching
- Intended use: prepared tabular export derived from EEG recordings such as DEAP
- Public source for raw benchmark context: <https://www.eecs.qmul.ac.uk/mmv/datasets/deap/>
- Preparation notes:
  - The canonical pipeline expects a prepared CSV, not raw EEG signals.
  - Document any preprocessing, windowing, feature extraction, or label alignment separately in the manuscript or supplement.
  - If redistribution of the prepared derivative is restricted, keep only the preparation instructions and obtain the raw data from the original source.

### Lorenz trajectories

- Status: generated synthetic dataset
- Expected file: `data/lorenz.csv`
- Article group: near-chaotic
- Generator script: `scripts/generate_lorenz_data.py`
- Example command:

```bash
python scripts/generate_lorenz_data.py --out_csv data/lorenz.csv
```

- Output columns: `t`, `x`, `y`, `z`
- Use case: controlled synthetic sanity checks and smoke validation

## External Versus Generated Data

- External public datasets: ETT, AirPassengers, Bitcoin, and raw-source EEG/DEAP materials
- Generated within this repository: Lorenz trajectories
- Derived/prepared artifacts: `EEG/train_with_header.csv` and any other preprocessed tabular exports

## Repository Policy

- Do not assume that third-party datasets can always be redistributed in a public GitHub or Zenodo archive.
- For public release, verify redistribution rights dataset by dataset.
- If legal or journal policy requires excluding large or restricted data, keep the filenames and preparation instructions unchanged so the pipeline remains reproducible after manual download.

## Minimal Data Checklist Before Running Experiments

- Every CSV should contain a header row.
- The target column should be resolvable as `OT`, `Close`, `y`, or the last non-constant numeric column.
- Non-numeric timestamp columns, if present, should either be the first column or be removed during preparation.
- File names should match the canonical names used in configs and README commands.
