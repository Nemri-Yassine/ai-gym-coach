# Data

Raw data is not stored in git. Download it into `data/raw/`:

| Dataset | Put it in | Source | Size |
|---|---|---|---|
| MM-Fit | `data/raw/mm-fit/` (folders `w00` … `w20`) | https://mmfit.github.io | 3.8 GB |
| EC3D | `data/raw/ec3d/` (`data_3D.pickle`, `data.pickle`) | https://drive.google.com/drive/folders/1Y00Qw6QyAnhrxUWelyFcTm2a7tnZB93y | 21 MB + 961 MB |
| Our recordings | `data/raw/own_recordings/` | recorded by us (Poco X3 Pro + Galaxy Watch 7) | — |

Pipeline stages:

- `raw/`: original files, never modified
- `interim/`: streams synced on the 30 fps frame clock, cleaned and normalized (Parquet, written by Spark)
- `processed/`: model-ready tensors (3 s windows for Model 1, one tensor per rep for Models 2 and 3)

Check the downloads with `python -m pytest tests`.
