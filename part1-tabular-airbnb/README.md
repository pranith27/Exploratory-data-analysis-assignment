# Part 1 — Tabular: NYC Airbnb Open Data 2019

**Author:** Pranith Varma Pakalapati — CMPE 255, SJSU, Fall 2026 — Exploratory Data Analysis assignment, Part 1 (tabular dataset).

End-to-end EDA, visualisation, cleaning/preparation, clustering, anomaly handling and AutoML
(AutoViML) on the NYC Airbnb Open Data 2019 file. Target: nightly `price`, modelled as `log1p(price)`.

## Headline results (all from the executed notebook)

- **48,895 → 48,825 rows:** 11 invalid $0 prices dropped, 59 anomaly rows removed (0.12%);
  `minimum_nights` capped at 45 (469 rows).
- **Missingness is structural:** 10,052 listings (20.6%) lack review date/rate because they were
  never reviewed → 0-fill + `is_never_reviewed` flag (asserted in code).
- **Price drivers:** room type (entire-home median $160 vs private $70 vs shared $45), borough
  (Manhattan $150 vs Bronx $65), neighbourhood, and distance to Midtown (r = −0.39 with log-price).
  No raw numeric column exceeds |r| = 0.15 with price.
- **Clusters (KMeans, k = 3, silhouette 0.217):** busy affordable outer-borough rooms (33.3%,
  mean $117); expensive Manhattan long-stay entire homes (8.8%, mean $232, 36-night stays); typical
  mid-priced core (57.9%, mean $161). GMM agrees only weakly (ARI 0.147).
- **AutoViML 0.2.2 (actually run, 7.2 min):** best member Bagging_Regressor — holdout log RMSE
  0.424, R² 0.602, $MAE $51.44. Returned model: tuned LinearSVR (log RMSE 0.461).
- **Best overall:** HistGradientBoosting (log RMSE 0.409, R² 0.629) and RandomForest (log RMSE
  0.410, lowest $MAE $49.52) on the same 9,765-row holdout. Full table: `model_comparison.csv`.
- See **`report.md`** for the detailed write-up with every number, and **`metrics.json`** for
  machine-readable results.

## Dataset

NYC Airbnb Open Data 2019 — Kaggle `dgomonov/new-york-city-airbnb-open-data`
(https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data), originally Inside Airbnb
(http://insideairbnb.com/get-the-data/). 48,895 rows x 16 columns. The notebook downloads the CSV
from a public GitHub mirror of the exact Kaggle file (7,077,973 bytes) with fallbacks; the full CSV
is not committed here. Column definitions: `data/DATA_DICTIONARY.md`. Quick-look sample:
`data/sample_500.csv`.

## How to run

### Google Colab (recommended for reviewers)

1. Upload `EDA_Airbnb_NYC_Tabular.ipynb` to Colab (Runtime → Python 3).
2. Run All. The setup cell installs AutoViML / ydata-profiling / SweetViz automatically if missing.
3. Runtime is roughly 20–30 minutes, mostly the AutoViML run. All outputs, figures and CSVs are
   written next to the notebook (in Colab, to `/content`).

### Locally

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace EDA_Airbnb_NYC_Tabular.ipynb
```

Python 3.12. The notebook was executed with `nbconvert` exit code 0, 42/42 code cells, 0 error
cells (see `nbconvert.log`). Seed 42 is used throughout.

Notes:

- `IPython==8.26.0` is pinned because autoviml 0.2.2 fails to import under IPython 9.
- AutoViML was fit on a documented random 15,000-row subsample of the training split (a full-frame
  run's linear search did not finish in 26+ minutes on the build machine); evaluation uses the full
  9,765-row holdout, and the sklearn baselines use the full training split. See `report.md` §9.
- `featurewiz` is not in `requirements.txt`: it cannot build on Python 3.12. Feature selection
  uses correlation + mutual information + RandomForest importance + RFE instead.

## Files

| File | What it is |
|---|---|
| `EDA_Airbnb_NYC_Tabular.ipynb` | Executed notebook (1.88 MB), Colab-ready, 0 errors |
| `report.md` | Detailed written report with real numbers |
| `metrics.json` | Row counts, missingness, clusters, anomalies, all model metrics |
| `model_comparison.csv` | Combined AutoViML + baseline holdout metrics |
| `autoviml_leaderboard.csv` | AutoViML-only holdout metrics |
| `autoviml_log.txt` + `autoviml_outputs/` | Captured AutoViML console output and its train/test/submission CSVs |
| `auto_eda_profile.html` | ydata-profiling report (minimal mode) |
| `sweetviz_report.html` | SweetViz report (target: price) |
| `figures/` (17 PNGs) | Every key chart as a file |
| `data/` | Data dictionary + 500-row sample |
| `requirements.txt` | Pinned versions that actually ran |
| `build_notebook.py` | Script that generated the notebook |

## Disclosure

A coding agent assisted with writing and executing this notebook, as the assignment allows. All
tables, charts and metrics are outputs of executed code on the dataset above.
