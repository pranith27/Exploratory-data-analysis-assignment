# Part 1 — Time Series: Household Electric Power Consumption

**Author:** Pranith Varma Pakalapati — CMPE 255, San Jose State University, Fall 2026

Time-series EDA, cleaning, clustering, anomaly detection, feature engineering/selection, and AutoML (AutoViML) on the UCI Individual Household Electric Power Consumption dataset — minute-level measurements from one household in France, Dec 2006 – Nov 2010. The modelling target is hourly mean `Global_active_power` (kW), forecast one hour ahead from calendar + lag/rolling features only (no same-hour measurements — no leakage).

## Headline results

- **Rows:** 2,075,259 minute-level → **34,589 hourly** (primary grain) → 1,442 daily → 34,421 modelling rows → train 27,470 / test 6,885 (time-based split, last 20% as test, no shuffling).
- **Missing data:** 25,979 missing minutes per column (1.25%), in 71 blocks; 6 multi-day outages hold 95% of them. At hourly grain: 421 missing hours — 1 filled by interpolation (gaps ≤ 3 h), 420 by hour-of-week medians.
- **Seasonality:** peak hour 20:00 (1.90 kW) vs 04:00 (0.44 kW); weekends 1.23 kW vs weekdays 1.04 kW; December 1.49 kW vs August 0.57 kW. ADF rejects a unit root (statistic −3.666, p = 0.0046); STL shows a strong weekly cycle plus an annual trend swing.
- **Clusters:** daily profiles split into 2 types (KMeans silhouette 0.203) — lower-use weekday days (915 days, ~20.6 kWh/day) vs higher-use weekend-heavy days (504 days, ~36.3 kWh/day). GMM agrees moderately (ARI 0.498).
- **Anomalies:** IsolationForest 346 h, rolling z-score 140 h, IQR 738 h; 77 h flagged by both IF and rolling-z; 66 training hours removed (test untouched).
- **Models (test RMSE, kW):** HistGradientBoosting **0.4693** (best overall) · RandomForest 0.4755 · **AutoViML best (Bagging_Regressor) 0.4810** · AutoViML chosen model (LinearSVR) 0.5090 · Ridge 0.5013 · persistence naive 0.5926 · seasonal naive 0.8060. Full table: `model_comparison.csv`.

See `report.md` for the detailed written report with all numbers, and `data/DATA_DICTIONARY.md` for column definitions.

## Contents

| File | What it is |
|---|---|
| `EDA_Household_Power_Timeseries.ipynb` | Executed Colab-ready notebook (42 code cells, 0 errors) — the main artifact |
| `report.md` | Detailed written report with real numbers from execution |
| `auto_eda_profile.html` | ydata-profiling report (minimal mode, full hourly frame) |
| `metrics.json` | Row counts, ADF/STL, clusters, anomalies, all model metrics, versions |
| `model_comparison.csv` | All models ranked by test RMSE |
| `requirements.txt` | Full `pip freeze` from the executing environment (153 packages) |
| `data/DATA_DICTIONARY.md` | Column definitions for raw + derived files |
| `data/hourly_clean.csv` | 34,589 cleaned hourly rows with imputation + anomaly flags |
| `data/daily_summary.csv` | 1,442 daily aggregates |
| `data/autoviml_test_predictions.csv` | Test-period actuals vs AutoViML + HGB predictions (6,885 rows) |
| `figures/*.png` | 20 charts (daily series, weekly profile, seasonality, ACF, heatmap, STL, clusters, anomalies, predictions) |
| `autoviml_output/target/*.csv` | AutoViML's own output files from the executed run |

The raw 2M-row file is **not** committed — the notebook downloads it from UCI at run time.

## How to run

### Locally

```bash
cd part1-timeseries-power
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook EDA_Household_Power_Timeseries.ipynb   # then Run All
```

Runtime is roughly 30–35 minutes on a 4-core machine; the AutoViML step dominates. The notebook downloads the ~20 MB zip from UCI on first run (or reuses `data/household_power_consumption.txt` if present) and regenerates every artifact in this folder. Seed 42 throughout.

> **Environment note:** AutoViML 0.2.2 does not import under IPython 9 (`from IPython.core.display import display` was removed), so `requirements.txt` pins `ipython==8.26.0`. Do not "upgrade" IPython or AutoViML will fail at import. AutoViML also pins scipy to 1.11.4 in this environment.

### In Google Colab

1. Upload this folder to Colab (or clone the repo and `cd` into this folder).
2. Run the install line at the top of the notebook (uncomment it): `!pip install -q -r requirements.txt`.
3. Run All. The notebook assumes the working directory is this folder (it reads/writes `data/` and `figures/` relatively).

## Dataset

- **UCI Individual Household Electric Power Consumption (#235):** https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip
- Citation: Dua, D. and Graff, C. (2019). UCI Machine Learning Repository. Data collected by Georges Hebrail and Alice Berard.
- Licence/terms: see the UCI dataset page. Missing values are `?` strings (plus an empty trailing field for `Sub_metering_3` on missing rows — documented in the notebook and `data/DATA_DICTIONARY.md`).

## Disclosure

A coding agent assisted with writing and debugging the notebook code; the analysis choices, interpretations, and verification are the author's, and every number comes from executed code (see the disclosure cell in the notebook).
