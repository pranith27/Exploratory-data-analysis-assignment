# Data dictionary — Household Electric Power Consumption (time-series part)

## Source file (not committed — downloaded by the notebook)

- **Dataset:** Individual Household Electric Power Consumption, UCI Machine Learning Repository (#235)
- **Download URL:** https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip
- **Contents:** one semicolon-separated text file, `household_power_consumption.txt` (~133 MB extracted)
- **Rows:** 2,075,259 data rows + 1 header row. One row per minute, 16 Dec 2006 17:24 → 26 Nov 2010 21:02. No duplicate timestamps, no missing timestamps (complete minute grid).
- **Missing values:** the literal string `?` in the seven measurement columns — except `Sub_metering_3`, whose missing entries are an *empty* trailing field (the raw line ends `?;?;`). Each measurement column has 25,979 missing values (1.25%), always all seven together (25,979 rows fully missing, 0 rows partially missing), in 71 blocks; 6 blocks longer than a day hold 24,647 of the missing minutes.
- **Household:** one house in Sceaux, France. Sub-meterings follow the UCI documentation below.

## Raw columns (minute level)

| Column | Type | Unit / meaning |
|---|---|---|
| `Date` | date (`d/m/Y`) | Calendar date, day-first |
| `Time` | time (`H:M:S`) | Time of day |
| `Global_active_power` | float | Household global minute-averaged active power, kW. **Modelling target (hourly mean).** Observed range 0.076–11.122 kW, mean 1.092 kW. No zero/negative values. |
| `Global_reactive_power` | float | Household global minute-averaged reactive power, kW. Range 0.000–1.390. |
| `Voltage` | float | Minute-averaged voltage, V. Range 223.2–254.15, mean 240.84. |
| `Global_intensity` | float | Household global minute-averaged current intensity, A. Range 0.2–48.4. Correlates 0.999 with active power — excluded as a same-hour predictor (leakage); see notebook §4. |
| `Sub_metering_1` | float | Energy sub-metering No. 1, Wh per minute. Kitchen: dishwasher, oven, microwave. Mostly 0; max 88. |
| `Sub_metering_2` | float | Energy sub-metering No. 2, Wh per minute. Laundry room: washing machine, tumble-drier, refrigerator, light. Max 80. |
| `Sub_metering_3` | float | Energy sub-metering No. 3, Wh per minute. Electric water-heater + air-conditioner. Max 31. |

Note: the three sub-meterings do not sum to the global total — remaining load (lighting, electronics, etc.) is unmetered. That is expected and is not treated as an error.

## `data/hourly_clean.csv` (created by the notebook — 34,589 rows, one per clock hour)

Hourly means of the minute values (`mean` resampling; see notebook §4 for the justification). Hours are complete clock hours from 2006-12-16 17:00 to 2010-11-26 21:00.

| Column | Meaning |
|---|---|
| `datetime` | Hour start (clock hour) |
| `Global_active_power`, `Global_reactive_power`, `Voltage`, `Global_intensity`, `Sub_metering_1/2/3` | Hourly mean of the minute values, after imputation (see below) |
| `n_minutes` | Number of observed minutes averaged into this hour (0–60; 60 for a complete hour; 36 for the first hour, 3 for the last) |
| `imputed` | `True` if the hour had no observed minutes and was filled (421 hours, 1.22%) |
| `impute_method` | `observed`, `interpolated` (linear, gaps ≤ 3 h — 1 hour), or `hour_of_week_median` (same weekday + hour median from observed hours — 420 hours) |
| `anomaly_if` | IsolationForest flag (contamination 1%, notebook §6) |
| `anomaly_rollz` | Rolling z-score flag (|z| > 4 vs trailing 168 h, notebook §6) |

## `data/daily_summary.csv` (created by the notebook — 1,442 rows, one per calendar day)

Daily aggregates of the cleaned hourly frame.

| Column | Meaning |
|---|---|
| `date` | Calendar date |
| `mean_kw` / `max_kw` / `min_kw` | Daily mean / max / min of hourly mean kW |
| `total_kwh_approx` | Sum of hourly mean kW ≈ daily energy in kWh (approximate for the partial first/last days) |
| `n_imputed_hours` | Imputed hours in the day (0–24) |
| `n_anomaly_if_hours` / `n_anomaly_rollz_hours` | Flagged hours in the day per method |

## `data/autoviml_test_predictions.csv` (created by the notebook — test-period hours only)

| Column | Meaning |
|---|---|
| `datetime` | Test hour |
| `actual_kw` | Observed hourly mean `Global_active_power` |
| `autoviml_pred_kw` | AutoViML best-model prediction (`target_predictions`) |
| `hgb_pred_kw` | HistGradientBoosting baseline prediction (same split, anomaly-cleaned train) |

## Modelling features (built in notebook §4; not saved as a separate file — regenerate by running the notebook)

Target: `target` = hourly mean `Global_active_power`. All features use only information available *before* the hour starts (no same-hour measurements). Lags/rolling values are computed on the cleaned hourly series before the time-based split, using only past values (`shift`), so no future information leaks into any row.

| Feature | Definition |
|---|---|
| `hour`, `dayofweek`, `month` | Calendar fields of the hour |
| `is_weekend` | 1 if Saturday/Sunday |
| `hour_sin`, `hour_cos` | Cyclic encoding of hour of day |
| `dow_sin`, `dow_cos` | Cyclic encoding of day of week |
| `gap_lag1/2/3/24/25/168` | `Global_active_power` 1, 2, 3, 24, 25, 168 hours earlier |
| `roll24_mean`, `roll24_std` | Mean / std of the previous 24 hours (shifted by 1) |
| `roll168_mean` | Mean of the previous 168 hours (shifted by 1) |
| `sub1_lag24`, `sub2_lag24`, `sub3_lag24`, `volt_lag24` | Sub-meterings / voltage 24 hours earlier |
| `imputed` | 1 if the target hour itself was imputed (kept as a candidate feature; selection decides) |

Final feature list after the three-method selection (correlation + mutual information + RandomForest importance, train period only) is recorded in `metrics.json` (`features_final` / `features_dropped`) and in `report.md`.
