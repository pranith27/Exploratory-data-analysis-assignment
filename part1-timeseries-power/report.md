# Report — Time-Series EDA: Household Electric Power Consumption

**Author:** Pranith Varma Pakalapati — CMPE 255, San Jose State University, Fall 2026
**Assignment:** Exploratory Data Analysis, Part 1 (time-series dataset)
**Notebook:** `EDA_Household_Power_Timeseries.ipynb` (executed; 42 code cells, 0 errors)
**Target:** `Global_active_power` — hourly mean active power (kW), one-hour-ahead forecasting framing

Every number in this report comes from the executed notebook or from `metrics.json` / `model_comparison.csv`, which the notebook itself wrote. Nothing here is copied from the dataset description or from another notebook.

## 1. Dataset and grain strategy

- **Dataset:** Individual Household Electric Power Consumption, UCI Machine Learning Repository (#235), one household in Sceaux, France.
- **Source:** https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip (HTTP 200, verified; ~20 MB zip, ~133 MB extracted text file).
- **Raw file:** 2,075,259 minute-level rows, 16 Dec 2006 17:24 → 26 Nov 2010 21:02, 9 columns (`Date`, `Time`, 7 measurements). No duplicate timestamps; no gaps in the timestamp grid (every minute in the span has a row).
- **Grain strategy** (stated in every notebook section; modelling is hourly, never claimed as minute-level):

| Grain | Rows | Used for |
|---|---|---|
| Minute (raw) | 2,075,259 | Load inspection, missing-value audit, gap analysis |
| Hourly (primary) | 34,589 | EDA charts, cleaning, imputation, clustering input, anomaly detection, modelling |
| Daily | 1,442 | Trend, seasonality, ADF/stationarity, STL decomposition, daily-profile clustering |

## 2. What the raw data looked like (Section 1)

- Each of the 7 measurement columns has **25,979 missing values (1.25%)**, and they always go missing together: 25,979 rows have all 7 missing, 0 rows are partially missing.
- **File quirk found by inspecting raw lines:** a missing row looks like `21/12/2006;11:23:00;?;?;?;?;?;?;` — the last field (`Sub_metering_3`) is an *empty* string, not `?`. A naive per-column count of `?` characters would wrongly report zero missing values for that column. The notebook parses with `na_values=['?']` plus pandas' default empty-field handling, which catches both.
- **Missingness is block-structured, not random:** 71 blocks at minute level (median length 1 minute, mean 366, max 7,226). Six blocks longer than a day hold 24,647 of the 25,979 missing minutes (94.9%). Longest: 17–22 Aug 2010 (7,226 minutes, ~5 days), then 25–28 Sep 2010 (5,237), 28–30 Apr 2007 (3,723), 13–15 Jun 2009 (3,305), 12–14 Jan 2010 (3,129), 20–21 Mar 2010 (2,027). This looks like the meter/logger being down, and it drives the imputation design in §5.
- **Ranges (minute level, observed):** active power 0.076–11.122 kW (mean 1.092); voltage 223.2–254.15 V (mean 240.84); intensity 0.2–48.4 A. No zero or negative power readings.
- **Minute-level correlations:** `Global_active_power` vs `Global_intensity` r = 0.999 (same physical quantity — leakage if used as a same-hour predictor); vs voltage r = −0.400; vs `Sub_metering_3` 0.639, `Sub_metering_1` 0.484, `Sub_metering_2` 0.435.

## 3. Manual time-series EDA (Section 2)

All charts are saved in `figures/` and embedded in the notebook, each with a written insight.

- **Full daily series (fig01):** daily mean 1.091 kW (std 0.420, min 0.174, max 3.315). No strong long-run trend — first 365 days average 1.126 kW, last 365 days 1.086 kW — but large day-to-day volatility and clear seasonal structure.
- **Representative week (fig02):** the first complete Mon–Sun week of 2008 with zero missing hours (7–13 Jan 2008): mean 1.477 kW, max 5.197 kW (Sat 12 Jan 18:00), min 0.261 kW. Clear daily rhythm: quiet nights, morning rise, evening peak.
- **Hour-of-day profile (fig03):** peak hour 20:00 (mean 1.899 kW), quietest 04:00 (0.444 kW), peak/trough ratio 4.28. The ±1 std band is wide at every hour — fixed-hour load still varies a lot day to day.
- **Weekly/yearly seasonality (fig04):** weekday mean 1.036 kW vs weekend 1.234 kW. Highest month December (1.490 kW), lowest August (0.573 kW) — winter heating/lighting vs summer lull, a 2.6× swing.
- **Rolling behaviour (fig05):** 30-day rolling mean ranges 0.251–1.763 kW (seasonal drift, not a steady trend); 30-day rolling std ranges 0.139–0.756 kW (volatility is not constant — supports rolling features over global averages).
- **Autocorrelation (fig06):** daily ACF 0.607 at lag 1, 0.544 at lag 7, 0.471 at lag 14, 0.442 at lag 21 — weekly bumps are visible directly in the ACF. Hourly: lag-1 correlation 0.713, lag-24 0.436, lag-168 0.461. High persistence is why the train/test split must be time-based (§6).
- **Correlation heatmap, hourly (fig07):** power–intensity 0.999; power–voltage −0.379; power vs sub-meterings 0.696 / 0.496 / 0.440 (3/1/2). Sub-meterings correlate only 0.12–0.20 with each other — distinct appliance information.
- **Distributions (fig08):** hourly power is right-skewed (mean 1.092, median 0.803, skew 1.30, p99 3.928, max 6.561 kW). Voltage is near-symmetric around 241 V (skew −0.32). Sub-metering 3 is zero-inflated and skewed.
- **Missingness over time (fig09):** only 8 of 48 months have any fully-missing hour (Aug 2010 worst at 119 hours, then Sep 2010 at 87).
- **Stationarity and decomposition (fig10):** ADF on the daily series (1,442 days, 9 missing days linearly interpolated for grid regularity): statistic −3.666, p = 0.0046, 22 lags — reject the unit-root null at 5% (and 1%): the series mean-reverts. But ADF does not remove seasonality: STL (weekly period 7, robust) shows a strong 7-day seasonal component (range −1.197 to +1.648 kW), a trend absorbing the annual swing (0.177–1.962 kW), and residual std 0.230 kW. Conclusion: no differencing needed; calendar + lag features encode the cycles instead. Trend values near the edges and outage days are least trustworthy and are not over-interpreted.

## 4. Auto-EDA (Section 3)

- **Tool:** `ydata-profiling` 4.12.2, minimal mode, on the full hourly frame (34,589 × 9). Report saved as `auto_eda_profile.html` (0.79 MB).
- **Agreement:** 421 missing values per column (1.2% of hours), zero duplicate rows, skewed power/sub-metering distributions, voltage concentrated near 241 V — all matching the manual audit.
- **What auto-EDA added:** per-column distinct-value and zero counts and a one-pass variable summary.
- **What only manual EDA caught:** everything about time — seasonality, autocorrelation, block-structured missingness, stationarity, and the intensity-leakage issue. Auto-EDA treats rows as independent, so it cannot see any of that. No second auto-EDA tool was added: SweetViz/dataprep pull conflicting dependencies against the autoviml + pandas 2.2 stack, and a second tabular profiler would add no time-aware insight.

## 5. Cleaning and preprocessing (Section 4)

Before/after audit at each step (all counts printed by the notebook):

1. **Parsing:** `?` → NaN (plus empty trailing field → NaN); `Date`+`Time` → one timestamp (explicit `%d/%m/%Y %H:%M:%S`); 7 measurement columns coerced to float. 0 duplicate timestamps removed (none existed).
2. **Resampling to hourly with `mean` for every column.** Justification: power/voltage/intensity are rates whose hourly average is the natural summary, and sub-meterings are Wh-per-minute rates, so their hourly mean is the average rate too. A sum would understate partial hours (first hour has 36 minutes, last has 3) and mix units across hours with different coverage. Coverage is kept visible in an `n_minutes` column: 34,085 hours have all 60 minutes, 83 have 1–59, 421 have 0.
3. **Hourly gap analysis:** 8 missing blocks totalling 421 hours, lengths 119, 87, 61, 54, 52, 33, 14, 1 hours.
4. **Time-aware imputation (gap-aware rule):**
   - Gaps of ≤ 3 hours → linear interpolation: **1 hour** filled.
   - Longer gaps → hour-of-week median (same weekday + same hour, computed from observed hours only): **420 hours** filled.
   - Total imputed: 421/34,589 hours (1.22%). Every imputed hour is flagged (`imputed`, `impute_method`) in `data/hourly_clean.csv`. Forward-fill was rejected (freezes one stale value for days); dropping gaps was rejected (deletes whole weeks, breaks lag continuity); interpolating across multi-day outages was rejected (invents a straight line through real daily/weekly cycles). Fig11 shows the longest outage before/after imputation.
   - Implementation note: the rule is enforced by explicit block-length computation, not by a bare `interpolate(limit=3)`, which would also fill the edge hours of long gaps (an early version did exactly that — 43 hours — and was corrected).
5. **Outlier investigation (no blanket removal):** 1.5×IQR bounds on hourly kW are [−1.514, 3.435]; 733 hours (2.15%) sit above the upper bound, 0 below. Hourly max 6.561 kW (23 Nov 2008 18:00). Voltage-vs-power scatter shows high-load hours on the same smooth voltage-sag trend as normal hours — high load is real behaviour, not error. No values were removed or winsorized here; anomalies are handled in §7 with a decision log.
6. **Daily frame:** 1,442 days, 9 fully-missing days (interpolated only for the ADF/STL analysis in §3; the saved `daily_summary.csv` is built from the cleaned hourly frame).

## 6. Feature engineering, leakage guards, feature selection, split (Section 4)

- **Target:** this hour's `Global_active_power` (hourly mean kW).
- **Leakage guards:** features use only information available *before* the hour starts — calendar fields plus lagged/rolling past values. Same-hour `Global_intensity` (r = 0.999), same-hour voltage, and same-hour sub-meterings are deliberately excluded: they are measured simultaneously with the target, so using them would be nowcasting, not forecasting. Rolling features use `.shift(1)` so the current hour never enters its own features. Lags are computed on the cleaned series before splitting, using only past values, so no future information leaks into any row.
- **Candidate features (22):** `hour`, `dayofweek`, `month`, `is_weekend`, `hour_sin/cos`, `dow_sin/cos`, `gap_lag1/2/3/24/25/168`, `roll24_mean`, `roll24_std`, `roll168_mean`, `sub1/sub2/sub3_lag24`, `volt_lag24`, `imputed`. Cyclic sin/cos encodings make hour 23 adjacent to hour 0. After the 168-hour lag warm-up: **34,421 modelling rows** (23 Dec 2006 17:00 → 26 Nov 2010 21:00).
- **Time-based split (no shuffling):** train 27,536 rows (→ 13 Feb 2010 00:00), test 6,885 rows (13 Feb 2010 01:00 → 26 Nov 2010 21:00). A random split would leak near-duplicate neighbouring hours across the boundary (hourly lag-1 correlation 0.713) and flatter the metrics.
- **Feature selection — three methods, train period only:**
  - *Correlation:* `gap_lag1` 0.718, `gap_lag2` 0.471, `gap_lag168` 0.470, `gap_lag24` 0.435, `gap_lag25` 0.386 lead; `month` −0.043 and `imputed` −0.011 look useless linearly.
  - *Mutual information:* `gap_lag1` 0.565, `gap_lag2` 0.290, `gap_lag24` 0.236, `hour` 0.221, `gap_lag168` 0.213; `month` rises to 0.137 (seasonal, non-monotonic effect that correlation misses).
  - *RandomForest importance (200 trees):* `gap_lag1` 0.552 dominates, then `gap_lag168` 0.056, `gap_lag2`/`gap_lag24` 0.034.
  - *Decision rule (fixed before seeing test results):* drop a feature only if weak by **all three** methods (|r| < 0.10 AND bottom-quartile MI AND bottom-quartile RF importance). Dropped (4): `dayofweek`, `dow_sin`, `dow_cos`, `imputed` — the day-of-week terms are redundant with `is_weekend` + the 168h lag, and the imputed flag carries no signal. **Final feature list (18):** `hour`, `month`, `is_weekend`, `hour_sin`, `hour_cos`, `gap_lag1`, `gap_lag2`, `gap_lag3`, `gap_lag24`, `gap_lag25`, `gap_lag168`, `roll24_mean`, `roll24_std`, `roll168_mean`, `sub1_lag24`, `sub2_lag24`, `sub3_lag24`, `volt_lag24`.
  - *Scaling/encoding:* tree models and AutoViML use raw numerics (AutoViML scales internally); the Ridge baseline uses a StandardScaler pipeline.

## 7. Clustering daily profiles (Section 5)

- **Input:** 1,419 daily profiles (24 hourly kW values each), from the cleaned hourly frame; days kept only with all 24 hourly slots and ≤ 6 imputed hours (23 of 1,442 days excluded: 2 partial edge days + 21 heavily-imputed days). Each hour standardized across days (level + shape both matter).
- **KMeans, k = 2..8 (elbow + silhouette):** silhouette 0.203 (k=2), 0.164 (k=3), 0.134 (k=4), 0.132 (k=5), 0.119 (k=6), 0.093 (k=7), 0.093 (k=8); inertia falls steadily with no sharp kink (28,230 → 18,875). **Chosen k = 2** (best silhouette; larger k adds clusters without separation). Fig13–fig15.
- **Clusters:**
  - Cluster 0 — 915 days (64%): mean 0.859 kW, ≈20.61 kWh/day, peak 21:00 (1.560 kW), trough 03:00, weekend share 19.3%. Lower-use, mostly-weekday days.
  - Cluster 1 — 504 days (36%): mean 1.513 kW, ≈36.31 kWh/day, peak 20:00 (2.636 kW), trough 04:00, weekend share 44.4%. Higher-use, weekend-heavy days.
  - Both share the same basic shape (quiet nights, evening peak) and differ mainly in level — day-to-day variation is more "how much" than "when". PCA (PC1 29.2%, PC2 14.6%) shows overlapping clusters: archetypes, not hard categories.
- **Second method — GMM (k = 2, diagonal covariance):** sizes 507/912, BIC 86,202.6. Adjusted Rand Index vs KMeans = **0.498** (moderate agreement; cross-tab shows the partitions mostly agree up to label swapping, with ~207 days assigned differently). The two-type structure is not an artefact of one algorithm, but exact boundaries should not be over-interpreted.

## 8. Anomaly detection and elimination (Section 6)

Three methods on the 34,589 cleaned hours, then a decision log (no deletion on one method's say-so):

| Method | Flagged hours | Notes |
|---|---|---|
| IsolationForest (7 features, contamination 1%) | 346 (1.00%) | Multivariate unusual combinations |
| Rolling z-score (\|z\| > 4 vs trailing 168 h, shifted) | 140 | Trailing std floored at 0.1 kW so flat weeks cannot produce absurd z-scores (without the floor one ordinary hour scored z = 50.9) |
| IQR rule (1.5×IQR on hourly kW) | 738 | Univariate, for comparison |

- **Overlap:** IF ∩ rolling-z = 77; IF ∩ IQR = 310; rolling-z ∩ IQR = 93; all three = 73; any method = 817. Agreement is modest — "anomaly" depends heavily on the definition.
- **Most extreme hours (rolling z):** 31 Aug 2008 15:00 (2.40 kW, z = 19.43), 14:00 (1.61 kW, z = 14.29), 17:00 (2.89 kW, z = 10.53); 4 Mar 2007 17:00 (4.09 kW, z = 9.55) and 19:00 (5.43 kW, z = 8.96). 272 of 1,442 days contain ≥ 1 flagged hour (fig16–fig17).
- **Sensor-error-like vs real-event-like:** of 409 IF-or-rolling-z flagged hours, 28 have voltage outside 232–248 V (sensor-error-like) and 381 have normal voltage (real-event-like heavy load).
- **Decision log:** remove from **training only** the hours flagged by *both* IsolationForest and rolling z-score: **66 training hours** removed (27,536 → 27,470). IQR-only and single-method flags are kept (plausibly real heavy-load events). Test set untouched (6,885 rows). No capping/winsorizing (max 6.56 kW hourly is plausible). Effect: excluding all IF-or-rolling-z hours moves mean hourly kW from 1.091 to 1.055 and std from 0.894 to 0.831 — anomalies never drove the headline statistics. The RMSE effect of the removal is measured in §9 (negligible: 0.4693 vs 0.4703 kW).

## 9. Modelling: baselines and AutoML with AutoViML (Section 7)

All models evaluated on the same untouched future test period (6,885 hours, 13 Feb → 26 Nov 2010; test target mean 1.004 kW, std 0.730 kW). Full table in `model_comparison.csv`.

### 9.1 Sklearn baselines (time-based split)

| Model | RMSE (kW) | MAE (kW) | R² |
|---|---|---|---|
| HistGradientBoosting (anomaly-cleaned train) | **0.4693** | 0.3204 | 0.5868 |
| HistGradientBoosting (full train, anomalies kept) | 0.4703 | 0.3226 | 0.5850 |
| RandomForest (200 trees) | 0.4755 | 0.3257 | 0.5758 |
| Ridge (scaled) | 0.5013 | 0.3504 | 0.5285 |
| Naive: persistence (lag-1h) | 0.5926 | 0.3879 | 0.3412 |
| Naive: seasonal (lag-168h) | 0.8060 | 0.5586 | −0.2187 |

Anomaly removal improves HGB test RMSE by only 0.001 kW (0.2%) — essentially no effect, which is itself a finding: the flagged hours were not distorting training. The seasonal naive is worse than predicting the test mean (negative R²) — last week is a poor forecast for this household, while last hour is a decent one.

### 9.2 AutoML with AutoViML 0.2.2 (actually executed)

- **Call:** `Auto_ViML(train_clean, "target", test, scoring_parameter="rmse", hyper_param="RS", verbose=0)` with the explicit time-split frames (27,470 × 19 train, 6,885 × 19 test). AutoViML prints "No shuffling of data set before training..." and runs its own variable classification, SULOV + XGBoost feature selection, and 5-fold CV *within the training frame only*, then refits the winner on the full training frame. Its printed CV scores are therefore not the reported metrics: the leaderboard below was computed by notebook code from AutoViML's returned per-model prediction columns against the held-out actuals, after verifying row alignment (correlation 1.0 on a returned feature column; 6,885 rows preserved in order).
- **AutoViML's internal selection:** 13 of 18 features kept (`gap_lag1`, `hour`, `gap_lag24`, `is_weekend`, `hour_cos`, `roll168_mean`, `hour_sin`, `gap_lag3`, `gap_lag168`, `month`, `volt_lag24`, `sub2_lag24`, `sub3_lag24`). Chosen single model: **LinearSVR** (C = 92.89, epsilon = 0.395, max_iter = 4000, random_state = 99) — its `target_predictions` column is identical to the `Linear` column.
- **Leaderboard (test metrics, computed independently):**

| AutoViML model | RMSE (kW) | MAE (kW) | R² |
|---|---|---|---|
| Bagging_Regressor | **0.4810** | 0.3324 | 0.5659 |
| Ensembled (weighted average of 5) | 0.4882 | 0.3587 | 0.5529 |
| Linear (= chosen model, LinearSVR) | 0.5090 | 0.3545 | 0.5139 |
| Decision_Tree | 0.5098 | 0.3542 | 0.5125 |
| KNN_Regressor | 0.5238 | 0.3629 | 0.4853 |
| Adaboost | 0.7495 | 0.6580 | −0.0538 |

- **Interpretation:** AutoViML's best test model (Bagging_Regressor, RMSE 0.4810) beats the seasonal naive by 40.3% and the persistence naive by 18.8%, but trails the tuned HistGradientBoosting baseline (0.4693) by 2.5% — competitive, not magic. Notably, AutoViML's *internal* CV picked the Linear model, which generalizes worse (0.5090) than Bagging on the future test period: in-sample CV and future-period performance disagree, which is exactly why the honest time-split comparison matters. Adaboost fails outright (negative R²) — AutoML tries weak models too, and the leaderboard shows it.
- **Prediction quality (chosen-model predictions, fig18–fig19):** residuals mean +0.0484 kW (slight under-prediction bias), std 0.5067 kW, mean |residual| 0.3545 kW. Worst residual-by-hour: 07:00 (+0.47 kW — the morning ramp is hardest to time), then 19:00/20:00 (+0.25). The scatter shows systematic under-prediction of the sharpest peaks: appliance switch-on spikes are only partly predictable from past values + calendar.
- **Environment fix (documented, not hidden):** AutoViML 0.2.2 fails to import under IPython 9 (`from IPython.core.display import display` was removed). The environment pins `ipython==8.26.0`; see `requirements.txt` and the notebook's Section 7 notes. AutoViML also downgraded scipy to 1.11.4 during installation; statsmodels/sklearn were re-verified working after that. AutoViML's own output CSVs are preserved in `autoviml_output/target/`.
- **Reproducibility check:** AutoViML was run twice end-to-end (an earlier notebook run failed after the AutoViML cell on a notebook syntax error in a later cell, before any outputs were saved). Both runs produced identical leaderboard numbers to 4 decimals, and the final executed notebook contains the complete successful run.

## 10. Conclusions

1. **Calendar cycles dominate.** Hour-of-day, day-of-week, and month, plus 1h/24h/168h lags, carry almost all the predictable signal. A model without them fights the data.
2. **The mess is structured.** Missingness arrives in multi-day outage blocks; the cleaning rules follow directly from that structure, and every imputed hour is flagged.
3. **Days form two soft archetypes** (lower-use weekday days vs higher-use weekend-heavy days), differing in level more than shape.
4. **AutoML works but does not work magic.** AutoViML beats naive baselines clearly on an honest future split, lands close to a tuned gradient-boosting baseline, and still misses sharp peaks. That is the right-sized conclusion for one household with no weather data.

## 11. Limitations

- One household, 2006–2010: behaviour, appliances, and tariffs have changed; nothing here generalizes to other homes or to today without new data.
- No weather data (temperature surely drives heating/cooling load) and no French holiday calendar (some "anomalous" days are probably holidays).
- Minute→hourly aggregation loses within-hour detail; an hourly model cannot forecast sub-hourly spikes.
- Imputed hours (1.22%) carry typical values, not truth, and slightly understate true variability in outage stretches.
- Single time-based split, no walk-forward validation: one test window is honest but noisy; a different window could shift RMSE noticeably.
- AutoViML's internal CV is not time-aware in the same strict sense (it cross-validates within the training frame), so its printed CV scores are optimistic relative to the independently computed test metrics reported here.

## 12. Reproducibility

- Seed 42 everywhere it applies. Package versions in `requirements.txt` (full `pip freeze` from the executing environment, 153 packages; key versions: pandas 2.2.3, numpy 1.26.4, scikit-learn 1.5.2, statsmodels 0.14.4, scipy 1.11.4, matplotlib 3.9.2, seaborn 0.13.2, ydata-profiling 4.12.2, autoviml 0.2.2, IPython 8.26.0, nbformat 5.11.1, nbconvert 7.17.2).
- The raw file is downloaded from the UCI URL at run time and never committed; cleaned aggregates are written by the notebook to `data/`.
- Execution verified: `jupyter nbconvert --execute --inplace` exit code 0; 42/42 code cells executed; 0 error outputs; notebook 3.1 MB (< 15 MB cap). Runtime ~34 minutes on a 4-core shared VM, dominated by the AutoViML step.
- Coding-agent disclosure: a coding agent assisted with writing/debugging the notebook code (including the IPython compatibility fix); analysis choices, interpretation, and verification are the author's, and all numbers come from executed code.
