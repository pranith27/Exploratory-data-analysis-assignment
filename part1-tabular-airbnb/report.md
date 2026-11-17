# Report — Part 1 (Tabular): EDA, Preparation and AutoML on NYC Airbnb Open Data 2019

**Author:** Pranith Varma Pakalapati — CMPE 255, San Jose State University, Fall 2026
**Assignment:** Exploratory Data Analysis, Part 1 (tabular dataset)
**Notebook:** `EDA_Airbnb_NYC_Tabular.ipynb` (executed, 78 cells: 42 code, 36 markdown, 0 error cells)
**Target:** `price`, nightly price in USD, modelled as `log1p(price)`.
**Seed:** 42 everywhere. All numbers below come from the executed notebook unless stated otherwise.

## 1. Dataset and provenance

- **Dataset:** NYC Airbnb Open Data 2019, Kaggle `dgomonov/new-york-city-airbnb-open-data`
  (https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data), originally Inside Airbnb
  (http://insideairbnb.com/get-the-data/), New York City snapshot, 2019.
- **Size verified on load:** 48,895 rows x 16 columns. Download URL used:
  `https://raw.githubusercontent.com/RonBalaban/CUNY-SPS-R/main/AB_NYC_2019.csv` (7,077,973 bytes),
  with two fallback mirrors of the same file and a local-copy fallback in the notebook. The full CSV is
  not committed to the repository; a 500-row sample is in `data/sample_500.csv` and the column
  definitions are in `data/DATA_DICTIONARY.md`.
- **One row = one listing.** Columns cover borough, neighbourhood, coordinates, room type, price,
  minimum stay, review activity and availability. Full data dictionary: `data/DATA_DICTIONARY.md`.

## 2. Row-count audit (no silent deletions)

| Stage | Rows | Change |
|---|---|---|
| Raw file | 48,895 | — |
| After dropping $0 prices | 48,884 | −11 (invalid $0 nightly rates) |
| After review-missingness handling | 48,884 | 0 (fills + flags only) |
| After rare-neighbourhood grouping | 48,884 | 0 (recoding only) |
| After anomaly removal | 48,825 | −59 (0.12%; see Section 7) |
| **Modelling frame** | **48,825** | |

## 3. Missingness

| Column | Missing | % | Treatment |
|---|---|---|---|
| `last_review` | 10,052 | 20.56 | Not imputed: no review date exists for never-reviewed listings. Information carried by `days_since_last_review` (−1 sentinel) + `is_never_reviewed`. Column excluded from model inputs. |
| `reviews_per_month` | 10,052 | 20.56 | Filled with 0 (true value for never-reviewed listings) + `is_never_reviewed` flag. |
| `name` | 16 | 0.03 | Column dropped (free text / identifier). |
| `host_name` | 21 | 0.04 | Column dropped (identifier). |
| All other 12 columns | 0 | 0.00 | — |

Key evidence: the 10,052 missing `last_review` values and the 10,052 missing `reviews_per_month` values
are the *same* listings, and every one of them has `number_of_reviews == 0`, while no reviewed listing
lacks a review date (asserted in code). The missingness is structural (never reviewed), not random —
median imputation would invent review activity for a fifth of the data and was rejected for that reason.
After dropping the 11 $0-priced listings, the never-reviewed count is 10,051.

## 4. Manual EDA findings (Section 3 of the notebook, 12 charts in `figures/`)

- **Price is extremely right-skewed:** mean $152.72, median $106, 90th percentile $269, 99th $799,
  99.9th $3,000, max $10,000; skew 19.12. On the `log1p` scale the distribution is near bell-shaped
  (mean 4.74, median 4.67, std 0.69, skew 0.63). All modelling uses `log1p(price)`.
- **Borough:** Manhattan median $150 (mean $196.90, n=21,661) > Brooklyn $90 ($124.40, n=20,104) >
  Queens $75 ($99.50, n=5,666) = Staten Island $75 ($114.80, n=373) > Bronx $65 ($87.50, n=1,091).
  Manhattan + Brooklyn hold 85% of listings.
- **Room type (strongest single signal):** entire home/apt median $160 (mean $211.80, n=25,409) >
  private room $70 ($89.80, n=22,326) > shared room $45 ($70.10, n=1,160).
- **Borough x room type interacts:** the entire-home premium exists everywhere but is largest in
  Manhattan ($191 vs $90 private). Thin cells: shared rooms in Staten Island (n=9) and the Bronx (n=60).
- **Geography:** listings concentrate in Manhattan and north-west Brooklyn; coordinates span
  lat 40.50–40.91, lon −74.24–−73.71, with no (0, 0) rows. Expensive listings cluster in lower/midtown
  Manhattan and the Brooklyn waterfront. This motivated the distance-to-Midtown feature, which
  correlates −0.39 with log-price — far stronger than any raw numeric column.
- **Reviews vs price:** weakly negative (r = −0.048 with price, −0.043 with log-price). The most-reviewed
  listings (max 629 reviews) sit mid-price; expensive listings rarely accumulate many reviews.
- **Availability is U-shaped:** 17,533 listings (35.9%) available 0 days, median 45 days, only 1,295
  (2.6%) available all 365 days. A large share of the file is effectively inactive supply.
- **Minimum nights:** median 3, mean 7.03, 90th percentile 28; most common values 1 (12,720), 2
  (11,696), 3 (7,999), then a spike at 30 (3,760 — monthly rentals); 747 listings require >30 nights,
  14 require >365, max 1,250.
- **Correlations with price (Pearson, numeric columns only):** longitude −0.15, availability +0.08,
  host-listings count +0.06, review count −0.05, minimum nights +0.04, latitude +0.03, reviews/month
  −0.03. No raw numeric column exceeds |r| = 0.15 — price is driven by categoricals and non-linear
  geography. Strongest numeric pair: `number_of_reviews` / `reviews_per_month` at +0.55 (expected —
  one derives from the other).

## 5. Auto-EDA (ydata-profiling + SweetViz)

- `auto_eda_profile.html` (2.83 MB, ydata-profiling 4.16.1, minimal mode, 37 s) and
  `sweetviz_report.html` (2.13 MB, SweetViz 2.3.1 with `price` as target, 40 s) are saved alongside
  the notebook; they are not embedded, to keep the notebook small.
- **Top 5 auto findings:** (1) price is the most skewed numeric column; (2) the two review columns
  share identical missingness (10,052 rows); (3) high cardinality in `neighbourhood` (221) and `host_id`
  (37,457); (4) zero-inflation in `availability_365` (35.9% at 0); (5) weak numeric but strong
  categorical associations with price.
- **Auto vs manual:** auto-EDA caught systematically what manual checks could miss (duplicate-free
  status, full per-column profiles, the 58.5 reviews/month outlier); manual EDA caught what auto-EDA
  cannot (the geographic price gradient — profilers never draw the map; the borough x room-type
  interaction; the *reason* for the missing review data; the $0-price data error). The notebook's
  Section 4 records this comparison in full.
- **Environment side effect found and fixed:** SweetViz sets numpy's global error handling to
  `all='raise'` and never restores it, which crashed the later GaussianMixture fit (underflow in
  `logsumexp` raised as `FloatingPointError`). Verified by direct probe; the notebook restores numpy
  defaults immediately after the SweetViz cell and wraps the GMM fit in `np.errstate(all='ignore')`.

## 6. Cleaning, feature engineering and feature selection

**Dropped columns (justified in the notebook):** `id` (row identifier), `name` (free text, near-unique;
NLP out of scope, leakage-prone), `host_id` / `host_name` (37,457 hosts; memorisation/overfit risk).
Host scale is retained via `calculated_host_listings_count` and the host-size band. No duplicated rows
or IDs existed, so nothing was deduplicated.

**Imputation experiment** (10,000-row sample of reviewed listings, 10% of observed `reviews_per_month`
values hidden at random, seed 42):

| Method | RMSE | MAE |
|---|---|---|
| Median fill | 1.630 | 1.042 |
| MICE / IterativeImputer (Bayesian ridge, 10 iters) | **1.264** | **0.865** |

MICE recovers randomly hidden values better, as expected. Neither method was used for the actual missing
values, which are structural (true value 0 + flag). The sklearn pipelines use median imputation as the
safe default for any genuinely numeric gaps.

**Rare neighbourhoods:** 106 of 221 neighbourhoods had <30 listings; grouped into `Other (rare)`,
affecting 1,158 listings (2.4%). Levels: 221 → 116.

**Engineered features (8):** `days_since_last_review` (ref. date 2019-07-08, −1 = never reviewed;
median 50 days for reviewed listings), `is_never_reviewed`, `review_density`
(reviews per available day), `host_listing_size_band` (Single 32,301 / Small 11,792 / Medium 2,586 /
Large 2,205), `min_nights_band` (Short 12,717 / Standard 28,836 / Long 6,584 / Extended 747),
`distance_to_midtown_km` (haversine from Times Square; mean 7.11 km, max 35.90 km),
`neighbourhood_freq` (counts-only, leakage-free), `log_minimum_nights` / `log_host_listings`.

**Encoding/scaling (inside pipelines, fit on train only):** one-hot for the 4 low-cardinality
categoricals; cross-fitted `TargetEncoder` for grouped neighbourhood (leakage-safe); median imputation
+ standard scaling for numerics in linear/distance models; trees unscaled.

**Feature selection — three methods on the training split + RFE check** (encoded matrix 39,107 x 31).
Top 5 by mean rank: entire-home room type (1.0), target-encoded neighbourhood (2.7), distance to
Midtown (4.3), longitude (5.0), private-room type (6.7). Correlation alone underrates categoricals;
mutual information and RandomForest importances agree with it on the top features. RFE (ridge, top 10)
includes neighbourhood encoding, distance, all three room-type indicators and availability. Final
feature list: all 14 numeric/engineered features + 4 one-hot categoricals + target-encoded
neighbourhood; raw neighbourhood enters only via target/frequency encoding, never one-hot.

## 7. Clustering

Features (standard-scaled): log-price, latitude, longitude, availability, reviews/month,
log-minimum-nights, log-host-listings, distance to Midtown, log-review-count. Silhouette on a 5,000-row
sample (the full 48,884² distance matrix is infeasible).

| k | Inertia | Silhouette (5k sample) |
|---|---|---|
| 2 | 371,755 | 0.186 |
| **3** | **318,258** | **0.217** |
| 4 | 283,420 | 0.181 |
| 5 | 257,437 | 0.177 |
| 6 | 238,953 | 0.180 |
| 7 | 222,684 | 0.185 |
| 8 | 209,891 | 0.181 |

**Chosen k = 3** (highest silhouette; elbow flattens after k = 3–4).

| Cluster | Size (share) | Mean / median price | Mean availability | Mean reviews (rpm) | Mean dist. to Midtown | Mean min. nights | Character |
|---|---|---|---|---|---|---|---|
| 0 | 16,264 (33.3%) | $116.76 / $89 | 181.7 days | 57.6 (2.54) | 9.48 km | 2.86 | Busy, affordable, available outer-borough rooms (Brooklyn 46.4%, Queens 21.1%; 54.4% private rooms) |
| 1 | 4,319 (8.8%) | $232.03 / $159 | 279.2 days | 4.6 (0.31) | 4.37 km | 35.91 | Expensive Manhattan long-stay entire homes (73.7% Manhattan, 68.0% entire homes) |
| 2 | 28,301 (57.9%) | $161.34 / $120 | 47.8 days | 6.4 (0.37) | 6.16 km | 5.02 | Typical mid-priced core (50.0% Manhattan, 41.5% Brooklyn; 55.0% entire homes) |

PCA (2 components, 42.4% of variance) shows heavy overlap — listings form a continuum, not islands.
**GaussianMixture comparison** (k = 3, full covariance, BIC 644,819): sizes 14,259 / 19,437 / 15,188,
one group being exactly the zero-availability listings (mean availability 0.0). Agreement with KMeans
is weak (adjusted Rand index **0.147**), showing the segmentation is method-dependent and soft. KMeans
is reported as the primary segmentation for interpretability.

## 8. Anomaly detection and elimination

| Method | Flagged | Detail |
|---|---|---|
| IsolationForest (contamination 0.02, 200 trees) | 978 (2.00%) | Most anomalous: far-flung Staten Island / eastern Queens listings 20–31 km from Midtown, and extreme-review listings (up to 629 reviews). |
| IQR on raw price (fence −$90 to $334) | 2,972 (6.08%) | Too aggressive on skewed price — rejected for removal. |
| IQR on log-price (fence 2.87–6.55, ≈ $701) | 626 (1.28%) | Used in the removal rule. |
| IQR on minimum_nights (fence −5 to 11) | 6,605 (13.51%) | Useless here (discrete, spiky distribution) — not used for removal. |
| Flagged by BOTH IsolationForest and log-price IQR | 36 | Jaccard overlap only 0.023 — the methods flag different kinds of unusualness. |
| Priced ≥ $5,000/night | 26 | Includes three $10,000 and two $9,999 listings. |

**Decision:** remove listings flagged by BOTH methods OR priced ≥ $5,000 → **59 rows removed (0.12%)**;
cap `minimum_nights` at its 99th percentile (**45 nights**, 469 listings capped) instead of deleting
long-stay listings. All other flagged listings kept (unusual ≠ wrong).
**Effect:** rows 48,884 → 48,825; price mean $152.76 → $148.12; median $106 → $105; max $10,000 → $4,500.

## 9. AutoML with AutoViML (mandatory tool — actually run)

- **Version:** autoviml 0.2.2, `Auto_ViML`, `hyper_param='RS'`, `feature_reduction=True`, verbose log
  in `autoviml_log.txt`, run time 7.2 minutes. Environment fixes required: `IPython==8.26.0` (0.2.2
  fails to import under IPython 9) and an explicit test frame (calling without one crashes at the end
  of an otherwise successful run — verified in a smoke test; it tries to attach holdout predictions to
  the full training frame).
- **Compute decision (documented, with evidence):** AutoViML was fit on a random **15,000-row subsample
  of the 39,060-row training split** (seed 42); evaluation uses the **full 9,765-row holdout**. A
  full-frame run was attempted first: its linear-model randomised search (15 iterations x 5-fold CV of
  LinearSVR on 31,248 rows) ran 26+ minutes without finishing that single stage on this shared machine
  and was stopped. The sklearn baselines use the full training split, so the comparison is conservative
  against AutoViML. AutoViML's frame excludes raw `neighbourhood` (its information enters via borough,
  frequency encoding and distance); 18 features in, **16 kept** after its SULOV + XGBoost reduction —
  it dropped `reviews_per_month` and `log_host_listings`.
- **Returned model:** `LinearSVR` (tuned). Largest |coefficients| (transformed space): distance to
  Midtown 1.64, room_type 1.40, longitude 0.67, number_of_reviews 0.60.
- **Internal CV result (AutoViML's own 80/20 split of the fit sample):** linear model RMSE 0.466,
  ensemble RMSE 0.442 (log scale); best algorithm type Bagging_Regressor.

**AutoViML leaderboard — full 9,765-row holdout** (from `autoviml_leaderboard.csv`; `predictions` is
AutoViML's final output and equals the Linear row — the returned LinearSVR):

| Model | RMSE (log) | MAE (log) | R² (log) | RMSE ($) | MAE ($) | R² ($) |
|---|---|---|---|---|---|---|
| **Bagging_Regressor** | **0.424** | **0.311** | **0.602** | **125.21** | **51.44** | **0.303** |
| Ensembled | 0.439 | 0.331 | 0.572 | 129.24 | 53.15 | 0.258 |
| KNN_Regressor | 0.454 | 0.337 | 0.543 | 129.43 | 55.02 | 0.256 |
| Decision_Tree | 0.456 | 0.338 | 0.540 | 132.58 | 55.12 | 0.219 |
| predictions (= LinearSVR final) | 0.461 | 0.348 | 0.530 | 130.89 | 55.27 | 0.239 |
| Linear | 0.461 | 0.348 | 0.530 | 130.89 | 55.27 | 0.239 |
| Adaboost | 0.562 | 0.445 | 0.302 | 135.94 | 65.97 | 0.179 |

Prediction sample (first 8 holdout rows, actual → predicted $): 45→67, 55→84, 99→177, 195→233,
100→110, 125→128, 60→74, 44→49 — the linear final model over-predicts cheap listings in this sample.

## 10. Transparent sklearn baselines and combined comparison

Same 80/20 split and holdout rows as AutoViML; full training split; leakage-safe pipelines (one-hot +
cross-fitted target encoding + median impute/scale). Full table in `model_comparison.csv`.

| Source | Model | RMSE (log) | MAE (log) | R² (log) | RMSE ($) | MAE ($) | R² ($) |
|---|---|---|---|---|---|---|---|
| Baseline | **HistGradientBoosting** | **0.409** | 0.301 | **0.629** | 123.63 | 49.82 | 0.321 |
| Baseline | **RandomForest** | 0.410 | **0.299** | 0.628 | **122.19** | **49.52** | **0.337** |
| AutoViML | Bagging_Regressor | 0.424 | 0.311 | 0.602 | 125.21 | 51.44 | 0.303 |
| Baseline | Ridge (linear) | 0.436 | 0.324 | 0.579 | 129.46 | 53.15 | 0.255 |
| AutoViML | Ensembled | 0.439 | 0.331 | 0.572 | 129.24 | 53.15 | 0.258 |
| AutoViML | KNN_Regressor | 0.454 | 0.337 | 0.543 | 129.43 | 55.02 | 0.256 |
| AutoViML | Decision_Tree | 0.456 | 0.338 | 0.540 | 132.58 | 55.12 | 0.219 |
| AutoViML | predictions / Linear | 0.461 | 0.348 | 0.530 | 130.89 | 55.27 | 0.239 |
| AutoViML | Adaboost | 0.562 | 0.445 | 0.302 | 135.94 | 65.97 | 0.179 |

**Reading:** tree/ensemble methods beat linear ones, but only modestly — most of the signal is in a few
strong features every method finds. AutoViML's ensemble is competitive with, not better than, the
hand-built baselines. Typical absolute error is ~$50/night: useful for pricing guidance, not precise
enough to price an individual listing without judgement. Dollar-scale R² is much lower than log-scale
R² because a few very expensive holdout listings dominate squared dollar errors — one more reason the
log scale is the honest modelling scale.

## 11. Conclusions and limitations

1. Price is driven by what the listing is and where it is (room type, borough/neighbourhood, distance
   to Midtown); no raw numeric column exceeds |r| = 0.15 with price.
2. The cleaning was substantive: $0 prices, structural review missingness, 221 neighbourhoods, and a
   1,250-night tail each needed a justified, counted decision.
3. Listings segment into soft, overlapping groups; KMeans and GMM agree only weakly (ARI 0.147).
4. AutoViML works and is competitive (best member log RMSE 0.424) but does not beat the baselines
   (0.409–0.410); its contribution is automation.
5. **Limitations:** 2019 snapshot (pre-dates major NYC short-term-rental rule changes — do not use for
   2026 pricing); no booking/payment outcomes (listed prices only); 35.9% zero-availability (stale)
   listings; neighbourhood/borough/coordinates confound each other (importances are not causal); no
   text/photo/amenity features; the most expensive listings remain hard to predict.

## 12. Reproducibility and issues log

- **Execution verified:** `jupyter nbconvert --to notebook --execute --inplace`, exit code 0,
  42/42 code cells executed, **0 error cells**, notebook 1.88 MB (< 15 MB limit). Runtime ~23 minutes
  on this machine (profiling/SweetViz ~1.5 min, AutoViML 7.2 min, RandomForest baseline ~2.8 min).
- **Versions (pinned in `requirements.txt`):** Python 3.12.3, pandas 2.2.3, numpy 1.26.4,
  matplotlib 3.9.2, seaborn 0.13.2, scikit-learn 1.5.2, scipy 1.11.4, autoviml 0.2.2,
  ydata-profiling 4.16.1, sweetviz 2.3.1, xgboost 1.7.6, lightgbm 4.7.0, catboost 1.2.8,
  IPython 8.26.0 (see below).
- **Issues encountered and resolved (all with evidence in the notebook/log):**
  1. autoviml 0.2.2 import fails under IPython 9 → pinned IPython 8.26.0.
  2. `Auto_ViML` without an explicit test frame crashes at the end (prediction/index length mismatch)
     → explicit 80/20 split passed.
  3. SweetViz sets `np.seterr(all='raise')` globally → restored defaults; GMM wrapped in `errstate`.
     (First full execution failed on this; fixed and re-run.)
  4. Full-frame AutoViML linear search did not finish in 26+ minutes → documented 15,000-row fit
     subsample, full holdout for evaluation.
  5. `featurewiz` cannot install on Python 3.12 (requires building old pandas from source; attempted
     twice, including `--no-build-isolation`) → feature selection used correlation + mutual
     information + RandomForest importance + RFE instead.
  6. `catboost` wheel download repeatedly interrupted → installed separately with resume retries
     (1.2.8) before autoviml.
- **Disclosure:** a coding agent assisted with writing and executing this notebook, as the assignment
  allows; all numbers are outputs of executed code. See the disclosure cell in the notebook.

## 13. Files in this folder

| File | Size | Contents |
|---|---|---|
| `EDA_Airbnb_NYC_Tabular.ipynb` | 1.88 MB | Executed notebook (Colab-ready), 0 errors |
| `report.md` | — | This report |
| `README.md` | — | How to run + headline results |
| `requirements.txt` | 1.0 KB | Pinned versions that actually ran |
| `metrics.json` | 6.2 KB | Row counts, missingness, clusters, anomalies, all model metrics |
| `model_comparison.csv` | 1.5 KB | Combined AutoViML + baseline holdout metrics |
| `autoviml_leaderboard.csv` | 1.0 KB | AutoViML-only holdout metrics |
| `autoviml_log.txt` | 6.4 KB | Captured AutoViML console output |
| `autoviml_outputs/` | 4.4 MB | AutoViML's modified train/test CSVs + submission file |
| `auto_eda_profile.html` | 2.83 MB | ydata-profiling report (minimal mode) |
| `sweetviz_report.html` | 2.13 MB | SweetViz report (target: price) |
| `figures/` (17 PNGs) | ~1.2 MB | Every key chart, saved as files |
| `data/DATA_DICTIONARY.md` | 3.9 KB | Column + engineered-feature definitions |
| `data/sample_500.csv` | 72 KB | 500-row sample for a quick look (analysis uses the full download) |
| `build_notebook.py` | 81 KB | Script that generated the notebook (pre-execution text; the .ipynb is authoritative) |
| `nbconvert.log` | 0.4 KB | Execution log (exit 0) |

The full 7 MB source CSV is intentionally not committed; the notebook downloads it from a public
mirror with fallbacks.
