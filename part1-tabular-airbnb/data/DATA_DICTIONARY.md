# Data Dictionary — NYC Airbnb Open Data 2019

Source: Kaggle `dgomonov/new-york-city-airbnb-open-data`, originally Inside Airbnb
(http://insideairbnb.com/get-the-data/), New York City snapshot, 2019.
File: `AB_NYC_2019.csv` — 48,895 rows x 16 columns, 7,077,973 bytes.
The notebook downloads this file from a public GitHub mirror (with fallbacks);
the full CSV is not committed to this repository.

## Raw columns

| Column | Type | Description | Missing |
|---|---|---|---|
| `id` | int | Listing identifier (unique per row). Dropped: row ID, no predictive meaning. | 0 |
| `name` | string | Listing title as written by the host. Dropped: free text, near-unique. | 16 (0.03%) |
| `host_id` | int | Host identifier (37,457 distinct hosts). Dropped: identifier, overfit risk. | 0 |
| `host_name` | string | Host display name. Dropped: identifier-like, adds nothing beyond host_id. | 21 (0.04%) |
| `neighbourhood_group` | string | Borough: Manhattan, Brooklyn, Queens, Bronx, Staten Island. | 0 |
| `neighbourhood` | string | Neighbourhood name, 221 distinct values. Rare levels (<30 listings) grouped into `Other (rare)`. | 0 |
| `latitude` | float | Listing latitude (40.50–40.91). | 0 |
| `longitude` | float | Listing longitude (−74.24–−73.71). | 0 |
| `room_type` | string | Entire home/apt, Private room, Shared room. | 0 |
| `price` | int | Nightly price in USD. **Modelling target** (as `log1p(price)`). 11 listings at $0 dropped as invalid; max $10,000. | 0 |
| `minimum_nights` | int | Minimum nights required for a booking (1–1,250; median 3). Capped at its 99th percentile in anomaly handling. | 0 |
| `number_of_reviews` | int | Total reviews received (0–629). Zero for all 10,052 never-reviewed listings. | 0 |
| `last_review` | date | Date of the most recent review (2011-03-28–2019-07-08). Missing exactly when `number_of_reviews == 0`. | 10,052 (20.56%) |
| `reviews_per_month` | float | Reviews per month over the listing's active period (0.01–58.5). Missing exactly when `number_of_reviews == 0`; filled with 0. | 10,052 (20.56%) |
| `calculated_host_listings_count` | int | Number of listings the host has (1–327; median 1). | 0 |
| `availability_365` | int | Days per year the listing is available (0–365; median 45). 35.9% of listings show 0. | 0 |

## Engineered features (added in notebook Section 5.5)

| Feature | Definition | Why |
|---|---|---|
| `days_since_last_review` | Days from `last_review` to the snapshot date (2019-07-08, the latest review in the file); −1 for never-reviewed listings. | Recency / staleness signal. |
| `is_never_reviewed` | 1 when `number_of_reviews == 0`, else 0. | Separates "no history" from "old history". |
| `review_density` | `number_of_reviews / (availability_365 + 1)`. | Demand proxy adjusted for bookable days. |
| `host_listing_size_band` | Single (1), Small (2–5), Medium (6–20), Large (21+) from `calculated_host_listings_count`. | Individual host vs professional operator. |
| `min_nights_band` | Short (1), Standard (2–7), Long (8–30), Extended (31+) from `minimum_nights`. | Captures the 30-night spike without the 1,250-night tail dominating. |
| `distance_to_midtown_km` | Haversine distance from Times Square (40.7580, −73.9855). | The smooth geographic price gradient as one number. |
| `neighbourhood_freq` | Share of all listings in the listing's (grouped) neighbourhood. | Leakage-free numeric summary of neighbourhood size (counts only). |
| `log_minimum_nights`, `log_host_listings` | `log1p` transforms of the two skewed inputs. | For linear / distance-based methods. |
| `log_price` | `log1p(price)`. | Modelling target: price is heavily right-skewed (skew 19.12 raw, 0.63 logged). |

## Small sample

A 500-row sample (`data/sample_500.csv`, same columns as the raw file) is included
for a quick look without downloading the full file. All analysis uses the full
downloaded dataset, not the sample.
