# Exploratory Data Analysis Assignment — Parts 1 & 2

**Author:** Pranith Varma Pakalapati
**Course:** CMPE 255, San José State University, Fall 2026
**Assignment:** Exploratory Data Analysis — Part 1 (manual + auto EDA, data preparation, AutoML with AutoViML) and Part 2 (agentic EDA with an AI data-analysis agent)
**Due:** Tuesday, November 17, 2026, 23:59 PT
**Submit in Canvas:** a website URL — submit the GitHub Pages site for this repo (see [Step 4](#4-submit-the-website-url-in-canvas) below). The repo itself is https://github.com/pranith27/Exploratory-data-analysis-assignment

## Video walkthroughs

Narrated walkthroughs covering the code snippets and the user journey for both parts, in detail. All three videos are built; YouTube links will be added here once the videos are uploaded:

- Part 1, tabular (Airbnb) — *CMPE 255 EDA Part 1 — NYC Airbnb: EDA, Cleaning, Clustering & AutoViML* (9:52) — YouTube link to be added
- Part 1, time series (household power) — *CMPE 255 EDA Part 1 — Household Power Time Series: EDA, Seasonality & AutoViML* (11:21) — YouTube link to be added
- Part 2, agentic EDA (AI Data Science Team) — *CMPE 255 EDA Part 2 — Agentic EDA with AI Data Science Team* (9:35) — YouTube link to be added

Each video walks through the executed notebook cell by cell: what the code does, what the output shows, and what decision it led to. The Part 2 video additionally walks through the agent user journey: prompt, plan, tool calls, generated code, executed charts, insight, and verification.

## What this repo is

Two complete EDA projects in Part 1, each on a different dataset, each following the same full pipeline — detailed manual EDA and visualisation, auto-EDA, cleaning, imputation, feature processing and selection, clustering, anomaly elimination, and AutoML with AutoViML — plus a Part 2 project that repeats EDA on the same datasets with an agentic tool instead of doing it manually. Every notebook is executed, so every number in the reports comes from code that actually ran. No placeholder results.

A coding agent assisted with building these notebooks, as the assignment allows. What the agent did, and what was checked by hand, is explained in each notebook and report. All findings below are from executed code (seed 42 throughout).

## The projects at a glance

| Part | Dataset | Size | Task | Headline result |
|---|---|---|---|---|
| 1 — Tabular | NYC Airbnb Open Data 2019 (Kaggle) | 48,895 rows × 16 columns → 48,825 modelling rows | Predict nightly price (log scale) | AutoViML Bagging_Regressor: log-RMSE 0.424, R² 0.602, $MAE 51.44; best baseline HistGradientBoosting: log-RMSE 0.409, R² 0.629 |
| 1 — Time series | UCI Individual Household Electric Power Consumption | 2,075,259 minute rows → 34,589 hourly rows → 34,421 modelling rows | Predict next-hour global active power | AutoViML Bagging_Regressor: RMSE 0.481 kW, R² 0.566; best baseline HistGradientBoosting: RMSE 0.469 kW, R² 0.587 (naive persistence: 0.593 kW) |
| 2 — Agentic | Same two datasets | 10 natural-language prompts, 20 tool calls, 1 retry | EDA via AI Data Science Team agents | 10/10 verification checks passed; agent findings match independent recomputation; see honesty note below |

Key findings in one line each:

- **Airbnb:** price is driven by where and what kind — Manhattan median $150 vs Bronx $65, entire home/apt $160 vs shared room $45 — while numeric correlations with price are all weak (|r| ≤ 0.08). The 10,052 missing review values are structural: every one belongs to a listing with zero reviews, so 0-fill (not mean-fill) is correct.
- **Household power:** a strong daily shape (20:00 peak 1.899 kW, 04:00 trough 0.444 kW), weekends heavier than weekdays (1.234 vs 1.036 kW), and December nearly triple August (1.490 vs 0.573 kW). Lag-1 power dominates every feature-selection method; same-hour Global_intensity (r = 0.999) was excluded as leakage.
- **Agentic EDA:** the supervisor pattern gets to first insight fast and leaves a fully auditable trace, but at L1/L2 (assisted, human-verified) depth — it matched manual findings on shared checks and missed the deeper steps (STL/ADF, geo mapping, AutoML search) that Part 1 covers.

## Part 1 — Tabular: NYC Airbnb (`part1-tabular-airbnb/`)

Notebook: `EDA_Airbnb_NYC_Tabular.ipynb` — 78 cells (42 code), executed, 0 errors.

- **Data:** Kaggle "New York City Airbnb Open Data" (dgomonov/new-york-city-airbnb-open-data), via a verified public mirror (see the notebook for the primary URL + two fallbacks). Diverse types: free text (`name`, `host_name`), categoricals (5 boroughs, 221 neighbourhoods, 3 room types), geo coordinates, a date (`last_review`), integers, and floats.
- **Manual EDA:** 17 saved figures — price and log-price distributions, price by borough and room type, geo scatters, reviews vs price, availability, minimum nights, correlation heatmap, missingness.
- **Auto-EDA:** ydata-profiling report + SweetViz report, both saved as HTML. The notebook compares auto vs manual findings explicitly.
- **Cleaning:** 11 invalid $0 prices dropped; structural review missingness 0-filled with a flag (asserted in code); MICE vs median imputation compared on masked values (MICE RMSE 1.264 vs median 1.630); 106 rare neighbourhoods grouped (221 → 116 levels); `minimum_nights` capped at the 99th percentile (45).
- **Feature selection:** correlation + mutual information + RandomForest importance + RFE. Top features: entire-home room type, target-encoded neighbourhood, distance to Midtown (r = −0.39 with log-price), longitude.
- **Clustering:** KMeans k = 3 chosen (silhouette 0.217, best of k = 2–8) — 16,264 busy affordable rooms (mean $117), 4,319 expensive Manhattan long-stay homes (mean $232), 28,301 mid-priced core (mean $161). GaussianMixture comparison (ARI 0.147).
- **Anomalies:** IsolationForest (978 flagged) vs IQR rules compared; only 59 rows (0.12%) removed after inspection. Price mean $152.76 → $148.12, max $10,000 → $4,500.
- **AutoML (AutoViML 0.2.2, genuinely run):** returned model LinearSVR (16/18 features kept). Because the full-frame AutoViML search did not finish on the shared machine, AutoViML ran on a documented random 15,000-row subsample of the training split (seed 42) with the full 9,765-row holdout for evaluation — stated in the notebook, report, and README. Baselines use the full split, so the comparison is conservative against AutoViML. Full leaderboard in `autoviml_leaderboard.csv` and `model_comparison.csv`.

## Part 1 — Time series: household power (`part1-timeseries-power/`)

Notebook: `EDA_Household_Power_Timeseries.ipynb` — 71 cells (42 code), executed, 0 errors.

- **Data:** UCI Individual Household Electric Power Consumption (Dec 2006–Nov 2010, minute level). Missing values are `?` strings; one file quirk found and documented: `Sub_metering_3` missing values are an empty trailing field, not `?`.
- **Grain strategy:** minute (2,075,259 rows) → hourly by mean (34,589 rows, primary modelling grain) → daily summaries (1,442 days). Every grain change and row count is logged.
- **Manual EDA:** 20 saved figures — daily series, representative week, hour-of-day profile, day-of-week/month seasonality, ACF, rolling stats, correlations, missingness over time, STL decomposition (period 7, robust). ADF test on daily data: statistic −3.666, p = 0.0046 (reject unit root).
- **Auto-EDA:** ydata-profiling report on the full hourly frame, compared with manual findings. (A second auto-EDA tool was not added: dependency conflicts with the AutoViML stack — documented in the report.)
- **Cleaning:** 25,979 missing minutes per column (1.25%), always all 7 columns together, in 71 blocks; 421 missing hours. Gap-aware imputation: 1 hour interpolated (gaps ≤ 3 h) + 420 hours hour-of-week median. Calendar + sin/cos + lag (1/2/3/24/25/168 h) + rolling features. Time-based split (last 20% as test, cutoff 13 Feb 2010), no shuffling.
- **Feature selection:** three methods, train-only. Final 18 features; lag-1 power dominates all three (r = 0.718, MI = 0.565, RF importance = 0.552).
- **Clustering:** daily 24-hour profiles, KMeans k = 2 (silhouette 0.203) — 915 lighter days (20.61 kWh/day) vs 504 heavier days (36.31 kWh/day). GMM comparison (ARI 0.498).
- **Anomalies:** IsolationForest (346 h) + rolling z-score (140 h) + IQR (738 h) compared; 66 train hours removed, test untouched. Sensor-like vs real-event-like anomalies distinguished.
- **AutoML (AutoViML 0.2.2, genuinely run, no shuffling):** chosen model LinearSVR. Best AutoViML member on the future holdout was Bagging_Regressor (RMSE 0.481 kW); the CV-chosen Linear model generalised worse (0.509 kW) — stated plainly in the report. Tuned HistGradientBoosting baseline: 0.469 kW. AutoViML's own output files are in `autoviml_output/`.

## Part 2 — Agentic EDA: AI Data Science Team (`part2-agentic-eda/`)

Notebook: `Agentic_EDA_AI_Data_Science_Team.ipynb` — 26 cells (13 code), executed, 0 errors. Code: `agents.py`. Full log: `agent_transcript.md`. Investigation: `investigation.md`. Journey: `user_journey.md`.

- **Tool chosen:** AI Data Science Team (https://github.com/business-science/ai-data-science-team), slide 226 of the course deck *"Fall 2026 cmpe 255 2. Data Preparation"* (agentic-analysis section, slides 222–228; summary slide 225). A supervisor-led multi-agent library (LangChain/LangGraph) with specialist agents for loading, cleaning, wrangling, EDA, visualisation, feature engineering, SQL, H2O AutoML, and MLflow, plus the AI Pipeline Studio low-code app.
- **Honesty note — read this first:** no LLM ran in this demo. The real library requires an LLM API key or a local Ollama server; neither was available and no key was invented (exact failure recorded in `investigation.md`). Part 2 runs a clearly labelled key-free local reproduction of the supervisor pattern: a deterministic planner replaces LLM routing, agents still generate real code, an executor runs it, the supervisor observes errors and retries once, and every number is independently re-verified. `run_real_library.py` runs the genuine library when `OPENAI_API_KEY` is set in the environment. The Pipeline Studio journey is described from the repo docs and labelled documented-not-run.
- **Demo:** 10 natural-language prompts across both datasets (profile, price drivers, missingness, cleaning, clusters/anomalies, price model, daily/weekly power pattern, gaps, trend/drivers, next-hour forecast) — 20 tool calls, 1 genuine retry (a real `TypeError` on resampling, retried successfully), 10/10 verification checks passed.
- **Comparison:** agent findings match independent recomputation on all 10 checks, and match the finished Part 1 notebooks where they measure the same thing (see `report.md` §4.1 for the full reconciliation, including where cleaning choices differ and why). The agent run missed deeper steps Part 1 covers (STL/ADF, geo mapping, AutoML search) — its strength is breadth-per-prompt and traceability, not depth.

## What's inside

```
Exploratory-data-analysis-assignment/
├── README.md                  ← this file
├── index.html                 ← submission website (GitHub Pages homepage)
├── part1-tabular-airbnb/
│   ├── EDA_Airbnb_NYC_Tabular.ipynb   executed notebook (Colab-ready)
│   ├── report.md              detailed report, real numbers only
│   ├── README.md              run instructions + headlines
│   ├── requirements.txt       pinned versions that ran
│   ├── metrics.json + model_comparison.csv + autoviml_leaderboard.csv
│   ├── auto_eda_profile.html + sweetviz_report.html
│   ├── figures/               17 charts
│   ├── data/                  data dictionary + 500-row sample (full CSV downloads from URL)
│   └── autoviml_outputs/ + autoviml_log.txt   AutoViML's own outputs
├── part1-timeseries-power/
│   ├── EDA_Household_Power_Timeseries.ipynb   executed notebook (Colab-ready)
│   ├── report.md / README.md / requirements.txt / metrics.json / model_comparison.csv
│   ├── auto_eda_profile.html
│   ├── figures/               20 charts
│   ├── data/                  dictionary + hourly_clean.csv + daily_summary.csv + predictions
│   └── autoviml_output/       AutoViML's own outputs
└── part2-agentic-eda/
    ├── Agentic_EDA_AI_Data_Science_Team.ipynb  executed notebook
    ├── agents.py              supervisor + specialist agents + executor
    ├── run_real_library.py    genuine-library driver (needs OPENAI_API_KEY)
    ├── agent_transcript.md    full prompt → plan → code → result log
    ├── investigation.md / report.md / user_journey.md / README.md
    ├── metrics.json / requirements.txt
    ├── figures/               9 agent charts + architecture diagram
    └── data/                  Airbnb CSV + hourly power CSV used by the demo
```

Notebook sizes, verified Nov 17, 2026: tabular 78 cells, time series 71 cells, agentic 26 cells — **0 error cells in any notebook**.

## How to run a notebook

### Option A — Google Colab (easiest, no install)

1. Open the notebook from its folder on GitHub.
2. Click **Open in Colab**, or paste the GitHub URL into https://colab.research.google.com/github — the notebooks are written to run top-to-bottom in Colab.
3. **Runtime → Run all.** Cell 1 installs the packages; the data cell downloads the dataset from its public URL. Seed is 42 throughout. Allow roughly 20–30 minutes for the tabular notebook (AutoViML) and a similar time for the time-series notebook on a standard Colab runtime.

Direct Colab links:

- Tabular: https://colab.research.google.com/github/pranith27/Exploratory-data-analysis-assignment/blob/main/part1-tabular-airbnb/EDA_Airbnb_NYC_Tabular.ipynb
- Time series: https://colab.research.google.com/github/pranith27/Exploratory-data-analysis-assignment/blob/main/part1-timeseries-power/EDA_Household_Power_Timeseries.ipynb
- Agentic: https://colab.research.google.com/github/pranith27/Exploratory-data-analysis-assignment/blob/main/part2-agentic-eda/Agentic_EDA_AI_Data_Science_Team.ipynb

### Option B — On your own computer

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r part1-tabular-airbnb/requirements.txt
jupyter nbconvert --to notebook --execute --inplace part1-tabular-airbnb/EDA_Airbnb_NYC_Tabular.ipynb
```

Swap in the other folders' `requirements.txt` and notebook paths for the other two projects. Notes that matter for re-runs: AutoViML 0.2.2 needs `IPython==8.26.0` (it fails to import under IPython 9) — this is pinned in both Part 1 requirements files. Part 2 needs no API key except for `run_real_library.py`.

## How this repo was pushed — step by step

### Step 1 — Repository (already created)

https://github.com/pranith27/Exploratory-data-analysis-assignment — public. This folder was pushed into it via the GitHub API.

### Step 2 — Colab links (already done)

The Colab links above point at `pranith27/Exploratory-data-analysis-assignment` on `main` — no placeholder step is needed.

### Step 3 — GitHub Pages (this creates your website URL)

1. On GitHub, open the repo → **Settings** → **Pages**.
2. Under "Build and deployment", Source: **Deploy from a branch**.
3. Branch: **main**, folder: **/ (root)**. Click **Save**.
4. Wait a minute, refresh the Pages settings page — it will show the site address:

```
https://pranith27.github.io/Exploratory-data-analysis-assignment/
```

### Step 4 — Submit the website URL in Canvas

Paste the GitHub Pages URL from Step 3 into the Canvas submission for the Exploratory Data Analysis assignment. The site links to both parts, all three notebooks, all reports, and the three walkthrough videos.

## Sources

- NYC Airbnb Open Data 2019 (Kaggle): https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data — original source Inside Airbnb: https://insideairbnb.com/get-the-data/
- UCI Individual Household Electric Power Consumption (data file used): https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip
- AutoViML: https://github.com/AutoViML/Auto_ViML — AutoViz / featurewiz / Auto_TS from the same project are referenced in the course deck
- Course deck, agentic-analysis section (slides 222–228): https://docs.google.com/presentation/d/12V4wXjPd0ymc4tcFSsQdpeQFYdtKwzQglG81nXYEO0k/
- AI Data Science Team (Part 2 tool, slide 226): https://github.com/business-science/ai-data-science-team
- ydata-profiling: https://github.com/ydataai/ydata-profiling — SweetViz: the `sweetviz` PyPI package (version 2.3.1, pinned in the tabular requirements file)
