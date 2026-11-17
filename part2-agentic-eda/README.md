# Part 2 — Agentic EDA with the AI Data Science Team

**Course:** CMPE 255, San Jose State University, Fall 2026 — Exploratory Data Analysis assignment, Part 2
**Author:** Pranith Varma Pakalapati
**Tool (slide 226):** AI Data Science Team — https://github.com/business-science/ai-data-science-team — from *"Fall 2026 cmpe 255 2. Data Preparation"* (deck: https://docs.google.com/presentation/d/12V4wXjPd0ymc4tcFSsQdpeQFYdtKwzQglG81nXYEO0k/ ; agentic-analysis section slides 222–228, summary slide 225 "Data Analysis Agents Summary").

## Honesty label

The real library needs an LLM API key (OpenAI) or a local Ollama server. Neither was available, and no key was invented or committed — the exact no-key failure is recorded in `investigation.md`. This folder therefore runs a **key-free local reproduction of the AI Data Science Team supervisor pattern**: a deterministic planner replaces LLM routing, templated code generation replaces LLM code writing, and a scikit-learn RandomForest stands in for H2O AutoML. All numbers and charts were computed by executed code and independently verified (10/10 checks). To run the **real** library, use `run_real_library.py` with `OPENAI_API_KEY` set in the environment.

## What is here

| File | What it is |
|---|---|
| `Agentic_EDA_AI_Data_Science_Team.ipynb` | Executed notebook (0 errors): 10 natural-language prompts, plans, generated-code excerpts, executed outputs, insights, verification notes, incl. a genuine error-recovery trace (P7) |
| `agents.py` | Supervisor + specialist agents + code executor + verification (the demo) |
| `run_real_library.py` | Driver for the REAL `EDAToolsAgent` / `SupervisorDSTeam` (needs `OPENAI_API_KEY` in env; documented, untested without a key) |
| `agent_transcript.md` | Full prompt → plan → tool-call → generated-code → result log with timestamps |
| `investigation.md` | Real-library investigation: install attempts, API surface, exact no-key behaviour, versions |
| `report.md` | Tool overview, findings with verified numbers, agent-vs-manual comparison, strengths/limitations, trust & guardrails |
| `user_journey.md` | Step-by-step journey with figures (§A notebook journey, actually run; §B Pipeline Studio, documented-not-run) |
| `make_architecture_figure.py` | Generates the architecture diagram |
| `requirements.txt` | Pinned dependencies |
| `metrics.json` | Prompt count, tool-call count, retries, verification results |
| `figures/*.png` | 9 agent figures + architecture diagram |
| `data/` | `AB_NYC_2019.csv` (48,895 × 16, as downloaded) + `household_power_hourly.csv` (34,589 hourly rows aggregated from the UCI minute-level file by hourly mean) |

## Headline results

- **Airbnb:** price median $106 (mean $152.72); Manhattan ($150) and Entire home/apt ($160) drive price; 10,052 missing review fields correctly 0-filled; price model R² = 0.638, MAE = $45.72.
- **Power:** peak 20:00 (1.899 kW), trough 04:00 (0.444 kW); weekend > weekday; Sub_metering_3 strongest real driver (r = 0.696); next-hour forecast MAE 0.360 kW beats naive persistence (0.396 kW).
- **Run counts:** 10 prompts, 20 tool calls, 1 retry, 10/10 verification checks passed.

## Run it (no key needed)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=600 \
  --output /tmp/executed.ipynb Agentic_EDA_AI_Data_Science_Team.ipynb
```

Seed 42 throughout. See `report.md` §7 for full reproducibility notes.
