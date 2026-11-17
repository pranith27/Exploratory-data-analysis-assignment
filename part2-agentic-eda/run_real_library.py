"""
Run the REAL AI Data Science Team library (business-science/ai-data-science-team)
against the same prompts used in the local reproduction.

REQUIREMENTS (not available in the original run environment -- see investigation.md):
  * An LLM API key in the ENVIRONMENT, e.g. OPENAI_API_KEY. Never hard-code a key,
    never commit one. This script reads ONLY from the environment and refuses to
    run if no key (and no local Ollama server) is configured.
  * pip install ai-data-science-team  (full install incl. streamlit/xgboost;
    the 0.0.0.9017 wheel also pulls large GPU deps via torch -- allow time/disk)
    OR: pip install -e . from a clone of the repo, per its README Quickstart.
  * Optional: a local Ollama server (ollama serve + ollama pull llama3.1:8b)
    if you prefer no cloud key -- set USE_OLLAMA=1.

STATUS: documented, UNTESTED without a key. The API calls below follow the
installed 0.0.0.9017 package source inspected on 2026-11-17:
  * EDAToolsAgent(model=llm).invoke_agent(user_instructions=..., data_raw=df)
  * SupervisorDSTeam(model, data_loader_agent, ..., temperature).invoke_messages(...)
  * Agents are constructed with a LangChain chat model (ChatOpenAI / ChatOllama).
Because the library is Beta (pre-0.1.0), check the repo README/examples for the
exact constructor signatures of the version you install before running.

Usage:
    export OPENAI_API_KEY="..."        # your key, your shell only
    python run_real_library.py         # EDA agent on both datasets
    python run_real_library.py --supervisor   # full SupervisorDSTeam (needs all sub-agents)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

PROMPTS = {
    "airbnb": [
        "Profile this dataset and flag data-quality issues.",
        "Which boroughs and room types drive price?",
        "Show missingness and suggest imputation.",
    ],
    "power": [
        "What is the daily and weekly pattern in power use?",
        "Profile the hourly power data, flag gaps and anomalies.",
    ],
}


def build_llm():
    """Build the chat model from the environment. No key => clear error, no run."""
    if os.environ.get("USE_OLLAMA") == "1":
        from langchain_ollama import ChatOllama

        print("Using local Ollama model llama3.1:8b (USE_OLLAMA=1).")
        return ChatOllama(model="llama3.1:8b")
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit(
            "ERROR: OPENAI_API_KEY is not set in the environment.\n"
            "Set it (export OPENAI_API_KEY=...) or run a local Ollama server and "
            "set USE_OLLAMA=1. This script never reads keys from files."
        )
    from langchain_openai import ChatOpenAI

    print("Using ChatOpenAI gpt-4.1-mini (key from OPENAI_API_KEY env var).")
    return ChatOpenAI(model="gpt-4.1-mini")


def run_eda_agent(llm):
    import pandas as pd
    from ai_data_science_team.ds_agents import EDAToolsAgent

    agent = EDAToolsAgent(model=llm)
    frames = {
        "airbnb": pd.read_csv(DATA_DIR / "AB_NYC_2019.csv"),
        "power": pd.read_csv(DATA_DIR / "household_power_hourly.csv"),
    }
    for key, prompts in PROMPTS.items():
        for prompt in prompts:
            print(f"\n===== {key}: {prompt} =====")
            agent.invoke_agent(user_instructions=prompt, data_raw=frames[key])
            print(agent.response)


def run_supervisor(llm):
    """Full supervisor team. Requires constructing every sub-agent; see the
    repo's examples/ and apps/supervisor-ds-team-app/ for the canonical
    wiring for your installed version."""
    import pandas as pd
    from langchain_core.messages import HumanMessage

    from ai_data_science_team.agents import (
        DataCleaningAgent,
        DataLoaderToolsAgent,
        DataVisualizationAgent,
        DataWranglingAgent,
        FeatureEngineeringAgent,
        SQLDatabaseAgent,
    )
    from ai_data_science_team.ds_agents import EDAToolsAgent
    from ai_data_science_team.ml_agents import H2OMLAgent, MLflowToolsAgent
    from ai_data_science_team.ml_agents.model_evaluation_agent import ModelEvaluationAgent
    from ai_data_science_team.multiagents.supervisor_ds_team import SupervisorDSTeam

    team = SupervisorDSTeam(
        model=llm,
        data_loader_agent=DataLoaderToolsAgent(model=llm),
        data_wrangling_agent=DataWranglingAgent(model=llm),
        data_cleaning_agent=DataCleaningAgent(model=llm),
        eda_tools_agent=EDAToolsAgent(model=llm),
        data_visualization_agent=DataVisualizationAgent(model=llm),
        sql_database_agent=SQLDatabaseAgent(model=llm),
        feature_engineering_agent=FeatureEngineeringAgent(model=llm),
        h2o_ml_agent=H2OMLAgent(model=llm),
        mlflow_tools_agent=MLflowToolsAgent(model=llm),
        model_evaluation_agent=ModelEvaluationAgent(model=llm),
    )
    df = pd.read_csv(DATA_DIR / "AB_NYC_2019.csv")
    team.invoke_messages(
        [HumanMessage(content="Profile this dataset and flag data-quality issues.")],
        artifacts={"data_raw": df},
    )
    print(team.response)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--supervisor", action="store_true",
                    help="run the full SupervisorDSTeam instead of the EDA agent")
    args = ap.parse_args()
    llm = build_llm()
    if args.supervisor:
        run_supervisor(llm)
    else:
        run_eda_agent(llm)


if __name__ == "__main__":
    main()
