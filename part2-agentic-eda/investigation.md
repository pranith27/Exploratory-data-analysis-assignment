# Investigation — AI Data Science Team (real library)

**Tool:** AI Data Science Team, https://github.com/business-science/ai-data-science-team
**Course source:** slide 226 of *"Fall 2026 cmpe 255 2. Data Preparation"* (deck: https://docs.google.com/presentation/d/12V4wXjPd0ymc4tcFSsQdpeQFYdtKwzQglG81nXYEO0k/ ; agentic-analysis section slides 222–228, summary slide 225 "Data Analysis Agents Summary").
**Investigated:** 2026-11-17, Python 3.12.3, fresh venv (`venv-part2`), by Pranith Varma Pakalapati (CMPE 255, SJSU, Fall 2026).
**Verdict:** the library is real, installs (with caveats below), imports, and exposes the supervisor + specialist-agent API described in the deck — but **it cannot run anything without an LLM**: constructing its `ChatOpenAI` model fails immediately with no API key, and no local Ollama server exists in this environment. No key was available, invented, or committed. The executed demo in this folder is therefore a labelled key-free local reproduction (`agents.py`); `run_real_library.py` is the driver for the real library when a key is available.

## 1. What the library is (from its README + installed source)

- **Status:** Beta, pre-0.1.0; breaking changes may occur until 0.1.0. License MIT. Repo created 2024-12-11; 5,511 stars / 947 forks at time of reading (2026-11-17).
- **Idea:** a Python library of specialised agents for data-science workflows, plus a flagship Streamlit app, **AI Pipeline Studio** (`streamlit run apps/ai-pipeline-studio-app/app.py`), with a pipeline-first workspace: Visual Editor, Table, Chart, EDA, Code, Model, Predictions, MLflow views; manual + AI steps with lineage and reproducible scripts; multi-dataset handling and merge workflows.
- **Requirements (README Quickstart):** Python 3.10+ and an **OpenAI API key (or Ollama for local models)**. Install: clone + `pip install -e .`.
- **LLM wiring (README):** `ChatOpenAI(model_name="gpt-4.1-mini")` from `langchain_openai`, or `ChatOllama(model="llama3.1:8b")` after `ollama serve` + `ollama pull llama3.1:8b`.

## 2. Agent inventory (verified in the installed 0.0.0.9017 package source)

Top-level exports (`ai_data_science_team/__init__.py`): `DataCleaningAgent`, `DataLoaderToolsAgent`, `DataVisualizationAgent`, `SQLDatabaseAgent`, `DataWranglingAgent`, `FeatureEngineeringAgent`, `EDAToolsAgent`, `H2OMLAgent`, `MLflowToolsAgent`, `SQLDataAnalyst`, `PandasDataAnalyst`.

| Agent (class) | Module | Role (from source/docstrings) |
|---|---|---|
| `DataLoaderToolsAgent` | `agents/data_loader_tools_agent.py` | Inspect filesystem, find/load files (`load_file`, `load_directory`, `search_files_by_pattern`, `list_directory_contents`) |
| `DataWranglingAgent` | `agents/data_wrangling_agent.py` | Join/merge, reshape, aggregate, encode (`recommend_wrangling_steps`, `create_data_wrangling_code`, `execute_data_wrangling_code`) |
| `DataCleaningAgent` | `agents/data_cleaning_agent.py` | Recommend + execute cleaning code (impute/clean, remove anomalies) |
| `EDAToolsAgent` | `ds_agents/eda_tools_agent.py` | ReAct agent over `explain_data`, `describe_dataset`, `visualize_missing`, `generate_correlation_funnel`, `generate_sweetviz_report`, `generate_dtale_report` |
| `DataVisualizationAgent` | `agents/data_visualization_agent.py` | Generate Plotly charts from instructions or default steps |
| `FeatureEngineeringAgent` | `agents/feature_engineering_agent.py` | Encoding, scaling, interactions, polynomial features |
| `SQLDatabaseAgent` | `agents/sql_database_agent.py` | Generate + execute SQL against a database connection |
| `H2OMLAgent` | `ml_agents/h2o_ml_agent.py` | H2O AutoML training/eval code generation + execution |
| `ModelEvaluationAgent` | `ml_agents/model_evaluation_agent.py` | Deterministic holdout evaluation: metrics, confusion matrix/ROC or residuals |
| `MLflowToolsAgent` | `ml_agents/mlflow_tools_agent.py` | MLflow experiments/runs/models/artifacts tools |
| `WorkflowPlannerAgent` | `agents/workflow_planner_agent.py` | Workflow planning for the supervisor team |
| `SupervisorDSTeam` | `multiagents/supervisor_ds_team.py` | LangGraph supervisor routing across 12 workers (see §3) |

## 3. Supervisor API (verified in installed source)

`SupervisorDSTeam.__init__(model, data_loader_agent, data_wrangling_agent, data_cleaning_agent, eda_tools_agent, data_visualization_agent, sql_database_agent, feature_engineering_agent, h2o_ml_agent, mlflow_tools_agent, model_evaluation_agent, workflow_planner_agent=None, checkpointer=None, temperature=1.0)`; invocation via `invoke_messages(messages, artifacts=...)` on a compiled LangGraph `StateGraph`.

The supervisor's system prompt routes on explicit intent across 12 workers: `Data_Loader_Tools_Agent`, `Data_Merge_Agent`, `Data_Wrangling_Agent`, `Data_Cleaning_Agent`, `EDA_Tools_Agent`, `Data_Visualization_Agent`, `SQL_Database_Agent`, `Feature_Engineering_Agent`, `H2O_ML_Agent`, `Model_Evaluation_Agent`, `MLflow_Logging_Agent`, `MLflow_Tools_Agent`, with the rule *"only route to workers when the user explicitly asks for their capabilities. Do not assume next steps."* State carries `messages`, shared payload slots (`data_raw`, `data_sql`, `chart_json`, `artifacts`, `errors`), and the implementation truncates conversation context (`TEAM_MAX_MESSAGES = 20`, `TEAM_MAX_MESSAGE_CHARS = 2000`) to avoid token/rate-limit errors.

`EDAToolsAgent.__init__(model, create_react_agent_kwargs={}, invoke_react_agent_kwargs={}, checkpointer=None, log_tool_calls=True)`; invocation via `invoke_agent(user_instructions=..., data_raw=df)` / `ainvoke_agent(...)`. Every agent takes a LangChain chat `model` — there is no key-free mode in the library.

## 4. Install attempts (exact outcomes)

**Attempt A — full PyPI install, `pip install ai-data-science-team`: FAILED on network, not on the package.** pip resolved the wheel (`ai_data_science_team-0.0.0.9017`, 188 kB) and began downloading the dependency tree, then failed downloading `nvidia_nccl_cu13-2.32.6` (206.6 MB, a GPU dependency pulled in via torch):

> `error: incomplete-download — Download failed after 6 attempts because not enough bytes were received (102.0 MB/206.6 MB)` — `URL: https://files.pythonhosted.org/packages/f4/80/.../nvidia_nccl_cu13-2.32.6-py3-none-manylinux_2_27_x86_64.whl`

`pip show ai-data-science-team` afterwards: "Package(s) not found". Lesson recorded: the full dependency closure is very heavy (torch/GPU wheels, streamlit, xgboost); a classroom install should budget time, disk, and a stable connection, or install from a repo clone per the README.

**Attempt B — `pip install --no-deps ai-data-science-team`: SUCCEEDED.** Installed `ai-data-science-team==0.0.0.9017` (`_version.py`: `__version__ = "0.0.0.9017"`). Declared requirements: `langchain, langchain-ollama, langchain_community, langchain_experimental, langchain_openai, langgraph, numpy, openai, openpyxl, pandas, plotly, psutil, scikit-learn, sqlalchemy, streamlit, streamlit-flow-component, xgboost`.

**Attempt C — import with `--no-deps` only: FAILED (expected).** `import ai_data_science_team` → `ModuleNotFoundError: No module named 'langchain_core'` (from `agents/data_cleaning_agent.py`).

**Attempt D — install LangChain stack only** (`langchain-core`, `langchain-openai`, `langchain-ollama`, `langgraph`, `langchain-community`, `langchain-experimental`, `openai`, `plotly`, `sqlalchemy`, `openpyxl`, `psutil`, then `langchain>=1.0.0`): **IMPORT SUCCEEDED.** `import ai_data_science_team` → OK; `from ai_data_science_team.ds_agents import EDAToolsAgent` and `from ai_data_science_team.multiagents.supervisor_ds_team import SupervisorDSTeam` → OK. Remaining resolver warnings only: `streamlit`, `streamlit-flow-component`, `xgboost` not installed (needed for the Pipeline Studio app, not for importing agents). Installed versions: langchain 1.5.0, langchain-core 1.8.1, langgraph 1.2.22, langchain-openai 1.8.0, langchain-ollama 1.1.0, openai 3.53.0, pandas 3.1.1, scikit-learn 1.9.1.

## 5. Behaviour without an API key (exact, reproduced 2026-11-17)

Environment had **no** `OPENAI_API_KEY` and **no** `ANTHROPIC_API_KEY` set; no `ollama` binary installed. Test script constructed the model exactly as the README shows:

```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4.1-mini")
```

**Result — construction fails immediately, before any agent is built:**

> `openai.OpenAIError: Missing credentials. Please pass an 'api_key', 'workload_identity', 'admin_api_key', or set the 'OPENAI_API_KEY' or 'OPENAI_ADMIN_KEY' environment variable.`

Consequences: `EDAToolsAgent(model=llm)` could not even be constructed (no `llm` object exists), so no agent invocation was possible. No network call to OpenAI was attempted; the failure is a local credential check. The Ollama alternative was also unavailable: no Ollama server/binary in the environment, and `ChatOllama(model="llama3.1:8b")` invocation failed in this sandboxed environment (`httpx.InvalidURL` from proxy URL parsing) before reaching any model. **No key was requested from the user mid-task and none was fabricated** — per the assignment setup, the honest path was to record this and build the labelled local reproduction.

## 6. What was NOT run (documented-not-run)

- AI Pipeline Studio UI (`streamlit run apps/ai-pipeline-studio-app/app.py`) — needs the full install (streamlit) + an LLM key; described from README/docs only in `user_journey.md` §B.
- Any real `EDAToolsAgent` / `SupervisorDSTeam` invocation — needs a key (see §5).
- H2O AutoML (`H2OMLAgent`) and MLflow tracking — need a Java/H2O server and/or MLflow server plus a key; the demo substitutes a scikit-learn RandomForest and labels it as a stand-in.
- Sweetviz / dtale EDA reports — tools referenced by `EDAToolsAgent`; not installed or run here.

## 7. Reproduce this investigation

```bash
python -m venv venv-part2 && source venv-part2/bin/activate
pip install --no-deps ai-data-science-team   # 0.0.0.9017
pip install "langchain>=1.0.0" langchain-core langchain-openai langchain-ollama \
  langgraph langchain-community langchain-experimental openai plotly sqlalchemy openpyxl psutil
python -c "import ai_data_science_team; print('IMPORT OK')"
env -u OPENAI_API_KEY python -c "from langchain_openai import ChatOpenAI; ChatOpenAI(model='gpt-4.1-mini')"
# -> openai.OpenAIError: Missing credentials. ...
```

With a key, run the real library via `run_real_library.py` (key from environment only).
