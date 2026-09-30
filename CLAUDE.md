# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the app
streamlit run streamlit_app.py

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_ml_helpers.py

# Run a single test by name
pytest tests/test_ml_helpers.py::test_function_name -v

# Install deep learning dependencies (optional, ~600 MB)
pip install -r requirements-dl.txt
```

## Configuration

`config.yaml` is the base config. Override locally (gitignored) via `config.local.yaml` or env vars:

```
AUTOML_<SECTION>_<KEY>=value   # e.g. AUTOML_PIPELINE_MAX_RETRIES=5
OPENAI_API_KEY=sk-...          # required for LLM agents
OPENAI_MODEL=gpt-4o-mini       # optional model override
ORCHESTRATOR_MODEL=gpt-4o      # override for orchestrator specifically
```

Access config in code via `from utils.config_loader import cfg; cfg("pipeline.max_retries")`.

## Architecture

### LangGraph Pipeline

The pipeline is defined in `agents/pipeline.py` and compiled with SQLite checkpointing (`db/automl_history.db`). Flow:

```
eda_agent → hitl_eda → leakage_agent → hitl_leakage → feature_agent → hitl_features
→ split_agent → hitl_split → hitl_model_selection → model_agent → hitl_models
→ eval_agent → ensemble_agent → hitl_ensemble → orchestrator_agent → hitl_loop
→ (accept → END) or (retry_features → feature_agent) or (retry_models → model_agent)
```

HITL nodes are pass-throughs (`return {}`) where LangGraph pauses via `interrupt_before`. The Streamlit UI resumes the graph after user approval. Every agent node has a conditional error-check edge that routes to `pipeline_error` if `_agent_error` is set.

### State Split: Two Layers

**LangGraph checkpointed state** (`agents/state.py` — `PipelineState` TypedDict): Only JSON-serialisable types (str, int, float, bool, list, dict, None). DataFrames are stored as base64-encoded Parquet strings (`df_parquet_b64`, `df_engineered_parquet_b64`). Sklearn models/arrays are stored as base64-joblib strings with `_` prefix (e.g. `_best_model_b64`).

**Runtime store** (`ui/runtime_store.py` — `RuntimeStore`): Thread-safe LRU+TTL (20 sessions, 1h TTL) for heavy non-serialisable objects (numpy arrays, fitted sklearn objects). Accessed by thread ID (`tid`). Use `rt.get_key(tid, key)` / `rt.update(tid, {...})`.

### UI State

`ui/state_store.py` provides `AppState` (a dataclass, not TypedDict) as the single source of truth for the Streamlit UI. Access via `get_store()` and `update_store()`. Phase navigation is validated in `streamlit_app.py:_last_valid_phase()` against `_compute_phase_gates()` from `ui/sidebar.py`.

Pages route by `st.session_state["phase"]`: fusion → ingest → eda → distributions → features → split → model_selection → models → eval → export → compare/run_compare.

### Agent Conventions

Every agent function:
1. Is decorated with `@agent_error_handler("Agent Name")` (from `utils/agent_utils.py`)
2. Returns a `dict` that goes through `sanitize_for_msgpack()` automatically (the decorator does this) — this converts numpy scalars to native Python types for LangGraph's msgpack checkpointer
3. Appends progress strings to `agent_messages` (capped at 200 entries via `_capped_add`)

LLM calls use `call_llm_json()` from `utils/agent_utils.py`, which has built-in exponential backoff (3 attempts, 1/2/4s delays).

### Serialization

- DataFrame ↔ `utils/serialization.py`: `df_to_b64()` / `b64_to_df()` (Parquet)
- Sklearn objects ↔ `df_to_b64` variant using joblib (for export only; never store in graph state directly)
- Always call `sanitize_for_msgpack(result)` on agent return dicts if not using the decorator

### Key Utilities

- `utils/column_intelligence.py` — LLM explains every column; `validate_target()` blocks unusable targets
- `utils/meta_memory.py` — records completed runs; provides warm-start hints for model selection
- `utils/ml_helpers.py` — preprocessing, Optuna tuning, SHAP computation
- `utils/advanced_features.py` — feature engineering (interactions, polynomial, target encoding)
- `utils/drift_detector.py` — PSI-based feature drift detection
- `utils/autoeda_report.py` — generates HTML AutoEDA report
- `script_exporter.py` — generates a self-contained reproducible Python script from a completed pipeline run
- `utils/code_executor.py` — sandboxed code execution (used by NL query panel)

### Database

`db/session_store.py` manages the SQLite DB path resolution and `init_db()`. The LangGraph checkpointer and session history both use this DB.
