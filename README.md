# Agentic-AutoML-Model-Lifecycle-Platform
End-to-end machine learning automation on tabular datasets using LangGraph, specialized agents, model optimization, explainability, and monitoring.

---

## Project Structure

```text
Agentic AutoML and Model Lifecycle Plateform /
│
├── agents/          # Specialized ML agents and workflow orchestration
├── api/             # FastAPI inference interface
├── config.py        # Configuration and pipeline settings
├── data/            # Dataset and data-processing utilities
├── models/          # Model training, selection, and optimization
├── monitoring/      # Drift detection and model monitoring
├── reports/         # Generated ML reports and experiment outputs
├── tests/           # Automated tests
├── requirements.txt
└── README.md
```
---

## Workflow
```text
Dataset
   ↓
EDA Agent
   ↓
Leakage Detection
   ↓
Feature Engineering
   ↓
Data Splitting
   ↓
Model Selection
   ↓
Hyperparameter Optimization
   ↓
Model Evaluation
   ↓
Explainability / Ensemble
   ↓
Final Model
   ↓
Inference & Monitoring
```
---

## Setup
```bash
pip install -r requirements.txt
```
---
## Usage
### 1. Run the ML Pipeline
```bash
python <pipeline_entry_file>.py
```
### 2. Launch the Streamlit Interface
```bash
streamlit run <streamlit_entry_file>.py
```
### 3. Start the Inference API
```bash
uvicorn <api_module>:app --reload
```

## Agentic ML Strategy

| Stage   | Agent / Component | Responsibility                                            |
| ------- | ----------------- | --------------------------------------------------------- |
| Phase 1 | EDA Agent         | Dataset profiling and exploratory analysis                |
| Phase 2 | Leakage Agent     | Detect potentially problematic or target-leaking features |
| Phase 3 | Feature Agent     | Feature engineering and preprocessing                     |
| Phase 4 | Split Agent       | Determine appropriate data splitting strategy             |
| Phase 5 | Model Agent       | Train and compare candidate ML models                     |
| Phase 6 | Optimization      | Hyperparameter tuning using Optuna                        |
| Phase 7 | Evaluation Agent  | Evaluate and compare model performance                    |
| Phase 8 | Orchestrator      | Coordinate iterations and select the next workflow step   |

## Key Design Decisions

- **Multi-Agent Architecture** — Different stages of the ML workflow are handled by specialized agents instead of a single monolithic pipeline.
- **LLM + Deterministic Execution** — Agents can generate analysis or ML actions, while the actual computation is executed using Python and established ML libraries.
- **Leakage Detection** — The pipeline explicitly checks for potentially problematic features before model training.
- **Automated Model Selection** — Multiple machine learning algorithms can be trained and compared automatically.
- **Hyperparameter Optimization** — Optuna is used to search for improved model configurations.
- **Explainability** — SHAP-based analysis provides insight into important model features and predictions.
- **Iterative Orchestration** — The workflow can route back to earlier stages when additional optimization is required.
- **Model Monitoring** — Statistical drift detection helps identify changes in incoming data.
- **Retraining Support** — Detected distribution changes can trigger a model retraining workflow.
- **Modular Structure** — Individual agents and ML components are separated, making the system easier to extend and maintain.

## Results

| Component           | Capability                         |
| ------------------- | ---------------------------------- |
| EDA                 | Automated dataset analysis         |
| Feature Engineering | Automated feature transformation   |
| Model Selection     | Multiple ML algorithms             |
| Optimization        | Optuna-based hyperparameter tuning |
| Explainability      | SHAP-based model interpretation    |
| Ensemble            | Candidate model combination        |
| Monitoring          | Statistical data-drift detection   |
| Retraining          | Automated retraining workflow      |
| API                 | FastAPI inference                  |
| Interface           | Streamlit-based interaction        |

## Technologies Used

- Python
- LangGraph
- Large Language Models (LLMs)
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Optuna
- SHAP
- Streamlit
- FastAPI
