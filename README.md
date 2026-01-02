# Mitgliedschafts-Voraussage

Lead generation pipeline that predicts membership conversion for Swiss companies. It trains on historical snapshots from SQL Server and ranks current non-members by probability of joining within a configurable horizon.

## Highlights
- Time-aware snapshot labeling and leakage-safe training
- Centralized preprocessing in `column_transformer_lead_gen.py` with explicit feature engineering and target encoding
- Time series cross-validation and probability calibration
- Ranked lead exports, gains table, and persisted model artifacts
- Optional visualization and dashboard utilities

## Repository layout
- `training_lead_generation_model.py` - main training, evaluation, and ranking workflow
- `column_transformer_lead_gen.py` - canonical preprocessing and feature engineering
- `time_estimation.py` - runtime estimate using the same preprocessing pipeline
- `run_visualizations.py` / `visualize_lead_model.py` - diagnostic plots and reports
- `analyze_current_snapshot_data.py` - descriptive analysis of the current snapshot
- `tools/benchmark_enhanced_sampling.py` - manual sampling benchmark (CPU and memory heavy)
- `tools/manual_preprocessor_check.py` - manual preprocessor validation
- `refresh_data.py` / `flask_seaborn.py` / `templates/` / `static/` - optional SHAB dashboard demo
- `tests/` - pytest smoke tests
- `docs/` - design, evaluation, deployment, and roadmap docs
- `outputs/`, `artifacts/`, `figures/` - generated results

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
   - `pip install -r requirements-dev.txt`
3. Optional DNN backend:
   - `pip install -r requirements-dnn.txt`
4. Install Microsoft ODBC Driver 17 for SQL Server and ensure access to the target DB (Windows integrated auth by default).

## Configuration
Update the constants near the top of these scripts as needed:
- `training_lead_generation_model.py` (primary)
- `time_estimation.py`
- `run_visualizations.py`

Key settings:
- `SERVER`, `DATABASE`, `SCHEMA` - SQL Server connection
- `HORIZON_MONTHS` - label window in months
- `LEAD_CAPACITY_K` - top-K precision target
- `N_ITER`, `N_SPLITS`, `CAL_SPLITS` - search and calibration budgets
- `SAMPLE_TRAINING_DATA`, `MAX_TRAINING_SAMPLES`, `USE_BUSINESS_LOGIC_SAMPLING`
- `MODEL_BACKEND` (env var) - `hgb`, `xgb_cpu`, `xgb_gpu`, `dnn`

## Pipeline at a glance
1. Load modeling snapshots with complete label windows.
2. Apply `temporal_feature_engineer` to add `Company_Age_Years`.
3. Use `create_lead_gen_preprocessor` for feature engineering and encoding.
4. Train with time-aware CV and calibrate probabilities.
5. Evaluate on the most recent snapshot and export the gains table.
6. Score current prospects and export ranked leads.

Imbalance handling uses `class_weight` when available (scikit-learn >= 1.5); otherwise it falls back to SMOTE (requires `imbalanced-learn`).

## Usage

### Train and rank leads
`python training_lead_generation_model.py`

Outputs:
- `outputs/ranked_leads_YYYYMMDD_HHMM.csv`
- `outputs/gains_table_test.csv`
- `artifacts/best_pipeline.joblib`, `artifacts/best_params.joblib`, `artifacts/search_metadata.joblib`
- `artifacts/calibrated_model.joblib`

Note: the script also writes to SQL table `mitgliederstatistik.lead_generation_rankings` with `if_exists="replace"`.

### Estimate runtime
`python time_estimation.py`

### Visualizations
`python run_visualizations.py`

Requires trained artifacts and database access. SHAP plots are skipped if `shap` is not installed.

### Analyze current snapshot
`python analyze_current_snapshot_data.py`

### Manual tools
- `python tools/manual_preprocessor_check.py`
- `python tools/benchmark_enhanced_sampling.py` (CPU and memory heavy)

### Optional SHAB dashboard demo
1. `python refresh_data.py`
2. `python flask_seaborn.py`
3. Open `http://localhost:5000`

## Testing
`pytest`

Tests are restricted to `tests/` via `pytest.ini`.

## Documentation
- `docs/Model_Development_Implementation.md`
- `docs/Data_Collection_and_Enrichment.md`
- `docs/Evaluation_and_Validation.md`
- `docs/Deployment_and_Infrastructure.md`
- `docs/User_Interfaces_and_Reporting.md`
- `docs/future_plans.md`

## Troubleshooting
- `pyodbc` driver errors: install Microsoft ODBC Driver 17 for SQL Server.
- SQL connection failures: verify `SERVER`, `DATABASE`, `SCHEMA`, and Windows auth permissions.
- Out-of-memory during training: lower `MAX_TRAINING_SAMPLES` or disable `USE_BUSINESS_LOGIC_SAMPLING`.
- Missing SHAP plots: `pip install shap`.

## License
License not specified in this repository.
