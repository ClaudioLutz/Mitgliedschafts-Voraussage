# Mitgliedschafts-Voraussage

**Lead Generation ML Pipeline for Membership Conversion Prediction**

A production-grade machine learning pipeline that predicts membership conversion for Swiss companies. The system trains on historical SQL Server snapshots and ranks current non-members by their probability of joining within a configurable time horizon.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Advanced Features](#advanced-features)
  - [Lookalike Modeling](#lookalike-modeling)
  - [Two-Stage Pipeline](#two-stage-pipeline)
  - [NOGA Hierarchy Features](#noga-hierarchy-features)
  - [Advanced Calibration](#advanced-calibration)
- [Model Backends](#model-backends)
- [Testing](#testing)
- [Recent Enhancements](#recent-enhancements)
- [Documentation](#documentation)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This pipeline addresses the challenge of identifying high-potential leads from a large pool of non-member companies. By analyzing historical membership patterns and company attributes, the model learns which characteristics are predictive of future membership conversion.

### Business Value

- **Prioritized Lead Lists**: Sales teams receive ranked prospects ordered by conversion probability
- **Resource Optimization**: Focus outreach efforts on companies most likely to convert
- **Data-Driven Decisions**: Replace intuition-based targeting with statistical evidence
- **Calibrated Probabilities**: Output scores reflect true conversion likelihood
- **Advanced Pattern Recognition**: Lookalike modeling identifies similar companies to past converters
- **Handles Extreme Imbalance**: Specialized techniques for conversion rates as low as 0.11%

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Time-Aware Training** | Snapshot-based labeling prevents temporal data leakage |
| **Leakage-Safe Design** | Strict separation of membership outcome fields from features |
| **Centralized Preprocessing** | Single source of truth for feature engineering in `column_transformer_lead_gen.py` |
| **Multiple Model Backends** | Support for HistGradientBoosting, BalancedBagging (HGB), LightGBM (CPU/GPU), XGBoost (CPU/GPU), and Deep Neural Networks |
| **Advanced Calibration** | Isotonic and optional Beta calibration for extreme imbalance scenarios |
| **Validation-Based Thresholds** | Automatic threshold optimization on validation data with diagnostics |
| **NOGA Hierarchy Features** | Automatically extracts section/division/group from industry codes and creates cross-features with cantons |
| **Lookalike Modeling** | K-Prototypes clustering and FAISS-based similarity features for capturing non-linear patterns |
| **Two-Stage Pipeline** | Optional filter-then-rank architecture for extreme class imbalance |
| **Time-Series Cross-Validation** | Respects chronological order during hyperparameter tuning |
| **Automated Checkpointing** | Saves search results to accelerate subsequent runs |
| **Memory Optimization** | Stratified sampling for large datasets with distribution preservation |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  SQL Server (PRODSVCREPORT70)                                               │
│  └── CAG_Analyse.mitgliederstatistik.MitgliederSegmentierung               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PREPROCESSING LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  column_transformer_lead_gen.py                                             │
│  ├── Temporal Feature Engineering (Company_Age_Years)                       │
│  ├── Missing Indicators & PLZ Grouping                                      │
│  ├── NOGA Industry Hierarchy (Section, Division, Group, Kanton interactions)│
│  ├── Log Transforms for Skewed Features                                     │
│  ├── Numeric: Median Imputation + Power Transform                           │
│  ├── Low-Cardinality: One-Hot Encoding (rare category grouping)             │
│  ├── High-Cardinality: Target Encoding (cross-fitted)                       │
│  └── Optional Lookalike Features (K-Prototypes + FAISS KNN)                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MODEL LAYER                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  Backend Options:                                                           │
│  ├── HGB: HistGradientBoostingClassifier (default)                         │
│  ├── HGB_Bagging: BalancedBagging with HGB (extreme imbalance)             │
│  ├── LGBM: LightGBM (CPU or GPU accelerated)                               │
│  ├── XGB: XGBoost (CPU or GPU accelerated)                                 │
│  └── DNN: Deep Neural Network (TensorFlow/SciKeras)                        │
│                                                                             │
│  Training Pipeline:                                                         │
│  ├── Time-Series Cross-Validation (chronological splits)                   │
│  ├── Randomized Hyperparameter Search with checkpointing                   │
│  ├── Class Imbalance Handling (class_weight, BalancedBagging, or SMOTE)    │
│  ├── Advanced Calibration (Isotonic or Beta)                               │
│  └── Validation-Based Threshold Optimization                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          OUTPUT LAYER                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  outputs/                                                                   │
│  ├── ranked_leads_YYYYMMDD_HHMM.csv    (Scored & ranked prospects)         │
│  └── gains_table_test.csv              (Model performance by decile)       │
│                                                                             │
│  artifacts/                                                                 │
│  ├── calibrated_model.joblib           (Production-ready model)            │
│  ├── best_pipeline.joblib              (Best hyperparameters)              │
│  ├── search_metadata.joblib            (Search history & versioning)       │
│  └── thresholds.json                   (Threshold & calibration diagnostics)│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### System Requirements

- **Python**: 3.9 or higher
- **Operating System**: Windows (recommended for SQL Server integration), Linux, or macOS
- **Memory**: 16 GB RAM minimum (32 GB recommended for full dataset)
- **Database Access**: SQL Server with Windows Authentication

### Database Driver

Install Microsoft ODBC Driver 17 for SQL Server:

**Windows:**
```powershell
# Download from Microsoft and run installer
# https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server
```

**Linux (Ubuntu/Debian):**
```bash
curl https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
curl https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/prod.list | sudo tee /etc/apt/sources.list.d/mssql-release.list
sudo apt-get update
sudo ACCEPT_EULA=Y apt-get install -y msodbcsql17
```

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd mitgliedschafts-voraussage
```

### 2. Create Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

### 3. Install Dependencies

**Core dependencies:**
```bash
pip install -r requirements.txt
```

**Development dependencies (for testing):**
```bash
pip install -r requirements-dev.txt
```

**Optional DNN backend:**
```bash
pip install -r requirements-dnn.txt
```

---

## Configuration

Configuration is managed through constants at the top of the main scripts. Key settings include:

### Database Connection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SERVER` | `PRODSVCREPORT70` | SQL Server hostname |
| `DATABASE` | `CAG_Analyse` | Database name |
| `SCHEMA` | `mitgliederstatistik` | Schema containing the data |

### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `HORIZON_MONTHS` | `12` | Prediction window (months until conversion) |
| `LEAD_CAPACITY_K` | `1000` | Top-K leads for precision metric |
| `N_ITER` | `10` | Hyperparameter search iterations |
| `N_SPLITS` | `4` | Time-series CV folds |
| `CAL_SPLITS` | `3` | Calibration CV folds |

### Memory Optimization

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SAMPLE_TRAINING_DATA` | `True` | Enable stratified sampling |
| `MAX_TRAINING_SAMPLES` | `2,500,000` | Maximum training samples |
| `USE_BUSINESS_LOGIC_SAMPLING` | `True` | Preserve rare valuable cases |

### Model Backend Selection

Set via environment variable:
```bash
# Options: hgb (default), hgb_bagging, lgbm_cpu, lgbm_gpu, xgb_cpu, xgb_gpu, dnn
export MODEL_BACKEND=xgb_gpu
```

### Advanced Features

| Parameter | Default | Description |
|-----------|---------|-------------|
| `USE_LOOKALIKE_FEATURES` | `False` | Enable K-Prototypes + FAISS similarity features |
| `USE_TWO_STAGE_PIPELINE` | `False` | Enable filter-then-rank architecture |
| `USE_BETA_CALIBRATION` | `False` | Use Beta calibration instead of Isotonic |

---

## Usage

### Training and Generating Leads

Run the complete pipeline:
```bash
python training_lead_generation_model.py
```

**Outputs:**
- `outputs/ranked_leads_YYYYMMDD_HHMM.csv` — Ranked prospect list
- `outputs/gains_table_test.csv` — Model performance metrics
- `artifacts/calibrated_model.joblib` — Serialized production model
- `artifacts/best_params.joblib` — Optimal hyperparameters
- `artifacts/thresholds.json` — Thresholds and calibration diagnostics
- `training_run_YYYYMMDD_HHMM.log` — Detailed training log
- SQL table `mitgliederstatistik.lead_generation_rankings` — Database export

### Estimating Runtime

Before running the full pipeline:
```bash
python time_estimation.py
```

### Generating Visualizations

After training:
```bash
python run_visualizations.py
```

**Generated plots** (in `figures/`):
- Feature importance (transformed and aggregated)
- SHAP summary and waterfall plots
- ROC curve with AUC
- Confusion matrix
- Calibration curve
- Gains and lift charts

### Analyzing Current Data

Descriptive statistics of the prospect pool:
```bash
python analyze_current_snapshot_data.py
```

### Optional: SHAB Dashboard Demo

```bash
python refresh_data.py
python flask_seaborn.py
# Open http://localhost:5000
```

---

## Project Structure

```
mitgliedschafts-voraussage/
│
├── training_lead_generation_model.py   # Main training pipeline
├── column_transformer_lead_gen.py      # Preprocessing & feature engineering
├── lookalike_features.py               # K-Prototypes + FAISS similarity features
├── two_stage_pipeline.py               # Filter-then-rank architecture
├── time_estimation.py                  # Runtime estimation utility
├── run_visualizations.py               # Visualization runner
├── visualize_lead_model.py             # Plotting functions
├── analyze_current_snapshot_data.py    # Data analysis script
│
├── model_backends/
│   ├── __init__.py
│   └── dnn_classifier.py               # Deep learning backend
│
├── tools/
│   ├── manual_preprocessor_check.py    # Preprocessor validation
│   └── benchmark_enhanced_sampling.py  # Sampling benchmark (CPU-intensive)
│
├── tests/
│   ├── test_pipeline_smoke.py          # Core pipeline tests
│   ├── test_xgb_pipeline.py            # XGBoost backend tests
│   ├── test_dnn_pipeline.py            # DNN backend tests
│   ├── test_new_features.py            # Lookalike/two-stage tests
│   └── verify_preprocessor_updates.py  # Preprocessor validation
│
├── docs/
│   ├── Model_Development_Implementation.md
│   ├── Feature_Engineering.md
│   ├── Evaluation_and_Validation.md
│   ├── Deployment_and_Infrastructure.md
│   ├── User_Interfaces_and_Reporting.md
│   ├── future_plans.md
│   └── stories/                        # Change documentation
│
├── deprecated/                         # Legacy code (archived)
│
├── outputs/                            # Generated results (gitignored)
├── artifacts/                          # Model artifacts (gitignored)
├── figures/                            # Visualization outputs (gitignored)
│
├── requirements.txt                    # Core dependencies
├── requirements-dev.txt                # Development dependencies
├── requirements-dnn.txt                # DNN backend dependencies
├── pytest.ini                          # Test configuration
└── AGENTS.md                           # AI agent guidelines
```

---

## Advanced Features

### Lookalike Modeling

The pipeline supports advanced similarity-based features to capture non-linear patterns:

**K-Prototypes Clustering** ([lookalike_features.py](lookalike_features.py)):
- Clusters companies using mixed categorical/numeric data
- Computes historical conversion rates per cluster with Bayesian smoothing
- Adds `cluster_conversion_rate` and `cluster_id` features

**FAISS KNN Similarity** ([lookalike_features.py](lookalike_features.py)):
- Fast approximate nearest neighbor search using FAISS
- Computes distance-based features to known converters
- Adds `mean_dist_converters`, `min_dist_converter`, `max_dist_converter`, `std_dist_converters`

**Installation:**
```bash
pip install kmodes faiss-cpu  # or faiss-gpu for GPU support
```

**Usage:**
Set `USE_LOOKALIKE_FEATURES=True` in [training_lead_generation_model.py](training_lead_generation_model.py) or configure via environment.

### Two-Stage Pipeline

For extreme class imbalance scenarios, the two-stage architecture improves performance:

**Stage 1 - Filter**: High-recall model (e.g., Logistic Regression) filters out obvious non-converters
- Uses out-of-fold predictions to prevent data leakage
- Targets 95% recall by default
- Reduces candidate pool to ~15-20% of data

**Stage 2 - Rank**: Precision-focused model (e.g., HGB) trains on filtered candidates
- Better class balance (~1-2% vs 0.11%)
- Can use expensive computed features
- Optional isotonic calibration

**Usage:**
Set `USE_TWO_STAGE_PIPELINE=True` in [training_lead_generation_model.py](training_lead_generation_model.py).

See [two_stage_pipeline.py](two_stage_pipeline.py) for implementation details.

### NOGA Hierarchy Features

Automatically extracts hierarchical industry classification features:

- **NOGA_section**: Single-digit section code (e.g., "C" for Manufacturing)
- **NOGA_division**: Two-digit division code
- **NOGA_group**: Three-digit group code
- **Kanton_NOGA_section**: Cross-feature combining canton and industry section

These features capture industry patterns at multiple granularities and regional industry interactions.

Implemented in [column_transformer_lead_gen.py](column_transformer_lead_gen.py:95-118).

### Advanced Calibration

**Beta Calibration**: For extreme imbalance scenarios where isotonic calibration may be unstable
- Fits a Beta distribution to calibrate probabilities
- More robust than isotonic for sparse positive classes
- Requires `betacal` package

**Usage:**
```bash
pip install betacal
```
Set `USE_BETA_CALIBRATION=True` in configuration.

**Validation-Based Thresholds**: Automatically optimizes classification threshold on validation data
- Computes optimal threshold for precision/recall trade-offs
- Saves diagnostics to `artifacts/thresholds.json`
- Reports include: threshold, precision, recall, F1, and calibration metrics

---

## Model Backends

### HistGradientBoosting (Default)

```bash
export MODEL_BACKEND=hgb
python training_lead_generation_model.py
```

- Native scikit-learn implementation
- Handles missing values natively
- Class weight support (sklearn ≥ 1.5)

### HistGradientBoosting BalancedBagging

```bash
export MODEL_BACKEND=hgb_bagging
python training_lead_generation_model.py
```

- BalancedBagging ensemble with HGB base estimator
- Undersamples the majority class per bag for extreme imbalance

### LightGBM CPU

```bash
export MODEL_BACKEND=lgbm_cpu
python training_lead_generation_model.py
```

- LightGBM gradient boosting with CPU backend
- Supports scale_pos_weight or is_unbalance

### LightGBM GPU

```bash
export MODEL_BACKEND=lgbm_gpu
python training_lead_generation_model.py
```

- GPU-accelerated LightGBM training
- Memory-optimized bins for 4GB VRAM

### XGBoost CPU

```bash
export MODEL_BACKEND=xgb_cpu
python training_lead_generation_model.py
```

- Optimized histogram-based algorithm
- Automatic class imbalance handling via `scale_pos_weight`

### XGBoost GPU

```bash
export MODEL_BACKEND=xgb_gpu
python training_lead_generation_model.py
```

- CUDA-accelerated training
- Requires NVIDIA GPU with CUDA support
- Verify GPU support: `python scripts/check_cuda_readiness.py`

### Deep Neural Network

```bash
pip install -r requirements-dnn.txt
export MODEL_BACKEND=dnn
python training_lead_generation_model.py
```

- TensorFlow/Keras backend via SciKeras
- Early stopping with validation monitoring
- Configurable architecture (hidden units, dropout, L2 regularization)

---

## Testing

Run the test suite:
```bash
pytest
```

Run with verbose output:
```bash
pytest -v
```

Run specific test files:
```bash
pytest tests/test_pipeline_smoke.py
pytest tests/test_xgb_pipeline.py
```

Test DNN backend (requires DNN dependencies):
```bash
pip install -r requirements-dnn.txt
MODEL_BACKEND=dnn pytest tests/test_dnn_pipeline.py
```

---

## Recent Enhancements

The pipeline has undergone significant improvements to handle extreme class imbalance:

| Release | Key Improvements |
|---------|------------------|
| **Latest (Jan 2026)** | Temporal features, lookalike modeling (K-Prototypes + FAISS), two-stage pipeline architecture |
| **Dec 2025** | BalancedBagging backend, LightGBM GPU/CPU support, Beta calibration, validation-based threshold optimization, NOGA hierarchy interaction features |
| **Nov 2025** | Deep Neural Network backend support via TensorFlow/SciKeras |
| **Oct 2025** | Forced hyperparameter search, comprehensive logging and monitoring |
| **Sep 2025** | XGBoost GPU acceleration support |

See [docs/stories/](docs/stories/) for detailed change documentation.

---

## Documentation

| Document | Description |
|----------|-------------|
| [Model Development](docs/Model_Development_Implementation.md) | Training pipeline details |
| [Feature Engineering](docs/Feature_Engineering.md) | Feature creation and preprocessing |
| [Evaluation & Validation](docs/Evaluation_and_Validation.md) | Metrics and validation strategy |
| [Deployment](docs/Deployment_and_Infrastructure.md) | Production deployment guide |
| [User Interfaces](docs/User_Interfaces_and_Reporting.md) | Reporting and output formats |
| [Future Plans](docs/future_plans.md) | Roadmap and enhancement ideas |

### Change Documentation

All functional changes are documented in `docs/stories/` following the format:
```
docs/stories/YYYYMMddHHmmss-topic-of-change.md
```

---

## Troubleshooting

### Database Connection Issues

**Error:** `pyodbc.Error: ... Driver not found`

**Solution:** Install Microsoft ODBC Driver 17 for SQL Server (see [Prerequisites](#prerequisites))

---

**Error:** SQL connection timeout or authentication failure

**Solution:** 
1. Verify `SERVER`, `DATABASE`, and `SCHEMA` constants
2. Confirm Windows Authentication permissions
3. Test connection with SQL Server Management Studio

---

### Memory Issues

**Error:** `MemoryError` or system slowdown during training

**Solution:**
1. Reduce `MAX_TRAINING_SAMPLES` in configuration
2. Disable `USE_BUSINESS_LOGIC_SAMPLING`
3. Use `MODEL_BACKEND=xgb_gpu` for GPU offloading

---

### Missing Visualizations

**Error:** SHAP plots not generated

**Solution:**
```bash
pip install shap
```

---

### DNN Backend Errors

**Error:** `ModuleNotFoundError: No module named 'tensorflow'`

**Solution:**
```bash
pip install -r requirements-dnn.txt
```

---

### Checkpoint Compatibility

**Warning:** `Checkpoint sklearn version differs from current`

**Solution:** Delete old checkpoints and re-run:
```bash
rm -rf artifacts/*.joblib
python training_lead_generation_model.py
```

---

## Contributing

Contributions are welcome. Please follow these guidelines:

1. **Read** `AGENTS.md` for development standards
2. **Create a branch** for your feature or fix
3. **Add tests** for any functional changes
4. **Document changes** in `docs/stories/` (see format in AGENTS.md)
5. **Run tests** before submitting: `pytest`
6. **Submit a pull request** with a clear description

### Code Style

- Follow existing patterns in the codebase
- Use type hints where practical
- Add logging for significant operations
- Prefer explicit over implicit

---

## License

License not specified. Please contact the repository maintainers for licensing information.

---

<div align="center">

**Built for Swiss Association Lead Generation**

</div>
