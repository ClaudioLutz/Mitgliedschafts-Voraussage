# Extensive Logging and Shape Tracing Implementation

## Summary
Implemented a multi-layered logging strategy across the training pipeline (`training_lead_generation_model.py`) and feature engineering (`column_transformer_lead_gen.py`). This includes console and file logging, execution timing decorators, memory usage monitoring, and shape tracing within the preprocessor.

## Context / Problem
The training pipeline performs long-running operations (database loading, feature engineering, model training) where silent failures or hangs were difficult to diagnose. There was a need for better visibility into progress, data shapes, and memory consumption to aid in debugging and performance monitoring.

## What Changed
- **`training_lead_generation_model.py`**:
  - Replaced basic logging setup with a robust configuration that logs to both console and a timestamped file.
  - Added a `@log_execution` decorator to time functions and log output shapes/sizes.
  - Added `log_memory_usage()` helper to monitor RAM usage at key checkpoints.
  - Wrapped heavy functions (`load_modeling_data`, `load_current_snapshot`, `split_by_unique_dates`, `temporal_feature_engineer`) with `@log_execution`.
  - Increased `RandomizedSearchCV` verbosity to 3.
  - Updated logging configuration to use the root logger to capture logs from imported modules.

- **`column_transformer_lead_gen.py`**:
  - Added logging to `FeatureEngineeringTransformer` to trace input/output shapes and feature expansion.
  - Added logging to `ToFloat32Transformer` to trace matrix shape before model input.

## How to Test
1. **Run verification script**:
   ```bash
   python verify_logging.py
   ```
   This script runs a synthetic test of the pipeline components and checks for the existence and content of the log file, ensuring specific log messages (timestamps, shapes, memory usage) are present.

2. **Run the full pipeline (if DB access is available)**:
   ```bash
   python training_lead_generation_model.py
   ```
   Check the generated `training_run_YYYYMMDD_HHMM.log` file for detailed execution traces.

## Risk / Rollback Notes
- **Risk**: Increased log volume might fill disk space if run frequently without cleanup (though log files are text and relatively small).
- **Risk**: If the root logger configuration conflicts with other libraries, output might be duplicated (mitigated by checking `hasHandlers`).
- **Rollback**: Revert changes to `training_lead_generation_model.py` and `column_transformer_lead_gen.py` to their previous state using git checkout.
