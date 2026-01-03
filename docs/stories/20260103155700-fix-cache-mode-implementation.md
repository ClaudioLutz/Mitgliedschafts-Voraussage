# Fix Cache Mode Issues in Training Pipeline

## Summary

Fixed missing `pickle` import and incomplete cache mode implementation in `training_lead_generation_model.py`. Cache mode now properly skips database operations while still completing training workflow.

## Context / Problem

When running the training pipeline with `use_cache=True`, multiple errors occurred:

1. **NameError**: `pickle` module was never imported despite being used for cache serialization
2. **UnboundLocalError**: `engine` variable was referenced but never created in cache mode
3. **Logic error**: Code attempted to load current snapshot and write to SQL even in offline cache mode

The cache mode was intended to support offline training but was incomplete.

## What Changed

### `training_lead_generation_model.py`:

1. **Added pickle import** (line 18)
   * Required for cache serialization/deserialization

2. **Restructured data loading section** (lines 1506-1523)
   * Set `engine = None` explicitly in cache mode
   * Moved `load_current_snapshot()` call inside the `else` block (non-cache mode only)
   * Cleaned up logging flow

3. **Updated feature engineering** (line 1540)
   * Added conditional: `df_curr_eng = temporal_feature_engineer(df_current) if df_current is not None else None`

4. **Wrapped scoring and export section** (lines 2101-2125)
   * Added outer condition: `if df_current is not None and df_curr_eng is not None:`
   * Added nested condition for SQL export: `if engine is not None:`
   * Added informative log message when skipping scoring in cache mode

## How to Test

Run the training pipeline with cached data:

```bash
wsl bash -c "cd '/mnt/c/Lokal_Code/Mitgliedschafts Voraussage/Mitgliedschafts Voraussage' && source venv_wsl/bin/activate && MODEL_BACKEND=dnn_gpu python3 -c 'from training_lead_generation_model import main; main(use_cache=True)'"
```

**Expected behavior:**
* Successfully loads cached training data
* Completes full training workflow (preprocessing, model training, calibration, evaluation)
* Skips current snapshot loading and scoring (logs warning message)
* Does not attempt database operations

## Risk / Rollback Notes

* **Risk:** Low - changes are defensive and maintain backward compatibility
* **Non-cache mode:** Completely unaffected; all database operations remain identical
* **Cache mode:** Now functional where it previously crashed
* **Rollback:** Revert the 4 changed sections if issues arise
* **Impact:** Enables offline training workflow for development/testing without database access
