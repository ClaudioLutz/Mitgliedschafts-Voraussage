# Implement Centralized Logging

## Summary

Created a centralized logging module (`log_utils.py`) and migrated all print statements and inconsistent logging setups across the codebase to use the unified logging infrastructure.

## Context / Problem

The codebase had inconsistent logging practices:
- Some files used `logging.basicConfig()` (e.g., `refresh_data.py`, `run_visualizations.py`)
- Some files used `logging.getLogger(__name__)` without setup (e.g., `column_transformer_lead_gen.py`)
- Many files used `print()` statements for status messages (14+ files)
- `training_lead_generation_model.py` had comprehensive logging but duplicated setup code
- No centralized configuration for log format, handlers, or file output

This made debugging difficult and log output inconsistent across pipeline components.

## What Changed

### New Module Created
- **`log_utils.py`**: Centralized logging configuration with:
  - `setup_logging()` - Configure root logger with console + file handlers
  - `get_logger(name)` - Get module-specific logger
  - `log_execution` decorator - Log function timing
  - `log_memory_usage()` - Memory checkpoint logging
  - Consistent format: `YYYY-MM-DD HH:MM:SS.mmm [LEVEL] function: message`
  - Timestamped log files: `{prefix}_YYYYMMDD_HHMM.log`

### Files Modified

| File | Changes |
|------|---------|
| `training_lead_generation_model.py` | Replaced inline logging setup with `log_utils` imports |
| `time_estimation.py` | Added logging, converted 17 print statements |
| `analyze_current_snapshot_data.py` | Added logging for status messages (kept formatted report prints) |
| `refresh_data.py` | Replaced `logging.basicConfig` with centralized setup |
| `run_visualizations.py` | Replaced `logging.basicConfig` with centralized setup |
| `column_transformer_lead_gen.py` | Updated logger import, converted `__main__` prints |
| `visualize_lead_model.py` | Added logging, converted print statements in functions |
| `two_stage_pipeline.py` | Updated logger import, converted `__main__` prints |
| `lookalike_features.py` | Updated logger import, converted `__main__` prints |
| `scripts/check_cuda_readiness.py` | Added logging setup and converted prints |
| `scripts/check_tensorflow_gpu.py` | Added logging setup and converted prints |
| `scripts/export_training_data.py` | Added logging setup and converted prints |

### Logging Conventions

- **Entry point scripts**: Call `setup_logging(log_prefix="script_name")` at module level
- **Library modules**: Use `from log_utils import get_logger; log = get_logger(__name__)`
- **Log levels**: INFO for status, WARNING for issues, ERROR for failures, DEBUG for details
- **Formatted output**: User-facing tables/reports kept as `print()` for readability

## How to Test

```bash
# Test 1: Verify log_utils imports correctly
python -c "from log_utils import setup_logging, get_logger; print('OK')"

# Test 2: Run time estimation - check log file created
python time_estimation.py

# Test 3: Run column transformer - verify logging works
python column_transformer_lead_gen.py

# Test 4: Run existing tests
pytest

# Test 5: Check log file is created
dir *.log
```

Verify:
- Log files created with timestamp in filename
- Console output shows formatted log messages
- No duplicate log messages
- Tests pass without errors

## Risk / Rollback Notes

**Low Risk**:
- All changes are additive (new module) or simple replacements
- Existing functionality unchanged
- Log output format similar to original `training_lead_generation_model.py`

**Rollback**:
- Revert all files to previous version
- Remove `log_utils.py`

**Potential Issues**:
- Scripts in `scripts/` folder add parent path to import `log_utils` - ensure working directory is correct
- If `psutil` not installed, `log_memory_usage()` silently skips (by design)
