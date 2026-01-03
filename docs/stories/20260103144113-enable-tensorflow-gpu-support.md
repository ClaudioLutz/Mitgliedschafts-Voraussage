# Enable TensorFlow GPU Support for DNN Backend

## Summary

Added GPU acceleration support for the DNN (Deep Neural Network) backend by upgrading from `tensorflow-cpu` to full `tensorflow` package and implementing GPU device configuration with separate `dnn_gpu` and `dnn_cpu` backend options.

## Context / Problem

The DNN backend was previously constrained to CPU-only execution using the `tensorflow-cpu` package. While XGBoost and LightGBM backends offered both GPU and CPU variants (`xgb_gpu`/`xgb_cpu`, `lgbm_gpu`/`lgbm_cpu`), the DNN backend lacked GPU acceleration capabilities, resulting in:

- Slower training times for neural network models
- Inability to leverage available GPU hardware for DNN training
- Inconsistent backend options (other backends had GPU variants, DNN did not)

## What Changed

### 1. Dependencies ([requirements-dnn.txt](requirements-dnn.txt))
- **Changed**: `tensorflow-cpu` → `tensorflow` (enables GPU support)
- Maintains existing `scikeras` and `cloudpickle` dependencies

### 2. DNN Classifier GPU Configuration ([model_backends/dnn_classifier.py](model_backends/dnn_classifier.py:67-98))
- **Added**: `use_gpu` parameter to `make_dnn_estimator()` function
- **Added**: GPU device detection and configuration logic
- **Added**: Automatic GPU memory growth configuration to prevent full GPU memory allocation
- **Added**: Graceful fallback to CPU if GPU is requested but unavailable
- **Added**: Explicit CPU-only mode when `use_gpu=False`
- **Added**: Informative logging for GPU/CPU backend selection

### 3. Main Training Script ([training_lead_generation_model.py](training_lead_generation_model.py))
- **Updated**: MODEL_BACKEND configuration comment to include `dnn_gpu` and `dnn_cpu` options (line 168-170)
- **Updated**: `get_dnn_classifier()` function to accept and forward `use_gpu` parameter (line 670)
- **Updated**: Backend selection logic to handle `dnn`, `dnn_gpu`, and `dnn_cpu` (line 1558-1573)
- **Updated**: All `MODEL_BACKEND == "dnn"` checks to `MODEL_BACKEND in ("dnn", "dnn_gpu", "dnn_cpu")` for:
  - DNN pipeline configuration (line 1558)
  - Class weight calculation (line 1772)
  - Best parameters selection (line 1799)
  - Hyperparameter search space (line 1877)
- **Added**: Automatic GPU detection with `use_gpu = MODEL_BACKEND == "dnn_gpu"` logic
- **Backward compatibility**: Legacy `dnn` backend defaults to CPU mode

### 4. GPU Readiness Check Utility ([scripts/check_tensorflow_gpu.py](scripts/check_tensorflow_gpu.py))
- **Created**: New diagnostic script to verify TensorFlow GPU configuration
- **Features**:
  - Detects available GPUs
  - Displays GPU details and CUDA/cuDNN versions
  - Tests simple GPU computation
  - Trains a minimal model on GPU to verify end-to-end functionality
  - Provides troubleshooting guidance if GPU is unavailable
  - Returns appropriate exit codes for scripting

## How to Test

### 1. Install GPU-enabled dependencies
```bash
pip install -r requirements-dnn.txt
```

### 2. Verify GPU availability
```bash
python scripts/check_tensorflow_gpu.py
```

Expected output if GPU is available:
- Lists detected GPUs
- Shows CUDA/cuDNN versions
- Reports successful GPU computation and model training

### 3. Train with GPU backend
```bash
# Use GPU acceleration
MODEL_BACKEND=dnn_gpu python training_lead_generation_model.py

# Or explicitly use CPU
MODEL_BACKEND=dnn_cpu python training_lead_generation_model.py

# Legacy mode (defaults to CPU for backward compatibility)
MODEL_BACKEND=dnn python training_lead_generation_model.py
```

### 4. Verify GPU usage in logs
Look for log messages:
- `"TensorFlow DNN backend: Using GPU acceleration (N GPU(s) detected)"` - GPU enabled
- `"TensorFlow DNN backend: Using CPU (GPU disabled)"` - CPU mode
- Warnings if GPU requested but unavailable

### 5. Run existing tests
```bash
pytest
```

All existing tests should pass (DNN tests default to CPU mode).

## Risk / Rollback Notes

### Risks

1. **Dependency size increase**: Full `tensorflow` package (~500MB) is larger than `tensorflow-cpu` (~200MB)
   - **Mitigation**: This is a conscious trade-off for GPU capability; users who don't need GPU can continue using CPU-only installations

2. **CUDA/cuDNN compatibility**: GPU functionality requires compatible CUDA toolkit and cuDNN libraries
   - **Mitigation**:
     - Graceful fallback to CPU if GPU unavailable
     - `check_tensorflow_gpu.py` diagnostic script helps verify setup
     - Detailed error messages with troubleshooting guidance

3. **GPU memory management**: Improper GPU memory configuration can cause OOM errors
   - **Mitigation**: Implemented `set_memory_growth()` to prevent full GPU memory allocation

4. **Backward compatibility**: Existing `MODEL_BACKEND=dnn` workflows might behave differently
   - **Mitigation**: Legacy `dnn` backend explicitly defaults to CPU mode; users must opt-in to GPU with `dnn_gpu`

### Rollback Procedure

If GPU support causes issues:

1. **Revert to CPU-only TensorFlow** (fastest, no code changes needed):
   ```bash
   pip uninstall tensorflow
   pip install tensorflow-cpu
   ```
   Use `MODEL_BACKEND=dnn` or `MODEL_BACKEND=dnn_cpu`

2. **Full rollback** (if code-level issues):
   ```bash
   git revert <this-commit-hash>
   pip install -r requirements-dnn.txt
   ```

3. **Emergency CPU-only mode** (force CPU without code changes):
   Set environment variable before running:
   ```bash
   CUDA_VISIBLE_DEVICES="" MODEL_BACKEND=dnn python training_lead_generation_model.py
   ```

### Performance Expectations

- **GPU mode**: 2-10x faster training for medium to large datasets (speedup varies by GPU, batch size, model architecture)
- **CPU mode**: Performance unchanged from previous implementation
- **Overhead**: GPU initialization adds ~2-5 seconds to startup time

### Compatibility Matrix

| TensorFlow Version | CUDA Version | cuDNN Version | Notes |
|-------------------|--------------|---------------|-------|
| 2.20.x | 12.x | 8.x | Current configuration |
| 2.18.x - 2.19.x | 12.x | 8.x | Compatible |
| < 2.18 | 11.x | 8.x | Requires older CUDA |

Check official compatibility: https://www.tensorflow.org/install/source#gpu
