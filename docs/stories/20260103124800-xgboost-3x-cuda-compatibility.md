# Summary
Updated XGBoost GPU backend to use modern XGBoost 3.x syntax (`device='cuda'` instead of deprecated `tree_method='gpu_hist'`), restoring CUDA GPU acceleration compatibility.

# Context / Problem
XGBoost 3.x removed the `gpu_hist` tree method in favor of a unified `device` parameter. The existing `xgb_gpu` backend implementation used the legacy `tree_method='gpu_hist'` syntax, causing GPU training to fail with:
```
Invalid Input: 'gpu_hist', valid values are: {'approx', 'auto', 'exact', 'hist'}
```

This prevented users with CUDA-capable GPUs from using GPU acceleration despite having the correct drivers and CUDA toolkit installed.

# What Changed
- **`training_lead_generation_model.py`**: Updated `get_xgb_classifier()` function (lines 643-656) to use `device='cuda'` with `tree_method='hist'` for the `xgb_gpu` backend instead of the deprecated `tree_method='gpu_hist'`.
- **`scripts/check_cuda_readiness.py`**: Updated CUDA verification script (lines 13-17) to test GPU support using the modern XGBoost 3.x syntax and added explanatory comment about the API change.

The fix maintains all existing GPU-specific optimizations:
- `max_bin=64` (reduced from 256 for GPU memory efficiency)
- `sampling_method='gradient_based'` (GPU-friendly sampling)

# How to Test
1. Verify CUDA readiness:
   ```bash
   python scripts/check_cuda_readiness.py
   ```
   Expected output: `SUCCESS: XGBoost trained with GPU support!`

2. Run training with GPU backend:
   ```cmd
   set MODEL_BACKEND=xgb_gpu
   python training_lead_generation_model.py
   ```

3. Monitor GPU utilization during training:
   ```bash
   nvidia-smi -l 1
   ```
   Verify GPU utilization increases to 50-100% and memory usage grows.

4. Run existing tests to ensure CPU backend unaffected:
   ```bash
   pytest tests/test_xgb_pipeline.py -v
   ```

# Risk / Rollback Notes
**Risk**: None. This is a compatibility fix for XGBoost 3.x. The new syntax is backward compatible with XGBoost 2.x (which also supports `device` parameter).

**Rollback**: If issues arise, revert to the old syntax:
```python
# Old (XGBoost < 3.0)
return XGBClassifier(tree_method="gpu_hist", **common_params)

# New (XGBoost >= 3.0)
return XGBClassifier(tree_method="hist", device="cuda", **common_params)
```

**Compatibility Note**: The codebase now requires XGBoost >= 3.0 for GPU support. CPU backends (`xgb_cpu`) remain unaffected by this change.
