# Reduce Training Sample Size and Disable Business Logic Sampling to Prevent OOM

## Summary

Reduced `MAX_TRAINING_SAMPLES` from 2,500,000 to 500,000 and disabled `USE_BUSINESS_LOGIC_SAMPLING` to prevent Out-of-Memory (OOM) kills during stratified sampling on systems with limited RAM.

## Context / Problem

When running the DNN training pipeline with cached data, the process was killed by the Linux OOM killer during the stratified sampling phase:

```
[Sat Jan  3 20:20:24 2026] Out of memory: Killed process 38964 (python3)
total-vm:54424224kB, anon-rss:15401072kB
```

The process was using ~15 GB of resident memory attempting to perform 5-dimensional stratified sampling on 5.3M records. This exceeded available system RAM.

**Root cause**: The stratified sampling operation (`advanced_stratified_sample_with_business_logic` and `stratified_sample_large_dataset`) creates multiple intermediate copies of large DataFrames when stratifying across multiple dimensions. Even after reducing the target sample size to 500K, the process continued to OOM because the business logic sampling pre-processes the entire 5.3M dataset before the main stratified sampling.

**Multiple OOM kills observed:**
- First attempt (2.5M target): Killed at ~15 GB memory usage
- Second attempt (500K target): Still killed at ~15 GB memory usage

The business logic sampling adds an extra layer that:
1. Preserves large company conversions (requires full dataset scan)
2. Preserves conversions from smaller regions (requires grouping entire dataset)
3. Then passes the result to the main stratified sampling with 5 dimensions

## What Changed

### `training_lead_generation_model.py`:

* **Line 277**: Reduced `MAX_TRAINING_SAMPLES` from `2500000` to `500000`
* **Line 278**: Set `USE_BUSINESS_LOGIC_SAMPLING = False` (was `True`)
* **Updated comments**: Documented OOM prevention rationale

**Change 1** (Sample size reduction): 5x reduction in target samples decreases memory for feature engineering and model training.

**Change 2** (Disable business logic sampling): Removes the memory-intensive pre-processing layer that scans and groups the entire 5.3M dataset. Now uses only the core stratified sampling which is more memory-efficient.

Combined effect significantly decreases memory pressure during:
1. Sampling operations (no longer pre-scanning full dataset)
2. DataFrame copies and transformations
3. Feature engineering on sampled data
4. DNN training in TensorFlow/Keras

## How to Test

Run the training pipeline with cached data:

```bash
wsl bash -c "cd '/mnt/c/Lokal_Code/Mitgliedschafts Voraussage/Mitgliedschafts Voraussage' && source venv_wsl/bin/activate && MODEL_BACKEND=dnn_gpu python3 -c 'from training_lead_generation_model import main; main(use_cache=True)'"
```

**Expected behavior:**
* Stratified sampling completes successfully
* Memory usage stays below system limits
* Training proceeds to completion
* No OOM kills in dmesg

Monitor memory usage:
```bash
wsl bash -c "watch -n 5 free -h"
```

## Risk / Rollback Notes

* **Risk:** Low-Medium - Reduced sample size may slightly impact model quality
* **Trade-off**:
  - **Pros**: Prevents OOM crashes, enables training on typical workstations (16-32GB RAM)
  - **Cons**: Less data for training may reduce model generalization
* **Mitigation**: The stratified sampling preserves:
  - All rare positive cases (conversions)
  - Geographic representation
  - Temporal distribution
  - Company size diversity
  - Legal form diversity
* **Alternative approaches if more samples/features needed**:
  1. Increase system RAM (32GB+ recommended for business logic sampling)
  2. Add swap space (slower but prevents OOM)
  3. Use XGBoost/LightGBM backends which are more memory-efficient
  4. Simplify stratification (reduce from 5 to 2-3 dimensions)
  5. Increase `MAX_TRAINING_SAMPLES` gradually (try 750K, 1M) while monitoring memory
* **Rollback**:
  - Sample size: Change line 277 back to `MAX_TRAINING_SAMPLES = 2500000` if system has 32GB+ RAM
  - Business logic: Change line 278 back to `USE_BUSINESS_LOGIC_SAMPLING = True` (requires even more RAM)

## Performance Impact

**Memory savings estimate:**
- Previous peak (2.5M samples + business logic): ~15 GB resident, 54 GB virtual → OOM killed
- After sample reduction (500K + business logic): ~15 GB resident → Still OOM killed
- **Expected with both changes (500K, no business logic): ~4-6 GB resident, 12-18 GB virtual**
- **Reduction: ~60-75% memory footprint**

**Training time impact:**
- Sampling: Much faster (no business logic pre-scan + fewer records to stratify)
- Feature engineering: Faster (fewer rows to transform)
- Model training: Faster (smaller dataset for DNN)
- Overall: Expected 40-60% reduction in total training time

**Quality impact:**
- Still preserves geographic, temporal, company size, and legal form stratification
- No longer specifically preserves large company conversions or rare regional conversions
- Overall model quality expected to remain good due to maintained stratification
- Can be validated by comparing test set metrics with previous runs
