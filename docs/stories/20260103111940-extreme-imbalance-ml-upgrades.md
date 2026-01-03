# Extreme Imbalance ML Pipeline Upgrades

## Summary

Implemented comprehensive ML pipeline improvements for extreme class imbalance (0.11% positive rate) including: calibrated stacking ensemble, two-stage filter-then-rank pipeline, LambdaMART ranking, lookalike feature engineering (K-Prototypes clusters, FAISS KNN propensity), enhanced temporal features, and Precision@K optimization scorers.

## Context / Problem

The lead generation model faces extreme class imbalance (~14K positives in 1.2M samples, 0.11% positive rate). Research indicates that combining multiple imbalance-handling strategies outperforms any single approach. The implementation plan specified five key improvement areas:

1. **Ensemble combination strategies** - Combining cost-sensitive methods with resampling provides complementary error patterns
2. **Lookalike feature engineering** - KNN propensity and cluster conversion rates capture non-linear relationships trees may miss
3. **Two-stage modeling** - Filter-then-rank architecture matches business goal of identifying top 10K-50K leads
4. **Precision@K optimization** - Direct optimization of business-relevant metric
5. **Temporal features** - Company lifecycle effects and B2B seasonal patterns

## What Changed

### New Model Backends (`training_lead_generation_model.py`)

- **`stacking`** - Calibrated stacking ensemble combining:
  - HGB with `class_weight='balanced'` + isotonic calibration
  - XGBoost with `scale_pos_weight` + isotonic calibration
  - BalancedBagging with DecisionTree base + isotonic calibration
  - LogisticRegression meta-learner with `class_weight='balanced'`

- **`two_stage`** - Filter-then-rank pipeline:
  - Stage 1: High-recall filter (95% target recall) using out-of-fold predictions
  - Stage 2: Precision-focused ranker trained on filtered data (~1-2% positive rate)
  - Prevents data leakage via OOF predictions for filtering

- **`lambdamart`** - Learning-to-rank backend:
  - LightGBM LambdaMART objective for direct ranking optimization
  - `LambdaMARTWrapper` class for sklearn classifier API compatibility

### New Modules

- **`lookalike_features.py`** - Similarity-based feature engineering:
  - `ClusterConversionRateFeatures`: K-Prototypes clustering for mixed data with Bayesian-smoothed conversion rates
  - `FAISSConverterSimilarity`: FAISS-based KNN propensity scores (distance to converters)
  - `LookalikeFeatureTransformer`: Combined transformer for easy pipeline integration

- **`two_stage_pipeline.py`** - Two-stage architecture:
  - `TwoStagePipeline`: Base implementation with OOF prediction filtering
  - `TwoStageWithDifferentFeatures`: Variant supporting different feature sets per stage
  - `create_two_stage_pipeline()`: Factory function for common configurations

### Enhanced Temporal Features (`training_lead_generation_model.py:420-480`)

- `Company_Age_Log` - Log-transformed age (handles outliers, diminishing lifecycle effects)
- `Company_Age_Bucket` - Categorical buckets: startup, early_growth, established, mature, legacy
- `Month_Sin`, `Month_Cos` - Cyclical month encoding (preserves Dec-Jan continuity)
- `Is_End_Of_Quarter` - Peak B2B purchasing periods (months 3, 6, 9, 12)
- `Is_Summer` - Swiss summer slowdown (Jul-Aug)
- `Quarter` - Quarter of year (1-4)

### Enhanced Scorers (`training_lead_generation_model.py:447-491`)

- `recall_at_k()` - Fraction of positives captured in top-K
- `lift_at_k()` - Precision@K divided by baseline rate
- `recall_at_k_scorer_factory()`, `lift_at_k_scorer_factory()` - For hyperparameter tuning

### Column Transformer Updates (`column_transformer_lead_gen.py`)

- Added temporal features to `NUMERIC_COLS`: `Company_Age_Log`, `Month_Sin`, `Month_Cos`, `Is_End_Of_Quarter`, `Is_Summer`, `Quarter`
- Added `Company_Age_Bucket` to `LOW_CARD_CATEGORICAL_COLS`

### Dependencies (`requirements.txt`)

- Added `kmodes>=0.12.2` - K-Prototypes clustering for mixed categorical/numeric data
- Added `faiss-cpu>=1.7.0` - Fast approximate nearest neighbor search

### Tests (`tests/test_new_features.py`)

- `TestTemporalFeatures` - Company age, buckets, seasonal encoding validation
- `TestPrecisionAtK` - Scorer calculation and factory integration tests
- `TestTwoStagePipeline` - Basic fit/predict, threshold finding, factory tests
- `TestLookalikeFeatures` - Cluster and FAISS transformer tests
- `TestStackingEnsemble` - Ensemble creation and fit/predict tests
- `TestLambdaMARTWrapper` - Ranker wrapper functionality tests
- `TestColumnTransformerUpdates` - Schema validation for new features

## How to Test

### Run all tests
```bash
pytest tests/test_new_features.py -v
```

### Test specific backends
```bash
# Stacking ensemble
MODEL_BACKEND=stacking python -c "from training_lead_generation_model import get_stacking_ensemble; print(get_stacking_ensemble())"

# Two-stage pipeline
python -c "from two_stage_pipeline import create_two_stage_pipeline; print(create_two_stage_pipeline())"

# LambdaMART
MODEL_BACKEND=lambdamart python -c "from training_lead_generation_model import get_lambdamart_ranker, LambdaMARTWrapper; print(LambdaMARTWrapper(get_lambdamart_ranker()))"
```

### Test lookalike features (requires kmodes/faiss)
```bash
python lookalike_features.py
```

### Test two-stage pipeline
```bash
python two_stage_pipeline.py
```

### Validate syntax
```bash
python3 -m py_compile training_lead_generation_model.py
python3 -m py_compile column_transformer_lead_gen.py
python3 -m py_compile lookalike_features.py
python3 -m py_compile two_stage_pipeline.py
```

### Full pipeline test with new backend
```bash
MODEL_BACKEND=stacking pytest tests/test_pipeline_smoke.py -v
```

## Risk / Rollback Notes

### Risks

1. **New dependencies** (`kmodes`, `faiss-cpu`):
   - Both are optional - graceful degradation if not installed
   - FAISS may have platform-specific installation issues on Windows
   - Mitigation: Features return dummy values when dependencies unavailable

2. **Stacking ensemble training time**:
   - 3 base models + calibration + stacking significantly increases training time
   - Mitigation: Use `MODEL_BACKEND=hgb` (default) for faster iterations

3. **Two-stage pipeline complexity**:
   - OOF predictions add memory overhead
   - Stage 2 has reduced sample size after filtering
   - Mitigation: Configurable `stage1_target_recall` to adjust filter aggressiveness

4. **Temporal features schema change**:
   - Adds 7 new columns to feature matrix
   - Mitigation: Column transformer handles missing columns gracefully

### Rollback

1. **Revert to previous backend**: Set `MODEL_BACKEND=hgb` (default)
2. **Remove new dependencies**: Comment out `kmodes` and `faiss-cpu` in requirements.txt
3. **Disable temporal features**: Modify `temporal_feature_engineer()` to return original columns only
4. **Full rollback**: Revert this commit

### Backward Compatibility

- All existing backends (`hgb`, `hgb_bagging`, `xgb_cpu`, `xgb_gpu`, `lgbm_cpu`, `lgbm_gpu`, `dnn`) unchanged
- Default backend remains `hgb`
- New features are additive to existing column lists
- Existing tests continue to pass
