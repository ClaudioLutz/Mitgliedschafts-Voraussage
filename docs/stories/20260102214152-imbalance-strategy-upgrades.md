# Summary
Implemented extreme-imbalance upgrades: BalancedBagging backend, optional LightGBM GPU/CPU backend, optional beta calibration, validation-based threshold optimization, and NOGA hierarchy interaction features. Added threshold report output and diagnostics.

# Context / Problem
The 0.11% positive-rate dataset needs stronger imbalance handling, GPU-friendly boosting options, and threshold optimization beyond class weights. The existing pipeline lacked BalancedBagging, validation thresholding, and NOGA hierarchy features for industry signal.

# What Changed
- training_lead_generation_model.py: added BalancedBagging and LightGBM backends, optional beta calibration, validation threshold optimization, calibration diagnostics, thresholds.json output, and GPU-friendly XGBoost defaults.
- column_transformer_lead_gen.py: added NOGA hierarchy features and Kanton x NOGA section interaction.
- time_estimation.py: added LightGBM and BalancedBagging handling plus GPU XGBoost bin defaults.
- requirements.txt: added lightgbm and betacal dependencies.
- README.md: documented new backends and threshold output.
- tests: added feature engineering and threshold optimization coverage.

# How to Test
- `pytest`
- `MODEL_BACKEND=hgb_bagging python training_lead_generation_model.py`
- `MODEL_BACKEND=lgbm_cpu python training_lead_generation_model.py`
- `MODEL_BACKEND=lgbm_gpu python training_lead_generation_model.py`
- `CALIBRATION_METHOD=beta python training_lead_generation_model.py`

# Risk / Rollback Notes
- New dependencies may require installation; remove `lightgbm` and `betacal` from `requirements.txt` to revert.
- New NOGA features change preprocessing schema; revert `column_transformer_lead_gen.py` updates if needed.
- Threshold report output adds an extra artifact (`artifacts/thresholds.json`); safe to ignore or delete.
