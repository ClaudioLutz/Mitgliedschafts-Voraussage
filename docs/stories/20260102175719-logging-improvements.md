# Summary
Added multi-layer logging, execution timing, and memory usage reporting to improve observability during long pipeline runs.

# Context / Problem
Long-running DB loads, preprocessing, and training steps provided limited visibility and made it hard to diagnose stalls or memory pressure.

# What Changed
- `training_lead_generation_model.py` now uses a console+file logger, adds a timing decorator to heavy functions, increases CV verbosity, and logs memory checkpoints in `main()`.
- `column_transformer_lead_gen.py` logs input/output shapes for feature engineering and float32 conversion.
- `requirements.txt` now includes `psutil` for RAM monitoring.

# How to Test
- `pytest`

# Risk / Rollback Notes
- Risk: increased log volume and a new `psutil` dependency could affect runtime or environments without updated requirements.
- Rollback: revert the logging blocks and remove `psutil` from `requirements.txt`.
