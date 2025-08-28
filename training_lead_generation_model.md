# Training Lead Generation Model

This document explains the purpose and inner workings of
`training_lead_generation_model.py`.  The script trains a time‑aware
classifier to rank current non‑member companies by their likelihood of
joining within the next year.  It operates on historical data snapshots
to prevent temporal leakage and produces a calibrated probability for
every prospect.

## Data snapshots and labeling
- `load_modeling_data` retrieves monthly snapshots from Microsoft SQL
  Server and only keeps those whose label window is fully observable.
  Each snapshot is labeled as positive when the company becomes a member
  within the next `HORIZON_MONTHS`; companies already members or founded
  after the snapshot are excluded【F:training_lead_generation_model.py†L135-L142】【F:training_lead_generation_model.py†L179-L195】.
- `load_current_snapshot` pulls the most recent snapshot without
  labels; these records are scored after training to generate the ranked
  lead list.

## Temporal splitting
The modeling dataset is sorted by `snapshot_date` and split by unique
dates into train, validation and test segments.  The chronologically last
date acts as a hold‑out test set, ensuring that evaluation reflects a
true forward‑looking deployment.

## Feature engineering
- `temporal_feature_engineer` derives extra variables such as
  `Company_Age_Years` and binary activity flags indicating the presence
  of employees or revenue, leaving all other non‑leakage columns intact【F:training_lead_generation_model.py†L292-L313】.
- `auto_column_groups` then inspects the engineered frame and partitions
  columns into numeric, low‑cardinality categorical and high‑cardinality
  categorical groups for downstream encoding【F:training_lead_generation_model.py†L263-L287】.

## Preprocessing and imbalance handling
- Numeric features are median‑imputed.  Low‑cardinality categoricals are
  one‑hot encoded, while high‑cardinality features use cross‑fitted
  `TargetEncoder`; all transformations are combined in a single
  `ColumnTransformer` to avoid leakage【F:training_lead_generation_model.py†L520-L549】.
- Class imbalance is mitigated by either supplying
  `class_weight='balanced'` to the `HistGradientBoostingClassifier` or
  by inserting a `SMOTE` step when class weights are unavailable【F:training_lead_generation_model.py†L551-L579】.

## Hyper‑parameter search and checkpointing
A multi‑tier strategy speeds up experimentation:
1. Reuse best known parameters when available.
2. Otherwise load a previously saved `RandomizedSearchCV` checkpoint.
3. As a last resort, run a fresh random search using `TimeSeriesSplit`
   with a gap that translates calendar months into sample offsets.  The
   search optimizes both PR‑AUC and Precision@K and immediately stores a
   checkpoint of successful results【F:training_lead_generation_model.py†L583-L647】.

## Calibration and evaluation
- The selected pipeline is wrapped in an isotonic
  `CalibratedClassifierCV` whose folds also respect time order, producing
  probabilities that better reflect deployment conditions.  The
  calibrated model is persisted for reuse in `artifacts/`【F:training_lead_generation_model.py†L672-L702】.
- Evaluation on the reserved test snapshot reports PR‑AUC and
  Precision@K and exports a gains table for further analysis【F:training_lead_generation_model.py†L707-L720】.

## Scoring current prospects
The calibrated model scores the latest snapshot of non‑members, ranks
them by conversion probability, assigns deciles, and writes the results
to `outputs/ranked_leads_<timestamp>.csv`.  An optional SQL export is
sketched for integration with downstream BI or CRM systems【F:training_lead_generation_model.py†L722-L737】.

## Usage
1. Install dependencies from `requirements.txt` and configure database
   constants at the top of `training_lead_generation_model.py`.
2. Execute `python training_lead_generation_model.py`.
3. Inspect `outputs/` for the gains table and ranked leads; artifacts for
   the calibrated model and search metadata are saved under `artifacts/`.

## Rationale
By reconstructing historical snapshots and respecting chronological
order during training, validation and calibration, the approach
approximates a real production deployment.  Hyper‑parameter
checkpointing accelerates iteration, while calibrated probabilities allow
sales teams to focus on companies with the highest estimated likelihood
of joining within the configured horizon.

