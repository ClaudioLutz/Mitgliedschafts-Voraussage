# time_estimation.py
# Estimate total runtime for your lead-gen training job by timing ONE fit on a small sample
# and extrapolating to (n_iter * n_splits + 1 refit + cal_splits).

import time
import math
import urllib.parse
import pandas as pd

from packaging.version import Version
from sqlalchemy import create_engine, URL, text

# --- Centralized logging
from log_utils import setup_logging, get_logger
setup_logging(log_prefix="time_estimate")
log = get_logger(__name__)

# --- scikit-learn bits
from sklearn import __version__ as sklearn_version
import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier

try:
    from xgboost import XGBClassifier
    HAVE_XGBOOST = True
except ImportError:
    HAVE_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAVE_LIGHTGBM = True
except ImportError:
    HAVE_LIGHTGBM = False

try:
    from imblearn.ensemble import BalancedBaggingClassifier
    HAVE_IMBLEARN_ENSEMBLE = True
except Exception:
    HAVE_IMBLEARN_ENSEMBLE = False

from column_transformer_lead_gen import create_lead_gen_preprocessor, DROP_COLS, ToFloat32Transformer
from training_lead_generation_model import temporal_feature_engineer

# ----------------- CONFIG -----------------
SERVER = "PRODSVCREPORT70"
DATABASE = "CAG_Analyse"
SCHEMA = "mitgliederstatistik"

# Model Backend Configuration
# Options: 'hgb' (default), 'hgb_bagging', 'xgb_gpu', 'xgb_cpu', 'lgbm_gpu', 'lgbm_cpu', 'dnn'
MODEL_BACKEND = os.environ.get("MODEL_BACKEND", "hgb").lower()

HORIZON_MONTHS = 12
SAMPLE_FRACTION = 0.02     # 2% probe (adjust if you want faster/slower)
N_ITER = 5                # matches your training
N_SPLITS = 4               # time-series CV folds in your training
CAL_SPLITS = 3             # calibration folds
# -----------------------------------------

def make_engine(server: str, database: str):
    # SQLAlchemy + pyodbc with URL.create and odbc_connect
    odbc = urllib.parse.quote_plus(
        "Driver=ODBC Driver 17 for SQL Server;"
        f"Server={server};"
        f"Database={database};"
        "Trusted_Connection=Yes;"
    )
    return create_engine(URL.create("mssql+pyodbc", query={"odbc_connect": odbc}))

def count_modeling_rows(engine, horizon_months: int) -> int:
    # Count rows in the same modeling CTE used by training (snapshots <= today - horizon)
    q = f"""
WITH snapshots AS (
    SELECT DATEADD(MONTH, -n, GETDATE()) AS snapshot_date
    FROM (VALUES (36),(30),(24),(18),(12)) t(n)
),
base AS (
    SELECT
        a.CrefoID, a.Gruendung_Jahr, a.Eintritt, a.Austritt
    FROM {DATABASE}.{SCHEMA}.MitgliederSegmentierung a
),
modeling AS (
    SELECT
        b.CrefoID,
        s.snapshot_date,
        CASE WHEN b.Eintritt IS NOT NULL
                  AND b.Eintritt >= s.snapshot_date
                  AND b.Eintritt < DATEADD(MONTH, {horizon_months}, s.snapshot_date)
             THEN 1 ELSE 0 END AS Target
    FROM base b
    CROSS JOIN snapshots s
    WHERE s.snapshot_date <= DATEADD(MONTH, -{horizon_months}, GETDATE())
      AND (b.Eintritt IS NULL OR b.Eintritt >= s.snapshot_date)
      AND (b.Gruendung_Jahr IS NULL OR b.Gruendung_Jahr <= YEAR(s.snapshot_date))
)
SELECT COUNT_BIG(*) AS nrows FROM modeling;
"""
    return int(pd.read_sql_query(text(q), engine)["nrows"][0])

def load_modeling_sample(engine, horizon_months: int, n_sample: int) -> pd.DataFrame:
    # Pull a random sample of modeling rows with needed columns for feature pipelines
    # NOTE: ORDER BY NEWID() is acceptable here because n_sample is small and we only run once.
    q = f"""
WITH snapshots AS (
    SELECT DATEADD(MONTH, -n, GETDATE()) AS snapshot_date
    FROM (VALUES (36),(30),(24),(18),(12)) t(n)
),
base AS (
    SELECT
        a.CrefoID,
        a.Name_Firma,
        a.Gruendung_Jahr,
        a.PLZ,
        a.Kanton,
        a.Rechtsform,
        a.BrancheText_06,
        a.BrancheCode_06,
        a.MitarbeiterBestand,
        a.MitarbeiterBestandKategorie,
        a.Umsatz,
        a.UmsatzKategorie,
        a.UmsatzKategorieOrder,
        a.Risikoklasse,
        a.Ort,
        a.RechtsCode,
        a.GroessenKategorie,
        a.V_Bestand_Kategorie,
        a.MitarbeiterBestandKategorieOrder,
        a.BrancheCode_02,
        a.BrancheCode_04,
        a.BrancheText_02,
        a.BrancheText_04,
        a.Eintritt,
        a.Austritt
    FROM {DATABASE}.{SCHEMA}.MitgliederSegmentierung a
),
modeling AS (
    SELECT
        b.*,
        s.snapshot_date,
        CASE WHEN b.Eintritt IS NOT NULL
                  AND b.Eintritt >= s.snapshot_date
                  AND b.Eintritt < DATEADD(MONTH, {horizon_months}, s.snapshot_date)
             THEN 1 ELSE 0 END AS Target
    FROM base b
    CROSS JOIN snapshots s
    WHERE s.snapshot_date <= DATEADD(MONTH, -{horizon_months}, GETDATE())
      AND (b.Eintritt IS NULL OR b.Eintritt >= s.snapshot_date)
      AND (b.Gruendung_Jahr IS NULL OR b.Gruendung_Jahr <= YEAR(s.snapshot_date))
)
SELECT TOP ({n_sample}) *
FROM modeling
ORDER BY NEWID();  -- random sample
"""
    return pd.read_sql_query(text(q), engine, parse_dates=["snapshot_date", "Eintritt", "Austritt"])

def main():
    log.info("Connecting to DB...")
    engine = make_engine(SERVER, DATABASE)

    log.info("Counting modeling rows (snapshots with complete labels)...")
    n_total = count_modeling_rows(engine, HORIZON_MONTHS)
    if n_total == 0:
        log.error("No modeling rows found (check DB/table names and date ranges).")
        raise SystemExit("No modeling rows found (check DB/table names and date ranges).")
    n_sample = max(2_000, int(math.ceil(SAMPLE_FRACTION * n_total)))
    log.info(f"Total modeling rows: {n_total:,}  |  Sampling: {n_sample:,} rows (~{100*SAMPLE_FRACTION:.1f}%)")

    log.info("Loading modeling SAMPLE from DB... (one-time)")
    df = load_modeling_sample(engine, HORIZON_MONTHS, n_sample)
    df = df.sort_values("snapshot_date").reset_index(drop=True)
    df = temporal_feature_engineer(df)

    y = df["Target"].to_numpy()
    feature_cols = [col for col in df.columns if col not in DROP_COLS]
    X = df[feature_cols].copy()

    # Configure Pipeline based on backend
    if MODEL_BACKEND.startswith("xgb"):
        log.info(f"Estimating time for backend: {MODEL_BACKEND.upper()}")
        if not HAVE_XGBOOST:
            raise RuntimeError("XGBoost not installed.")

        pre = create_lead_gen_preprocessor(onehot_sparse=True)
        to_float = ToFloat32Transformer()

        tree_method = "gpu_hist" if MODEL_BACKEND == "xgb_gpu" else "hist"
        max_bin = 64 if MODEL_BACKEND == "xgb_gpu" else 256

        clf_params = {
            "tree_method": tree_method,
            "n_estimators": 100, # Conservative for timing
            "learning_rate": 0.1,
            "max_depth": 6,
            "objective": "binary:logistic",
            "random_state": 42,
            "max_bin": max_bin,
            "n_jobs": 1, # Single thread estimate often safer for extrapolation
        }
        if MODEL_BACKEND == "xgb_gpu":
            clf_params["sampling_method"] = "gradient_based"
        clf = XGBClassifier(**clf_params)
        pipe = Pipeline([("pre", pre), ("to_float", to_float), ("clf", clf)])

    elif MODEL_BACKEND.startswith("lgbm"):
        log.info(f"Estimating time for backend: {MODEL_BACKEND.upper()}")
        if not HAVE_LIGHTGBM:
            raise RuntimeError("LightGBM not installed.")

        pre = create_lead_gen_preprocessor(onehot_sparse=True)
        to_float = ToFloat32Transformer()

        device = "gpu" if MODEL_BACKEND == "lgbm_gpu" else "cpu"
        clf = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            num_leaves=63,
            max_depth=7,
            objective="binary",
            random_state=42,
            device=device,
        )
        pipe = Pipeline([("pre", pre), ("to_float", to_float), ("clf", clf)])

    elif MODEL_BACKEND == "dnn":
        log.info("Estimating time for backend: DNN")
        try:
            from model_backends.dnn_classifier import make_dnn_estimator
        except Exception as exc:
            raise RuntimeError(
                "DNN backend requires tensorflow and scikeras. "
                "Install requirements-dnn.txt."
            ) from exc

        pre = create_lead_gen_preprocessor(onehot_sparse=False)
        to_float = ToFloat32Transformer()

        clf = make_dnn_estimator(
            batch_size=4096,
            epochs=5,
            patience=2,
            validation_split=0.1,
            verbose=0,
        )
        pipe = Pipeline([("pre", pre), ("to_float", to_float), ("clf", clf)])

    elif MODEL_BACKEND == "hgb_bagging":
        log.info("Estimating time for backend: HGB BalancedBagging")
        if not HAVE_IMBLEARN_ENSEMBLE:
            raise RuntimeError("imbalanced-learn not installed for BalancedBagging.")

        pre = create_lead_gen_preprocessor(onehot_sparse=False)
        use_class_weight = Version(sklearn_version) >= Version("1.5")
        base = HistGradientBoostingClassifier(
            random_state=42,
            early_stopping=False,
            class_weight=("balanced" if use_class_weight else None),
            max_iter=200,
            max_depth=8,
            min_samples_leaf=50,
            l2_regularization=0.5,
        )
        try:
            bagger = BalancedBaggingClassifier(estimator=base, n_estimators=10, random_state=42)
        except TypeError:
            bagger = BalancedBaggingClassifier(base_estimator=base, n_estimators=10, random_state=42)
        pipe = Pipeline([("pre", pre), ("clf", bagger)])

    else:
        log.info("Estimating time for backend: HGB (Legacy)")
        pre = create_lead_gen_preprocessor(onehot_sparse=False)

        # Pick ONE imbalance strategy: class_weight if available in this sklearn, else none (since this is only a probe)
        use_class_weight = Version(sklearn_version) >= Version("1.5")
        clf = HistGradientBoostingClassifier(
            random_state=42,
            early_stopping=False,
            class_weight=("balanced" if use_class_weight else None),
            max_iter=200, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.1
        )
        pipe = Pipeline([("pre", pre), ("clf", clf)])

    log.info("Timing ONE fit on the sample...")
    t0 = time.time()
    pipe.fit(X, y)
    t1 = time.time()
    minutes_per_fit_on_sample = (t1 - t0) / 60.0

    scale = n_total / max(len(X), 1)
    est_minutes_per_full_fit = minutes_per_fit_on_sample * scale
    total_fits = N_ITER * N_SPLITS + 1 + CAL_SPLITS
    est_total_minutes = est_minutes_per_full_fit * total_fits

    log.info("--- ESTIMATE ---")
    log.info(f"Sample rows: {len(X):,} | Minutes per fit on sample: ~{minutes_per_fit_on_sample:.2f}")
    log.info(f"Scale factor to full dataset: x{scale:.1f}")
    log.info(f"Estimated minutes per FULL fit: ~{est_minutes_per_full_fit:.1f}")
    log.info(f"Total fits = N_ITER*N_SPLITS + 1 + CAL_SPLITS = {N_ITER}*{N_SPLITS} + 1 + {CAL_SPLITS} = {total_fits}")
    log.info(f"Estimated TOTAL runtime: ~{est_total_minutes:.0f} minutes")
    log.info("----------------")
    log.info("Note: This is a ballpark. If you change N_ITER/N_SPLITS/CAL_SPLITS or data size, re-run this script.")

if __name__ == "__main__":
    main()
