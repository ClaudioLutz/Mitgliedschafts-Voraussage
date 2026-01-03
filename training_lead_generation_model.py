# training_lead_generation_model_refactored.py
# -------------------------------------------
# Lead-generation oriented, leakage-safe training with time-aware CV & calibration.
# - Uses only snapshots with complete labels (<= today - horizon).
# - Splits by unique snapshot dates; computes time-meaningful gap for TSCV.
# - TargetEncoder inside pipeline (internal cross-fitting).
# - Imbalance strategies: class_weight (HGB), BalancedBagging, scale_pos_weight (XGB/LGBM), SMOTE fallback.
# - Time-aware calibration via CalibratedClassifierCV(TimeSeriesSplit).
# - Ranks current prospects (latest snapshot) without using labels.

import os
import sys
import math
import json
import logging
import warnings
from datetime import datetime, timedelta
import psutil

import numpy as np
import pandas as pd

from sqlalchemy import create_engine, URL, text
import urllib.parse

from sklearn import __version__ as sklearn_version
from packaging.version import Version

from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import average_precision_score, precision_recall_curve, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

# --------------------
# Extensive Logging setup
# --------------------
import time
from functools import wraps

# Create a custom logger
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)  # Capture everything

# Create handlers
c_handler = logging.StreamHandler() # Console
f_handler = logging.FileHandler(f'training_run_{datetime.now().strftime("%Y%m%d_%H%M")}.log') # File

# Create formatters and add it to handlers
# We add milliseconds (%(msecs)03d) to see if steps are hanging quickly
log_format = logging.Formatter('%(asctime)s.%(msecs)03d [%(levelname)s] %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
c_handler.setFormatter(log_format)
f_handler.setFormatter(log_format)

# Add handlers to the logger
# Avoid adding handlers multiple times if script is re-run in interactive env
if not log.hasHandlers():
    log.addHandler(c_handler)
    log.addHandler(f_handler)
if log.handlers:
    log.propagate = False

# Attach handlers to root so logs from other modules are captured.
root_logger = logging.getLogger()
if not root_logger.hasHandlers():
    root_logger.addHandler(c_handler)
    root_logger.addHandler(f_handler)
root_logger.setLevel(logging.DEBUG)

# --------------------
# Helper: Execution Timer Decorator
# --------------------
def log_execution(func):
    """Decorator to log start, end, and duration of functions automatically."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        log.info(f"🟢 START: {func.__name__}")
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            
            # If result is a DataFrame/Array, log its shape
            meta = ""
            if hasattr(result, 'shape'):
                meta = f" | Output Shape: {result.shape}"
            elif isinstance(result, (list, tuple)):
                meta = f" | Output Len: {len(result)}"
                
            log.info(f"✅ FINISHED: {func.__name__} in {duration:.2f}s{meta}")
            return result
        except Exception as e:
            log.error(f"❌ FAILED: {func.__name__} after {time.time() - start_time:.2f}s with error: {str(e)}")
            raise e
    return wrapper


def log_memory_usage(tag=""):
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / 1024 / 1024 / 1024
    log.info(f"💾 RAM USAGE [{tag}]: {mem_gb:.2f} GB")

warnings.filterwarnings("ignore")

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import inspect

# XGBoost support
try:
    from xgboost import XGBClassifier
    HAVE_XGBOOST = True
except ImportError:
    HAVE_XGBOOST = False

# Optional: Beta calibration (betacal)
try:
    from betacal import BetaCalibration
    HAVE_BETACAL = True
except ImportError:
    HAVE_BETACAL = False

# Optional: LightGBM backend
try:
    from lightgbm import LGBMClassifier
    HAVE_LIGHTGBM = True
except ImportError:
    HAVE_LIGHTGBM = False

# Import new Lead-Gen preprocessor
from column_transformer_lead_gen import create_lead_gen_preprocessor, DROP_COLS, validate_preprocessor, ToFloat32Transformer

# Import two-stage pipeline
try:
    from two_stage_pipeline import TwoStagePipeline, create_two_stage_pipeline
    HAVE_TWO_STAGE = True
except ImportError:
    HAVE_TWO_STAGE = False

# Import lookalike features (optional)
try:
    from lookalike_features import LookalikeFeatureTransformer, HAVE_KPROTOTYPES, HAVE_FAISS
    HAVE_LOOKALIKE = True
except ImportError:
    HAVE_LOOKALIKE = False
    HAVE_KPROTOTYPES = False
    HAVE_FAISS = False

# Imbalance handling
USE_CLASS_WEIGHT = Version(sklearn_version) >= Version("1.5")
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    HAVE_IMBLEARN = True
except Exception:
    HAVE_IMBLEARN = False

try:
    from imblearn.ensemble import BalancedBaggingClassifier
    HAVE_IMBLEARN_ENSEMBLE = True
except Exception:
    HAVE_IMBLEARN_ENSEMBLE = False

# Model Backend Configuration
# Options: 'hgb' (default), 'hgb_bagging', 'xgb_gpu', 'xgb_cpu', 'lgbm_gpu', 'lgbm_cpu', 'dnn',
#          'stacking' (calibrated ensemble), 'two_stage' (filter-then-rank), 'lambdamart' (ranking)
MODEL_BACKEND = os.environ.get("MODEL_BACKEND", "hgb").lower()

# Model persistence for checkpointing
try:
    from joblib import dump, load
    HAVE_JOBLIB = True
except ImportError:
    import pickle
    HAVE_JOBLIB = False
    log.warning("joblib not available, using pickle for model persistence")

RANDOM_STATE = 42

# --------------------
# Fast Path Configuration & Best Known Parameters
# --------------------
# Configuration flags
USE_BEST_KNOWN_PARAMS = True   # Use proven best parameters (fastest)
FORCE_NEW_SEARCH = True       # Override to always run hyperparameter search
ENABLE_CHECKPOINTING = True    # Save/load search results to avoid re-work

# Best hyperparameters from previous 3-hour search (HGB)
BEST_PARAMS_HGB = {
    "classifier__min_samples_leaf": 50,
    "classifier__max_leaf_nodes": 63,
    "classifier__max_iter": 300,
    "classifier__learning_rate": 0.06963974029624322,
    "classifier__l2_regularization": 0.5,
}

# Initial conservative best params for XGBoost (Placeholder)
BEST_PARAMS_XGB = {
    "classifier__max_depth": 6,
    "classifier__learning_rate": 0.05,
    "classifier__n_estimators": 600,
    "classifier__subsample": 0.8,
    "classifier__colsample_bytree": 0.8,
    "classifier__min_child_weight": 1,
    "classifier__reg_lambda": 1.0,
    "classifier__reg_alpha": 0.0,
    "classifier__gamma": 0.0
}

# Initial defaults for DNN (no search)
BEST_PARAMS_DNN = {}

# Optional defaults for LightGBM and BalancedBagging
BEST_PARAMS_LGBM = {}
BEST_PARAMS_HGB_BAGGING = {}

# DNN default training configuration
DNN_HIDDEN_UNITS = (128, 64)
DNN_DROPOUT = 0.2
DNN_L2 = 0.0001
DNN_LEARNING_RATE = 1e-3
DNN_BATCH_SIZE = 4096
DNN_EPOCHS = 20
DNN_PATIENCE = 3
DNN_VALIDATION_SPLIT = 0.1
DNN_VERBOSE = 0

# Calibration + threshold optimization
CALIBRATION_METHOD = os.environ.get("CALIBRATION_METHOD", "isotonic").lower()
BETA_CALIBRATION_PARAMETERS = os.environ.get("BETA_CALIBRATION_PARAMETERS", "abm")
THRESHOLD_BETA = float(os.environ.get("THRESHOLD_BETA", "2.0"))
ENABLE_THRESHOLD_OPTIMIZATION = True

# Imbalance handling parameters
SMOTE_SAMPLING_STRATEGY = os.environ.get("SMOTE_SAMPLING_STRATEGY", "0.1")
BALANCED_BAGGING_N_ESTIMATORS = int(os.environ.get("BALANCED_BAGGING_N_ESTIMATORS", "30"))
BALANCED_BAGGING_SAMPLING_STRATEGY = os.environ.get("BALANCED_BAGGING_SAMPLING_STRATEGY", "auto")
BALANCED_BAGGING_N_JOBS = int(os.environ.get("BALANCED_BAGGING_N_JOBS", "-1"))

# LightGBM defaults (optional backend)
LGBM_USE_UNBALANCE = os.environ.get("LGBM_USE_UNBALANCE", "true").lower() in {"1", "true", "yes"}


# --------------------
# Config
# --------------------
SERVER = "PRODSVCREPORT70"
DATABASE = "CAG_Analyse"
SCHEMA = "mitgliederstatistik"

# Files
OUTDIR = "./outputs"
ARTIFACTS_DIR = "./artifacts"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Lead-gen horizon & capacity
HORIZON_MONTHS = 12
LEAD_CAPACITY_K = 1000  # set to weekly capacity for Precision@K, adjust to your team

# Search budget
N_ITER = 10     # trimmed for stability & speed; increase as needed
N_SPLITS = 4    # time-aware CV folds for tuning
CAL_SPLITS = 3  # time-aware folds for calibration

N_JOBS_SEARCH = 1  # avoid worker crashes on huge folds (increase if RAM allows)

# Memory optimization settings
SAMPLE_TRAINING_DATA = True  # Use stratified sampling for large datasets
MAX_TRAINING_SAMPLES = 2500000  # Maximum samples for training (increased for better stratification)
USE_BUSINESS_LOGIC_SAMPLING = True  # Use advanced business-logic sampling
PRESERVE_RARE_POSITIVES = True  # Preserve rare but valuable positive cases


# --------------------
# DB connection helper (pyodbc via SQLAlchemy URL)
# --------------------
def make_engine(server: str, database: str):
    odbc = urllib.parse.quote_plus(
        "Driver=ODBC Driver 17 for SQL Server;"
        f"Server={server};"
        f"Database={database};"
        "Trusted_Connection=Yes;"
    )
    eng = create_engine(URL.create("mssql+pyodbc", query={"odbc_connect": odbc}))
    return eng


# --------------------
# Data loading
# --------------------
@log_execution
def load_modeling_data(engine, horizon_months: int = 12) -> pd.DataFrame:
    """
    Load only snapshots with COMPLETE labels: snapshot_date <= GETDATE() - horizon.
    Label = 1 if Eintritt in (snapshot_date, snapshot_date + horizon), else 0.
    Risk set = non-members at snapshot (Eintritt IS NULL or Eintritt >= snapshot_date),
               founding year <= snapshot year.
    NOTE: Adjust table/column names to your schema if needed.
    """
    query = f"""
WITH snapshots AS (
    -- Choose historical monthly snapshots (e.g., last 36..12 months) for MODELING
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
        -- membership table fields
        a.Eintritt,
        a.Austritt,
        a.DT_LoeschungAusfall
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
    WHERE
        -- only snapshots with COMPLETE label window
        s.snapshot_date <= DATEADD(MONTH, -{horizon_months}, GETDATE())
        -- risk set: must be non-member AT snapshot
        AND (b.Eintritt IS NULL OR b.Eintritt >= s.snapshot_date)
        -- existence: company founded on/before snapshot year
        AND (b.Gruendung_Jahr IS NULL OR b.Gruendung_Jahr <= YEAR(s.snapshot_date))
        -- realistic snapshots: exclude companies deleted/bankrupt before or at snapshot
        -- Note: 1888-12-31 is used as NULL sentinel value for DT_LoeschungAusfall
        AND (b.DT_LoeschungAusfall IS NULL 
             OR b.DT_LoeschungAusfall = '1888-12-31' 
             OR b.DT_LoeschungAusfall > s.snapshot_date)
)
SELECT * FROM modeling;
"""
    log.info("Loading modeling snapshots (labels complete) from DB...")
    df = pd.read_sql_query(text(query), engine, parse_dates=["snapshot_date", "Eintritt", "Austritt"])
    return df


@log_execution
def load_current_snapshot(engine) -> pd.DataFrame:
    """
    Load ONLY the most recent snapshot (today) for SCORING.
    No label filtering here. Risk set = non-members today.
    """
    query = f"""
WITH snapshot_today AS (
    SELECT CAST(GETDATE() AS datetime) AS snapshot_date
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
        a.Austritt,
        a.DT_LoeschungAusfall
    FROM {DATABASE}.{SCHEMA}.MitgliederSegmentierung a
)
SELECT
    b.*,
    s.snapshot_date
FROM base b
CROSS JOIN snapshot_today s
WHERE
    (b.Eintritt IS NULL OR b.Eintritt > s.snapshot_date) -- non-members TODAY
    AND (b.Gruendung_Jahr IS NULL OR b.Gruendung_Jahr <= YEAR(s.snapshot_date))
    -- realistic snapshots: exclude companies deleted/bankrupt before or at snapshot
    -- Note: 1888-12-31 is used as NULL sentinel value for DT_LoeschungAusfall
    AND (b.DT_LoeschungAusfall IS NULL 
         OR b.DT_LoeschungAusfall = '1888-12-31' 
         OR b.DT_LoeschungAusfall > s.snapshot_date);
"""
    log.info("Loading latest snapshot (current prospects) from DB...")
    df = pd.read_sql_query(text(query), engine, parse_dates=["snapshot_date", "Eintritt", "Austritt"])
    return df

# --------------------
# Feature utilities
# --------------------
LEAKAGE_COLS = {
    "Target", "Eintritt", "Austritt", "snapshot_date", "DT_LoeschungAusfall"
}

@log_execution
def temporal_feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal features for lead generation model.

    Features added:
    - Company_Age_Years: Age of company at snapshot
    - Company_Age_Log: Log-transformed age (handles outliers, captures diminishing effects)
    - Company_Age_Bucket: Categorical age buckets (startup, early_growth, established, mature, legacy)
    - Month_Sin, Month_Cos: Cyclical encoding of month (preserves continuity)
    - Is_End_Of_Quarter: Flag for Q4 end months (3, 6, 9, 12) - peak B2B purchasing periods
    - Is_Summer: Flag for Jul-Aug (slowdown period)
    - Quarter: Quarter of the year (1-4)
    """
    out = df.copy()

    # Company age in years at snapshot
    if "Gruendung_Jahr" in out.columns and "snapshot_date" in out.columns:
        snap_year = out["snapshot_date"].dt.year
        out["Company_Age_Years"] = (
            snap_year - out["Gruendung_Jahr"].fillna(snap_year)
        ).clip(lower=0)

        # Log-transformed age (handles outliers, captures diminishing lifecycle effects)
        out["Company_Age_Log"] = np.log1p(out["Company_Age_Years"])

        # Age buckets for interpretability
        out["Company_Age_Bucket"] = pd.cut(
            out["Company_Age_Years"],
            bins=[-0.1, 2, 5, 10, 20, np.inf],
            labels=['startup', 'early_growth', 'established', 'mature', 'legacy']
        ).astype(str)
    else:
        out["Company_Age_Years"] = 0
        out["Company_Age_Log"] = 0.0
        out["Company_Age_Bucket"] = 'unknown'

    # Seasonal B2B patterns from snapshot_date
    if "snapshot_date" in out.columns:
        month = out["snapshot_date"].dt.month

        # Cyclical encoding for months (preserves continuity: Dec is close to Jan)
        out["Month_Sin"] = np.sin(2 * np.pi * month / 12)
        out["Month_Cos"] = np.cos(2 * np.pi * month / 12)

        # End of quarter flag (Q4 and Q1 are peak B2B purchasing periods)
        out["Is_End_Of_Quarter"] = month.isin([3, 6, 9, 12]).astype(int)

        # Summer slowdown (Jul-Aug in Switzerland)
        out["Is_Summer"] = month.isin([7, 8]).astype(int)

        # Quarter of the year
        out["Quarter"] = out["snapshot_date"].dt.quarter
    else:
        out["Month_Sin"] = 0.0
        out["Month_Cos"] = 1.0
        out["Is_End_Of_Quarter"] = 0
        out["Is_Summer"] = 0
        out["Quarter"] = 1

    return out

# --------------------
# Scorers & metrics
# --------------------
def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Compute precision at top-K predictions."""
    k = int(min(k, len(y_score)))
    if k <= 0:
        return 0.0
    order = np.argsort(-y_score)[:k]
    return float(np.mean(np.asarray(y_true)[order] == 1))


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Compute recall at top-K predictions (what fraction of positives are in top-K)."""
    y_true = np.asarray(y_true)
    k = int(min(k, len(y_score)))
    if k <= 0:
        return 0.0
    n_positives = np.sum(y_true == 1)
    if n_positives == 0:
        return 0.0
    order = np.argsort(-y_score)[:k]
    return float(np.sum(y_true[order] == 1) / n_positives)


def lift_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Compute lift at top-K (precision@K / baseline rate)."""
    y_true = np.asarray(y_true)
    baseline = np.mean(y_true)
    if baseline == 0:
        return 0.0
    p_at_k = precision_at_k(y_true, y_score, k)
    return float(p_at_k / baseline)


def precision_at_k_scorer_factory(k: int):
    """Create a scorer compatible with RandomizedSearchCV (signature: estimator, X, y)."""
    def _score(estimator, X, y):
        proba = estimator.predict_proba(X)[:, 1]
        return precision_at_k(np.asarray(y), proba, k)
    return _score


def recall_at_k_scorer_factory(k: int):
    """Create a recall@K scorer for hyperparameter tuning."""
    def _score(estimator, X, y):
        proba = estimator.predict_proba(X)[:, 1]
        return recall_at_k(np.asarray(y), proba, k)
    return _score


def lift_at_k_scorer_factory(k: int):
    """Create a lift@K scorer for hyperparameter tuning."""
    def _score(estimator, X, y):
        proba = estimator.predict_proba(X)[:, 1]
        return lift_at_k(np.asarray(y), proba, k)
    return _score


def gains_table(df_scores: pd.DataFrame, score_col="p_convert", target_col="Target", bins=10):
    df = df_scores.dropna(subset=[score_col]).copy()
    df["rank"] = df[score_col].rank(method="first", ascending=False)
    df["decile"] = pd.qcut(df["rank"], q=bins, labels=list(range(1, bins + 1)))
    agg = (df.groupby("decile")[target_col]
             .agg(["count", "sum"])
             .rename(columns={"sum": "positives"}))
    agg["rate"] = agg["positives"] / agg["count"].clip(lower=1)
    baseline = df[target_col].mean()
    agg["lift"] = agg["rate"] / max(baseline, 1e-9)
    return agg.sort_index()


# --------------------
# Split helpers
# --------------------
@log_execution
def split_by_unique_dates(df: pd.DataFrame, date_col: str = "snapshot_date"):
    """Train = all but last 2 unique dates; Val = second-last; Test = last."""
    unique_dates = sorted(df[date_col].dropna().unique().tolist())
    if len(unique_dates) < 3:
        raise ValueError("Need at least 3 distinct snapshot dates for train/val/test.")

    train_dates = unique_dates[:-2]
    val_dates   = [unique_dates[-2]]
    test_dates  = [unique_dates[-1]]

    df_train = df[df[date_col].isin(train_dates)].copy()
    df_val   = df[df[date_col].isin(val_dates)].copy()
    df_test  = df[df[date_col].isin(test_dates)].copy()

    return df_train, df_val, df_test, train_dates, val_dates, test_dates


def compute_ts_gap_samples(df_train_val: pd.DataFrame, date_col="snapshot_date", months_gap: int = 2) -> int:
    """Convert a time gap (months) into an approximate sample gap for TimeSeriesSplit."""
    # avg samples per month
    periods = df_train_val[date_col].dt.to_period("M")
    samples_per_month = len(df_train_val) / max(periods.nunique(), 1)
    gap = int(max(100, min(months_gap * samples_per_month, 2000)))
    return gap


def make_calibrated_classifier(estimator, method: str, cv):
    """
    Handles sklearn API differences: some versions use 'estimator', older use 'base_estimator'.
    """
    params = inspect.signature(CalibratedClassifierCV).parameters
    if "estimator" in params:
        return CalibratedClassifierCV(estimator=estimator, method=method, cv=cv)
    return CalibratedClassifierCV(base_estimator=estimator, method=method, cv=cv)


def ensure_chronological_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort rows chronologically for time-series CV. Uses snapshot_date and a stable tie-breaker if available.
    """
    if "snapshot_date" not in df.columns:
        return df.reset_index(drop=True)

    sort_cols = ["snapshot_date"]
    if "CrefoID" in df.columns:
        sort_cols.append("CrefoID")

    return df.sort_values(sort_cols).reset_index(drop=True)


def get_xgb_classifier(backend, random_state=42, scale_pos_weight=1.0):
    """
    Factory to create XGBClassifier with appropriate parameters for GPU or CPU.
    """
    if not HAVE_XGBOOST:
        raise RuntimeError("XGBoost is not installed. Please install 'xgboost'.")

    # Common parameters
    common_params = {
        "n_estimators": 100, # default, will be tuned
        "learning_rate": 0.1,
        "max_depth": 6,
        "objective": "binary:logistic",
        "random_state": random_state,
        "scale_pos_weight": scale_pos_weight,
        "max_bin": 256,
        "eval_metric": "logloss" # avoid warning
    }

    if backend == "xgb_gpu":
        # GPU-friendly defaults: reduce binning and enable gradient-based sampling
        common_params["max_bin"] = 64
        common_params["sampling_method"] = "gradient_based"
        return XGBClassifier(
            tree_method="gpu_hist",
            **common_params
        )
    else: # xgb_cpu
        return XGBClassifier(
            tree_method="hist",
            **common_params
        )


def get_dnn_classifier(
    hidden_units=None,
    dropout=None,
    l2=None,
    learning_rate=None,
    batch_size=None,
    epochs=None,
    patience=None,
    validation_split=None,
    random_state=None,
    verbose=None,
):
    """Factory to create a SciKeras-compatible DNN classifier."""
    try:
        from model_backends.dnn_classifier import make_dnn_estimator
    except Exception as exc:
        raise RuntimeError(
            "DNN backend requires tensorflow and scikeras. "
            "Install requirements-dnn.txt."
        ) from exc

    if hidden_units is None:
        hidden_units = DNN_HIDDEN_UNITS
    if dropout is None:
        dropout = DNN_DROPOUT
    if l2 is None:
        l2 = DNN_L2
    if learning_rate is None:
        learning_rate = DNN_LEARNING_RATE
    if batch_size is None:
        batch_size = DNN_BATCH_SIZE
    if epochs is None:
        epochs = DNN_EPOCHS
    if patience is None:
        patience = DNN_PATIENCE
    if validation_split is None:
        validation_split = DNN_VALIDATION_SPLIT
    if random_state is None:
        random_state = RANDOM_STATE
    if verbose is None:
        verbose = DNN_VERBOSE

    return make_dnn_estimator(
        hidden_units=hidden_units,
        dropout=dropout,
        l2=l2,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        validation_split=validation_split,
        random_state=random_state,
        verbose=verbose,
    )


def parse_sampling_strategy(value):
    """Parse sampling strategy values from env/config into usable types."""
    if isinstance(value, (float, int)):
        return value
    if value is None:
        return value
    value_str = str(value).strip().lower()
    if value_str in {"auto", "minority", "majority"}:
        return value_str
    try:
        return float(value_str)
    except ValueError:
        return value


def hgb_supports_class_weight() -> bool:
    """Return True if HistGradientBoostingClassifier supports class_weight."""
    params = inspect.signature(HistGradientBoostingClassifier).parameters
    return "class_weight" in params


def make_hgb_classifier(
    random_state: int,
    class_weight=None,
    **kwargs,
):
    """Create an HGB classifier with safe class_weight handling."""
    params = {
        "random_state": random_state,
        "early_stopping": False,
        "max_depth": None,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 20,
        "l2_regularization": 0.1,
    }
    params.update(kwargs)
    if class_weight is not None and hgb_supports_class_weight():
        params["class_weight"] = class_weight
    return HistGradientBoostingClassifier(**params)


def get_balanced_bagging_estimator_param_name() -> str:
    """Return estimator parameter name for BalancedBaggingClassifier across versions."""
    if not HAVE_IMBLEARN_ENSEMBLE:
        return "estimator"
    params = inspect.signature(BalancedBaggingClassifier).parameters
    return "estimator" if "estimator" in params else "base_estimator"


def make_balanced_bagging_classifier(
    base_estimator,
    n_estimators: int,
    sampling_strategy,
    random_state: int,
    n_jobs: int,
):
    if not HAVE_IMBLEARN_ENSEMBLE:
        raise RuntimeError("imbalanced-learn ensemble is not available.")
    estimator_param = get_balanced_bagging_estimator_param_name()
    kwargs = {
        estimator_param: base_estimator,
        "n_estimators": n_estimators,
        "sampling_strategy": sampling_strategy,
        "random_state": random_state,
    }
    if n_jobs is not None:
        kwargs["n_jobs"] = n_jobs
    return BalancedBaggingClassifier(**kwargs)


def get_lgbm_classifier(backend, random_state=42, scale_pos_weight=1.0):
    """Factory to create LightGBM classifier with GPU-friendly defaults."""
    if not HAVE_LIGHTGBM:
        raise RuntimeError("LightGBM is not installed. Please install 'lightgbm'.")

    params = {
        "n_estimators": 800,
        "learning_rate": 0.05,
        "num_leaves": 127,
        "max_depth": 7,
        "subsample": 0.7,
        "colsample_bytree": 0.8,
        "random_state": random_state,
        "n_jobs": -1,
        "objective": "binary",
    }

    if backend == "lgbm_gpu":
        params["device"] = "gpu"
        params["max_bin"] = 63
    else:
        params["device"] = "cpu"

    if LGBM_USE_UNBALANCE:
        params["is_unbalance"] = True
    else:
        params["scale_pos_weight"] = scale_pos_weight

    return LGBMClassifier(**params)


def get_stacking_ensemble(random_state=42, scale_pos_weight=1.0, cv=5):
    """
    Create a calibrated stacking ensemble combining multiple imbalance-handling strategies.

    The ensemble combines:
    1. Cost-sensitive HGB with class_weight='balanced'
    2. Cost-sensitive XGBoost with scale_pos_weight
    3. BalancedBagging with undersampling (different strategy = diversity)

    Each base model is calibrated with isotonic regression before stacking.
    The meta-learner is a class-weighted logistic regression.

    Reference: Research shows combining resampling with cost-sensitive learning
    consistently outperforms either approach alone.
    """
    log.info("Building calibrated stacking ensemble...")

    # Base 1: Cost-sensitive HistGradientBoosting (with calibration)
    hgb_base = make_hgb_classifier(
        random_state=random_state,
        class_weight="balanced" if hgb_supports_class_weight() else None,
        max_iter=200,
        learning_rate=0.07,
        max_leaf_nodes=63,
        min_samples_leaf=50,
        l2_regularization=0.5,
    )
    hgb_calibrated = CalibratedClassifierCV(
        estimator=hgb_base,
        method='isotonic',
        cv=cv
    )

    # Base 2: Cost-sensitive XGBoost (with calibration)
    if HAVE_XGBOOST:
        xgb_base = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            eval_metric="logloss",
            random_state=random_state
        )
        xgb_calibrated = CalibratedClassifierCV(
            estimator=xgb_base,
            method='isotonic',
            cv=cv
        )
    else:
        xgb_calibrated = None
        log.warning("XGBoost not available, using only HGB and BalancedBagging in stack.")

    # Base 3: BalancedBagging (different strategy = diversity)
    # Uses undersampling per bag - complementary to cost-sensitive methods
    if HAVE_IMBLEARN_ENSEMBLE:
        # Use DecisionTree as base to maximize diversity (different from HGB/XGB)
        dt_base = DecisionTreeClassifier(
            max_depth=10,
            min_samples_leaf=20,
            random_state=random_state
        )
        bbc_base = make_balanced_bagging_classifier(
            base_estimator=dt_base,
            n_estimators=50,
            sampling_strategy='auto',
            random_state=random_state,
            n_jobs=-1
        )
        bbc_calibrated = CalibratedClassifierCV(
            estimator=bbc_base,
            method='isotonic',
            cv=cv
        )
    else:
        bbc_calibrated = None
        log.warning("imbalanced-learn not available, using only cost-sensitive methods in stack.")

    # Build estimator list
    estimators = [('hgb', hgb_calibrated)]
    if xgb_calibrated is not None:
        estimators.append(('xgb', xgb_calibrated))
    if bbc_calibrated is not None:
        estimators.append(('bbc', bbc_calibrated))

    if len(estimators) < 2:
        raise RuntimeError("Stacking requires at least 2 estimators. Install xgboost or imbalanced-learn.")

    log.info(f"Stacking ensemble with {len(estimators)} calibrated base models: {[e[0] for e in estimators]}")

    # Stack with class-weighted meta-learner
    ensemble = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=random_state
        ),
        cv=cv,
        stack_method='predict_proba',
        passthrough=False,  # Use only base model predictions
        n_jobs=1  # Avoid nested parallelism issues
    )

    return ensemble


def get_lambdamart_ranker(random_state=42, truncation_level=55000):
    """
    Create a LightGBM LambdaMART ranker for learning-to-rank optimization.

    LambdaMART optimizes ranking directly, which can be better than
    classification when the goal is to rank the top-K leads.

    Note: The ranker treats all samples as a single query group.
    """
    if not HAVE_LIGHTGBM:
        raise RuntimeError("LightGBM is not installed. Required for LambdaMART.")

    from lightgbm import LGBMRanker

    ranker = LGBMRanker(
        objective='lambdarank',
        lambdarank_truncation_level=truncation_level,
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1
    )

    return ranker


class LambdaMARTWrapper:
    """
    Wrapper to make LGBMRanker compatible with sklearn classifier API.

    Converts ranker scores to pseudo-probabilities for compatibility
    with calibration and scoring functions.
    """

    def __init__(self, ranker):
        self.ranker = ranker
        self.classes_ = np.array([0, 1])
        self._is_fitted = False

    def fit(self, X, y):
        # LGBMRanker requires group parameter
        # Treat all samples as one query group
        if hasattr(X, 'values'):
            X = X.values
        y = np.asarray(y)
        group = [len(y)]  # Single group with all samples
        self.ranker.fit(X, y, group=group)
        self._is_fitted = True
        return self

    def predict_proba(self, X):
        if hasattr(X, 'values'):
            X = X.values
        # Get raw ranker scores
        scores = self.ranker.predict(X)
        # Convert to pseudo-probabilities using sigmoid
        probs = 1 / (1 + np.exp(-scores))
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


def optimize_threshold_fbeta(y_true: np.ndarray, y_score: np.ndarray, beta: float = 2.0):
    """Find threshold that maximizes F-beta."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if thresholds.size == 0:
        return 0.5, float("nan")
    thresholds = np.concatenate([thresholds, [1.0]])
    beta_sq = beta ** 2
    f_beta = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall + 1e-12)
    best_idx = int(np.nanargmax(f_beta))
    return float(thresholds[best_idx]), float(f_beta[best_idx])


def threshold_for_top_k(y_score: np.ndarray, k: int) -> float:
    """Return the score threshold corresponding to the top-k highest scores."""
    scores = np.asarray(y_score)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return 1.0
    k = int(min(max(k, 0), scores.size))
    if k <= 0:
        return 1.0
    return float(np.partition(scores, -k)[-k])


def metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float, beta: float = 2.0):
    """Compute precision/recall/F-beta at a fixed threshold."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_score) >= threshold
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    beta_sq = beta ** 2
    f_beta = (1 + beta_sq) * (precision * recall) / max(beta_sq * precision + recall, 1e-12)

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f_beta": float(f_beta),
        "support": int(len(y_true)),
        "positives": int(np.sum(y_true == 1)),
        "predicted_positives": int(np.sum(y_pred)),
    }


def class_stratified_brier(y_true: np.ndarray, y_score: np.ndarray):
    """Compute separate Brier scores for positives and negatives."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos_mask = y_true == 1
    neg_mask = y_true == 0

    bs_pos = float(np.mean((y_score[pos_mask] - 1.0) ** 2)) if np.any(pos_mask) else float("nan")
    bs_neg = float(np.mean((y_score[neg_mask] - 0.0) ** 2)) if np.any(neg_mask) else float("nan")
    return bs_pos, bs_neg


def ece_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int, n_bins: int = 10):
    """Expected calibration error on the top-k scored samples."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_score.size == 0 or k <= 0:
        return float("nan")

    order = np.argsort(-y_score)
    top_idx = order[: min(k, len(order))]
    y_top = y_true[top_idx]
    p_top = y_score[top_idx]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (p_top >= bins[i]) & (p_top <= bins[i + 1])
        else:
            mask = (p_top >= bins[i]) & (p_top < bins[i + 1])
        if not np.any(mask):
            continue
        bin_frac = np.sum(mask) / max(len(p_top), 1)
        avg_pred = float(np.mean(p_top[mask]))
        avg_true = float(np.mean(y_top[mask]))
        ece += bin_frac * abs(avg_pred - avg_true)
    return float(ece)


def _safe_index(X, idx):
    """Index X with numpy indices for pandas or numpy arrays."""
    if hasattr(X, "iloc"):
        return X.iloc[idx]
    return X[idx]


class BetaCalibratedModel:
    """Wrapper to apply BetaCalibration to a fitted probabilistic estimator."""

    def __init__(self, estimator, calibrator):
        self.estimator = estimator
        self.calibrator = calibrator
        self.classes_ = getattr(estimator, "classes_", np.array([0, 1]))

    def predict_proba(self, X):
        proba = self.estimator.predict_proba(X)[:, 1]
        calibrated = self.calibrator.predict(proba)
        calibrated = np.clip(calibrated, 0.0, 1.0)
        return np.column_stack([1.0 - calibrated, calibrated])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def fit_beta_calibrated_model(estimator, X, y, cv):
    """Fit BetaCalibration using out-of-fold predictions for leakage-safe calibration."""
    if not HAVE_BETACAL:
        raise RuntimeError("betacal is not installed.")

    oof_pred = np.full(len(y), np.nan, dtype=float)
    for train_idx, val_idx in cv.split(X):
        est = clone(estimator)
        X_train = _safe_index(X, train_idx)
        y_train = np.asarray(y)[train_idx]
        X_val = _safe_index(X, val_idx)

        est.fit(X_train, y_train)
        oof_pred[val_idx] = est.predict_proba(X_val)[:, 1]

    valid_mask = np.isfinite(oof_pred)
    if np.sum(valid_mask) < 10:
        raise RuntimeError("Not enough calibration samples for beta calibration.")

    calibrator = BetaCalibration(parameters=BETA_CALIBRATION_PARAMETERS)
    calibrator.fit(oof_pred[valid_mask], np.asarray(y)[valid_mask])

    estimator.fit(X, y)
    return BetaCalibratedModel(estimator=estimator, calibrator=calibrator)

def stratified_sample_large_dataset(df: pd.DataFrame, target_col: str, max_samples: int, random_state: int = 42) -> pd.DataFrame:
    """
    Enhanced multi-dimensional stratified sampling for large datasets.
    Preserves distributions across multiple key dimensions simultaneously.
    
    Args:
        df: Input dataframe
        target_col: Name of target column
        max_samples: Maximum number of samples to retain
        random_state: Random seed for reproducibility
    
    Returns:
        Multi-dimensionally stratified sampled dataframe
    """
    if len(df) <= max_samples:
        return df.copy()
    
    df_temp = df.copy()

    log.info(f"Dataset too large ({len(df):,} samples). Applying enhanced stratified sampling to {max_samples:,} samples...")
    
    # Define stratification strategy based on business importance
    stratification_features = []
    
    # 1. Always stratify by target (most important)
    stratification_features.append(target_col)
    
    # 2. Geographic stratification (preserve regional representation)
    if 'Kanton' in df.columns:
        # Group smaller cantons together to avoid over-fragmentation
        major_cantons = ['ZH', 'BE', 'VD', 'GE', 'AG', 'SG', 'TI', 'VS', 'LU', 'ZG']
        df_temp['Kanton_Grouped'] = df_temp['Kanton'].apply(
            lambda x: x if x in major_cantons else 'OTHER'
        )
        stratification_features.append('Kanton_Grouped')
        log.info("✅ Geographic stratification enabled (major cantons + OTHER)")
    
    # 3. Company size stratification (critical business dimension)
    if 'GroessenKategorie' in df.columns:
        stratification_features.append('GroessenKategorie')
        log.info("✅ Company size stratification enabled")
    
    # 4. Temporal stratification (preserve time-series properties)
    if 'snapshot_date' in df.columns:
        df_temp['snapshot_year'] = df_temp['snapshot_date'].dt.year
        stratification_features.append('snapshot_year')
        log.info("✅ Temporal stratification enabled")
    
    # 5. Legal form stratification (different business types)
    if 'Rechtsform' in df.columns:
        # Group less common legal forms to avoid over-fragmentation
        major_forms = ['Einzelunternehmen', 'GmbH', 'Aktiengesellschaft', 'Verein','Genossenschaft']
        df_temp['Rechtsform_Grouped'] = df_temp['Rechtsform'].apply(
            lambda x: x if x in major_forms else 'OTHER'
        )
        stratification_features.append('Rechtsform_Grouped')
        log.info("✅ Legal form stratification enabled")
    
    log.info(f"Stratifying across {len(stratification_features)} dimensions: {stratification_features}")
    
    # Create stratification groups
    if len(stratification_features) == 1:
        # Simple stratification (fallback)
        df_temp['strat_group'] = df_temp[stratification_features[0]].astype(str)
    else:
        # Multi-dimensional stratification
        df_temp['strat_group'] = df_temp[stratification_features].astype(str).agg('|'.join, axis=1)
    
    # Calculate sampling strategy
    group_counts = df_temp['strat_group'].value_counts()
    total_groups = len(group_counts)
    sample_ratio = max_samples / len(df)
    
    log.info(f"Created {total_groups:,} stratification groups")
    log.info(f"Base sampling ratio: {sample_ratio:.4f}")
    
    # Enhanced sampling with minimum guarantees
    sampled_dfs = []
    min_samples_per_group = max(1, int(max_samples / total_groups * 0.1))  # At least 10% of equal allocation
    
    for group_id, group_size in group_counts.items():
        group_df = df_temp[df_temp['strat_group'] == group_id]
        
        # Calculate target sample size with minimum guarantee
        target_samples = max(min_samples_per_group, int(group_size * sample_ratio))
        
        # Don't over-sample small groups
        actual_samples = min(target_samples, len(group_df))
        
        if actual_samples >= len(group_df):
            sampled_group = group_df.copy()
        else:
            sampled_group = group_df.sample(n=actual_samples, random_state=random_state)
        
        sampled_dfs.append(sampled_group)
    
    # Combine all sampled groups
    result = pd.concat(sampled_dfs, ignore_index=True)
    
    # If we're over the target, do proportional reduction
    if len(result) > max_samples:
        log.info(f"Initial sample ({len(result):,}) exceeds target. Applying proportional reduction...")
        reduction_ratio = max_samples / len(result)
        
        final_dfs = []
        for group_id in result['strat_group'].unique():
            group_df = result[result['strat_group'] == group_id]
            reduced_size = max(1, int(len(group_df) * reduction_ratio))
            
            if reduced_size >= len(group_df):
                final_group = group_df.copy()
            else:
                final_group = group_df.sample(n=reduced_size, random_state=random_state)
            
            final_dfs.append(final_group)
        
        result = pd.concat(final_dfs, ignore_index=True)
    
    # Remove temporary columns and shuffle
    columns_to_drop = ['strat_group']
    if 'Kanton_Grouped' in result.columns:
        columns_to_drop.append('Kanton_Grouped')
    if 'Rechtsform_Grouped' in result.columns:
        columns_to_drop.append('Rechtsform_Grouped')
    if 'snapshot_year' in result.columns:
        columns_to_drop.append('snapshot_year')
    
    result = result.drop(columns=columns_to_drop, errors='ignore')
    # Do not shuffle; preserve chronological order for time-series workflows
    result = ensure_chronological_order(result)
    
    # Enhanced reporting
    log.info(f"✅ Enhanced stratified sampling complete: {len(df):,} -> {len(result):,} samples")
    log.info(f"📊 Target distribution preservation:")
    
    orig_target_dist = df[target_col].value_counts(normalize=True).sort_index()
    sampled_target_dist = result[target_col].value_counts(normalize=True).sort_index()
    
    for target_val in orig_target_dist.index:
        orig_pct = orig_target_dist.get(target_val, 0) * 100
        sampled_pct = sampled_target_dist.get(target_val, 0) * 100
        difference = abs(orig_pct - sampled_pct)
        log.info(f"  Target={target_val}: {orig_pct:.2f}% -> {sampled_pct:.2f}% (Δ={difference:.2f}%)")
    
    # Report on other key dimensions
    if 'Kanton' in df.columns and 'Kanton' in result.columns:
        log.info("📍 Geographic distribution preservation (top cantons):")
        orig_geo = df['Kanton'].value_counts(normalize=True).head(5)
        sampled_geo = result['Kanton'].value_counts(normalize=True).head(5)
        for canton in orig_geo.index:
            orig_pct = orig_geo.get(canton, 0) * 100
            sampled_pct = sampled_geo.get(canton, 0) * 100
            log.info(f"  {canton}: {orig_pct:.1f}% -> {sampled_pct:.1f}%")
    
    if 'GroessenKategorie' in df.columns and 'GroessenKategorie' in result.columns:
        log.info("🏢 Company size distribution preservation:")
        orig_size = df['GroessenKategorie'].value_counts(normalize=True)
        sampled_size = result['GroessenKategorie'].value_counts(normalize=True)
        for size_cat in ['MICRO', 'KLEIN', 'MITTEL', 'GROSS']:
            if size_cat in orig_size.index:
                orig_pct = orig_size.get(size_cat, 0) * 100
                sampled_pct = sampled_size.get(size_cat, 0) * 100
                log.info(f"  {size_cat}: {orig_pct:.1f}% -> {sampled_pct:.1f}%")
    
    return result


def advanced_stratified_sample_with_business_logic(df: pd.DataFrame, target_col: str, max_samples: int, 
                                                  random_state: int = 42, 
                                                  preserve_rare_positives: bool = True) -> pd.DataFrame:
    """
    Advanced stratified sampling with business-specific logic for lead generation.
    
    Args:
        df: Input dataframe
        target_col: Name of target column
        max_samples: Maximum number of samples to retain
        random_state: Random seed for reproducibility
        preserve_rare_positives: Whether to ensure rare positive cases are preserved
    
    Returns:
        Business-logic enhanced stratified sampled dataframe
    """
    if len(df) <= max_samples:
        return df.copy()
    
    log.info(f"🎯 Applying business-logic enhanced sampling to {max_samples:,} samples...")
    
    # Step 1: Identify and preserve rare but valuable cases
    rare_cases = pd.DataFrame()
    
    if preserve_rare_positives and target_col in df.columns:
        # Preserve all positive cases from underrepresented segments
        positive_cases = df[df[target_col] == 1]
        
        if len(positive_cases) > 0:
            # Identify rare positive segments (e.g., large companies that converted)
            if 'GroessenKategorie' in df.columns:
                large_company_positives = positive_cases[
                    positive_cases['GroessenKategorie'].isin(['GROSS', 'SEHR GROSS', 'MITTEL'])
                ]
                if len(large_company_positives) > 0:
                    rare_cases = pd.concat([rare_cases, large_company_positives])
                    log.info(f"🔸 Preserved {len(large_company_positives)} large company conversions")
            
            # Preserve positive cases from smaller cantons
            if 'Kanton' in df.columns:
                small_canton_positives = positive_cases[
                    ~positive_cases['Kanton'].isin(['ZH', 'BE', 'VD', 'GE', 'AG'])
                ]
                small_canton_sample = small_canton_positives.head(min(1000, len(small_canton_positives)))
                rare_cases = pd.concat([rare_cases, small_canton_sample])
                log.info(f"🔸 Preserved {len(small_canton_sample)} conversions from smaller regions")
    
    # Step 2: Apply main stratified sampling to remaining data
    remaining_df = df.drop(rare_cases.index) if len(rare_cases) > 0 else df
    remaining_budget = max_samples - len(rare_cases)
    
    if remaining_budget > 0:
        main_sample = stratified_sample_large_dataset(
            remaining_df, target_col, remaining_budget, random_state
        )
    else:
        main_sample = pd.DataFrame()
    
    # Step 3: Combine rare cases with main sample
    if len(rare_cases) > 0:
        final_result = pd.concat([rare_cases, main_sample], ignore_index=True)
        final_result = ensure_chronological_order(final_result)
        log.info(f"🎯 Business-logic sampling complete: {len(rare_cases)} rare + {len(main_sample)} stratified = {len(final_result)} total")
    else:
        final_result = ensure_chronological_order(main_sample)
        log.info(f"🎯 Business-logic sampling complete: {len(final_result)} samples (no rare cases found)")
    
    return final_result


# --------------------
# Model Checkpointing Functions
# --------------------
def save_checkpoint(search_result, metadata=None):
    """Save hyperparameter search results to avoid re-work."""
    if not ENABLE_CHECKPOINTING:
        return
    
    try:
        # Save fitted pipeline and metadata
        pipeline_path = os.path.join(ARTIFACTS_DIR, "best_pipeline.joblib")
        params_path = os.path.join(ARTIFACTS_DIR, "best_params.joblib")
        metadata_path = os.path.join(ARTIFACTS_DIR, "search_metadata.joblib")
        
        if HAVE_JOBLIB:
            dump(search_result.best_estimator_, pipeline_path)
            dump(search_result.best_params_, params_path)
            
            # Save metadata including sklearn version, search score, timestamp
            save_metadata = {
                "sklearn_version": sklearn_version,
                "best_score": search_result.best_score_,
                "timestamp": datetime.now().isoformat(),
                "n_iter": getattr(search_result, 'n_iter', N_ITER),
                "cv_results_keys": list(search_result.cv_results_.keys()) if hasattr(search_result, 'cv_results_') else []
            }
            if metadata:
                save_metadata.update(metadata)
            dump(save_metadata, metadata_path)
            
        else:
            # Fallback to pickle
            with open(pipeline_path.replace('.joblib', '.pkl'), 'wb') as f:
                pickle.dump(search_result.best_estimator_, f)
            with open(params_path.replace('.joblib', '.pkl'), 'wb') as f:
                pickle.dump(search_result.best_params_, f)
                
        log.info(f"Checkpoint saved: pipeline and params to {ARTIFACTS_DIR}")
        
    except Exception as e:
        log.warning(f"Failed to save checkpoint: {e}")


def load_checkpoint():
    """Load previous hyperparameter search results if available and valid."""
    if not ENABLE_CHECKPOINTING:
        return None, None, None
        
    pipeline_path = os.path.join(ARTIFACTS_DIR, "best_pipeline.joblib") 
    params_path = os.path.join(ARTIFACTS_DIR, "best_params.joblib")
    metadata_path = os.path.join(ARTIFACTS_DIR, "search_metadata.joblib")
    
    # Fallback paths for pickle
    if not HAVE_JOBLIB:
        pipeline_path = pipeline_path.replace('.joblib', '.pkl')
        params_path = params_path.replace('.joblib', '.pkl') 
        metadata_path = metadata_path.replace('.joblib', '.pkl')
    
    # Check if checkpoint files exist
    if not all(os.path.exists(p) for p in [pipeline_path, params_path]):
        return None, None, None
        
    try:
        # Load checkpoint
        if HAVE_JOBLIB:
            pipeline = load(pipeline_path)
            params = load(params_path)
            metadata = load(metadata_path) if os.path.exists(metadata_path) else {}
        else:
            with open(pipeline_path, 'rb') as f:
                pipeline = pickle.load(f)
            with open(params_path, 'rb') as f:
                params = pickle.load(f)
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
        
        # Version compatibility check
        saved_version = metadata.get("sklearn_version", "unknown")
        if saved_version != sklearn_version:
            log.warning(f"Checkpoint sklearn version ({saved_version}) differs from current ({sklearn_version})")
            log.warning("Checkpoint may be incompatible - consider re-running search")
            
        log.info(f"Checkpoint loaded: {len(params)} parameters from {metadata.get('timestamp', 'unknown date')}")
        log.info(f"Checkpoint best score: {metadata.get('best_score', 'unknown')}")
        
        return pipeline, params, metadata
        
    except Exception as e:
        log.warning(f"Failed to load checkpoint: {e}")
        return None, None, None


def checkpoint_exists():
    """Check if valid checkpoint files exist."""
    pipeline, params, metadata = load_checkpoint()
    return pipeline is not None and params is not None


# --------------------
# Main
# --------------------
def main():
    log_memory_usage("Start")
    log.info("=== Starting Lead Generation Model Training (refactored) ===")
    engine = make_engine(SERVER, DATABASE)
    log.info("DB connection OK")

    # 1) Load modeling (complete labels) + current prospects (latest snapshot)
    df_model = load_modeling_data(engine, horizon_months=HORIZON_MONTHS)
    log.info(f"Modeling rows: {len(df_model):,}, snapshots: {df_model['snapshot_date'].nunique()} "
             f"from {df_model['snapshot_date'].min()} to {df_model['snapshot_date'].max()}")
    log.info(f"Overall conversion rate (modeling): {df_model['Target'].mean():.4f}")

    df_current = load_current_snapshot(engine)
    latest_snapshot = df_current["snapshot_date"].max()
    log.info(f"Current prospects loaded for snapshot: {latest_snapshot}")
    log_memory_usage("After Data Load")

    # 2) Split modeling data by unique dates
    df_model = df_model.sort_values("snapshot_date").reset_index(drop=True)
    df_train, df_val, df_test, train_dates, val_dates, test_dates = split_by_unique_dates(df_model)

    def rate(df): return df["Target"].mean() if len(df) else float("nan")
    log.info(f"Train: {len(df_train):,} (pos rate={rate(df_train):.4f}) "
             f"| Val: {len(df_val):,} (pos rate={rate(df_val):.4f}) "
             f"| Test: {len(df_test):,} (pos rate={rate(df_test):.4f})")

    # 3) Basic feature engineering (before ColumnTransformer)
    #    - add Company_Age_Years
    #    - no leakage columns downstream
    df_train_eng = temporal_feature_engineer(df_train)
    df_val_eng   = temporal_feature_engineer(df_val)
    df_test_eng  = temporal_feature_engineer(df_test)
    df_curr_eng  = temporal_feature_engineer(df_current)
    log_memory_usage("After Feature Engineering")

    # 4) Optional: Validate preprocessor on sample data
    try:
        validation_results = validate_preprocessor(df_train_eng.head(1000))
        log.info(f"✅ Preprocessor validation successful!")
        log.info(f"   Features created: {validation_results['n_features']}")
        log.info(f"   Missing indicators: {len(validation_results['missing_indicators_added'])}")
        log.info(f"   PLZ groupings: {len(validation_results['plz_groupings_added'])}")
    except Exception as e:
        log.warning(f"⚠️  Preprocessor validation failed: {e}")

    # 5) Create lead-gen specific preprocessor

    # Determine configuration based on backend
    if MODEL_BACKEND.startswith("xgb"):
        # XGBoost path: Sparse output + Float32 + scale_pos_weight
        log.info(f"Using {MODEL_BACKEND.upper()} backend.")

        # Preprocessor: Sparse OHE, Float32 output preferred
        pre = create_lead_gen_preprocessor(onehot_sparse=True)
        log.info("Using Lead-Gen ColumnTransformer (Sparse Mode, Float32)")

        clf = get_xgb_classifier(MODEL_BACKEND, random_state=RANDOM_STATE, scale_pos_weight=1.0)

        # Pipeline: Preprocess -> ToFloat32 -> Classifier
        # No SMOTE for XGBoost
        steps = [
            ("preprocessor", pre),
            ("to_float32", ToFloat32Transformer()),
            ("classifier", clf)
        ]
        PipelineClass = Pipeline
        log.info("Imbalance handling via XGBoost scale_pos_weight (no SMOTE).")

    elif MODEL_BACKEND.startswith("lgbm"):
        log.info(f"Using {MODEL_BACKEND.upper()} backend.")

        pre = create_lead_gen_preprocessor(onehot_sparse=True)
        log.info("Using Lead-Gen ColumnTransformer (Sparse Mode, Float32)")

        clf = get_lgbm_classifier(MODEL_BACKEND, random_state=RANDOM_STATE, scale_pos_weight=1.0)

        steps = [
            ("preprocessor", pre),
            ("to_float32", ToFloat32Transformer()),
            ("classifier", clf)
        ]
        PipelineClass = Pipeline
        if LGBM_USE_UNBALANCE:
            log.info("Imbalance handling via LightGBM is_unbalance.")
        else:
            log.info("Imbalance handling via LightGBM scale_pos_weight.")

    elif MODEL_BACKEND == "dnn":
        log.info("Using DNN backend.")
        pre = create_lead_gen_preprocessor(onehot_sparse=False)
        log.info("Using Lead-Gen ColumnTransformer (Dense Mode, Float32) for DNN")

        clf = get_dnn_classifier()

        steps = [
            ("preprocessor", pre),
            ("to_float32", ToFloat32Transformer()),
            ("classifier", clf)
        ]
        PipelineClass = Pipeline
        log.info("Imbalance handling via class_weight passed to DNN (set after sampling).")

    elif MODEL_BACKEND == "hgb_bagging":
        if not HAVE_IMBLEARN_ENSEMBLE:
            raise RuntimeError("imbalanced-learn not installed; needed for BalancedBagging.")
        pre = create_lead_gen_preprocessor(onehot_sparse=False)
        log.info("Using Lead-Gen ColumnTransformer with engineered features (BalancedBagging).")

        base_estimator = make_hgb_classifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            max_iter=200,
            learning_rate=0.05,
            max_depth=8,
            min_samples_leaf=50,
            l2_regularization=0.5,
        )

        clf = make_balanced_bagging_classifier(
            base_estimator=base_estimator,
            n_estimators=BALANCED_BAGGING_N_ESTIMATORS,
            sampling_strategy=parse_sampling_strategy(BALANCED_BAGGING_SAMPLING_STRATEGY),
            random_state=RANDOM_STATE,
            n_jobs=BALANCED_BAGGING_N_JOBS,
        )

        steps = [("preprocessor", pre), ("classifier", clf)]
        PipelineClass = Pipeline
        log.info("Imbalance handling via BalancedBagging (undersampling).")

    elif MODEL_BACKEND == "stacking":
        # Calibrated stacking ensemble: combines HGB, XGB, BalancedBagging
        log.info("Using STACKING (calibrated ensemble) backend.")
        pre = create_lead_gen_preprocessor(onehot_sparse=False)
        log.info("Using Lead-Gen ColumnTransformer (Dense Mode for Stacking)")

        # scale_pos_weight will be set dynamically after sampling
        clf = get_stacking_ensemble(
            random_state=RANDOM_STATE,
            scale_pos_weight=1.0,  # Updated later
            cv=5
        )

        steps = [("preprocessor", pre), ("classifier", clf)]
        PipelineClass = Pipeline
        log.info("Imbalance handling via calibrated stacking (multiple strategies combined).")

    elif MODEL_BACKEND == "lambdamart":
        # LambdaMART ranking backend
        log.info("Using LAMBDAMART (learning-to-rank) backend.")
        pre = create_lead_gen_preprocessor(onehot_sparse=False)
        log.info("Using Lead-Gen ColumnTransformer (Dense Mode for LambdaMART)")

        ranker = get_lambdamart_ranker(
            random_state=RANDOM_STATE,
            truncation_level=max(LEAD_CAPACITY_K + 5000, 55000)
        )
        clf = LambdaMARTWrapper(ranker)

        steps = [
            ("preprocessor", pre),
            ("to_float32", ToFloat32Transformer()),
            ("classifier", clf)
        ]
        PipelineClass = Pipeline
        log.info("Optimizing ranking directly via LambdaMART (no explicit imbalance handling).")

    elif MODEL_BACKEND == "two_stage":
        # Two-stage filter-then-rank pipeline
        if not HAVE_TWO_STAGE:
            raise RuntimeError("two_stage_pipeline module not available.")
        log.info("Using TWO_STAGE (filter-then-rank) backend.")
        pre = create_lead_gen_preprocessor(onehot_sparse=False)
        log.info("Using Lead-Gen ColumnTransformer (Dense Mode for Two-Stage)")

        # Create two-stage pipeline with logistic filter and HGB ranker
        clf = create_two_stage_pipeline(
            stage1_type='logistic',
            stage2_type='hgb',
            target_recall=0.95,
            random_state=RANDOM_STATE
        )

        steps = [("preprocessor", pre), ("classifier", clf)]
        PipelineClass = Pipeline
        log.info("Using two-stage filter-then-rank with 95% recall Stage 1 filter.")

    else:
        # HGB (Legacy) path
        pre = create_lead_gen_preprocessor(onehot_sparse=False)
        log.info("Using Lead-Gen ColumnTransformer with engineered features")

        # 6) Estimator & imbalance strategy
        #    Choose ONE: class_weight if supported (sklearn >= 1.5), else SMOTE
        if USE_CLASS_WEIGHT:
            clf = make_hgb_classifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                max_depth=None,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=0.1,
            )
            steps = [("preprocessor", pre), ("classifier", clf)]
            PipelineClass = Pipeline  # standard sklearn pipeline
            log.info("Imbalance via class_weight='balanced' (no SMOTE).")
        else:
            if not HAVE_IMBLEARN:
                raise RuntimeError("imbalanced-learn not installed; install or upgrade scikit-learn to >=1.5 for class_weight.")
            clf = make_hgb_classifier(
                random_state=RANDOM_STATE,
                class_weight=None,
                max_depth=None,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=0.1,
            )
            steps = [
                ("preprocessor", pre),
                ("smote", SMOTE(random_state=RANDOM_STATE, sampling_strategy=parse_sampling_strategy(SMOTE_SAMPLING_STRATEGY))),
                ("classifier", clf)
            ]
            PipelineClass = ImbPipeline  # imbalanced-learn pipeline
            log.info("Imbalance via SMOTE (no class_weight).")

    pipe = PipelineClass(steps=steps)

    # 7) Smart hyperparameter handling with multi-tier approach
    # Train+Val set for CV; hold out Test by date
    X_train_val = pd.concat([df_train_eng, df_val_eng], ignore_index=True)
    y_train_val = pd.concat([df_train["Target"], df_val["Target"]], ignore_index=True)
    feature_cols = [c for c in X_train_val.columns if c not in DROP_COLS]
    
    # Memory optimization: Apply advanced stratified sampling if dataset is too large
    if SAMPLE_TRAINING_DATA and len(X_train_val) > MAX_TRAINING_SAMPLES:
        # Combine features and target for stratified sampling
        combined_data = X_train_val.copy()
        combined_data["Target"] = y_train_val
        
        # Choose sampling strategy based on configuration
        if USE_BUSINESS_LOGIC_SAMPLING:
            log.info("🎯 Using business-logic enhanced stratified sampling...")
            sampled_data = advanced_stratified_sample_with_business_logic(
                df=combined_data, 
                target_col="Target", 
                max_samples=MAX_TRAINING_SAMPLES, 
                random_state=RANDOM_STATE,
                preserve_rare_positives=PRESERVE_RARE_POSITIVES
            )
        else:
            log.info("📊 Using multi-dimensional stratified sampling...")
            sampled_data = stratified_sample_large_dataset(
                df=combined_data, 
                target_col="Target", 
                max_samples=MAX_TRAINING_SAMPLES, 
                random_state=RANDOM_STATE
            )
        
        sampled_data = ensure_chronological_order(sampled_data)

        # Keep an ordering frame for time-series logic
        ordering_df = sampled_data[["snapshot_date"] + (["CrefoID"] if "CrefoID" in sampled_data.columns else [])].copy()

        # Build feature columns excluding leakage/ID/date fields
        feature_cols = [c for c in sampled_data.columns if c not in DROP_COLS]

        # Train matrices
        X_train_val = sampled_data[feature_cols]
        y_train_val = sampled_data["Target"].astype(int).values
        
        log.info(f"🎯 Memory-optimized training data: {len(X_train_val):,} samples")
    else:
        log.info(f"📊 Using full training dataset: {len(X_train_val):,} samples")
        # Ensure chronological order even if not sampling
        combined_data = X_train_val.copy()
        combined_data["Target"] = y_train_val
        combined_data = ensure_chronological_order(combined_data)

        ordering_df = combined_data[["snapshot_date"] + (["CrefoID"] if "CrefoID" in combined_data.columns else [])].copy()
        X_train_val = combined_data[feature_cols]
        y_train_val = combined_data["Target"].astype(int).values

        sampled_data = combined_data # For consistency

    log_memory_usage("After Train/Val Prep")

    # Update scale_pos_weight for XGBoost/LightGBM if used
    ratio = None
    if MODEL_BACKEND.startswith("xgb") or (MODEL_BACKEND.startswith("lgbm") and not LGBM_USE_UNBALANCE):
        n_pos = np.sum(y_train_val)
        n_neg = len(y_train_val) - n_pos
        ratio = n_neg / max(n_pos, 1)
        log.info(f"Calculated scale_pos_weight: {ratio:.4f} (Neg={n_neg}, Pos={n_pos})")
        # Update the parameter in the pipeline
        if MODEL_BACKEND.startswith("xgb"):
            pipe.set_params(classifier__scale_pos_weight=ratio)
        elif MODEL_BACKEND.startswith("lgbm") and not LGBM_USE_UNBALANCE:
            pipe.set_params(classifier__scale_pos_weight=ratio)

    if MODEL_BACKEND == "dnn":
        classes = np.unique(y_train_val)
        if len(classes) == 2:
            weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train_val)
            class_weight = {int(classes[0]): float(weights[0]), int(classes[1]): float(weights[1])}
            pipe.set_params(classifier__fit__class_weight=class_weight)
            log.info(f"Using DNN class_weight: {class_weight}")
        else:
            log.warning("DNN class_weight skipped: only one class present in training data.")

    # Multi-tier approach: Known params > Checkpoint > Full search
    if USE_BEST_KNOWN_PARAMS and not FORCE_NEW_SEARCH:
        # TIER 1: Use proven best parameters (fastest path - ~30x faster)
        log.info("🚀 FAST PATH: Using known best parameters from previous 3-hour search")
        
        if MODEL_BACKEND.startswith("xgb"):
            current_best_params = BEST_PARAMS_XGB
            # Ensure scale_pos_weight is kept from dynamic calculation if not in BEST_PARAMS
            if "classifier__scale_pos_weight" not in current_best_params:
                current_best_params = current_best_params.copy()
                current_best_params["classifier__scale_pos_weight"] = ratio
        elif MODEL_BACKEND.startswith("lgbm"):
            current_best_params = BEST_PARAMS_LGBM
            if not LGBM_USE_UNBALANCE and ratio is not None:
                if "classifier__scale_pos_weight" not in current_best_params:
                    current_best_params = current_best_params.copy()
                    current_best_params["classifier__scale_pos_weight"] = ratio
        elif MODEL_BACKEND == "dnn":
            current_best_params = BEST_PARAMS_DNN
        elif MODEL_BACKEND == "hgb_bagging":
            current_best_params = BEST_PARAMS_HGB_BAGGING
        else:
            current_best_params = BEST_PARAMS_HGB

        log.info(f"Best params: {current_best_params}")

        pipe.set_params(**current_best_params)
        fitted_pipeline = pipe.fit(X_train_val[feature_cols], y_train_val)
        
        # Create search result structure for compatibility with downstream code
        class BestParamsResult:
            def __init__(self):
                self.best_estimator_ = fitted_pipeline
                self.best_params_ = current_best_params
                self.best_score_ = None  # Unknown, will be determined during calibration
        
        search = BestParamsResult()
        log.info("✅ Pipeline fitted with optimal parameters (skipped 3-hour search)")
        
    elif ENABLE_CHECKPOINTING and checkpoint_exists() and not FORCE_NEW_SEARCH:
        # TIER 2: Load previous search results from checkpoint (~10x faster)
        log.info("⚡ CHECKPOINT PATH: Loading previous hyperparameter search results")
        fitted_pipeline, best_params, metadata = load_checkpoint()
        
        class CheckpointResult:
            def __init__(self, pipeline, params, meta):
                self.best_estimator_ = pipeline
                self.best_params_ = params
                self.best_score_ = meta.get("best_score")
        
        search = CheckpointResult(fitted_pipeline, best_params, metadata)
        log.info(f"Best params: {search.best_params_}")
        if search.best_score_:
            log.info(f"Best CV AP (PR-AUC): {search.best_score_:.5f}")

    else:
        # TIER 3: Full hyperparameter search with automatic checkpointing (original path)
        log.info("⏳ FULL SEARCH PATH: Running complete hyperparameter search (this may take time)")
        if FORCE_NEW_SEARCH:
            log.info("🔄 FORCE_NEW_SEARCH=True: Ignoring existing checkpoints")
        
        if MODEL_BACKEND.startswith("xgb"):
            # XGBoost Search Space
            param_distributions = {
                "classifier__max_depth": [3, 4, 6, 8],
                "classifier__learning_rate": np.logspace(-2.0, -0.5, 6), # 0.01 .. 0.3
                "classifier__n_estimators": [200, 400, 600, 800],
                "classifier__min_child_weight": [1, 5, 10],
                "classifier__subsample": [0.6, 0.8, 1.0],
                "classifier__colsample_bytree": [0.6, 0.8, 1.0],
                "classifier__reg_lambda": [0.1, 1.0, 5.0],
                "classifier__gamma": [0.0, 0.1, 1.0],
            }
        elif MODEL_BACKEND.startswith("lgbm"):
            # LightGBM Search Space (keep compact)
            param_distributions = {
                "classifier__learning_rate": np.logspace(-2.3, -0.7, 6),
                "classifier__n_estimators": [400, 800, 1200],
                "classifier__num_leaves": [31, 63, 127],
                "classifier__max_depth": [5, 7, 9],
                "classifier__subsample": [0.6, 0.8, 1.0],
                "classifier__colsample_bytree": [0.6, 0.8, 1.0],
                "classifier__min_child_samples": [20, 50, 100],
            }
        elif MODEL_BACKEND == "hgb_bagging":
            # BalancedBagging Search Space (base estimator params)
            estimator_param = get_balanced_bagging_estimator_param_name()
            param_distributions = {
                "classifier__n_estimators": [20, 30, 40],
                f"classifier__{estimator_param}__learning_rate": np.logspace(-2.0, -0.7, 4),
                f"classifier__{estimator_param}__max_iter": [200, 300],
                f"classifier__{estimator_param}__max_depth": [6, 8, None],
                f"classifier__{estimator_param}__min_samples_leaf": [20, 50],
                f"classifier__{estimator_param}__l2_regularization": [0.1, 0.5, 1.0],
            }
        elif MODEL_BACKEND == "dnn":
            # DNN Search Space (keep small; early stopping controls training time)
            param_distributions = {
                "classifier__model__hidden_units": [(128, 64), (256, 128), (256, 128, 64)],
                "classifier__model__dropout": [0.0, 0.2, 0.4],
                "classifier__model__l2": [0.0, 1e-4, 1e-3],
                "classifier__model__learning_rate": [1e-3, 3e-4, 1e-4],
                "classifier__batch_size": [2048, 4096, 8192],
                "classifier__epochs": [10, 20, 30],
            }
        else:
            # HGB Search Space
            param_distributions = {
                "classifier__learning_rate": np.logspace(-2.3, -0.7, 8),  # ~0.005..0.2
                "classifier__max_iter": [100, 200, 300],
                "classifier__max_leaf_nodes": [15, 31, 63],
                "classifier__min_samples_leaf": [10, 20, 50],
                "classifier__l2_regularization": [0.0, 0.1, 0.5, 1.0],
            }

        # Compute time-meaningful gap on TRAIN+VAL only (≈2 months)
        gap_samples = compute_ts_gap_samples(sampled_data, months_gap=2)
        
        tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=gap_samples)
        scoring = {
            "average_precision": "average_precision",
            "precision_at_k": precision_at_k_scorer_factory(LEAD_CAPACITY_K)
        }

        log.info(f"Setting up RandomizedSearchCV: n_iter={N_ITER}, folds={N_SPLITS}, gap={gap_samples} samples")
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_distributions,
            n_iter=N_ITER,
            scoring=scoring,
            refit="average_precision",
            cv=tscv,
            n_jobs=N_JOBS_SEARCH,
            random_state=RANDOM_STATE,
            verbose=3,
            pre_dispatch=1,
            error_score="raise"  # fail fast to surface memory errors
        )

        # 8) Fit search and save checkpoint immediately
        search.fit(X_train_val, y_train_val)
        
        log.info(f"Best params: {search.best_params_}")
        log.info(f"Best CV AP (PR-AUC): {search.best_score_:.5f}")
        
        # Save checkpoint immediately after successful search
        save_checkpoint(search, metadata={"feature_count": len(feature_cols)})
        log.info("💾 Checkpoint saved: Future runs will use fast path")

    # 9) Time-aware calibration on Train+Val
    # sampled_data should include snapshot_date; ensure it is ordered
    sampled_data = ensure_chronological_order(sampled_data)

    # Compute gap from the ordered frame that includes snapshot_date
    gap = compute_ts_gap_samples(sampled_data, months_gap=2)
    cal_cv = TimeSeriesSplit(n_splits=CAL_SPLITS, gap=gap)
    
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    calibration_method = CALIBRATION_METHOD
    if calibration_method not in {"isotonic", "sigmoid", "beta"}:
        log.warning(f"Unknown calibration method '{CALIBRATION_METHOD}', defaulting to isotonic.")
        calibration_method = "isotonic"

    if calibration_method == "beta":
        if not HAVE_BETACAL:
            log.warning("Beta calibration requested but betacal not installed; falling back to isotonic.")
            calibration_method = "isotonic"
        else:
            log.info("Calibrating with beta calibration (out-of-fold time-series predictions).")
            calibrated = fit_beta_calibrated_model(
                estimator=search.best_estimator_,
                X=X_train_val,
                y=y_train_val,
                cv=cal_cv
            )

    if calibration_method in {"isotonic", "sigmoid"}:
        calibrated = make_calibrated_classifier(
            estimator=search.best_estimator_,
            method=calibration_method,
            cv=cal_cv
        )
        calibrated.fit(X_train_val, y_train_val)

    # Save calibrated model for future use without re-calibration
    try:
        calibrated_path = os.path.join(ARTIFACTS_DIR, "calibrated_model.joblib")
        if HAVE_JOBLIB:
            dump(calibrated, calibrated_path)
        else:
            # Fallback to pickle
            with open(calibrated_path.replace('.joblib', '.pkl'), 'wb') as f:
                pickle.dump(calibrated, f)
        log.info(f"💾 Calibrated model saved to {calibrated_path}")
    except Exception as e:
        log.warning(f"Failed to save calibrated model: {e}")

    # 9b) Threshold optimization on validation snapshot
    X_val = df_val_eng[feature_cols]
    y_val = df_val["Target"].values
    p_val = calibrated.predict_proba(X_val)[:, 1]
    ap_val = average_precision_score(y_val, p_val)
    p_at_k_val = precision_at_k(y_val, p_val, LEAD_CAPACITY_K)

    log.info(f"VAL AP (PR-AUC): {ap_val:.5f}")
    log.info(f"VAL Precision@{LEAD_CAPACITY_K}: {p_at_k_val:.5f}")

    if ENABLE_THRESHOLD_OPTIMIZATION:
        fbeta_threshold, fbeta_best = optimize_threshold_fbeta(y_val, p_val, beta=THRESHOLD_BETA)
    else:
        fbeta_threshold, fbeta_best = 0.5, float("nan")

    topk_threshold = threshold_for_top_k(p_val, LEAD_CAPACITY_K)
    val_fbeta_metrics = metrics_at_threshold(y_val, p_val, fbeta_threshold, beta=THRESHOLD_BETA)
    val_topk_metrics = metrics_at_threshold(y_val, p_val, topk_threshold, beta=THRESHOLD_BETA)
    val_brier = brier_score_loss(y_val, p_val)
    val_bs_pos, val_bs_neg = class_stratified_brier(y_val, p_val)
    val_ece = ece_at_k(y_val, p_val, LEAD_CAPACITY_K)

    log.info(f"VAL F{THRESHOLD_BETA:.1f} optimal threshold: {fbeta_threshold:.6f} (score={fbeta_best:.6f})")
    log.info(f"VAL Top-K threshold (K={LEAD_CAPACITY_K}): {topk_threshold:.6f}")
    # 10) Evaluate on Test (chronologically last unique date)
    X_test = df_test_eng[feature_cols]
    y_test = df_test["Target"].values
    p_test = calibrated.predict_proba(X_test)[:, 1]
    ap_test = average_precision_score(y_test, p_test)
    p_at_k = precision_at_k(y_test, p_test, LEAD_CAPACITY_K)

    log.info(f"TEST AP (PR-AUC): {ap_test:.5f}")
    log.info(f"TEST Precision@{LEAD_CAPACITY_K}: {p_at_k:.5f}")

    test_fbeta_metrics = metrics_at_threshold(y_test, p_test, fbeta_threshold, beta=THRESHOLD_BETA)
    test_topk_metrics = metrics_at_threshold(y_test, p_test, topk_threshold, beta=THRESHOLD_BETA)
    test_brier = brier_score_loss(y_test, p_test)
    test_bs_pos, test_bs_neg = class_stratified_brier(y_test, p_test)
    test_ece = ece_at_k(y_test, p_test, LEAD_CAPACITY_K)

    log.info(f"TEST F{THRESHOLD_BETA:.1f} at val-opt threshold: {test_fbeta_metrics['f_beta']:.6f}")
    log.info(f"TEST Top-K threshold precision: {test_topk_metrics['precision']:.6f}")

    threshold_report = {
        "calibration_method": calibration_method,
        "fbeta_beta": THRESHOLD_BETA,
        "fbeta_threshold": fbeta_threshold,
        "top_k_threshold": topk_threshold,
        "k": LEAD_CAPACITY_K,
        "validation": {
            "ap": ap_val,
            "precision_at_k": p_at_k_val,
            "fbeta_metrics": val_fbeta_metrics,
            "topk_metrics": val_topk_metrics,
            "brier": val_brier,
            "brier_pos": val_bs_pos,
            "brier_neg": val_bs_neg,
            "ece_at_k": val_ece,
        },
        "test": {
            "ap": ap_test,
            "precision_at_k": p_at_k,
            "fbeta_metrics": test_fbeta_metrics,
            "topk_metrics": test_topk_metrics,
            "brier": test_brier,
            "brier_pos": test_bs_pos,
            "brier_neg": test_bs_neg,
            "ece_at_k": test_ece,
        },
    }

    try:
        thresholds_path = os.path.join(ARTIFACTS_DIR, "thresholds.json")
        with open(thresholds_path, "w", encoding="utf-8") as handle:
            json.dump(threshold_report, handle, indent=2, ensure_ascii=True)
        log.info(f"Saved threshold report to {thresholds_path}")
    except Exception as e:
        log.warning(f"Failed to save threshold report: {e}")

    # Gains table on test
    gt = gains_table(pd.DataFrame({"Target": y_test, "p_convert": p_test}))
    gt.to_csv(os.path.join(OUTDIR, "gains_table_test.csv"), index=True)
    log.info("Saved gains_table_test.csv")

    # 11) Score CURRENT prospects (latest snapshot)
    X_curr = df_curr_eng[feature_cols]
    p_curr = calibrated.predict_proba(X_curr)[:, 1]
    scored = df_current[["CrefoID", "Name_Firma", "snapshot_date"]].copy()
    scored["p_convert"] = p_curr

    # Rank & deciles for sales
    scored = scored.sort_values("p_convert", ascending=False).reset_index(drop=True)
    scored["rank"] = np.arange(1, len(scored) + 1)
    scored["decile"] = pd.qcut(scored["rank"], q=10, labels=False) + 1

    # Export CSV
    ts_tag = datetime.now().strftime("%Y%m%d_%H%M")
    out_csv = os.path.join(OUTDIR, f"ranked_leads_{ts_tag}.csv")
    scored.to_csv(out_csv, index=False)
    log.info(f"Saved ranked leads to {out_csv}")

    #12) (Optional) Write to SQL table for BI/CRM pickup
    scored.to_sql("lead_generation_rankings", con=engine, schema=SCHEMA,
                  if_exists="replace", index=False)
    log.info("Exported ranked leads to SQL (mitgliederstatistik.lead_generation_rankings)")

    log.info("=== Done ===")


if __name__ == "__main__":
    main()
