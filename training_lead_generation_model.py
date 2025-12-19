

# training_lead_generation_model_refactored.py
# -------------------------------------------
# Lead-generation oriented, leakage-safe training with time-aware CV & calibration.
# - Uses only snapshots with complete labels (<= today - horizon).
# - Splits by unique snapshot dates; computes time-meaningful gap for TSCV.
# - TargetEncoder inside pipeline (internal cross-fitting).
# - ONE imbalance strategy: class_weight if available; else SMOTE.
# - Time-aware calibration via CalibratedClassifierCV(TimeSeriesSplit).
# - Ranks current prospects (latest snapshot) without using labels.

import os
import sys
import math
import logging
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from sqlalchemy import create_engine, URL, text
import urllib.parse

from sklearn import __version__ as sklearn_version
from packaging.version import Version

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import average_precision_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer

# --------------------
# Logging setup (must be before any log.warning calls)
# --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# TargetEncoder: prefer sklearn (has internal cross-fitting); fallback to category_encoders.
try:
    from sklearn.preprocessing import TargetEncoder  # sklearn >= 1.3
    SKLEARN_TARGET_ENCODER = True
except Exception:
    SKLEARN_TARGET_ENCODER = False
    from category_encoders.target_encoder import TargetEncoder  # type: ignore

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
import inspect

# Import new Lead-Gen preprocessor
from column_transformer_lead_gen import create_lead_gen_preprocessor, DROP_COLS, validate_preprocessor

# Imbalance handling
USE_CLASS_WEIGHT = Version(sklearn_version) >= Version("1.5")
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    HAVE_IMBLEARN = True
except Exception:
    HAVE_IMBLEARN = False

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
FORCE_NEW_SEARCH = False       # Override to always run hyperparameter search
ENABLE_CHECKPOINTING = True    # Save/load search results to avoid re-work

# Best hyperparameters from previous 3-hour search
BEST_PARAMS = {
    "classifier__min_samples_leaf": 50,
    "classifier__max_leaf_nodes": 63,
    "classifier__max_iter": 300,
    "classifier__learning_rate": 0.06963974029624322,
    "classifier__l2_regularization": 0.5,
}


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
MAX_TRAINING_SAMPLES = 1500000  # Maximum samples for training (increased for better stratification)
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

def auto_column_groups(df: pd.DataFrame,
                       high_card_threshold: int = 20,
                       numeric_override: list[str] | None = None):
    """Infer numeric/low-card/high-card categorical columns; drop leakage columns."""
    cols = [c for c in df.columns if c not in LEAKAGE_COLS]
    if numeric_override is None:
        numeric_override = []

    num_cols, cat_cols = [], []
    for c in cols:
        if c in numeric_override:
            num_cols.append(c)
        elif pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    # split categoricals by cardinality
    low_card, high_card = [], []
    for c in cat_cols:
        nunique = df[c].nunique(dropna=True)
        if nunique <= high_card_threshold:
            low_card.append(c)
        else:
            high_card.append(c)

    return num_cols, low_card, high_card


def temporal_feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add Company_Age_Years feature. Other feature engineering handled by Lead-Gen preprocessor."""
    out = df.copy()
    # Company age in years at snapshot (required by Lead-Gen ColumnTransformer)
    if "Gruendung_Jahr" in out.columns:
        snap_year = out["snapshot_date"].dt.year
        out["Company_Age_Years"] = (
            snap_year - out["Gruendung_Jahr"].fillna(snap_year)
        ).clip(lower=0)
    else:
        out["Company_Age_Years"] = 0

    return out




# --------------------
# Scorers & metrics
# --------------------
def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    k = int(min(k, len(y_score)))
    if k <= 0:
        return 0.0
    order = np.argsort(-y_score)[:k]
    return float(np.mean(np.asarray(y_true)[order] == 1))


def precision_at_k_scorer_factory(k: int):
    # scorer compatible with RandomizedSearchCV (signature: estimator, X, y)
    def _score(estimator, X, y):
        proba = estimator.predict_proba(X)[:, 1]
        return precision_at_k(np.asarray(y), proba, k)
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

    # 2) Split modeling data by unique dates
    df_model = df_model.sort_values("snapshot_date").reset_index(drop=True)
    df_train, df_val, df_test, train_dates, val_dates, test_dates = split_by_unique_dates(df_model)

    def rate(df): return df["Target"].mean() if len(df) else float("nan")
    log.info(f"Train: {len(df_train):,} (pos rate={rate(df_train):.4f}) "
             f"| Val: {len(df_val):,} (pos rate={rate(df_val):.4f}) "
             f"| Test: {len(df_test):,} (pos rate={rate(df_test):.4f})")

    # 3) Basic feature engineering (before ColumnTransformer)
    #    - add Company_Age_Years, Has_Employees, Has_Revenue
    #    - no leakage columns downstream
    df_train_eng = temporal_feature_engineer(df_train)
    df_val_eng   = temporal_feature_engineer(df_val)
    df_test_eng  = temporal_feature_engineer(df_test)
    df_curr_eng  = temporal_feature_engineer(df_current)

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
    pre = create_lead_gen_preprocessor()
    log.info("🔧 Using Lead-Gen ColumnTransformer with engineered features")

    # 6) Estimator & imbalance strategy
    #    Choose ONE: class_weight if supported (sklearn >= 1.5), else SMOTE
    if USE_CLASS_WEIGHT:
        clf = HistGradientBoostingClassifier(
            random_state=RANDOM_STATE,
            early_stopping=False,
            class_weight="balanced",
            max_depth=None,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.1
        )
        steps = [("preprocessor", pre), ("classifier", clf)]
        PipelineClass = Pipeline  # standard sklearn pipeline
        log.info("Imbalance via class_weight='balanced' (no SMOTE).")
    else:
        if not HAVE_IMBLEARN:
            raise RuntimeError("imbalanced-learn not installed; install or upgrade scikit-learn to >=1.5 for class_weight.")
        clf = HistGradientBoostingClassifier(
            random_state=RANDOM_STATE,
            early_stopping=False,
            max_depth=None,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.1
        )
        steps = [("preprocessor", pre), ("smote", SMOTE(random_state=RANDOM_STATE)), ("classifier", clf)]
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

    # Multi-tier approach: Known params > Checkpoint > Full search
    if USE_BEST_KNOWN_PARAMS and not FORCE_NEW_SEARCH:
        # TIER 1: Use proven best parameters (fastest path - ~30x faster)
        log.info("🚀 FAST PATH: Using known best parameters from previous 3-hour search")
        log.info(f"Best params: {BEST_PARAMS}")
        
        pipe.set_params(**BEST_PARAMS)
        fitted_pipeline = pipe.fit(X_train_val[feature_cols], y_train_val)
        
        # Create search result structure for compatibility with downstream code
        class BestParamsResult:
            def __init__(self):
                self.best_estimator_ = fitted_pipeline
                self.best_params_ = BEST_PARAMS
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
            verbose=1,
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

    calibrated = make_calibrated_classifier(
        estimator=search.best_estimator_,
        method="isotonic",   # or "sigmoid" if you prefer and have enough data
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

    # 10) Evaluate on Test (chronologically last unique date)
    X_test = df_test_eng[feature_cols]
    y_test = df_test["Target"].values
    p_test = calibrated.predict_proba(X_test)[:, 1]
    ap_test = average_precision_score(y_test, p_test)
    p_at_k = precision_at_k(y_test, p_test, LEAD_CAPACITY_K)

    log.info(f"TEST AP (PR-AUC): {ap_test:.5f}")
    log.info(f"TEST Precision@{LEAD_CAPACITY_K}: {p_at_k:.5f}")

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
