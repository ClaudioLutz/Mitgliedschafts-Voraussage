#!/usr/bin/env python3
"""
Script to run visualizations for the trained membership prediction model.

This script loads the trained model and attempts to recreate the train/test data
to generate comprehensive visualizations including feature importance, SHAP plots,
ROC curves, confusion matrices, and more.
"""

import os
import sys
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, URL, text
import urllib.parse

from joblib import load
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Import the visualization functions
from visualize_lead_model import make_all_viz

# Configuration (same as training script)
SERVER = "PRODSVCREPORT70"
DATABASE = "CAG_Analyse"
SCHEMA = "mitgliederstatistik"
HORIZON_MONTHS = 12
ARTIFACTS_DIR = "./artifacts"
FIGURES_DIR = "./figures"

def make_engine(server: str, database: str):
    """Create database engine with same config as training script."""
    odbc = urllib.parse.quote_plus(
        "Driver=ODBC Driver 17 for SQL Server;"
        f"Server={server};"
        f"Database={database};"
        "Trusted_Connection=Yes;"
    )
    eng = create_engine(URL.create("mssql+pyodbc", query={"odbc_connect": odbc}))
    return eng

def load_modeling_data(engine, horizon_months: int = 12) -> pd.DataFrame:
    """
    Load the same modeling data as used in training.
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
        AND (b.DT_LoeschungAusfall IS NULL OR b.DT_LoeschungAusfall > s.snapshot_date)
)
SELECT * FROM modeling;
"""
    log.info("Loading modeling data from database...")
    df = pd.read_sql_query(text(query), engine, parse_dates=["snapshot_date", "Eintritt", "Austritt"])
    return df

def split_by_unique_dates(df: pd.DataFrame, date_col: str = "snapshot_date"):
    """Split data the same way as training script."""
    unique_dates = sorted(df[date_col].dropna().unique().tolist())
    if len(unique_dates) < 3:
        raise ValueError("Need at least 3 distinct snapshot dates for train/val/test.")

    train_dates = unique_dates[:-2]
    val_dates   = [unique_dates[-2]]
    test_dates  = [unique_dates[-1]]

    df_train = df[df[date_col].isin(train_dates)].copy()
    df_val   = df[df[date_col].isin(val_dates)].copy()
    df_test  = df[df[date_col].isin(test_dates)].copy()

    return df_train, df_val, df_test

def temporal_feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal features like in training script."""
    out = df.copy()
    # Company age in years at snapshot
    if "Gruendung_Jahr" in out.columns:
        snap_year = out["snapshot_date"].dt.year
        out["Company_Age_Years"] = (
            snap_year - out["Gruendung_Jahr"].fillna(snap_year)
        ).clip(lower=0)
    else:
        out["Company_Age_Years"] = 0
    return out

def create_demo_data():
    """
    Create synthetic demo data if database connection fails.
    This allows the visualization to run even without database access.
    """
    log.info("Creating synthetic demo data...")
    
    # Create synthetic data with similar structure
    np.random.seed(42)
    n_samples = 5000
    
    # Generate synthetic features
    data = {
        'PLZ': np.random.choice([8000, 8001, 8002, 8003, 8004, 8005, 9000, 9001], n_samples),
        'Kanton': np.random.choice(['ZH', 'BE', 'GE', 'BS', 'SG'], n_samples),
        'Rechtsform': np.random.choice(['AG', 'GmbH', 'Einzelfirma'], n_samples),
        'BrancheText_06': np.random.choice(['Handel', 'Dienstleistung', 'Industrie', 'IT'], n_samples),
        'MitarbeiterBestand': np.random.lognormal(2, 1, n_samples).astype(int),
        'Umsatz': np.random.lognormal(14, 1, n_samples),
        'Risikoklasse': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'Company_Age_Years': np.random.exponential(10, n_samples),
    }
    
    # Create target (membership conversion) - make it realistic with ~5% positive rate
    target_prob = 0.05 + 0.02 * (data['Company_Age_Years'] / np.max(data['Company_Age_Years']))
    target_prob = np.clip(target_prob, 0.01, 0.15)
    data['Target'] = np.random.binomial(1, target_prob)
    
    df = pd.DataFrame(data)
    
    # Split into train/test (80/20)
    n_train = int(0.8 * len(df))
    df_train = df.iloc[:n_train].copy()
    df_test = df.iloc[n_train:].copy()
    
    return df_train, df_test

def main():
    """Main function to run visualizations."""
    log.info("=== Running Model Visualizations ===")
    
    # Load the trained model
    model_path = os.path.join(ARTIFACTS_DIR, "calibrated_model.joblib")
    if not os.path.exists(model_path):
        log.error(f"Trained model not found at {model_path}")
        log.error("Please run the training script first: python training_lead_generation_model.py")
        return
    
    log.info(f"Loading trained model from {model_path}")
    try:
        model = load(model_path)
        log.info("✅ Model loaded successfully")
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        return
    
    # Try to load real data from database
    try:
        engine = make_engine(SERVER, DATABASE)
        df_model = load_modeling_data(engine, horizon_months=HORIZON_MONTHS)
        df_model = df_model.sort_values("snapshot_date").reset_index(drop=True)
        
        # Split data same way as training
        df_train, df_val, df_test = split_by_unique_dates(df_model)
        
        # Add temporal features
        df_train_eng = temporal_feature_engineer(df_train)
        df_test_eng = temporal_feature_engineer(df_test)
        
        # Get feature columns (exclude leakage columns)
        DROP_COLS = {
            "Target", "Eintritt", "Austritt", "snapshot_date", "DT_LoeschungAusfall",
            "CrefoID", "Name_Firma"  # Also drop ID columns
        }
        feature_cols = [c for c in df_train_eng.columns if c not in DROP_COLS]
        
        X_train = df_train_eng[feature_cols]
        y_train = df_train["Target"].values
        X_test = df_test_eng[feature_cols] 
        y_test = df_test["Target"].values
        
        log.info(f"Loaded real data: {len(X_train)} train, {len(X_test)} test samples")
        log.info(f"Features: {len(feature_cols)} columns")
        log.info(f"Train conversion rate: {y_train.mean():.4f}")
        log.info(f"Test conversion rate: {y_test.mean():.4f}")
        
    except Exception as e:
        log.warning(f"Could not load real data from database: {e}")
        log.info("Falling back to synthetic demo data...")
        
        # Use synthetic data for demo
        df_train, df_test = create_demo_data()
        
        feature_cols = [c for c in df_train.columns if c != 'Target']
        X_train = df_train[feature_cols]
        y_train = df_train["Target"].values
        X_test = df_test[feature_cols]
        y_test = df_test["Target"].values
        
        log.info(f"Created synthetic data: {len(X_train)} train, {len(X_test)} test samples")
        log.info(f"Features: {len(feature_cols)} columns")
    
    # Ensure figures directory exists
    os.makedirs(FIGURES_DIR, exist_ok=True)
    log.info(f"Saving visualizations to: {os.path.abspath(FIGURES_DIR)}")
    
    # Run all visualizations
    try:
        make_all_viz(
            estimator=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_cols,
            figures_dir=FIGURES_DIR,
            shap_sample=1000
        )
        
        log.info("✅ All visualizations completed successfully!")
        log.info(f"📊 Check the '{FIGURES_DIR}' directory for the generated plots:")
        
        # List generated files
        if os.path.exists(FIGURES_DIR):
            files = os.listdir(FIGURES_DIR)
            for file in sorted(files):
                if file.endswith(('.png', '.jpg', '.pdf')):
                    log.info(f"   📈 {file}")
        
    except Exception as e:
        log.error(f"❌ Visualization failed: {e}")
        raise

if __name__ == "__main__":
    main()
