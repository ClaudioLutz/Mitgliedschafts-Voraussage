#!/usr/bin/env python3
"""
Script to run visualizations for the trained membership prediction model.

This script loads the trained model and attempts to recreate the train/test data
to generate comprehensive visualizations including feature importance, SHAP plots,
ROC curves, confusion matrices, and more.
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, URL, text
import urllib.parse

from joblib import load
import warnings
warnings.filterwarnings('ignore')

# --- Centralized logging
from log_utils import setup_logging, get_logger
setup_logging(log_prefix="visualizations")
log = get_logger(__name__)

# Import the visualization functions
from visualize_lead_model import make_all_viz
from column_transformer_lead_gen import DROP_COLS

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
        AND (b.DT_LoeschungAusfall IS NULL
             OR b.DT_LoeschungAusfall = '1888-12-31'
             OR b.DT_LoeschungAusfall > s.snapshot_date)
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

def create_demo_data(n_samples: int = 5000, n_snapshots: int = 5):
    """
    Create synthetic demo data with the SAME schema expected by the trained pipeline.
    """
    log.info("Creating synthetic demo data (schema-aligned)...")
    rng = np.random.default_rng(42)

    # Ensure >= 3 unique snapshot dates for split_by_unique_dates()
    snapshot_dates = pd.date_range(
        end=pd.Timestamp.today().normalize(),
        periods=n_snapshots,
        freq="MS"
    )

    df = pd.DataFrame({
        "CrefoID": rng.integers(1_000_000, 9_999_999, size=n_samples),
        "Name_Firma": [f"DemoFirma_{i:05d}" for i in range(n_samples)],
        "Gruendung_Jahr": rng.integers(1950, pd.Timestamp.today().year + 1, size=n_samples),
        "PLZ": rng.integers(1000, 9999, size=n_samples),
        "Kanton": rng.choice(["ZH", "BE", "VD", "GE", "AG", "SG", "TI", "VS", "LU", "ZG"], size=n_samples),
        "Rechtsform": rng.choice(["Einzelunternehmen", "GmbH", "Aktiengesellschaft", "Verein", "Genossenschaft"], size=n_samples),
        "BrancheCode_06": rng.choice(["620100", "471100", "692000", "412000", "862100"], size=n_samples),
        "BrancheText_06": rng.choice(["IT", "Handel", "Treuhand", "Bau", "Gesundheit"], size=n_samples),

        "MitarbeiterBestand": rng.lognormal(2.0, 1.0, size=n_samples).round().astype(int),
        "MitarbeiterBestandKategorieOrder": rng.integers(1, 7, size=n_samples),
        "MitarbeiterBestandKategorie": rng.choice(["0", "1-9", "10-49", "50-249", "250+"], size=n_samples),

        "Umsatz": rng.lognormal(14.0, 1.0, size=n_samples),
        "UmsatzKategorieOrder": rng.integers(1, 7, size=n_samples),
        "UmsatzKategorie": rng.choice(["<1M", "1-5M", "5-20M", "20-100M", ">100M"], size=n_samples),

        # Keep numeric (your pipeline treats this as numeric)
        "Risikoklasse": rng.integers(1, 5, size=n_samples),

        "Ort": rng.choice(["Zürich", "Bern", "Lausanne", "Genève", "Basel", "St. Gallen"], size=n_samples),
        "RechtsCode": rng.choice(["AG", "GMBH", "EINF", "VEREIN", "GEN"], size=n_samples),
        "GroessenKategorie": rng.choice(["MICRO", "KLEIN", "MITTEL", "GROSS"], size=n_samples),
        "V_Bestand_Kategorie": rng.choice(["NEU", "BESTAND", "ALT"], size=n_samples),

        "BrancheCode_02": rng.choice(["62", "47", "69", "41", "86"], size=n_samples),
        "BrancheCode_04": rng.choice(["6201", "4711", "6920", "4120", "8621"], size=n_samples),
        "BrancheText_02": rng.choice(["IT", "Handel", "Treuhand", "Bau", "Gesundheit"], size=n_samples),
        "BrancheText_04": rng.choice(["IT", "Handel", "Treuhand", "Bau", "Gesundheit"], size=n_samples),

        # Leakage columns present in real data; set to NaT in demo
        "Eintritt": pd.NaT,
        "Austritt": pd.NaT,
        "DT_LoeschungAusfall": pd.NaT,

        "snapshot_date": pd.to_datetime(rng.choice(snapshot_dates, size=n_samples)),
    })

    # Create a realistic-ish target (~3–12% depending on some drivers)
    age_years = (df["snapshot_date"].dt.year - df["Gruendung_Jahr"]).clip(lower=0)
    prob = (
        0.03
        + 0.02 * (df["GroessenKategorie"].isin(["MITTEL", "GROSS"]).astype(int))
        + 0.01 * (df["Risikoklasse"] <= 2).astype(int)
        + 0.02 * np.tanh((age_years - 5) / 10)
    )
    prob = np.clip(prob, 0.01, 0.20)
    df["Target"] = rng.binomial(1, prob)

    # Sort by time and split like training logic expects
    df = df.sort_values("snapshot_date").reset_index(drop=True)
    df_train, _, df_test = split_by_unique_dates(df)

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
        
        df_train_eng = temporal_feature_engineer(df_train)
        df_test_eng = temporal_feature_engineer(df_test)

        feature_cols = [c for c in df_train_eng.columns if c not in DROP_COLS]

        X_train = df_train_eng[feature_cols]
        y_train = df_train["Target"].values
        X_test = df_test_eng[feature_cols]
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
