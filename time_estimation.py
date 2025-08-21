# time_estimation.py
# Estimate total runtime for your lead-gen training job by timing ONE fit on a small sample
# and extrapolating to (n_iter * n_splits + 1 refit + cal_splits).

import time
import math
import urllib.parse
from datetime import datetime
import numpy as np
import pandas as pd

from packaging.version import Version
from sqlalchemy import create_engine, URL, text

# --- scikit-learn bits
from sklearn import __version__ as sklearn_version
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier

# Prefer sklearn TargetEncoder; fallback to category_encoders if missing
try:
    from sklearn.preprocessing import TargetEncoder  # sklearn >= 1.3
except Exception:
    from category_encoders.target_encoder import TargetEncoder  # type: ignore

# ----------------- CONFIG -----------------
SERVER = "PRODSVCREPORT70"
DATABASE = "CAG_Analyse"
SCHEMA = "mitgliederstatistik"

HORIZON_MONTHS = 12
SAMPLE_FRACTION = 0.02     # 2% probe (adjust if you want faster/slower)
N_ITER = 5                # matches your training
N_SPLITS = 4               # time-series CV folds in your training
CAL_SPLITS = 3             # calibration folds
# -----------------------------------------

LEAKAGE_COLS = {"Target", "Eintritt", "Austritt", "snapshot_date"}

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

def temporal_feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "snapshot_date" not in out or "Gruendung_Jahr" not in out:
        return out
    snap_year = out["snapshot_date"].dt.year
    out["Company_Age_Years"] = (snap_year - out["Gruendung_Jahr"].fillna(snap_year)).clip(lower=0)
    out["Has_Employees"] = (out.get("MitarbeiterBestand", pd.Series(0)).fillna(0) > 0).astype(int)
    out["Has_Revenue"] = (out.get("Umsatz", pd.Series(0)).fillna(0) > 0).astype(int)
    return out

def auto_column_groups(df: pd.DataFrame, high_card_threshold: int = 20):
    cols = [c for c in df.columns if c not in LEAKAGE_COLS]
    num_cols, cat_cols = [], []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)
    low_card, high_card = [], []
    for c in cat_cols:
        nunique = df[c].nunique(dropna=True)
        (low_card if nunique <= high_card_threshold else high_card).append(c)
    return num_cols, low_card, high_card

def main():
    print("Connecting to DB…")
    engine = make_engine(SERVER, DATABASE)

    print("Counting modeling rows (snapshots with complete labels)…")
    n_total = count_modeling_rows(engine, HORIZON_MONTHS)
    if n_total == 0:
        raise SystemExit("No modeling rows found (check DB/table names and date ranges).")
    n_sample = max(2_000, int(math.ceil(SAMPLE_FRACTION * n_total)))
    print(f"Total modeling rows: {n_total:,}  |  Sampling: {n_sample:,} rows (~{100*SAMPLE_FRACTION:.1f}%)")

    print("Loading modeling SAMPLE from DB… (one-time)")
    df = load_modeling_sample(engine, HORIZON_MONTHS, n_sample)
    df = df.sort_values("snapshot_date").reset_index(drop=True)
    df = temporal_feature_engineer(df)

    y = df["Target"].to_numpy()
    X = df.drop(columns=list(LEAKAGE_COLS))

    # Column groups
    num_cols, low_cat_cols, high_cat_cols = auto_column_groups(df)

    # OneHotEncoder: version-safe flag
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median"))]), num_cols),
        ("lowcat", Pipeline([("ohe", ohe)]), low_cat_cols),
        ("highcat", Pipeline([("te", TargetEncoder())]), high_cat_cols),
    ], remainder="drop", sparse_threshold=0.3)

    # Pick ONE imbalance strategy: class_weight if available in this sklearn, else none (since this is only a probe)
    use_class_weight = Version(sklearn_version) >= Version("1.5")
    clf = HistGradientBoostingClassifier(
        random_state=42,
        early_stopping=False,
        class_weight=("balanced" if use_class_weight else None),
        max_iter=200, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.1
    )

    pipe = Pipeline([("pre", pre), ("clf", clf)])

    print("Timing ONE fit on the sample…")
    t0 = time.time()
    pipe.fit(X, y)
    t1 = time.time()
    minutes_per_fit_on_sample = (t1 - t0) / 60.0

    scale = n_total / max(len(X), 1)
    est_minutes_per_full_fit = minutes_per_fit_on_sample * scale
    total_fits = N_ITER * N_SPLITS + 1 + CAL_SPLITS
    est_total_minutes = est_minutes_per_full_fit * total_fits

    print("\n--- ESTIMATE ---")
    print(f"Sample rows: {len(X):,} | Minutes per fit on sample: ~{minutes_per_fit_on_sample:.2f}")
    print(f"Scale factor to full dataset: x{scale:.1f}")
    print(f"Estimated minutes per FULL fit: ~{est_minutes_per_full_fit:.1f}")
    print(f"Total fits ≈ N_ITER*N_SPLITS + 1 + CAL_SPLITS = {N_ITER}*{N_SPLITS} + 1 + {CAL_SPLITS} = {total_fits}")
    print(f"Estimated TOTAL runtime: ~{est_total_minutes:.0f} minutes")
    print("----------------")
    print("Note: This is a ballpark. If you change N_ITER/N_SPLITS/CAL_SPLITS or data size, re-run this script.")

if __name__ == "__main__":
    main()
