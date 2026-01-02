import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = "static/data"
STATUS_FILE = "static/status.json"

def generate_mock_data(n_samples=5000, n_months=24):
    """
    Generates synthetic SHAB data.
    """
    log.info("Generating synthetic SHAB data...")
    rng = np.random.default_rng(42)

    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.DateOffset(months=n_months)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    # Cantons
    cantons = ["ZH", "BE", "LU", "UR", "SZ", "OW", "NW", "GL", "ZG", "FR", "SO", "BS", "BL", "SH", "AR", "AI", "SG", "GR", "AG", "TG", "TI", "VD", "VS", "NE", "GE", "JU"]

    # Generate random events
    data = {
        "publicationDate": rng.choice(date_range, size=n_samples),
        "kanton": rng.choice(cantons, size=n_samples),
        # Weighted choice for event type: More new entries (HR01) than deletions (HR03)
        "shab_id": rng.choice(["HR01", "HR03"], size=n_samples, p=[0.6, 0.4])
    }

    df = pd.DataFrame(data)

    # Normalize
    df["publicationDate"] = pd.to_datetime(df["publicationDate"])
    df["kanton"] = df["kanton"].astype(str).str.upper().str.strip()
    df["shab_id"] = df["shab_id"].astype(str).str.upper()

    return df

def export_dashboard_data(df_shab, out_dir=DATA_DIR):
    """
    Aggregates and exports data for the dashboard.
    """
    os.makedirs(out_dir, exist_ok=True)

    log.info("Processing data for dashboard export...")

    # 1. Normalize Date to Month Start
    df_shab["month"] = df_shab["publicationDate"].dt.to_period("M").dt.to_timestamp()

    # 2. Aggregate Canton Monthly
    canton_counts = df_shab.groupby(["month", "kanton", "shab_id"]).size().reset_index(name="count")
    canton_counts["geo"] = "KT"

    # 3. Aggregate CH Monthly
    ch_counts = df_shab.groupby(["month", "shab_id"]).size().reset_index(name="count")
    ch_counts["geo"] = "CH"
    ch_counts["kanton"] = None

    # Combine
    all_counts = pd.concat([canton_counts, ch_counts], ignore_index=True)

    # 4. Compute NET (HR01 - HR03)
    # Pivot to get columns for HR01 and HR03
    # Note: pivot_table drops rows with NaN in index. We need to handle kanton=None for CH.
    # We replace None with a placeholder "CH_TOTAL" before pivot, then revert.
    all_counts["kanton_filled"] = all_counts["kanton"].fillna("CH_TOTAL")

    pivot_df = all_counts.pivot_table(index=["month", "geo", "kanton_filled"], columns="shab_id", values="count", fill_value=0).reset_index()

    # Revert kanton column
    pivot_df["kanton"] = pivot_df["kanton_filled"].replace("CH_TOTAL", None)
    pivot_df = pivot_df.drop(columns=["kanton_filled"])

    if "HR01" not in pivot_df.columns:
        pivot_df["HR01"] = 0
    if "HR03" not in pivot_df.columns:
        pivot_df["HR03"] = 0

    pivot_df["NET"] = pivot_df["HR01"] - pivot_df["HR03"]

    # Melt back to long format for NET
    net_df = pivot_df.melt(id_vars=["month", "geo", "kanton"], value_vars=["NET"], var_name="shab_id", value_name="count")

    # Combine all
    final_df = pd.concat([all_counts, net_df], ignore_index=True)

    # Rename shab_id to hr for consistency with plan
    final_df = final_df.rename(columns={"shab_id": "hr"})

    # Convert dates to string
    final_df["month"] = final_df["month"].dt.strftime("%Y-%m-%d")

    # Sort
    final_df = final_df.sort_values(["month", "geo", "kanton", "hr"])

    # Export shab_monthly.json
    out_file = os.path.join(out_dir, "shab_monthly.json")
    final_df.to_json(out_file, orient="records")
    log.info(f"Exported {len(final_df)} records to {out_file}")

    # Export dimensions.json
    dimensions = {
        "metrics": ["HR01", "HR03", "NET"],
        "cantons": sorted(df_shab["kanton"].unique().tolist()),
        "months": sorted(final_df["month"].unique().tolist())
    }
    dim_file = os.path.join(out_dir, "dimensions.json")
    with open(dim_file, "w") as f:
        json.dump(dimensions, f)
    log.info(f"Exported dimensions to {dim_file}")

    return len(final_df), dimensions

def update_status(record_count, dimensions):
    """
    Updates status.json
    """
    status = {
        "data_updated_at": datetime.utcnow().isoformat(),
        "records": record_count,
        "months_covered": len(dimensions["months"]),
        "cantons_covered": len(dimensions["cantons"]),
        "data_files": ["shab_monthly.json", "dimensions.json"]
    }

    with open(STATUS_FILE, "w") as f:
        json.dump(status, f, indent=2)
    log.info(f"Updated status file at {STATUS_FILE}")

def main():
    try:
        df = generate_mock_data(n_samples=10000)
        record_count, dimensions = export_dashboard_data(df)
        update_status(record_count, dimensions)
        log.info("Refresh completed successfully.")
    except Exception as e:
        log.error(f"Refresh failed: {e}")
        raise

if __name__ == "__main__":
    main()
