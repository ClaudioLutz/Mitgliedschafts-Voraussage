#!/usr/bin/env python3
"""
Export training data from database to pickle for WSL/offline training.
Run this on Windows where DB connection works.
"""
import sys
import os
import pickle
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training_lead_generation_model import (
    load_modeling_data,
    make_engine,
    SERVER,
    DATABASE,
    HORIZON_MONTHS,
)

def main():
    """Export training data to pickle file."""
    print("=" * 60)
    print("🗄️  Exporting Training Data for WSL")
    print("=" * 60)
    
    # Create cache directory
    cache_dir = Path(__file__).parent.parent / "cache"
    cache_dir.mkdir(exist_ok=True)
    
    cache_file = cache_dir / "training_data.pkl"
    
    print(f"\n📦 Connecting to database: {SERVER}/{DATABASE}")
    engine = make_engine(SERVER, DATABASE)
    
    print(f"📥 Loading modeling data (horizon={HORIZON_MONTHS} months)...")
    df_model = load_modeling_data(engine, horizon_months=HORIZON_MONTHS)
    
    print(f"✅ Loaded {len(df_model):,} records")
    print(f"💾 Saving to: {cache_file}")
    
    # Save data and metadata
    cache_data = {
        "df_model": df_model,
        "horizon_months": HORIZON_MONTHS,
        "server": SERVER,
        "database": DATABASE,
        "export_timestamp": str(Path(cache_file).stat().st_mtime) if cache_file.exists() else "N/A"
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"✅ Export complete!")
    print(f"📊 Data shape: {df_model.shape}")
    print(f"📁 Cache file: {cache_file}")
    print(f"💽 File size: {cache_file.stat().st_size / 1024 / 1024:.1f} MB")
    print("\n🚀 Now you can run training in WSL without database!")
    print(f"   Use: --use-cache flag")
    print("=" * 60)

if __name__ == "__main__":
    main()
