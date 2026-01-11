"""
Data Analysis Script for load_current_snapshot DataFrame
Extracts comprehensive numerical statistics, ranges, counts, and data insights
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# --- Centralized logging
from log_utils import setup_logging, get_logger
setup_logging(log_prefix="data_analysis")
log = get_logger(__name__)

# Import the database connection and function from the main script
from training_lead_generation_model import make_engine, load_current_snapshot, SERVER, DATABASE

def comprehensive_data_analysis(df, output_dir="./outputs"):
    """
    Perform comprehensive numerical analysis of the DataFrame
    """
    print("=" * 80)
    print("COMPREHENSIVE DATA ANALYSIS: load_current_snapshot DataFrame")
    print("=" * 80)
    
    # Basic info
    print(f"\n📊 DATASET OVERVIEW")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Data types distribution:")
    print(df.dtypes.value_counts().to_dict())
    
    # Missing values analysis
    print(f"\n🔍 MISSING VALUES ANALYSIS")
    missing_stats = []
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        missing_stats.append({
            'column': col,
            'missing_count': missing_count,
            'missing_percentage': missing_pct
        })
    
    missing_df = pd.DataFrame(missing_stats).sort_values('missing_percentage', ascending=False)
    print(f"Columns with missing values:")
    for _, row in missing_df[missing_df['missing_count'] > 0].iterrows():
        print(f"  {row['column']}: {row['missing_count']:,} ({row['missing_percentage']:.1f}%)")
    
    # Numerical columns analysis
    print(f"\n📈 NUMERICAL VARIABLES ANALYSIS")
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numerical_cols:
        print(f"Found {len(numerical_cols)} numerical columns:")
        
        for col in numerical_cols:
            print(f"\n  📊 {col.upper()}")
            stats = df[col].describe()
            print(f"    Count: {stats['count']:,.0f}")
            print(f"    Mean: {stats['mean']:,.2f}")
            print(f"    Std: {stats['std']:,.2f}")
            print(f"    Min: {stats['min']:,.2f}")
            print(f"    25%: {stats['25%']:,.2f}")
            print(f"    50%: {stats['50%']:,.2f}")
            print(f"    75%: {stats['75%']:,.2f}")
            print(f"    Max: {stats['max']:,.2f}")
            
            # Additional insights
            non_null_data = df[col].dropna()
            if len(non_null_data) > 0:
                print(f"    Range: {non_null_data.max() - non_null_data.min():,.2f}")
                print(f"    Unique values: {df[col].nunique():,}")
                print(f"    Zeros: {(df[col] == 0).sum():,} ({((df[col] == 0).sum() / len(df)) * 100:.1f}%)")
    
    # Categorical columns analysis
    print(f"\n📊 CATEGORICAL VARIABLES ANALYSIS")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_cols:
        print(f"\n  🏷️ {col.upper()}")
        value_counts = df[col].value_counts()
        print(f"    Unique values: {df[col].nunique():,}")
        print(f"    Most frequent: '{value_counts.index[0]}' ({value_counts.iloc[0]:,} times, {(value_counts.iloc[0]/len(df)*100):.1f}%)")
        
        # Show top 10 categories
        print(f"    Top categories:")
        for i, (value, count) in enumerate(value_counts.head(10).items()):
            print(f"      {i+1:2d}. {str(value)[:50]:<50} {count:>8,} ({count/len(df)*100:>5.1f}%)")
    
    # Date columns analysis
    print(f"\n📅 DATE VARIABLES ANALYSIS")
    date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    for col in date_cols:
        print(f"\n  📅 {col.upper()}")
        non_null_dates = df[col].dropna()
        if len(non_null_dates) > 0:
            print(f"    Count: {len(non_null_dates):,}")
            print(f"    Min date: {non_null_dates.min()}")
            print(f"    Max date: {non_null_dates.max()}")
            print(f"    Date range: {(non_null_dates.max() - non_null_dates.min()).days:,} days")
    
    # Year-based analysis for founding years
    if 'Gruendung_Jahr' in df.columns:
        print(f"\n  🏗️ FOUNDING YEAR ANALYSIS")
        founding_years = df['Gruendung_Jahr'].dropna()
        if len(founding_years) > 0:
            print(f"    Companies with founding year: {len(founding_years):,}")
            print(f"    Earliest: {founding_years.min():.0f}")
            print(f"    Latest: {founding_years.max():.0f}")
            print(f"    Most common decade:")
            
            # Decade analysis
            decades = (founding_years // 10 * 10).value_counts().head(10)
            for decade, count in decades.items():
                print(f"      {decade:.0f}s: {count:,} companies ({count/len(founding_years)*100:.1f}%)")
    
    # Geographic analysis
    print(f"\n🌍 GEOGRAPHIC DISTRIBUTION ANALYSIS")
    
    if 'Kanton' in df.columns:
        print(f"\n  🏞️ CANTON DISTRIBUTION")
        canton_counts = df['Kanton'].value_counts()
        print(f"    Total cantons: {df['Kanton'].nunique()}")
        print(f"    Top cantons:")
        for i, (canton, count) in enumerate(canton_counts.head(10).items()):
            print(f"      {i+1:2d}. {str(canton):<20} {count:>8,} ({count/len(df)*100:>5.1f}%)")
    
    if 'PLZ' in df.columns:
        print(f"\n  📮 POSTAL CODE ANALYSIS")
        print(f"    Unique postal codes: {df['PLZ'].nunique():,}")
        
        # Convert PLZ to numeric for analysis, handle non-numeric values
        plz_numeric = pd.to_numeric(df['PLZ'], errors='coerce').dropna()
        if len(plz_numeric) > 0:
            print(f"    Range: {plz_numeric.min():.0f} - {plz_numeric.max():.0f}")
            print(f"    Numeric PLZ values: {len(plz_numeric):,} ({len(plz_numeric)/len(df)*100:.1f}%)")
        
        # PLZ regions (rough Swiss regions) - work with original string values
        plz_regions = []
        for plz in df['PLZ'].dropna():
            try:
                plz_num = int(str(plz))
                if 1000 <= plz_num < 2000:
                    plz_regions.append('West (1xxx)')
                elif 2000 <= plz_num < 3000:
                    plz_regions.append('West (2xxx)')
                elif 3000 <= plz_num < 4000:
                    plz_regions.append('Bern (3xxx)')
                elif 4000 <= plz_num < 5000:
                    plz_regions.append('Basel (4xxx)')
                elif 5000 <= plz_num < 6000:
                    plz_regions.append('Aargau (5xxx)')
                elif 6000 <= plz_num < 7000:
                    plz_regions.append('Central (6xxx)')
                elif 7000 <= plz_num < 8000:
                    plz_regions.append('Graubünden (7xxx)')
                elif 8000 <= plz_num < 9000:
                    plz_regions.append('Zurich (8xxx)')
                elif 9000 <= plz_num < 10000:
                    plz_regions.append('East (9xxx)')
                else:
                    plz_regions.append('Other')
            except (ValueError, TypeError):
                plz_regions.append('Invalid/Other')
        
        if plz_regions:
            region_counts = pd.Series(plz_regions).value_counts()
            print(f"    Regional distribution:")
            for region, count in region_counts.items():
                print(f"      {region:<20} {count:>8,} ({count/len(plz_regions)*100:>5.1f}%)")
    
    # Industry analysis
    print(f"\n🏭 INDUSTRY ANALYSIS")
    industry_cols = [col for col in df.columns if 'Branche' in col and 'Text' in col]
    
    for col in industry_cols:
        if col in df.columns:
            print(f"\n  🏢 {col.upper()}")
            industry_counts = df[col].value_counts()
            print(f"    Unique industries: {df[col].nunique():,}")
            print(f"    Top industries:")
            for i, (industry, count) in enumerate(industry_counts.head(10).items()):
                print(f"      {i+1:2d}. {str(industry)[:50]:<50} {count:>8,} ({count/len(df)*100:>5.1f}%)")
    
    # Save detailed statistics to files
    os.makedirs(output_dir, exist_ok=True)
    
    # Numerical summary
    if numerical_cols:
        num_summary = df[numerical_cols].describe()
        num_summary.to_csv(os.path.join(output_dir, 'numerical_statistics.csv'))
        print(f"\n💾 Saved numerical statistics to: {output_dir}/numerical_statistics.csv")
    
    # Categorical summaries
    categorical_summaries = {}
    for col in categorical_cols:
        categorical_summaries[col] = df[col].value_counts()
    
    # Save top categories for each categorical column
    with open(os.path.join(output_dir, 'categorical_summaries.txt'), 'w', encoding='utf-8') as f:
        for col, counts in categorical_summaries.items():
            f.write(f"\n{col.upper()}\n")
            f.write("=" * len(col) + "\n")
            f.write(f"Unique values: {len(counts)}\n")
            f.write("Top 20 categories:\n")
            for i, (value, count) in enumerate(counts.head(20).items()):
                f.write(f"{i+1:3d}. {str(value):<50} {count:>8,} ({count/len(df)*100:>5.1f}%)\n")
    
    print(f"� Saved categorical summaries to: {output_dir}/categorical_summaries.txt")
    
    # Overall summary
    summary_stats = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'numerical_columns': len(numerical_cols),
        'categorical_columns': len(categorical_cols),
        'date_columns': len(date_cols),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'analysis_timestamp': datetime.now().isoformat()
    }
    
    pd.DataFrame([summary_stats]).to_csv(os.path.join(output_dir, 'data_summary.csv'), index=False)
    print(f"💾 Saved data summary to: {output_dir}/data_summary.csv")
    
    return df

def main():
    """
    Main execution function
    """
    log.info("Connecting to database...")
    try:
        engine = make_engine(SERVER, DATABASE)
        log.info("Database connection successful")

        log.info("Loading current snapshot data...")
        df = load_current_snapshot(engine)
        log.info(f"Data loaded successfully: {len(df):,} rows")

        # Perform comprehensive analysis
        analyzed_df = comprehensive_data_analysis(df)

        log.info("Analysis complete!")
        log.info("Check the './outputs' directory for detailed statistics files")

        return analyzed_df

    except Exception as e:
        log.error(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    df = main()
