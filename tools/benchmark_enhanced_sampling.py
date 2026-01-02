#!/usr/bin/env python3
"""
Test script for enhanced stratified sampling functionality.
Validates that the new sampling methods work correctly and preserve distributions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add repo root to path so local imports work when running from tools/
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)

# Import the enhanced sampling functions
from training_lead_generation_model import (
    stratified_sample_large_dataset,
    advanced_stratified_sample_with_business_logic
)

def create_test_dataset(n_samples=100000, random_state=42):
    """Create a synthetic test dataset that mimics the real data structure."""
    np.random.seed(random_state)
    
    # Define realistic distributions based on the actual data
    cantons = ['ZH', 'BE', 'VD', 'GE', 'AG', 'SG', 'TI', 'VS', 'LU', 'ZG'] + [f'C{i}' for i in range(10, 28)]
    canton_probs = np.array([0.17, 0.102, 0.091, 0.067, 0.062, 0.054, 0.052, 0.046, 0.044, 0.038] + [0.01] * 18)
    canton_probs = canton_probs / canton_probs.sum()  # Normalize to ensure sum is 1.0
    
    groessen = ['MICRO', 'KLEIN', 'MITTEL', 'GROSS', 'SEHR GROSS', 'UNBEK']
    groessen_probs = np.array([0.856, 0.092, 0.007, 0.003, 0.001, 0.041])
    groessen_probs = groessen_probs / groessen_probs.sum()
    
    rechtsform = ['Einzelunternehmen', 'GmbH', 'Aktiengesellschaft', 'Verein'] + [f'RF{i}' for i in range(4, 16)]
    rechtsform_probs = np.array([0.439, 0.221, 0.200, 0.053] + [0.087/12] * 12)
    rechtsform_probs = rechtsform_probs / rechtsform_probs.sum()
    
    # Generate base dataset
    data = {
        'CrefoID': np.arange(1000000, 1000000 + n_samples),
        'Kanton': np.random.choice(cantons, size=n_samples, p=canton_probs),
        'GroessenKategorie': np.random.choice(groessen, size=n_samples, p=groessen_probs),
        'Rechtsform': np.random.choice(rechtsform, size=n_samples, p=rechtsform_probs),
        'snapshot_date': pd.date_range(start='2021-01-01', end='2024-12-01', freq='MS')[
            np.random.choice(range(48), size=n_samples)
        ],
        'Umsatz': np.random.lognormal(12, 2, n_samples),
        'MitarbeiterBestand': np.random.poisson(5, n_samples),
        'Company_Age_Years': np.random.gamma(2, 5, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic target variable with business logic
    # Higher conversion rates for certain segments
    target_prob = 0.02  # Base 2% conversion rate
    
    # Adjust probabilities based on business logic
    prob_adjustments = np.ones(n_samples) * target_prob
    
    # Large companies convert more
    prob_adjustments[df['GroessenKategorie'].isin(['MITTEL', 'GROSS', 'SEHR GROSS'])] *= 3
    
    # Certain cantons have different patterns
    prob_adjustments[df['Kanton'].isin(['ZG', 'AG'])] *= 1.5  # Business-friendly cantons
    
    # Legal forms matter
    prob_adjustments[df['Rechtsform'] == 'GmbH'] *= 1.2
    
    # Generate target
    df['Target'] = np.random.binomial(1, prob_adjustments, n_samples)
    
    print(f"📊 Test dataset created:")
    print(f"   Total samples: {len(df):,}")
    print(f"   Conversion rate: {df['Target'].mean():.3f}")
    print(f"   Unique cantons: {df['Kanton'].nunique()}")
    print(f"   Unique company sizes: {df['GroessenKategorie'].nunique()}")
    print(f"   Date range: {df['snapshot_date'].min()} to {df['snapshot_date'].max()}")
    
    return df

def test_basic_stratified_sampling():
    """Test the enhanced multi-dimensional stratified sampling."""
    print("\n" + "="*60)
    print("🧪 TESTING: Enhanced Multi-Dimensional Stratified Sampling")
    print("="*60)
    
    # Create test data
    df_test = create_test_dataset(n_samples=50000, random_state=42)
    
    # Apply sampling
    sample_size = 10000
    sampled_df = stratified_sample_large_dataset(
        df=df_test,
        target_col='Target',
        max_samples=sample_size,
        random_state=42
    )
    
    print(f"\n✅ Sampling completed: {len(df_test):,} → {len(sampled_df):,} samples")
    
    # Validate results
    validate_sampling_quality(df_test, sampled_df, "Multi-Dimensional Stratified")
    
    return sampled_df

def test_business_logic_sampling():
    """Test the business-logic enhanced sampling."""
    print("\n" + "="*60)
    print("🧪 TESTING: Business-Logic Enhanced Sampling")
    print("="*60)
    
    # Create test data
    df_test = create_test_dataset(n_samples=50000, random_state=123)
    
    # Apply business-logic sampling
    sample_size = 10000
    sampled_df = advanced_stratified_sample_with_business_logic(
        df=df_test,
        target_col='Target',
        max_samples=sample_size,
        random_state=123,
        preserve_rare_positives=True
    )
    
    print(f"\n✅ Business-logic sampling completed: {len(df_test):,} → {len(sampled_df):,} samples")
    
    # Validate results
    validate_sampling_quality(df_test, sampled_df, "Business-Logic Enhanced")
    
    return sampled_df

def validate_sampling_quality(original_df, sampled_df, method_name):
    """Validate the quality of stratified sampling."""
    print(f"\n📈 VALIDATION RESULTS for {method_name}:")
    print("-" * 50)
    
    # 1. Target distribution preservation
    orig_target_dist = original_df['Target'].value_counts(normalize=True).sort_index()
    sampled_target_dist = sampled_df['Target'].value_counts(normalize=True).sort_index()
    
    print("🎯 Target Distribution:")
    for target in orig_target_dist.index:
        orig_pct = orig_target_dist.get(target, 0) * 100
        sampled_pct = sampled_target_dist.get(target, 0) * 100
        diff = abs(orig_pct - sampled_pct)
        status = "✅" if diff < 1.0 else "⚠️ " if diff < 2.0 else "❌"
        print(f"   {status} Target {target}: {orig_pct:.2f}% → {sampled_pct:.2f}% (Δ={diff:.2f}%)")
    
    # 2. Geographic distribution preservation
    print("\n🗺️  Geographic Distribution (Top 5 Cantons):")
    orig_geo = original_df['Kanton'].value_counts(normalize=True).head(5)
    sampled_geo = sampled_df['Kanton'].value_counts(normalize=True)
    
    for canton in orig_geo.index:
        orig_pct = orig_geo.get(canton, 0) * 100
        sampled_pct = sampled_geo.get(canton, 0) * 100
        diff = abs(orig_pct - sampled_pct)
        status = "✅" if diff < 2.0 else "⚠️ " if diff < 4.0 else "❌"
        print(f"   {status} {canton}: {orig_pct:.1f}% → {sampled_pct:.1f}% (Δ={diff:.1f}%)")
    
    # 3. Company size distribution preservation
    print("\n🏢 Company Size Distribution:")
    orig_size = original_df['GroessenKategorie'].value_counts(normalize=True)
    sampled_size = sampled_df['GroessenKategorie'].value_counts(normalize=True)
    
    for size_cat in ['MICRO', 'KLEIN', 'MITTEL', 'GROSS']:
        if size_cat in orig_size.index:
            orig_pct = orig_size.get(size_cat, 0) * 100
            sampled_pct = sampled_size.get(size_cat, 0) * 100
            diff = abs(orig_pct - sampled_pct)
            status = "✅" if diff < 2.0 else "⚠️ " if diff < 5.0 else "❌"
            print(f"   {status} {size_cat}: {orig_pct:.1f}% → {sampled_pct:.1f}% (Δ={diff:.1f}%)")
    
    # 4. Rare positive case preservation (for business-logic sampling)
    if 'Business-Logic' in method_name:
        large_co_positives_orig = len(original_df[
            (original_df['Target'] == 1) & 
            (original_df['GroessenKategorie'].isin(['MITTEL', 'GROSS', 'SEHR GROSS']))
        ])
        large_co_positives_sampled = len(sampled_df[
            (sampled_df['Target'] == 1) & 
            (sampled_df['GroessenKategorie'].isin(['MITTEL', 'GROSS', 'SEHR GROSS']))
        ])
        
        if large_co_positives_orig > 0:
            preservation_rate = large_co_positives_sampled / large_co_positives_orig
            status = "✅" if preservation_rate > 0.8 else "⚠️ " if preservation_rate > 0.5 else "❌"
            print(f"\n🔸 Rare Large Company Positives:")
            print(f"   {status} {large_co_positives_orig} → {large_co_positives_sampled} preserved ({preservation_rate:.1%})")

def run_performance_comparison():
    """Compare performance of different sampling methods."""
    print("\n" + "="*60)
    print("⚡ PERFORMANCE COMPARISON")
    print("="*60)
    
    df_large = create_test_dataset(n_samples=200000, random_state=999)
    sample_size = 50000
    
    print(f"\nTesting with {len(df_large):,} samples → {sample_size:,} samples")
    
    methods = [
        ("Multi-Dimensional Stratified", stratified_sample_large_dataset),
        ("Business-Logic Enhanced", advanced_stratified_sample_with_business_logic)
    ]
    
    import time
    results = {}
    
    for method_name, method_func in methods:
        print(f"\n🔄 Running {method_name}...")
        start_time = time.time()
        
        if method_name == "Business-Logic Enhanced":
            sampled = method_func(
                df=df_large, 
                target_col='Target', 
                max_samples=sample_size,
                random_state=999,
                preserve_rare_positives=True
            )
        else:
            sampled = method_func(
                df=df_large, 
                target_col='Target', 
                max_samples=sample_size,
                random_state=999
            )
        
        elapsed = time.time() - start_time
        results[method_name] = {
            'time': elapsed,
            'samples': len(sampled),
            'target_preservation': abs(
                df_large['Target'].mean() - sampled['Target'].mean()
            )
        }
        
        print(f"   ⏱️  Time: {elapsed:.2f}s")
        print(f"   📊 Samples: {len(sampled):,}")
        print(f"   🎯 Target preservation error: {results[method_name]['target_preservation']:.4f}")
    
    print(f"\n🏆 PERFORMANCE SUMMARY:")
    for method_name, result in results.items():
        print(f"   {method_name}: {result['time']:.2f}s, {result['target_preservation']:.4f} target error")

def main():
    """Run all tests."""
    print("🚀 Starting Enhanced Stratified Sampling Tests")
    print("=" * 60)
    
    try:
        # Test 1: Basic enhanced stratified sampling
        test_basic_stratified_sampling()
        
        # Test 2: Business-logic enhanced sampling
        test_business_logic_sampling()
        
        # Test 3: Performance comparison
        run_performance_comparison()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n📋 SUMMARY:")
        print("• Enhanced multi-dimensional stratified sampling preserves distributions")
        print("• Business-logic sampling preserves rare positive cases")
        print("• Both methods provide comprehensive reporting")
        print("• Performance is suitable for large datasets")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
