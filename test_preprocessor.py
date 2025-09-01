"""
Test Script: Validate Lead-Gen ColumnTransformer
===============================================

This script tests the ColumnTransformer implementation with sample data
to ensure it works correctly before integration.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from column_transformer_lead_gen import create_lead_gen_preprocessor, validate_preprocessor

def create_sample_data(n_samples=1000):
    """Create realistic sample data matching the database schema."""
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Sample data matching your database schema
    data = {
        # Identifiers (will be dropped)
        'CrefoID': [f'CH-{i:06d}' for i in range(n_samples)],
        'Name_Firma': [f'Company_{i}' for i in range(n_samples)],
        
        # Numeric features (keep as numeric, handle skewness)
        'MitarbeiterBestand': np.random.lognormal(mean=1.5, sigma=1.2, size=n_samples),
        'Umsatz': np.random.lognormal(mean=12, sigma=1.5, size=n_samples),
        'Risikoklasse': np.random.randint(1, 6, size=n_samples),
        'Gruendung_Jahr': np.random.randint(1980, 2020, size=n_samples),
        
        # Ordinal features (use as integers)
        'MitarbeiterBestandKategorieOrder': np.random.randint(1, 6, size=n_samples),
        'UmsatzKategorieOrder': np.random.randint(1, 8, size=n_samples),
        
        # Low-cardinality categoricals (One-Hot Encoding)
        'Kanton': np.random.choice(['ZH', 'BE', 'VD', 'AG', 'SG', 'LU', 'TI', 'BS'], size=n_samples),
        'Rechtsform': np.random.choice(['AG', 'GmbH', 'Einzelfirma', 'Kollektiv', 'Kommandit'], size=n_samples),
        'GroessenKategorie': np.random.choice(['Micro', 'Klein', 'Mittel', 'Gross'], size=n_samples),
        'V_Bestand_Kategorie': np.random.choice(['A', 'B', 'C', 'D'], size=n_samples),
        'RechtsCode': np.random.choice(['01', '02', '03', '04', '05'], size=n_samples),
        
        # High-cardinality categoricals (Target Encoding)
        'PLZ': np.random.randint(1000, 9999, size=n_samples),
        'BrancheCode_06': np.random.choice([f'{i:06d}' for i in range(100000, 999999)], size=n_samples),
        
        # Optional industry codes (usually redundant)
        'BrancheCode_02': np.random.choice([f'{i:02d}' for i in range(10, 99)], size=n_samples),
        'BrancheCode_04': np.random.choice([f'{i:04d}' for i in range(1000, 9999)], size=n_samples),
        
        # Redundant categorical versions (will be dropped)
        'MitarbeiterBestandKategorie': np.random.choice(['1-9', '10-49', '50-249', '250+'], size=n_samples),
        'UmsatzKategorie': np.random.choice(['<100K', '100K-1M', '1M-10M', '10M+'], size=n_samples),
        
        # Text versions (will be dropped - redundant with codes)
        'BrancheText_02': [f'Industry_{np.random.randint(1,20)}' for _ in range(n_samples)],
        'BrancheText_04': [f'SubIndustry_{np.random.randint(1,100)}' for _ in range(n_samples)],
        'BrancheText_06': [f'DetailedIndustry_{np.random.randint(1,500)}' for _ in range(n_samples)],
        
        # Location
        'Ort': [f'City_{np.random.randint(1,100)}' for _ in range(n_samples)],
        
        # Temporal data
        'snapshot_date': [datetime.now() - timedelta(days=np.random.randint(30, 1000)) for _ in range(n_samples)],
        
        # Membership data (leakage columns)
        'Eintritt': [None if np.random.random() > 0.1 else 
                    datetime.now() + timedelta(days=np.random.randint(1, 365)) 
                    for _ in range(n_samples)],
        'Austritt': [None] * n_samples,
        
        # Target variable
        'Target': np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    }
    
    df = pd.DataFrame(data)
    
    # Add some missing values to key columns (realistic)
    missing_mask = np.random.random(n_samples) < 0.15
    df.loc[missing_mask, 'MitarbeiterBestand'] = np.nan
    
    missing_mask = np.random.random(n_samples) < 0.12
    df.loc[missing_mask, 'Umsatz'] = np.nan
    
    missing_mask = np.random.random(n_samples) < 0.05
    df.loc[missing_mask, 'Gruendung_Jahr'] = np.nan
    
    # Add Company_Age_Years (this would normally be done by temporal_feature_engineer)
    snap_year = df['snapshot_date'].dt.year
    df['Company_Age_Years'] = (snap_year - df['Gruendung_Jahr'].fillna(snap_year)).clip(lower=0)
    
    return df


def test_column_transformer():
    """Test the ColumnTransformer with sample data."""
    
    print("🧪 Testing Lead-Gen ColumnTransformer")
    print("=" * 50)
    
    # 1. Create sample data
    print("1. Creating sample data...")
    df = create_sample_data(n_samples=1000)
    print(f"   ✅ Created {len(df)} samples with {len(df.columns)} columns")
    print(f"   ✅ Target distribution: {df['Target'].value_counts().to_dict()}")
    print(f"   ✅ Missing values: MitarbeiterBestand={df['MitarbeiterBestand'].isnull().sum()}, Umsatz={df['Umsatz'].isnull().sum()}")
    
    # 2. Test preprocessor creation
    print("\n2. Creating preprocessor...")
    try:
        preprocessor = create_lead_gen_preprocessor()
        print("   ✅ Preprocessor created successfully")
    except Exception as e:
        print(f"   ❌ Failed to create preprocessor: {e}")
        return False
    
    # 3. Test validation function
    print("\n3. Running validation...")
    try:
        results = validate_preprocessor(df, target_col='Target')
        print("   ✅ Validation successful!")
        print(f"   📊 Original shape: {results['original_shape']}")
        print(f"   📊 Transformed shape: {results['transformed_shape']}")
        print(f"   📊 Features created: {results['n_features']}")
        print(f"   📊 Missing indicators: {results['missing_indicators_added']}")
        print(f"   📊 Log transforms: {results['log_transforms_added']}")  
        print(f"   📊 PLZ groupings: {results['plz_groupings_added']}")
        
        # Show some feature names
        feature_names = results['feature_names']
        print(f"\n   🔍 Sample feature names (first 10):")
        for i, name in enumerate(feature_names[:10]):
            print(f"      {i+1:2d}. {name}")
        if len(feature_names) > 10:
            print(f"      ... and {len(feature_names)-10} more features")
            
    except Exception as e:
        print(f"   ❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. Test fit/transform process manually
    print("\n4. Testing fit/transform process...")
    try:
        # Remove leakage columns
        from column_transformer_lead_gen import DROP_COLS
        feature_cols = [col for col in df.columns if col not in DROP_COLS]
        X = df[feature_cols].copy()
        y = df['Target'].values
        
        print(f"   📋 Input features: {X.shape[1]} columns")
        print(f"   📋 Columns to process: {list(X.columns)}")
        
        # Fit and transform
        X_transformed = preprocessor.fit_transform(X, y)
        print(f"   ✅ Transform successful: {X.shape} → {X_transformed.shape}")
        
        # Check for any issues
        if np.isnan(X_transformed).any():
            print(f"   ⚠️  Warning: {np.isnan(X_transformed).sum()} NaN values in output")
        else:
            print("   ✅ No NaN values in transformed output")
            
        if np.isinf(X_transformed).any():
            print(f"   ⚠️  Warning: {np.isinf(X_transformed).sum()} infinite values in output")
        else:
            print("   ✅ No infinite values in transformed output")
            
    except Exception as e:
        print(f"   ❌ Fit/transform failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. Test with sklearn estimator
    print("\n5. Testing integration with sklearn estimator...")
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import cross_val_score
        
        # Create complete pipeline
        clf = HistGradientBoostingClassifier(
            random_state=42, 
            class_weight='balanced',
            max_iter=50  # Small for testing
        )
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        # Test cross-validation
        scores = cross_val_score(pipeline, X, y, cv=3, scoring='average_precision')
        print(f"   ✅ Pipeline integration successful")
        print(f"   📊 Cross-val AP scores: {scores}")
        print(f"   📊 Mean AP: {scores.mean():.4f} ± {scores.std():.4f}")
        
    except Exception as e:
        print(f"   ❌ Pipeline integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 All tests passed! ColumnTransformer is ready for integration.")
    return True


def test_specific_features():
    """Test specific feature engineering aspects."""
    
    print("\n🔬 Testing specific feature engineering...")
    print("=" * 50)
    
    # Create minimal test data with known values
    test_data = pd.DataFrame({
        'PLZ': [8001, 1000, None, 9999],
        'MitarbeiterBestand': [10, None, 50, 100],
        'Umsatz': [1000000, 500000, None, 2000000],
        'Company_Age_Years': [5, 10, 15, 20],
        'Target': [1, 0, 1, 0]
    })
    
    print("Input data:")
    print(test_data)
    
    # Test missing indicators
    from column_transformer_lead_gen import create_missing_indicators
    data_with_indicators = create_missing_indicators(test_data)
    print(f"\nMissing indicators added: {[col for col in data_with_indicators.columns if 'is_missing_' in col]}")
    
    # Test PLZ grouping
    from column_transformer_lead_gen import add_plz_grouping
    data_with_plz = add_plz_grouping(test_data)
    print(f"PLZ groupings added: {[col for col in data_with_plz.columns if 'PLZ' in col and col != 'PLZ']}")
    print("PLZ2 values:", data_with_plz['PLZ2'].tolist() if 'PLZ2' in data_with_plz.columns else None)
    print("PLZ3 values:", data_with_plz['PLZ3'].tolist() if 'PLZ3' in data_with_plz.columns else None)
    
    # Test log transforms
    from column_transformer_lead_gen import log_transform_skewed
    data_with_log = log_transform_skewed(test_data)
    print(f"Log transforms added: {[col for col in data_with_log.columns if '_log1p' in col]}")


if __name__ == "__main__":
    # Run all tests
    success = test_column_transformer()
    
    if success:
        test_specific_features()
        print(f"\n✅ SUCCESS: Lead-Gen ColumnTransformer is ready!")
        print("   → You can now integrate it into your training pipeline")
        print("   → See integration_guide.py for step-by-step instructions")
    else:
        print(f"\n❌ FAILED: Please fix the issues above before integration")
