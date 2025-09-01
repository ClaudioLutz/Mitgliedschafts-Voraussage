"""
Lead-Gen Oriented ColumnTransformer Implementation
==================================================
Implements the exact preprocessing strategy specified for membership prediction:
- Numeric features with missing indicators
- Ordinal features as integers  
- Low-cardinality → One-Hot Encoding
- High-cardinality → Target Encoding with cross-fitting
- Feature engineering (Company_Age_Years, missing flags, PLZ grouping)
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, FunctionTransformer
from sklearn.impute import SimpleImputer

# TargetEncoder: prefer sklearn (has internal cross-fitting); fallback to category_encoders
try:
    from sklearn.preprocessing import TargetEncoder  # sklearn >= 1.3
    SKLEARN_TARGET_ENCODER = True
except ImportError:
    from category_encoders.target_encoder import TargetEncoder
    SKLEARN_TARGET_ENCODER = False


class FeatureEngineeringTransformer:
    """Custom transformer for feature engineering with proper feature names support."""
    
    def __init__(self):
        self.feature_names_in_ = None
        self.feature_names_out_ = None
    
    def fit(self, X, y=None):
        """Fit the transformer and determine output feature names."""
        self.feature_names_in_ = list(X.columns) if hasattr(X, 'columns') else [f'x{i}' for i in range(X.shape[1])]
        
        # Determine output feature names
        output_names = list(self.feature_names_in_)
        
        # Add missing indicator names
        missing_cols = ['MitarbeiterBestand', 'Umsatz']
        for col in missing_cols:
            if col in output_names:
                output_names.append(f'is_missing_{col}')
        
        # Add PLZ grouping names
        if 'PLZ' in output_names:
            output_names.extend(['PLZ2', 'PLZ3'])
        
        # Add log transform names  
        skewed_cols = ['MitarbeiterBestand', 'Umsatz']
        for col in skewed_cols:
            if col in output_names:
                output_names.append(f'{col}_log1p')
        
        self.feature_names_out_ = output_names
        return self
    
    def transform(self, X):
        """Apply all feature engineering steps."""
        X_out = X.copy() if hasattr(X, 'copy') else pd.DataFrame(X, columns=self.feature_names_in_)
        
        # 1. Create missing indicators
        missing_cols = ['MitarbeiterBestand', 'Umsatz']
        for col in missing_cols:
            if col in X_out.columns:
                X_out[f'is_missing_{col}'] = X_out[col].isnull().astype(int)
        
        # 2. Add PLZ grouping
        if 'PLZ' in X_out.columns:
            # Convert to string to handle NaN properly
            plz_str = X_out['PLZ'].astype(str)
            
            # Extract PLZ2 and PLZ3 prefixes (more stable than full PLZ)
            X_out['PLZ2'] = plz_str.str[:2]
            X_out['PLZ3'] = plz_str.str[:3]
            
            # Keep NaN/invalid values as strings to avoid mixed types
            X_out['PLZ2'] = X_out['PLZ2'].replace('na', 'missing')
            X_out['PLZ3'] = X_out['PLZ3'].replace('na', 'missing')
        
        # 3. Apply log transforms to skewed features
        skewed_cols = ['MitarbeiterBestand', 'Umsatz']
        for col in skewed_cols:
            if col in X_out.columns:
                # log(1+x) transform to handle zeros and skewness
                X_out[f'{col}_log1p'] = np.log1p(X_out[col].fillna(0).clip(lower=0))
        
        return X_out
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names for output."""
        return np.array(self.feature_names_out_)


# Keep the original functions for backward compatibility
def create_missing_indicators(X):
    """Create missing value indicators for specified columns."""
    X_out = X.copy()
    
    # Add missing indicators for key business metrics
    missing_cols = ['MitarbeiterBestand', 'Umsatz']
    for col in missing_cols:
        if col in X_out.columns:
            X_out[f'is_missing_{col}'] = X_out[col].isnull().astype(int)
    
    return X_out


def add_plz_grouping(X):
    """Add PLZ2/PLZ3 geographical grouping to reduce cardinality."""
    X_out = X.copy()
    
    if 'PLZ' in X_out.columns:
        # Convert to string to handle NaN properly
        plz_str = X_out['PLZ'].astype(str)
        
        # Extract PLZ2 and PLZ3 prefixes (more stable than full PLZ)
        X_out['PLZ2'] = plz_str.str[:2]
        X_out['PLZ3'] = plz_str.str[:3]
        
        # Keep NaN/invalid values as strings to avoid mixed types
        X_out['PLZ2'] = X_out['PLZ2'].replace('na', 'missing')
        X_out['PLZ3'] = X_out['PLZ3'].replace('na', 'missing')
    
    return X_out


def log_transform_skewed(X):
    """Apply log(1+x) transform to highly skewed numeric features."""
    X_out = X.copy()
    
    # Columns that are often highly skewed
    skewed_cols = ['MitarbeiterBestand', 'Umsatz']
    
    for col in skewed_cols:
        if col in X_out.columns:
            # log(1+x) transform to handle zeros and skewness
            X_out[f'{col}_log1p'] = np.log1p(X_out[col].fillna(0).clip(lower=0))
    
    return X_out


# =============================================================================
# Column Definitions (based on your specification)
# =============================================================================

# Numeric features (keep as numeric, consider transforms if skewed)
NUMERIC_COLS = [
    'MitarbeiterBestand',
    'Umsatz', 
    'Risikoklasse',
    'Company_Age_Years',           # Engineered feature
    'is_missing_MitarbeiterBestand',  # Missing indicator (engineered)
    'is_missing_Umsatz',              # Missing indicator (engineered)
    'MitarbeiterBestand_log1p',       # Log transform (engineered)
    'Umsatz_log1p'                    # Log transform (engineered)
]

# Ordinal features (use as integers) 
ORDINAL_COLS = [
    'MitarbeiterBestandKategorieOrder',
    'UmsatzKategorieOrder'
]

# Low-cardinality categoricals → One-Hot Encoding
LOW_CARD_CATEGORICAL_COLS = [
    'Kanton',
    'Rechtsform', 
    'GroessenKategorie',
    'V_Bestand_Kategorie',
    'RechtsCode'
]

# High-cardinality categoricals → Target Encoding
HIGH_CARD_CATEGORICAL_COLS = [
    'PLZ',           # Full PLZ for target encoding
    'PLZ2',          # Grouped PLZ (engineered)  
    'PLZ3',          # Grouped PLZ (engineered)
    'BrancheCode_06' # Most granular industry code
]

# Optional: Keep other industry codes only if they show extra signal
# 'BrancheCode_02', 'BrancheCode_04'  # Usually redundant with 06

# Columns to explicitly drop (leakage/identifiers/redundant)
DROP_COLS = [
    'CrefoID', 'Name_Firma',           # Identifiers
    'Eintritt', 'Austritt',            # Pure leakage (membership outcomes)
    'snapshot_date', 'Target',         # Leakage/target
    'DT_LoeschungAusfall',             # Used for snapshot filtering only, not as feature
    'BrancheText_02', 'BrancheText_04', 'BrancheText_06',  # Redundant with codes
    'MitarbeiterBestandKategorie',     # Redundant with Order version
    'UmsatzKategorie'                  # Redundant with Order version
]


# =============================================================================
# Pipeline Components  
# =============================================================================

# Numeric preprocessing pipeline (feature engineering done earlier in pipeline)
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', PowerTransformer(method='yeo-johnson', standardize=True))  # Handle skewness + scale
])

# Ordinal preprocessing (treat as integers, impute with median)
ordinal_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])

# Custom transformer to ensure categorical data is consistently string type
class CategoricalTypeConverter:
    """Ensure all categorical columns are string type to avoid mixed type errors."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Convert all values to string type and handle NaN properly."""
        X_out = X.copy() if hasattr(X, 'copy') else X
        
        # Convert to DataFrame if it's not already
        if not hasattr(X_out, 'columns'):
            import pandas as pd
            X_out = pd.DataFrame(X_out)
        
        # Convert all columns to string, handling NaN values
        for col in X_out.columns:
            X_out[col] = X_out[col].astype(str).replace('nan', 'missing')
        
        return X_out
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

# Low-cardinality categorical preprocessing with type conversion
try:
    # Handle sklearn version differences
    low_card_pipeline = Pipeline([
        ('type_converter', CategoricalTypeConverter()),
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False,        # sklearn >= 1.2
            min_frequency=0.01,         # Group rare categories (1% threshold)
            max_categories=20           # Cap categories per feature
        ))
    ])
except TypeError:
    # Fallback for older sklearn versions
    low_card_pipeline = Pipeline([
        ('type_converter', CategoricalTypeConverter()),
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

# High-cardinality categorical preprocessing with type conversion
high_card_pipeline = Pipeline([
    ('type_converter', CategoricalTypeConverter()),
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('target_encoder', TargetEncoder(
        smooth='auto',          # Automatic smoothing based on category frequency
        random_state=42
    ))
])


# =============================================================================
# Main ColumnTransformer 
# =============================================================================

def create_lead_gen_preprocessor():
    """
    Create the complete preprocessing pipeline for lead generation model.
    
    Implements the exact strategy from specification:
    - Numeric features with missing indicators and skewness handling
    - Ordinal features as integers
    - Low-card categoricals with One-Hot Encoding (rare category grouping)
    - High-card categoricals with Target Encoding (internal cross-fitting)
    - Feature engineering (Company_Age_Years, PLZ grouping, missing flags)
    
    Returns:
        sklearn.compose.ColumnTransformer: Complete preprocessing pipeline
    """
    
    # Add feature engineering step with proper feature names support
    feature_engineering = FeatureEngineeringTransformer()
    
    # Main preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_pipeline, NUMERIC_COLS),
            ('ordinal', ordinal_pipeline, ORDINAL_COLS), 
            ('low_card_cat', low_card_pipeline, LOW_CARD_CATEGORICAL_COLS),
            ('high_card_cat', high_card_pipeline, HIGH_CARD_CATEGORICAL_COLS)
        ],
        remainder='drop',  # Drop all other columns (including identifiers/leakage)
        sparse_threshold=0.3,  # Allow sparse output if dominated by one-hot encoding
        verbose_feature_names_out=False  # Cleaner feature names
    )
    
    # Combine feature engineering + preprocessing 
    complete_pipeline = Pipeline([
        ('feature_engineering', feature_engineering),
        ('preprocessing', preprocessor)
    ])
    
    return complete_pipeline


# =============================================================================
# Usage Example & Integration
# =============================================================================

def example_usage():
    """Example of how to integrate with existing training pipeline."""
    
    # Create the preprocessor
    preprocessor = create_lead_gen_preprocessor()
    
    # Example of how to integrate with model training
    from sklearn.ensemble import HistGradientBoostingClassifier
    
    # Complete pipeline: preprocessing + model
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', HistGradientBoostingClassifier(
            random_state=42,
            class_weight='balanced',  # Handle class imbalance
            early_stopping=False
        ))
    ])
    
    return model_pipeline


# =============================================================================
# Validation & Testing
# =============================================================================

def validate_preprocessor(df_sample, target_col='Target'):
    """
    Validate the preprocessor on a sample dataset.
    
    Args:
        df_sample: Sample dataframe with all expected columns
        target_col: Name of target column
        
    Returns:
        dict: Validation results and feature info
    """
    
    preprocessor = create_lead_gen_preprocessor()
    
    # Prepare data (remove leakage columns)
    feature_cols = [col for col in df_sample.columns if col not in DROP_COLS]
    X = df_sample[feature_cols].copy()
    y = df_sample[target_col].values if target_col in df_sample.columns else None
    
    # Fit and transform
    if y is not None:
        X_transformed = preprocessor.fit_transform(X, y)
    else:
        X_transformed = preprocessor.fit_transform(X)
    
    # Get feature names after transformation
    feature_names = preprocessor.named_steps['preprocessing'].get_feature_names_out()
    
    results = {
        'original_shape': X.shape,
        'transformed_shape': X_transformed.shape,
        'feature_names': feature_names.tolist(),
        'n_features': len(feature_names),
        'missing_indicators_added': [name for name in feature_names if 'is_missing_' in name],
        'log_transforms_added': [name for name in feature_names if '_log1p' in name],
        'plz_groupings_added': [name for name in feature_names if 'PLZ2' in name or 'PLZ3' in name]
    }
    
    return results

if __name__ == "__main__":
    # Example usage
    preprocessor = create_lead_gen_preprocessor()
    print("Lead-Gen ColumnTransformer created successfully!")
    print(f"Numeric columns: {NUMERIC_COLS}")
    print(f"Ordinal columns: {ORDINAL_COLS}")
    print(f"Low-card categorical: {LOW_CARD_CATEGORICAL_COLS}")
    print(f"High-card categorical: {HIGH_CARD_CATEGORICAL_COLS}")
