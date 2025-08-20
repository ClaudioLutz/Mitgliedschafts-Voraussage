"""
Lead-Generation Propensity Model Training Script
===============================================

This script trains a leakage-safe, business-ready model that predicts:
"Will a current non-member convert to membership within the next 12 months?"

Key improvements implemented:
- Forward-looking labels with temporal horizon (12-month conversion window)
- Chronological time-based splits prevent temporal leakage
- Target encoding with cross-fitting prevents label leakage  
- Lead-optimized metrics (Precision@K, PR-AUC) for sales capacity planning
- Calibrated probabilities for business decisions with proper held-out validation
- Production-ready pipeline with lead rankings and decile analysis
- Proper temporal feature engineering before preprocessing
"""

import pandas as pd
import numpy as np
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.parse
from datetime import datetime, timedelta
from sqlalchemy import create_engine, URL, text

# ML imports
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, OrdinalEncoder, 
    TargetEncoder, FunctionTransformer
)
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, 
    classification_report
)
from sklearn.base import BaseEstimator, TransformerMixin

# Imbalanced learning
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("lead_generation_model.log"), 
        logging.StreamHandler()
    ]
)

def log_with_timestamp(message):
    """Log a message with timestamp."""
    logging.info(message)

# Database connection helpers - Fixed with proper pyodbc approach
def create_db_engine(server, db):
    """Create database engine with proper pyodbc connection string."""
    odbc_connection_string = urllib.parse.quote_plus(
        f"Driver=ODBC Driver 17 for SQL Server;"
        f"Server={server};"
        f"Database={db};"
        f"Trusted_Connection=Yes;"
    )
    
    try:
        engine = create_engine(
            URL.create("mssql+pyodbc", query={"odbc_connect": odbc_connection_string})
        )
        with engine.connect() as conn:
            log_with_timestamp("Database connection successful")
        return engine
    except Exception as e:
        log_with_timestamp(f"Database connection failed: {e}")
        return None

def load_data(engine, query):
    """Load data from database using pandas for proper DataFrame typing."""
    return pd.read_sql_query(query, engine)

# Custom metrics for lead generation - Fixed scorer signature
def precision_at_k(y_true, y_proba, k=2000):
    """Calculate Precision@K for lead generation."""
    if len(y_true) < k:
        k = len(y_true)
    
    # Get top K predictions
    top_k_indices = np.argsort(-y_proba)[:k]
    top_k_true = y_true.iloc[top_k_indices] if hasattr(y_true, 'iloc') else y_true[top_k_indices]
    
    return np.mean(top_k_true)

def precision_at_k_scorer_factory(k=2000):
    """Create sklearn-compatible Precision@K scorer with correct signature."""
    def _score(estimator, X, y):
        y_proba = estimator.predict_proba(X)[:, 1]
        return precision_at_k(y, y_proba, k)
    return _score

# Custom transformers - Fixed to work before ColumnTransformer
class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Engineer temporal features safely before preprocessing."""
    
    def __init__(self, date_col='Snapshot_Date'):
        self.date_col = date_col
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        # Use snapshot date from each row for temporal features
        if self.date_col in X_copy.columns and 'Gruendung_Jahr' in X_copy.columns:
            snapshot_years = pd.to_datetime(X_copy[self.date_col]).dt.year
            X_copy['Company_Age_Years'] = snapshot_years - X_copy['Gruendung_Jahr']
            X_copy['Company_Age_Years'] = X_copy['Company_Age_Years'].clip(0, 150)  # Sanity check
        
        # Economic activity score
        activity_cols = [col for col in X_copy.columns if 'BezugArchiv' in col]
        if activity_cols:
            X_copy['Economic_Activity_Score'] = X_copy[activity_cols].fillna(0).sum(axis=1)
            X_copy['Is_Economically_Active'] = (X_copy['Economic_Activity_Score'] > 0).astype(int)
        
        return X_copy

class LiftAnalyzer:
    """Calculate lift and gains tables for business analysis."""
    
    @staticmethod
    def calculate_lift_table(y_true, y_proba, n_deciles=10):
        """Calculate lift table with decile analysis."""
        df = pd.DataFrame({
            'y_true': y_true,
            'y_proba': y_proba
        }).sort_values('y_proba', ascending=False).reset_index(drop=True)
        
        df['decile'] = pd.qcut(df.index, n_deciles, labels=False, duplicates='drop') + 1
        
        # Calculate metrics by decile
        lift_table = df.groupby('decile').agg({
            'y_true': ['count', 'sum', 'mean']
        }).round(4)
        
        lift_table.columns = ['Total_Count', 'Conversions', 'Conversion_Rate']
        
        # Calculate lift vs baseline
        overall_rate = df['y_true'].mean()
        lift_table['Lift'] = lift_table['Conversion_Rate'] / overall_rate
        
        # Calculate cumulative metrics
        lift_table['Cumulative_Conversions'] = lift_table['Conversions'].cumsum()
        lift_table['Cumulative_Count'] = lift_table['Total_Count'].cumsum()
        lift_table['Cumulative_Rate'] = (
            lift_table['Cumulative_Conversions'] / lift_table['Cumulative_Count']
        )
        
        return lift_table.reset_index()

def main():
    """Main training pipeline with temporal modeling and leakage prevention."""
    log_with_timestamp("=== Starting Lead Generation Model Training ===")
    
    # Configuration
    PREDICTION_HORIZON_MONTHS = 12
    LEAD_CAPACITY_K = 2000
    RANDOM_STATE = 42
    
    # Database setup - Fixed connection approach
    server = 'PRODSVCREPORT70'
    db = 'CAG_Analyse'
    
    engine = create_db_engine(server, db)
    if engine is None:
        raise ConnectionError("Failed to connect to database")
    
    # Enhanced query with proper temporal modeling - Fixed: Add current snapshot (0)
    # Note: This is a simulation - in production you'd have historical snapshots
    query = """
    WITH historical_snapshots AS (
        -- Simulate historical snapshots for proper temporal modeling
        -- In production, you'd have actual historical data snapshots
        SELECT 
            m.CrefoID,
            m.Eintritt,
            m.Gruendung_Jahr,
            m.Kanton,
            LEFT(m.PLZ, 2) as PLZ_Prefix,
            m.BrancheText_06,
            m.BrancheText_04, 
            m.BrancheText_02,
            m.Rechtsform,
            m.MitarbeiterBestandKategorie,
            m.UmsatzKategorie,
            
            -- Economic activity (archive data only - available for prospects)
            ISNULL(m.BezugArchivProduktGross, 0) as BezugArchivProduktGross,
            ISNULL(m.BezugArchivProduktMittel, 0) as BezugArchivProduktMittel,
            ISNULL(m.BezugArchivProduktKlein, 0) as BezugArchivProduktKlein,
            ISNULL(m.BezugArchivProduktMatch, 0) as BezugArchivProduktMatch,
            ISNULL(m.BezugArchivProduktBA, 0) as BezugArchivProduktBA,
            ISNULL(m.Anzahl_ArchivMonitoringRegistriert, 0) as ArchivMonitoring,
            ISNULL(m.Anzahl_AuftragBestellung, 0) as AuftragBestellung,
            ISNULL(m.Anzahl_OnlineAuftrag, 0) as OnlineAuftrag,
            
            -- Simulate multiple snapshot dates for temporal modeling
            -- In production, you'd have actual historical snapshot table
            snapshot_date,
            
            -- Forward-looking label: "Will convert within 12 months after snapshot?"
            CASE 
                WHEN m.Eintritt >= snapshot_date 
                     AND m.Eintritt < DATEADD(MONTH, 12, snapshot_date)
                THEN 1 
                ELSE 0 
            END as Target
            
        FROM CAG_Analyse.mitgliederstatistik.MitgliederSegmentierung m
        CROSS JOIN (
            -- Simulate historical snapshots - Fixed: Add current snapshot (0 months)
            SELECT DATEADD(MONTH, -n, GETDATE()) as snapshot_date
            FROM (VALUES (36), (30), (24), (18), (12), (6), (0)) t(n)
        ) snapshots
        WHERE 
            (m.BrancheCode_06 IS NOT NULL AND m.BrancheCode_06 != 0)
            AND m.ArchivStatus = 20
            -- Only include firms that are NON-members at the snapshot
            AND (m.Eintritt IS NULL OR m.Eintritt >= snapshot_date)
            -- Also ensure the firm existed at the snapshot (founding year not after snapshot)
            AND (m.Gruendung_Jahr IS NULL OR m.Gruendung_Jahr <= YEAR(snapshot_date))
    )
    SELECT * FROM historical_snapshots
    WHERE snapshot_date >= DATEADD(MONTH, -36, GETDATE())  -- Last 36 months of snapshots
    """
    
    log_with_timestamp("Loading temporal data from database...")
    df = load_data(engine, query)
    log_with_timestamp(f"Loaded {len(df):,} records across time periods")
    
    # Data preprocessing and temporal validation
    log_with_timestamp("Processing temporal data...")
    
    # Convert date columns
    df['Eintritt'] = pd.to_datetime(df['Eintritt'], errors='coerce')
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
    
    # Sort by snapshot_date for proper temporal splits
    df = df.sort_values('snapshot_date').reset_index(drop=True)
    
    log_with_timestamp(f"Temporal data shape: {df.shape}")
    log_with_timestamp(f"Snapshot date range: {df['snapshot_date'].min()} to {df['snapshot_date'].max()}")
    log_with_timestamp(f"Overall conversion rate: {df['Target'].mean():.3f}")
    
    # Chronological train/validation/test split - Fixed: use dates, not row counts
    log_with_timestamp("Creating chronological splits...")
    
    # Split by actual dates (handles uneven sampling per month)
    cut1 = df['snapshot_date'].quantile(0.70)
    cut2 = df['snapshot_date'].quantile(0.85)
    
    df_train = df[df['snapshot_date'] <= cut1].copy()
    df_val = df[(df['snapshot_date'] > cut1) & (df['snapshot_date'] <= cut2)].copy()
    df_test = df[df['snapshot_date'] > cut2].copy()
    
    log_with_timestamp(f"Train: {len(df_train):,} records ({df_train['Target'].mean():.3f} conversion rate)")
    log_with_timestamp(f"Validation: {len(df_val):,} records ({df_val['Target'].mean():.3f} conversion rate)")
    log_with_timestamp(f"Test: {len(df_test):,} records ({df_test['Target'].mean():.3f} conversion rate)")
    
    # Feature engineering and preprocessing setup
    log_with_timestamp("Setting up feature engineering...")
    
    # Handle missing values
    categorical_cols = [
        'MitarbeiterBestandKategorie', 'UmsatzKategorie', 
        'BrancheText_06', 'BrancheText_04', 'BrancheText_02', 'Rechtsform'
    ]
    
    for col in categorical_cols:
        df_train[col] = df_train[col].fillna('Unknown')
        df_val[col] = df_val[col].fillna('Unknown')
        df_test[col] = df_test[col].fillna('Unknown')
    
    # Define feature groups for preprocessing
    numeric_features = [
        'Gruendung_Jahr', 'BezugArchivProduktGross', 'BezugArchivProduktMittel',
        'BezugArchivProduktKlein', 'BezugArchivProduktMatch', 'BezugArchivProduktBA',
        'ArchivMonitoring', 'AuftragBestellung', 'OnlineAuftrag'
    ]
    
    low_cardinality_categorical = ['Kanton', 'PLZ_Prefix', 'Rechtsform']
    
    high_cardinality_categorical = ['BrancheText_06', 'BrancheText_04', 'BrancheText_02']
    
    ordinal_features = ['MitarbeiterBestandKategorie', 'UmsatzKategorie']
    
    # Define ordinal mappings
    mitarbeiter_order = [
        'Unknown', 'Unbekannt', '0', '1', '2-4', '5-9', '0-9 (OptingOut)', 
        '10-24', '25-49', '50-149', '150-249', '250-499', '500-999', 
        '1000-4999', '5000-9999', '>= 10000'
    ]
    
    umsatz_order = [
        'Unknown', 'Unbekannt', '0', "1 - 199'999", "200'000 - 999'999", 
        '1 Mio. - < 2 Mio.', '2 Mio. - < 5 Mio.', '5 Mio. - < 10 Mio.', 
        '10 Mio. - < 30 Mio.', '30 Mio. - < 50 Mio.', '50 Mio. - < 100 Mio.', 
        '100 Mio. - < 200 Mio.', '200 Mio. - < 1 Mia.', '1 Mia. - < 2 Mia.', '>= 2 Mia.'
    ]
    
    # Build preprocessing pipeline - temporal features BEFORE ColumnTransformer
    log_with_timestamp("Building preprocessing pipeline...")
    
    # Define all feature columns including snapshot_date for temporal engineering
    feature_cols = (
        numeric_features + low_cardinality_categorical + 
        high_cardinality_categorical + ordinal_features + ['snapshot_date']
    )
    
    # Create version-safe OneHotEncoder
    try:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        log_with_timestamp("Using OneHotEncoder with sparse_output=False (sklearn >= 1.2)")
    except TypeError:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
        log_with_timestamp("Using OneHotEncoder with sparse=False (sklearn < 1.2)")
    
    # Preprocessor without temporal block (temporal features added before)
    preprocessor = ColumnTransformer(
        transformers=[
            # Numeric features (including new temporal features)
            ('numeric', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())  # Keep for consistency, though not needed for trees
            ]), numeric_features + ['Company_Age_Years', 'Economic_Activity_Score']),
            
            # Low cardinality categoricals - version-safe OneHotEncoder
            ('low_cat', ohe, low_cardinality_categorical),
            
            # High cardinality categoricals with target encoding
            ('high_cat', TargetEncoder(
                smooth='auto'
                # sklearn's TargetEncoder has built-in cross-fitting for leakage prevention
            ), high_cardinality_categorical),
            
            # Ordinal features
            ('ordinal', OrdinalEncoder(
                categories=[mitarbeiter_order, umsatz_order],
                handle_unknown='use_encoded_value',
                unknown_value=-1
            ), ordinal_features),
            
            # Binary features
            ('binary', 'passthrough', ['Is_Economically_Active'])
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    
    # Build complete pipeline - Fixed: temporal engineering FIRST, disable early stopping
    log_with_timestamp("Building ML pipeline...")
    
    # Check sklearn version for HGB class_weight support
    import sklearn
    sklearn_version = [int(x) for x in sklearn.__version__.split('.')]
    supports_class_weight = sklearn_version >= [1, 5, 0]  # Available in sklearn >= 1.5.0
    
    if supports_class_weight:
        log_with_timestamp("Using HGB with class_weight='balanced' - no SMOTE (sklearn >= 1.5.0)")
        classifier_params = {
            'random_state': RANDOM_STATE,
            'early_stopping': False,
            'max_depth': 5,
            'min_samples_leaf': 20,
            'l2_regularization': 0.1,
            'class_weight': 'balanced'
        }
        # Option A: Use class_weight, no SMOTE to avoid double compensation
        pipeline = Pipeline([
            ('feateng', TemporalFeatureEngineer(date_col='snapshot_date')),
            ('preprocessor', preprocessor),
            ('classifier', HistGradientBoostingClassifier(**classifier_params))
        ])
    else:
        log_with_timestamp("Using SMOTE without class_weight (sklearn < 1.5.0)")
        classifier_params = {
            'random_state': RANDOM_STATE,
            'early_stopping': False,
            'max_depth': 5,
            'min_samples_leaf': 20,
            'l2_regularization': 0.1
        }
        # Option B: Use SMOTE, no class_weight
        pipeline = ImbPipeline([
            ('feateng', TemporalFeatureEngineer(date_col='snapshot_date')),
            ('preprocessor', preprocessor),
            ('sampler', SMOTE(random_state=RANDOM_STATE)),
            ('classifier', HistGradientBoostingClassifier(**classifier_params))
        ])
    
    # Prepare training data
    X_train = df_train[feature_cols].copy()
    y_train = df_train['Target'].copy()
    
    X_val = df_val[feature_cols].copy()
    y_val = df_val['Target'].copy()
    
    X_test = df_test[feature_cols].copy()
    y_test = df_test['Target'].copy()
    
    log_with_timestamp(f"Training feature matrix shape: {X_train.shape}")
    log_with_timestamp(f"Training class distribution: {y_train.value_counts().to_dict()}")
    
    # Time-based cross-validation for hyperparameter tuning
    log_with_timestamp("Setting up time-based cross-validation...")
    
    # Combine train and validation for CV, but keep test held out
    X_train_val = pd.concat([X_train, X_val], ignore_index=True)
    y_train_val = pd.concat([y_train, y_val], ignore_index=True)
    
    # TimeSeriesSplit with gap to prevent leakage - Fixed: build DF for time-aware gap calc
    # Calculate approximate gap in samples based on 2-month time gap
    train_val_df = pd.concat([df_train, df_val], ignore_index=True)
    samples_per_month = (
        len(train_val_df) /
        train_val_df['snapshot_date'].dt.to_period('M').nunique()
    )
    time_gap_samples = int(2 * samples_per_month)  # ≈2-month gap
    time_gap_samples = max(100, min(time_gap_samples, 2000))
    
    tscv = TimeSeriesSplit(n_splits=4, gap=time_gap_samples)  # no test_size needed
    log_with_timestamp(f"Using TimeSeriesSplit with gap of {time_gap_samples} samples (~2 months)")
    
    # Hyperparameter tuning
    log_with_timestamp("Hyperparameter tuning with lead-focused metrics...")
    
    # Conditional hyperparameters based on pipeline type - Fixed: account for different pipelines
    param_distributions = {
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7, 10],
        'classifier__min_samples_leaf': [10, 20, 50, 100],
        'classifier__l2_regularization': [0.0, 0.1, 1.0]
    }
    
    # Add SMOTE parameters only if using SMOTE pipeline
    if not supports_class_weight:
        param_distributions['sampler__k_neighbors'] = [3, 5, 7]
    
    # Custom scoring focused on lead generation - Fixed scorer signature
    scoring = {
        'average_precision': 'average_precision',  # Primary metric for imbalanced data
        'precision_at_k': precision_at_k_scorer_factory(LEAD_CAPACITY_K)
    }
    
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=20,
        scoring=scoring,
        refit='average_precision',  # Primary metric for model selection
        cv=tscv,  # Time-aware cross-validation
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )
    
    log_with_timestamp("Fitting model with hyperparameter search...")
    search.fit(X_train_val, y_train_val)
    
    log_with_timestamp("Best parameters found:")
    for param, value in search.best_params_.items():
        log_with_timestamp(f"  {param}: {value}")
    
    log_with_timestamp(f"Best CV Average Precision: {search.best_score_:.4f}")
    
    # Calibrate on training data, evaluate on held-out test
    log_with_timestamp("Calibrating probabilities on CV folds...")
    
    # Fit calibrator on training data using time-aware CV - Fixed: temporal calibration
    cal_cv = TimeSeriesSplit(n_splits=3)
    calibrated_model = CalibratedClassifierCV(
        base_estimator=search.best_estimator_,
        method='isotonic',  # Better for non-monotonic calibration
        cv=cal_cv  # Time-aware cross-validation for calibration
    )
    
    calibrated_model.fit(X_train_val, y_train_val)
    
    # Final evaluation on held-out test set
    log_with_timestamp("Evaluating on held-out test set...")
    
    y_proba_test = calibrated_model.predict_proba(X_test)[:, 1]
    y_pred_test = calibrated_model.predict(X_test)
    
    # Calculate test metrics
    test_avg_precision = average_precision_score(y_test, y_proba_test)
    test_precision_at_k = precision_at_k(y_test, y_proba_test, LEAD_CAPACITY_K)
    
    log_with_timestamp(f"Final Test Set Performance:")
    log_with_timestamp(f"  Average Precision (PR-AUC): {test_avg_precision:.4f}")
    log_with_timestamp(f"  Precision@{LEAD_CAPACITY_K}: {test_precision_at_k:.4f}")
    
    # Generate lift analysis on test set
    log_with_timestamp("Generating lift analysis on test set...")
    
    lift_analyzer = LiftAnalyzer()
    lift_table = lift_analyzer.calculate_lift_table(y_test, y_proba_test)
    
    log_with_timestamp("Test Set Lift Table by Decile:")
    log_with_timestamp(f"\n{lift_table.to_string(index=False)}")
    
    # Save lift table
    lift_table.to_csv('lead_generation_lift_table.csv', index=False)
    log_with_timestamp("Lift table saved to 'lead_generation_lift_table.csv'")
    
    # Feature importance analysis - Get from tuned pipeline, not calibrated wrapper
    log_with_timestamp("Analyzing feature importance...")
    
    try:
        # Get feature names from the best (uncalibrated) pipeline
        best_pipeline = search.best_estimator_
        feature_names = best_pipeline.named_steps['preprocessor'].get_feature_names_out()
        importances = best_pipeline.named_steps['classifier'].feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(20)
        
        log_with_timestamp("Top 10 Most Important Features:")
        for idx, row in importance_df.head(10).iterrows():
            log_with_timestamp(f"  {row['Feature']}: {row['Importance']:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
        plt.title('Top 20 Feature Importances - Lead Generation Model')
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        plt.savefig('lead_generation_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        log_with_timestamp("Feature importance plot saved to 'lead_generation_feature_importance.png'")
        
        # Save feature importance
        importance_df.to_csv('lead_generation_feature_importance.csv', index=False)
        
    except Exception as e:
        log_with_timestamp(f"Could not generate feature importance analysis: {e}")
    
    # Generate lead rankings for current prospects
    log_with_timestamp("Generating lead rankings for current prospects...")
    
    # Get current prospects (most recent snapshot, non-members) - Fixed: use status not label
    latest_snapshot = df['snapshot_date'].max()
    current_prospects = df[df['snapshot_date'] == latest_snapshot].copy()
    # Do NOT filter by Target here - Target is forward-looking, not current status
    
    if len(current_prospects) > 0:
        X_prospects = current_prospects[feature_cols].copy()
        
        # Handle missing values in prospects
        for col in categorical_cols:
            if col in X_prospects.columns:
                X_prospects[col] = X_prospects[col].fillna('Unknown')
        
        # Generate predictions
        prospect_probabilities = calibrated_model.predict_proba(X_prospects)[:, 1]
        
        # Create lead ranking
        current_prospects['Propensity_Score'] = prospect_probabilities
        current_prospects['Lead_Rank'] = current_prospects['Propensity_Score'].rank(ascending=False, method='first')
        current_prospects['Decile'] = pd.qcut(
            current_prospects['Propensity_Score'], 
            q=10, 
            labels=[f'D{i}' for i in range(1, 11)],
            duplicates='drop'
        )
        
        # Select top leads
        top_leads = current_prospects.nlargest(LEAD_CAPACITY_K * 2, 'Propensity_Score')[
            ['CrefoID', 'Kanton', 'PLZ_Prefix', 'BrancheText_06', 'MitarbeiterBestandKategorie',
             'UmsatzKategorie', 'Propensity_Score', 'Lead_Rank', 'Decile']
        ].copy()
        
        # Save lead rankings
        top_leads.to_csv('lead_generation_rankings.csv', index=False)
        log_with_timestamp(f"Top {len(top_leads):,} leads saved to 'lead_generation_rankings.csv'")
        
        # Summary by region
        regional_summary = top_leads.groupby('Kanton').agg({
            'Propensity_Score': ['count', 'mean', 'min', 'max'],
            'CrefoID': 'count'
        }).round(4)
        
        regional_summary.columns = ['Lead_Count', 'Avg_Score', 'Min_Score', 'Max_Score', 'Total_Companies']
        regional_summary = regional_summary.sort_values('Avg_Score', ascending=False)
        
        log_with_timestamp("Lead Summary by Canton (Top Regions):")
        log_with_timestamp(f"\n{regional_summary.head(10).to_string()}")
        
        regional_summary.to_csv('lead_generation_regional_summary.csv')
    
    # Save the complete pipeline
    log_with_timestamp("Saving trained model...")
    
    model_artifacts = {
        'calibrated_model': calibrated_model,
        'feature_columns': feature_cols,
        'preprocessing_pipeline': search.best_estimator_.named_steps['preprocessor'],
        'model_metadata': {
            'training_date': datetime.now().isoformat(),
            'prediction_horizon_months': PREDICTION_HORIZON_MONTHS,
            'lead_capacity': LEAD_CAPACITY_K,
            'best_params': search.best_params_,
            'performance_metrics': {
                'test_average_precision': test_avg_precision,
                'test_precision_at_k': test_precision_at_k,
                'cv_average_precision': search.best_score_
            },
            'data_splits': {
                'train_size': len(df_train),
                'val_size': len(df_val),
                'test_size': len(df_test)
            }
        }
    }
    
    joblib.dump(model_artifacts, 'lead_generation_model_complete.pkl')
    log_with_timestamp("Complete model saved to 'lead_generation_model_complete.pkl'")
    
    # Summary report
    log_with_timestamp("=== Training Complete ===")
    log_with_timestamp(f"Model Type: Calibrated HistGradientBoosting with SMOTE")
    log_with_timestamp(f"Training Approach: Chronological time-based splits")
    log_with_timestamp(f"Total Records: {len(df):,} across {len(df['snapshot_date'].unique())} time periods")
    log_with_timestamp(f"Train/Val/Test: {len(df_train):,}/{len(df_val):,}/{len(df_test):,}")
    log_with_timestamp(f"Features: {len(feature_cols)}")
    log_with_timestamp(f"Final Test Average Precision: {test_avg_precision:.4f}")
    log_with_timestamp(f"Final Test Precision@{LEAD_CAPACITY_K}: {test_precision_at_k:.4f}")
    log_with_timestamp(f"Top Decile Lift: {lift_table.iloc[0]['Lift']:.2f}x")
    
    engine.dispose()
    log_with_timestamp("Temporal lead generation pipeline completed successfully!")

if __name__ == "__main__":
    main()
