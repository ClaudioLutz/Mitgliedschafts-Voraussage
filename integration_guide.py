"""
Integration Guide: Replace Auto Column Grouping with Lead-Gen ColumnTransformer
================================================================================

This shows exactly how to integrate the lead-gen specific ColumnTransformer into your
existing training_lead_generation_model.py file, replacing the auto column grouping approach.
"""

# =============================================================================
# STEP 1: Import the new preprocessor at the top of your training file
# =============================================================================

# Add this import after your existing imports (around line 25):
from column_transformer_lead_gen import create_lead_gen_preprocessor


# =============================================================================
# STEP 2: Replace the entire preprocessing section (lines ~180-260)
# =============================================================================

def integrate_preprocessing_replacement():
    """
    REPLACE THIS ENTIRE SECTION in your training_lead_generation_model.py:
    
    From line ~180:  "# 4) Column groups (auto)" 
    To line ~260:    "PipelineClass = Pipeline"
    
    With the code below:
    """
    
    # 4) Create lead-gen specific preprocessor (replaces auto column grouping)
    preprocessor = create_lead_gen_preprocessor()
    
    # 5) Estimator & imbalance strategy (simplified - no more complex pipeline building)
    if USE_CLASS_WEIGHT:
        clf = HistGradientBoostingClassifier(
            random_state=RANDOM_STATE,
            early_stopping=False,
            class_weight="balanced",
            max_depth=None,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.1
        )
        # Simple pipeline: preprocessor + classifier
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        log.info("Imbalance via class_weight='balanced' (no SMOTE).")
    else:
        if not HAVE_IMBLEARN:
            raise RuntimeError("imbalanced-learn not installed; install or upgrade scikit-learn to >=1.5 for class_weight.")
        clf = HistGradientBoostingClassifier(
            random_state=RANDOM_STATE,
            early_stopping=False,
            max_depth=None,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.1
        )
        # Pipeline with SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        pipe = ImbPipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("classifier", clf)
        ])
        log.info("Imbalance via SMOTE (no class_weight).")


# =============================================================================
# STEP 3: Update feature column selection (around line 270-275)
# =============================================================================

def update_feature_selection():
    """
    REPLACE this section around line 270-275:
    
    OLD:
        feature_cols = [c for c in X_train_val.columns if c not in LEAKAGE_COLS]
    
    NEW:
        # Feature selection is now handled inside the preprocessor
        # Just remove the explicit leakage columns
        from column_transformer_lead_gen import DROP_COLS
        feature_cols = [c for c in X_train_val.columns if c not in DROP_COLS]
    """
    pass


# =============================================================================
# STEP 4: Remove redundant feature engineering (around line 140-180)
# =============================================================================

def remove_redundant_engineering():
    """
    The temporal_feature_engineer() function is still needed for Company_Age_Years,
    but you can simplify it since missing indicators and PLZ grouping are now
    handled in the preprocessor.
    
    REPLACE the temporal_feature_engineer function (around line 140-180) with:
    """
    
    def temporal_feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
        """Add Company_Age_Years (other engineering moved to preprocessor)."""
        out = df.copy()
        
        # Company age in years at snapshot (still needed as base feature)
        if "Gruendung_Jahr" in out.columns:
            snap_year = out["snapshot_date"].dt.year
            out["Company_Age_Years"] = (
                snap_year - out["Gruendung_Jahr"].fillna(snap_year)
            ).clip(lower=0)
        else:
            out["Company_Age_Years"] = 0
            
        # Note: missing indicators, PLZ grouping now handled in preprocessor
        return out


# =============================================================================
# STEP 5: Complete integration example
# =============================================================================

def complete_integration_example():
    """
    Here's how the key section of your main() function should look after integration:
    """
    
    # After loading and splitting data...
    
    # 3) Basic feature engineering (simplified)
    df_train_eng = temporal_feature_engineer(df_train)
    df_val_eng   = temporal_feature_engineer(df_val)
    df_test_eng  = temporal_feature_engineer(df_test)
    df_curr_eng  = temporal_feature_engineer(df_current)

    # 4) Create lead-gen specific preprocessor (replaces column grouping)
    preprocessor = create_lead_gen_preprocessor()

    # 5) Create pipeline with imbalance handling
    if USE_CLASS_WEIGHT:
        clf = HistGradientBoostingClassifier(
            random_state=RANDOM_STATE,
            early_stopping=False,
            class_weight="balanced",
            max_depth=None
        )
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
    else:
        # SMOTE version...
        pass

    # 6) Feature selection (simplified)
    from column_transformer_lead_gen import DROP_COLS
    X_train_val = pd.concat([df_train_eng, df_val_eng], ignore_index=True)
    y_train_val = pd.concat([df_train["Target"], df_val["Target"]], ignore_index=True)
    feature_cols = [c for c in X_train_val.columns if c not in DROP_COLS]

    # Continue with hyperparameter search as before...
    # The rest of your pipeline remains the same!


# =============================================================================
# STEP 6: Validation and Testing
# =============================================================================

def validation_example():
    """
    Add this validation code to test the new preprocessor:
    """
    
    # Test the preprocessor before training
    from column_transformer_lead_gen import validate_preprocessor
    
    def test_preprocessor_integration(df_sample):
        try:
            results = validate_preprocessor(df_sample)
            log.info(f"✅ Preprocessor validation successful!")
            log.info(f"   Original shape: {results['original_shape']}")  
            log.info(f"   Transformed shape: {results['transformed_shape']}")
            log.info(f"   Features created: {results['n_features']}")
            log.info(f"   Missing indicators: {len(results['missing_indicators_added'])}")
            log.info(f"   Log transforms: {len(results['log_transforms_added'])}")
            log.info(f"   PLZ groupings: {len(results['plz_groupings_added'])}")
            return True
        except Exception as e:
            log.error(f"❌ Preprocessor validation failed: {e}")
            return False
    
    # Add this test in your main() function after loading data:
    # if not test_preprocessor_integration(df_train_eng.head(1000)):
    #     raise RuntimeError("Preprocessor validation failed")


# =============================================================================
# Summary of Changes
# =============================================================================

"""
SUMMARY OF INTEGRATION:

1. ✅ Import: Add `from column_transformer_lead_gen import create_lead_gen_preprocessor`

2. ✅ Replace preprocessing: Remove auto_column_groups() and complex ColumnTransformer 
   building with single call to create_lead_gen_preprocessor()

3. ✅ Simplify pipelines: No more complex pipeline building - just preprocessor + classifier

4. ✅ Update feature selection: Use DROP_COLS instead of LEAKAGE_COLS

5. ✅ Simplify feature engineering: Keep Company_Age_Years, move other engineering to preprocessor

6. ✅ Add validation: Optional test to ensure preprocessor works correctly

BENEFITS:
- ✅ Implements exact lead-gen strategy from specification
- ✅ Handles missing indicators automatically  
- ✅ Target encoding with built-in cross-fitting (no leakage)
- ✅ One-hot encoding with rare category grouping
- ✅ PLZ geographical grouping for stability
- ✅ All preprocessing safely inside pipeline (no leakage)
- ✅ Cleaner, more maintainable code
- ✅ Follows scikit-learn best practices
"""

if __name__ == "__main__":
    print("Integration guide loaded!")
    print("Follow the steps above to integrate the lead-gen ColumnTransformer")
    print("into your existing training_lead_generation_model.py file.")
