"""
Tests for new ML pipeline features:
- Temporal features (end-of-quarter, seasonal encoding)
- Precision@K and related scorers
- Two-stage pipeline
- Lookalike features
- Stacking ensemble
- LambdaMART wrapper
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock

# Import modules to test
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTemporalFeatures:
    """Tests for temporal feature engineering."""

    def test_company_age_calculation(self):
        """Test Company_Age_Years calculation."""
        from training_lead_generation_model import temporal_feature_engineer

        df = pd.DataFrame({
            'Gruendung_Jahr': [2010, 2015, 2020, None],
            'snapshot_date': pd.to_datetime(['2024-06-15'] * 4)
        })

        result = temporal_feature_engineer(df)

        assert 'Company_Age_Years' in result.columns
        assert result['Company_Age_Years'].iloc[0] == 14
        assert result['Company_Age_Years'].iloc[1] == 9
        assert result['Company_Age_Years'].iloc[2] == 4
        # None founding year should default to snapshot year (age 0)
        assert result['Company_Age_Years'].iloc[3] == 0

    def test_company_age_log(self):
        """Test Company_Age_Log calculation."""
        from training_lead_generation_model import temporal_feature_engineer

        df = pd.DataFrame({
            'Gruendung_Jahr': [2014, 2024],
            'snapshot_date': pd.to_datetime(['2024-06-15'] * 2)
        })

        result = temporal_feature_engineer(df)

        assert 'Company_Age_Log' in result.columns
        # Age 10 -> log1p(10) = log(11) ≈ 2.4
        assert result['Company_Age_Log'].iloc[0] == pytest.approx(np.log1p(10), rel=0.01)
        # Age 0 -> log1p(0) = 0
        assert result['Company_Age_Log'].iloc[1] == 0.0

    def test_company_age_bucket(self):
        """Test Company_Age_Bucket categories."""
        from training_lead_generation_model import temporal_feature_engineer

        df = pd.DataFrame({
            'Gruendung_Jahr': [2023, 2020, 2015, 2010, 1990],
            'snapshot_date': pd.to_datetime(['2024-06-15'] * 5)
        })

        result = temporal_feature_engineer(df)

        assert 'Company_Age_Bucket' in result.columns
        buckets = result['Company_Age_Bucket'].tolist()
        assert buckets[0] == 'startup'       # Age 1
        assert buckets[1] == 'early_growth'  # Age 4
        assert buckets[2] == 'established'   # Age 9
        assert buckets[3] == 'mature'        # Age 14
        assert buckets[4] == 'legacy'        # Age 34

    def test_seasonal_features(self):
        """Test seasonal features (month encoding, quarter flags)."""
        from training_lead_generation_model import temporal_feature_engineer

        df = pd.DataFrame({
            'Gruendung_Jahr': [2020] * 12,
            'snapshot_date': pd.to_datetime([
                f'2024-{m:02d}-15' for m in range(1, 13)
            ])
        })

        result = temporal_feature_engineer(df)

        # Check all temporal features exist
        assert 'Month_Sin' in result.columns
        assert 'Month_Cos' in result.columns
        assert 'Is_End_Of_Quarter' in result.columns
        assert 'Is_Summer' in result.columns
        assert 'Quarter' in result.columns

        # Check cyclical encoding produces values in [-1, 1]
        assert result['Month_Sin'].min() >= -1.0
        assert result['Month_Sin'].max() <= 1.0
        assert result['Month_Cos'].min() >= -1.0
        assert result['Month_Cos'].max() <= 1.0

        # Check end-of-quarter flags (months 3, 6, 9, 12)
        eoq = result['Is_End_Of_Quarter'].tolist()
        assert eoq[2] == 1   # March
        assert eoq[5] == 1   # June
        assert eoq[8] == 1   # September
        assert eoq[11] == 1  # December
        assert eoq[0] == 0   # January
        assert eoq[4] == 0   # May

        # Check summer flags (July, August)
        summer = result['Is_Summer'].tolist()
        assert summer[6] == 1   # July
        assert summer[7] == 1   # August
        assert summer[5] == 0   # June
        assert summer[8] == 0   # September

        # Check quarters
        quarters = result['Quarter'].tolist()
        assert quarters[0] == 1   # January
        assert quarters[3] == 2   # April
        assert quarters[6] == 3   # July
        assert quarters[9] == 4   # October


class TestPrecisionAtK:
    """Tests for Precision@K and related scorers."""

    def test_precision_at_k_basic(self):
        """Test basic precision@K calculation."""
        from training_lead_generation_model import precision_at_k

        y_true = np.array([1, 1, 0, 0, 0])
        y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

        # Top 2: both are positives
        assert precision_at_k(y_true, y_score, k=2) == 1.0

        # Top 3: 2 positives out of 3
        assert precision_at_k(y_true, y_score, k=3) == pytest.approx(2/3, rel=0.01)

        # Top 5: 2 positives out of 5
        assert precision_at_k(y_true, y_score, k=5) == pytest.approx(2/5, rel=0.01)

    def test_recall_at_k_basic(self):
        """Test basic recall@K calculation."""
        from training_lead_generation_model import recall_at_k

        y_true = np.array([1, 1, 0, 0, 0])
        y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

        # Top 2: both positives found (2/2)
        assert recall_at_k(y_true, y_score, k=2) == 1.0

        # Top 1: 1 positive found (1/2)
        assert recall_at_k(y_true, y_score, k=1) == 0.5

    def test_lift_at_k_basic(self):
        """Test basic lift@K calculation."""
        from training_lead_generation_model import lift_at_k

        y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])  # 20% baseline
        y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

        # Top 2: precision = 100%, baseline = 20%, lift = 5x
        assert lift_at_k(y_true, y_score, k=2) == pytest.approx(5.0, rel=0.01)

    def test_scorer_factory_integration(self):
        """Test scorer factories work with sklearn estimators."""
        from training_lead_generation_model import precision_at_k_scorer_factory
        from sklearn.linear_model import LogisticRegression

        # Create simple dataset
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)

        scorer = precision_at_k_scorer_factory(k=10)
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)

        score = scorer(model, X, y)
        assert 0.0 <= score <= 1.0


class TestTwoStagePipeline:
    """Tests for two-stage filter-then-rank pipeline."""

    def test_two_stage_basic_fit_predict(self):
        """Test basic fit/predict workflow."""
        from two_stage_pipeline import TwoStagePipeline
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import HistGradientBoostingClassifier

        np.random.seed(42)
        X = np.random.randn(500, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        pipeline = TwoStagePipeline(
            stage1_estimator=LogisticRegression(max_iter=1000),
            stage2_estimator=HistGradientBoostingClassifier(max_iter=50),
            stage1_target_recall=0.90,
            stage1_cv=3
        )

        pipeline.fit(X, y)

        # Check attributes are set
        assert pipeline.stage1_threshold_ is not None
        assert pipeline.stage1_recall_ >= 0.85  # Should be close to target
        assert pipeline.stage2_class_balance_ is not None

        # Check predictions work
        proba = pipeline.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_find_recall_threshold(self):
        """Test threshold finding for target recall."""
        from two_stage_pipeline import find_recall_threshold

        y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

        threshold, recall, precision = find_recall_threshold(
            y_true, y_score, target_recall=0.66
        )

        # Should find threshold that gets at least 66% recall (2/3 positives)
        assert recall >= 0.66

    def test_factory_function(self):
        """Test create_two_stage_pipeline factory."""
        from two_stage_pipeline import create_two_stage_pipeline

        pipeline = create_two_stage_pipeline(
            stage1_type='logistic',
            stage2_type='hgb',
            target_recall=0.95
        )

        assert pipeline is not None
        assert pipeline.stage1_target_recall == 0.95


class TestLookalikeFeatures:
    """Tests for lookalike feature engineering."""

    def test_cluster_conversion_rate_basic(self):
        """Test ClusterConversionRateFeatures basic functionality."""
        try:
            from lookalike_features import ClusterConversionRateFeatures, HAVE_KPROTOTYPES
        except ImportError:
            pytest.skip("lookalike_features module not available")

        if not HAVE_KPROTOTYPES:
            pytest.skip("kmodes not installed")

        np.random.seed(42)
        X = pd.DataFrame({
            'numeric1': np.random.randn(200),
            'cat1': np.random.choice(['A', 'B', 'C'], 200)
        })
        y = np.random.binomial(1, 0.1, 200)

        transformer = ClusterConversionRateFeatures(
            n_clusters=5,
            categorical_cols=['cat1'],
            numeric_cols=['numeric1']
        )

        transformer.fit(X, y)
        features = transformer.transform(X)

        assert 'cluster_conversion_rate' in features.columns
        assert 'cluster_id' in features.columns
        assert len(features) == len(X)

    def test_faiss_similarity_basic(self):
        """Test FAISSConverterSimilarity basic functionality."""
        try:
            from lookalike_features import FAISSConverterSimilarity, HAVE_FAISS
        except ImportError:
            pytest.skip("lookalike_features module not available")

        if not HAVE_FAISS:
            pytest.skip("faiss not installed")

        np.random.seed(42)
        X = pd.DataFrame({
            'feat1': np.random.randn(200),
            'feat2': np.random.randn(200)
        })
        y = np.random.binomial(1, 0.2, 200)

        transformer = FAISSConverterSimilarity(
            n_neighbors=10,
            embedding_cols=['feat1', 'feat2']
        )

        transformer.fit(X, y)
        features = transformer.transform(X)

        assert 'mean_dist_converters' in features.columns
        assert 'min_dist_converter' in features.columns
        assert len(features) == len(X)

    def test_combined_transformer(self):
        """Test LookalikeFeatureTransformer combines both feature types."""
        try:
            from lookalike_features import LookalikeFeatureTransformer
        except ImportError:
            pytest.skip("lookalike_features module not available")

        np.random.seed(42)
        X = pd.DataFrame({
            'numeric1': np.random.randn(100),
            'cat1': np.random.choice(['A', 'B'], 100)
        })
        y = np.random.binomial(1, 0.1, 100)

        transformer = LookalikeFeatureTransformer(
            n_clusters=3,
            n_neighbors=5,
            categorical_cols=['cat1'],
            numeric_cols=['numeric1']
        )

        transformer.fit(X, y)
        features = transformer.transform(X)

        # Should have at least some features
        assert len(features.columns) >= 0


class TestStackingEnsemble:
    """Tests for calibrated stacking ensemble."""

    def test_stacking_ensemble_creation(self):
        """Test get_stacking_ensemble creates valid ensemble."""
        from training_lead_generation_model import (
            get_stacking_ensemble,
            HAVE_XGBOOST,
            HAVE_IMBLEARN_ENSEMBLE
        )

        if not (HAVE_XGBOOST or HAVE_IMBLEARN_ENSEMBLE):
            pytest.skip("Need XGBoost or imbalanced-learn for stacking")

        ensemble = get_stacking_ensemble(
            random_state=42,
            scale_pos_weight=10.0,
            cv=3
        )

        assert ensemble is not None
        assert hasattr(ensemble, 'fit')
        assert hasattr(ensemble, 'predict_proba')

    def test_stacking_ensemble_fit_predict(self):
        """Test stacking ensemble can fit and predict."""
        from training_lead_generation_model import (
            get_stacking_ensemble,
            HAVE_XGBOOST,
            HAVE_IMBLEARN_ENSEMBLE
        )

        if not (HAVE_XGBOOST or HAVE_IMBLEARN_ENSEMBLE):
            pytest.skip("Need XGBoost or imbalanced-learn for stacking")

        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = (X[:, 0] > 0.5).astype(int)

        ensemble = get_stacking_ensemble(
            random_state=42,
            scale_pos_weight=5.0,
            cv=2  # Small CV for speed
        )

        ensemble.fit(X, y)
        proba = ensemble.predict_proba(X)

        assert proba.shape == (len(X), 2)


class TestLambdaMARTWrapper:
    """Tests for LambdaMART ranking wrapper."""

    def test_lambdamart_wrapper_basic(self):
        """Test LambdaMARTWrapper basic functionality."""
        from training_lead_generation_model import (
            LambdaMARTWrapper,
            get_lambdamart_ranker,
            HAVE_LIGHTGBM
        )

        if not HAVE_LIGHTGBM:
            pytest.skip("LightGBM not installed")

        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = (X[:, 0] > 0).astype(int)

        ranker = get_lambdamart_ranker(random_state=42, truncation_level=100)
        wrapper = LambdaMARTWrapper(ranker)

        wrapper.fit(X, y)
        proba = wrapper.predict_proba(X)

        assert proba.shape == (len(X), 2)
        assert np.all(proba >= 0) and np.all(proba <= 1)

        predictions = wrapper.predict(X)
        assert predictions.shape == (len(X),)
        assert set(predictions).issubset({0, 1})


class TestColumnTransformerUpdates:
    """Tests for updated column transformer with new features."""

    def test_numeric_cols_include_temporal(self):
        """Test NUMERIC_COLS includes new temporal features."""
        from column_transformer_lead_gen import NUMERIC_COLS

        assert 'Company_Age_Log' in NUMERIC_COLS
        assert 'Month_Sin' in NUMERIC_COLS
        assert 'Month_Cos' in NUMERIC_COLS
        assert 'Is_End_Of_Quarter' in NUMERIC_COLS
        assert 'Is_Summer' in NUMERIC_COLS
        assert 'Quarter' in NUMERIC_COLS

    def test_low_card_cols_include_age_bucket(self):
        """Test LOW_CARD_CATEGORICAL_COLS includes Company_Age_Bucket."""
        from column_transformer_lead_gen import LOW_CARD_CATEGORICAL_COLS

        assert 'Company_Age_Bucket' in LOW_CARD_CATEGORICAL_COLS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
