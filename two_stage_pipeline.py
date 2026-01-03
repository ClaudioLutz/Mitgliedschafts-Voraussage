"""
Two-Stage Filter-Then-Rank Pipeline for Lead Generation
========================================================
Implements a two-stage architecture for extreme class imbalance:
- Stage 1: High-recall filter to reduce candidates (~15-20% of data)
- Stage 2: Precision-focused ranker on filtered data

Key design decisions:
- Out-of-fold predictions for Stage 1 to prevent data leakage
- Stage 2 trains on filtered data with better class balance (~1-2% vs 0.11%)
- Different feature sets can be used for each stage

Reference: This architecture is standard in ad-tech (Twitter, Meta)
where rankers train on retrieved candidates.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

log = logging.getLogger(__name__)


def find_recall_threshold(y_true, y_score, target_recall=0.95):
    """
    Find the threshold that achieves a target recall.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_score : array-like
        Predicted probabilities.
    target_recall : float, default=0.95
        Target recall to achieve.

    Returns
    -------
    float : Threshold value.
    float : Actual recall achieved.
    float : Precision at that threshold.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    # Find index where recall >= target_recall
    valid_indices = np.where(recall >= target_recall)[0]

    if len(valid_indices) == 0:
        # Can't achieve target recall, use minimum threshold
        log.warning(f"Cannot achieve {target_recall:.0%} recall. Using minimum threshold.")
        return float(thresholds[0]) if len(thresholds) > 0 else 0.0, recall[0], precision[0]

    # Get the highest threshold that still achieves target recall
    # (this maximizes precision while maintaining recall)
    idx = valid_indices[-1]

    # Handle edge case where idx is at the end
    if idx >= len(thresholds):
        idx = len(thresholds) - 1

    return float(thresholds[idx]), float(recall[idx]), float(precision[idx])


class TwoStagePipeline(BaseEstimator, ClassifierMixin):
    """
    Two-stage filter-then-rank pipeline for extreme class imbalance.

    Stage 1 (Filter): High-recall model to filter out obvious non-converters.
    Stage 2 (Rank): Precision-focused model trained on filtered candidates.

    Critical: Uses out-of-fold predictions for Stage 1 filtering during training
    to prevent data leakage.

    Parameters
    ----------
    stage1_estimator : estimator, optional
        High-recall classifier for filtering. Default: LogisticRegression with balanced weights.
    stage2_estimator : estimator, optional
        Precision-focused classifier for ranking. Default: HistGradientBoostingClassifier.
    stage1_target_recall : float, default=0.95
        Target recall for Stage 1 threshold selection.
    stage1_cv : int, default=5
        Number of CV folds for out-of-fold predictions.
    calibrate_stage2 : bool, default=True
        Whether to calibrate Stage 2 probabilities.
    random_state : int, default=42
        Random state for reproducibility.

    Attributes
    ----------
    stage1_threshold_ : float
        Threshold for Stage 1 filtering.
    stage1_recall_ : float
        Actual recall achieved by Stage 1.
    stage1_precision_ : float
        Precision at Stage 1 threshold.
    stage2_class_balance_ : float
        Positive rate in Stage 2 training data.
    """

    def __init__(
        self,
        stage1_estimator=None,
        stage2_estimator=None,
        stage1_target_recall=0.95,
        stage1_cv=5,
        calibrate_stage2=True,
        random_state=42
    ):
        self.stage1_estimator = stage1_estimator
        self.stage2_estimator = stage2_estimator
        self.stage1_target_recall = stage1_target_recall
        self.stage1_cv = stage1_cv
        self.calibrate_stage2 = calibrate_stage2
        self.random_state = random_state

        # Fitted attributes
        self.stage1_ = None
        self.stage2_ = None
        self.stage1_threshold_ = None
        self.stage1_recall_ = None
        self.stage1_precision_ = None
        self.stage2_class_balance_ = None
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        """
        Fit the two-stage pipeline.

        Uses out-of-fold predictions for Stage 1 filtering to prevent leakage.
        """
        X = self._ensure_array(X)
        y = np.asarray(y)

        log.info("Fitting TwoStagePipeline...")
        log.info(f"Input data: {len(y)} samples, {np.sum(y)} positives ({np.mean(y):.4%})")

        # Initialize estimators
        stage1 = self.stage1_estimator
        if stage1 is None:
            stage1 = LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=self.random_state
            )

        stage2 = self.stage2_estimator
        if stage2 is None:
            stage2 = HistGradientBoostingClassifier(
                class_weight='balanced',
                max_iter=200,
                learning_rate=0.05,
                max_depth=8,
                random_state=self.random_state
            )

        # Stage 1: Get OUT-OF-FOLD predictions (prevents leakage!)
        log.info(f"Stage 1: Computing {self.stage1_cv}-fold out-of-fold predictions...")
        cv = StratifiedKFold(n_splits=self.stage1_cv, shuffle=True, random_state=self.random_state)

        stage1_oof = cross_val_predict(
            stage1, X, y,
            cv=cv,
            method='predict_proba'
        )[:, 1]

        # Find threshold for target recall
        self.stage1_threshold_, self.stage1_recall_, self.stage1_precision_ = find_recall_threshold(
            y, stage1_oof, target_recall=self.stage1_target_recall
        )

        log.info(f"Stage 1 threshold: {self.stage1_threshold_:.6f} "
                 f"(recall={self.stage1_recall_:.2%}, precision={self.stage1_precision_:.2%})")

        # Fit Stage 1 on full data for inference
        self.stage1_ = clone(stage1)
        self.stage1_.fit(X, y)

        # Stage 2: Train on filtered data using OOF predictions
        filtered_mask = stage1_oof >= self.stage1_threshold_
        X_filtered = X[filtered_mask] if hasattr(X, '__getitem__') else X.iloc[filtered_mask]
        y_filtered = y[filtered_mask]

        n_filtered = np.sum(filtered_mask)
        n_positives_retained = np.sum(y_filtered)
        self.stage2_class_balance_ = np.mean(y_filtered) if len(y_filtered) > 0 else 0

        log.info(f"Stage 2: Training on {n_filtered} filtered samples "
                 f"({n_positives_retained} positives, {self.stage2_class_balance_:.2%} rate)")

        if n_filtered < 100:
            log.warning("Very few samples after filtering. Stage 2 may not train well.")

        self.stage2_ = clone(stage2)
        self.stage2_.fit(X_filtered, y_filtered)

        # Optional calibration of Stage 2
        if self.calibrate_stage2 and n_positives_retained > 20:
            log.info("Calibrating Stage 2 probabilities...")
            try:
                calibrated = CalibratedClassifierCV(
                    estimator=self.stage2_,
                    method='isotonic',
                    cv=min(3, self.stage1_cv)
                )
                calibrated.fit(X_filtered, y_filtered)
                self.stage2_ = calibrated
            except Exception as e:
                log.warning(f"Calibration failed: {e}. Using uncalibrated Stage 2.")

        log.info("TwoStagePipeline fitted successfully.")
        return self

    def predict_proba(self, X):
        """
        Predict probabilities using two-stage pipeline.

        Samples that don't pass Stage 1 filter get Stage 1 probability.
        Samples that pass get Stage 2 probability.
        """
        X = self._ensure_array(X)

        # Stage 1 predictions
        stage1_proba = self.stage1_.predict_proba(X)[:, 1]

        # Identify samples that pass Stage 1
        pass_stage1 = stage1_proba >= self.stage1_threshold_

        # Initialize final probabilities with Stage 1 probabilities
        final_proba = stage1_proba.copy()

        # For samples passing Stage 1, use Stage 2 probabilities
        if np.any(pass_stage1):
            X_pass = X[pass_stage1] if hasattr(X, '__getitem__') else X.iloc[pass_stage1]
            stage2_proba = self.stage2_.predict_proba(X_pass)[:, 1]
            final_proba[pass_stage1] = stage2_proba

        return np.column_stack([1 - final_proba, final_proba])

    def predict(self, X):
        """Predict class labels."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)

    def get_stage1_filtered_mask(self, X):
        """Get mask of samples that pass Stage 1 filter."""
        X = self._ensure_array(X)
        stage1_proba = self.stage1_.predict_proba(X)[:, 1]
        return stage1_proba >= self.stage1_threshold_

    def get_diagnostic_info(self):
        """Return diagnostic information about the pipeline."""
        return {
            'stage1_threshold': self.stage1_threshold_,
            'stage1_recall': self.stage1_recall_,
            'stage1_precision': self.stage1_precision_,
            'stage2_class_balance': self.stage2_class_balance_,
        }

    def _ensure_array(self, X):
        """Ensure X is in array format."""
        if hasattr(X, 'values'):
            return X.values
        return np.asarray(X)


class TwoStageWithDifferentFeatures(TwoStagePipeline):
    """
    Two-stage pipeline with different feature sets for each stage.

    Stage 1 can use coarse demographic features for fast filtering.
    Stage 2 can use expensive computed features (e.g., lookalike scores).

    Parameters
    ----------
    stage1_features : list, optional
        Column names for Stage 1 features. If None, uses all.
    stage2_features : list, optional
        Column names for Stage 2 features. If None, uses all.
    **kwargs
        Additional arguments passed to TwoStagePipeline.
    """

    def __init__(
        self,
        stage1_features=None,
        stage2_features=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.stage1_features = stage1_features
        self.stage2_features = stage2_features
        self._feature_names = None

    def fit(self, X, y):
        """Fit with different feature sets per stage."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self._feature_names = list(X.columns)

        # Extract feature subsets
        X_stage1 = self._get_stage1_features(X)
        X_stage2 = X  # Stage 2 uses all features by default

        y = np.asarray(y)

        log.info(f"TwoStageWithDifferentFeatures: Stage 1 uses {X_stage1.shape[1]} features, "
                 f"Stage 2 uses {X_stage2.shape[1]} features")

        # Initialize estimators
        stage1 = self.stage1_estimator
        if stage1 is None:
            stage1 = LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=self.random_state
            )

        stage2 = self.stage2_estimator
        if stage2 is None:
            stage2 = HistGradientBoostingClassifier(
                class_weight='balanced',
                max_iter=200,
                random_state=self.random_state
            )

        # Stage 1: Out-of-fold predictions
        cv = StratifiedKFold(n_splits=self.stage1_cv, shuffle=True, random_state=self.random_state)
        stage1_oof = cross_val_predict(stage1, X_stage1, y, cv=cv, method='predict_proba')[:, 1]

        # Find threshold
        self.stage1_threshold_, self.stage1_recall_, self.stage1_precision_ = find_recall_threshold(
            y, stage1_oof, target_recall=self.stage1_target_recall
        )

        log.info(f"Stage 1 threshold: {self.stage1_threshold_:.6f}")

        # Fit Stage 1 on full data
        self.stage1_ = clone(stage1)
        self.stage1_.fit(X_stage1, y)

        # Stage 2: Train on filtered data with Stage 2 features
        filtered_mask = stage1_oof >= self.stage1_threshold_
        X2_filtered = self._get_stage2_features(X)[filtered_mask]
        y_filtered = y[filtered_mask]

        self.stage2_class_balance_ = np.mean(y_filtered) if len(y_filtered) > 0 else 0

        log.info(f"Stage 2: {len(y_filtered)} samples ({self.stage2_class_balance_:.2%} positive)")

        self.stage2_ = clone(stage2)
        self.stage2_.fit(X2_filtered, y_filtered)

        if self.calibrate_stage2 and np.sum(y_filtered) > 20:
            try:
                calibrated = CalibratedClassifierCV(
                    estimator=self.stage2_, method='isotonic', cv=3
                )
                calibrated.fit(X2_filtered, y_filtered)
                self.stage2_ = calibrated
            except Exception as e:
                log.warning(f"Stage 2 calibration failed: {e}")

        return self

    def predict_proba(self, X):
        """Predict with different feature sets per stage."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self._feature_names)

        X_stage1 = self._get_stage1_features(X)
        X_stage2 = self._get_stage2_features(X)

        stage1_proba = self.stage1_.predict_proba(X_stage1)[:, 1]
        pass_stage1 = stage1_proba >= self.stage1_threshold_

        final_proba = stage1_proba.copy()

        if np.any(pass_stage1):
            X2_pass = X_stage2[pass_stage1]
            stage2_proba = self.stage2_.predict_proba(X2_pass)[:, 1]
            final_proba[pass_stage1] = stage2_proba

        return np.column_stack([1 - final_proba, final_proba])

    def _get_stage1_features(self, X):
        """Extract Stage 1 features."""
        if self.stage1_features is None:
            return X
        available = [f for f in self.stage1_features if f in X.columns]
        return X[available]

    def _get_stage2_features(self, X):
        """Extract Stage 2 features."""
        if self.stage2_features is None:
            return X
        available = [f for f in self.stage2_features if f in X.columns]
        return X[available]


def create_two_stage_pipeline(
    stage1_type='logistic',
    stage2_type='hgb',
    target_recall=0.95,
    random_state=42
):
    """
    Factory function to create a two-stage pipeline.

    Parameters
    ----------
    stage1_type : str
        Type of Stage 1 model: 'logistic', 'rf' (random forest), or 'hgb'.
    stage2_type : str
        Type of Stage 2 model: 'hgb', 'xgb', or 'logistic'.
    target_recall : float
        Target recall for Stage 1 filtering.
    random_state : int
        Random state.

    Returns
    -------
    TwoStagePipeline
    """
    # Stage 1: High-recall model
    if stage1_type == 'logistic':
        stage1 = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            C=0.1,  # More regularization for simpler model
            random_state=random_state
        )
    elif stage1_type == 'rf':
        stage1 = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
    elif stage1_type == 'hgb':
        stage1 = HistGradientBoostingClassifier(
            class_weight='balanced',
            max_iter=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown stage1_type: {stage1_type}")

    # Stage 2: Precision-focused model
    if stage2_type == 'hgb':
        stage2 = HistGradientBoostingClassifier(
            class_weight='balanced',
            max_iter=300,
            learning_rate=0.05,
            max_depth=8,
            min_samples_leaf=20,
            random_state=random_state
        )
    elif stage2_type == 'xgb':
        try:
            from xgboost import XGBClassifier
            stage2 = XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=8,
                scale_pos_weight=10,  # Will be adjusted
                tree_method='hist',
                random_state=random_state
            )
        except ImportError:
            log.warning("XGBoost not available, using HGB for Stage 2")
            stage2 = HistGradientBoostingClassifier(
                class_weight='balanced',
                max_iter=300,
                random_state=random_state
            )
    elif stage2_type == 'logistic':
        stage2 = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown stage2_type: {stage2_type}")

    return TwoStagePipeline(
        stage1_estimator=stage1,
        stage2_estimator=stage2,
        stage1_target_recall=target_recall,
        random_state=random_state
    )


if __name__ == "__main__":
    # Example usage and testing
    from sklearn.datasets import make_classification

    print("Two-Stage Pipeline Example")
    print("=" * 50)

    # Create imbalanced dataset
    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        weights=[0.99, 0.01],  # 1% positive rate
        random_state=42
    )

    # Split data
    train_idx = np.random.choice(len(X), size=int(0.7 * len(X)), replace=False)
    test_idx = np.setdiff1d(np.arange(len(X)), train_idx)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"Train: {len(y_train)} samples, {np.sum(y_train)} positives ({np.mean(y_train):.2%})")
    print(f"Test: {len(y_test)} samples, {np.sum(y_test)} positives ({np.mean(y_test):.2%})")

    # Create and fit pipeline
    pipeline = create_two_stage_pipeline(
        stage1_type='logistic',
        stage2_type='hgb',
        target_recall=0.95
    )

    pipeline.fit(X_train, y_train)

    # Evaluate
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    ap = average_precision_score(y_test, y_proba)

    # Precision@K
    k = 100
    top_k_idx = np.argsort(y_proba)[-k:]
    p_at_k = np.mean(y_test[top_k_idx])

    print(f"\nResults:")
    print(f"  Average Precision: {ap:.4f}")
    print(f"  Precision@{k}: {p_at_k:.2%}")
    print(f"\nDiagnostics:")
    for key, value in pipeline.get_diagnostic_info().items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
