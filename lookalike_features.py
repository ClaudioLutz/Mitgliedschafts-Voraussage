"""
Lookalike Feature Engineering for Lead Generation
==================================================
Implements similarity-based features for B2B lead scoring:
- Cluster conversion rates using K-Prototypes (mixed categorical/numeric)
- FAISS KNN propensity scores (distance to known converters)
- Segment centroid similarity

These features capture non-linear relationships that tree models may miss.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder

log = logging.getLogger(__name__)

# Optional dependencies
try:
    from kmodes.kprototypes import KPrototypes
    HAVE_KPROTOTYPES = True
except ImportError:
    HAVE_KPROTOTYPES = False
    log.warning("kmodes not installed. Cluster features will be disabled. Install with: pip install kmodes")

try:
    import faiss
    HAVE_FAISS = True
except ImportError:
    HAVE_FAISS = False
    log.warning("faiss not installed. KNN propensity features will be disabled. Install with: pip install faiss-cpu")


class ClusterConversionRateFeatures(BaseEstimator, TransformerMixin):
    """
    Compute cluster-based conversion rates as features.

    Uses K-Prototypes clustering for mixed categorical/numeric data,
    then computes historical conversion rate per cluster with Bayesian smoothing.

    Parameters
    ----------
    n_clusters : int, default=20
        Number of clusters to create.
    prior_weight : int, default=15
        Weight for global rate in Bayesian smoothing (number of pseudo-observations).
    categorical_cols : list, optional
        Names of categorical columns. If None, auto-detected.
    numeric_cols : list, optional
        Names of numeric columns. If None, auto-detected.
    random_state : int, default=42
        Random state for reproducibility.
    """

    def __init__(
        self,
        n_clusters=20,
        prior_weight=15,
        categorical_cols=None,
        numeric_cols=None,
        random_state=42
    ):
        self.n_clusters = n_clusters
        self.prior_weight = prior_weight
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols
        self.random_state = random_state

        self.kproto_ = None
        self.cluster_rates_ = None
        self.global_rate_ = None
        self.scaler_ = None
        self.label_encoders_ = None
        self._fitted_categorical_cols = None
        self._fitted_numeric_cols = None

    def fit(self, X, y):
        """Fit K-Prototypes and compute cluster conversion rates."""
        if not HAVE_KPROTOTYPES:
            log.warning("kmodes not available. Returning dummy features.")
            self.global_rate_ = np.mean(y)
            return self

        X = self._ensure_dataframe(X)
        y = np.asarray(y)

        # Identify column types
        self._fitted_categorical_cols = self.categorical_cols or self._detect_categorical(X)
        self._fitted_numeric_cols = self.numeric_cols or self._detect_numeric(X)

        log.info(f"ClusterFeatures: Using {len(self._fitted_categorical_cols)} categorical, "
                 f"{len(self._fitted_numeric_cols)} numeric columns")

        # Prepare data for K-Prototypes
        X_prepared, cat_indices = self._prepare_data(X)

        # Fit K-Prototypes
        self.kproto_ = KPrototypes(
            n_clusters=self.n_clusters,
            init='Huang',
            n_init=3,
            max_iter=100,
            random_state=self.random_state,
            n_jobs=-1
        )

        clusters = self.kproto_.fit_predict(X_prepared, categorical=cat_indices)

        # Compute conversion rates per cluster with Bayesian smoothing
        self.global_rate_ = np.mean(y)
        self.cluster_rates_ = {}

        for cluster_id in range(self.n_clusters):
            mask = clusters == cluster_id
            cluster_count = np.sum(mask)
            cluster_positives = np.sum(y[mask]) if cluster_count > 0 else 0

            if cluster_count > 0:
                cluster_rate = cluster_positives / cluster_count
                # Bayesian smoothing: combine cluster rate with global rate
                smoothed_rate = (
                    cluster_rate * cluster_count + self.global_rate_ * self.prior_weight
                ) / (cluster_count + self.prior_weight)
            else:
                smoothed_rate = self.global_rate_

            self.cluster_rates_[cluster_id] = smoothed_rate

        log.info(f"ClusterFeatures: Fitted {self.n_clusters} clusters. "
                 f"Rate range: {min(self.cluster_rates_.values()):.4f} - {max(self.cluster_rates_.values()):.4f}")

        return self

    def transform(self, X):
        """Add cluster conversion rate features."""
        X = self._ensure_dataframe(X)

        if not HAVE_KPROTOTYPES or self.kproto_ is None:
            # Return dummy features if not fitted
            return pd.DataFrame({
                'cluster_conversion_rate': np.full(len(X), self.global_rate_ or 0.0),
                'cluster_id': np.zeros(len(X), dtype=int)
            }, index=X.index)

        # Prepare data and predict clusters
        X_prepared, cat_indices = self._prepare_data(X)
        clusters = self.kproto_.predict(X_prepared, categorical=cat_indices)

        # Map clusters to conversion rates
        conversion_rates = np.array([
            self.cluster_rates_.get(c, self.global_rate_) for c in clusters
        ])

        return pd.DataFrame({
            'cluster_conversion_rate': conversion_rates,
            'cluster_id': clusters
        }, index=X.index)

    def _ensure_dataframe(self, X):
        if not isinstance(X, pd.DataFrame):
            return pd.DataFrame(X)
        return X

    def _detect_categorical(self, X):
        """Auto-detect categorical columns."""
        cat_cols = []
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                cat_cols.append(col)
            elif X[col].nunique() < 20 and X[col].dtype in ['int64', 'int32']:
                # Low-cardinality integers might be categorical
                pass  # Keep as numeric for now
        return cat_cols

    def _detect_numeric(self, X):
        """Auto-detect numeric columns."""
        num_cols = []
        for col in X.columns:
            if col not in (self._fitted_categorical_cols or []):
                if X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    num_cols.append(col)
        return num_cols

    def _prepare_data(self, X):
        """Prepare data for K-Prototypes (numeric + encoded categorical)."""
        # Initialize encoders and scaler if needed
        if self.label_encoders_ is None:
            self.label_encoders_ = {}
            self.scaler_ = StandardScaler()

        # Select columns
        cat_cols = [c for c in self._fitted_categorical_cols if c in X.columns]
        num_cols = [c for c in self._fitted_numeric_cols if c in X.columns]

        if not cat_cols and not num_cols:
            raise ValueError("No valid columns found for clustering.")

        parts = []
        cat_indices = []

        # Process numeric columns
        if num_cols:
            X_num = X[num_cols].fillna(0).values.astype(float)
            if hasattr(self.scaler_, 'mean_'):
                X_num = self.scaler_.transform(X_num)
            else:
                X_num = self.scaler_.fit_transform(X_num)
            parts.append(X_num)

        # Process categorical columns
        if cat_cols:
            X_cat = []
            for i, col in enumerate(cat_cols):
                values = X[col].fillna('missing').astype(str)
                if col not in self.label_encoders_:
                    le = LabelEncoder()
                    encoded = le.fit_transform(values)
                    self.label_encoders_[col] = le
                else:
                    le = self.label_encoders_[col]
                    # Handle unseen categories
                    known = set(le.classes_)
                    encoded = np.array([
                        le.transform([v])[0] if v in known else -1
                        for v in values
                    ])
                X_cat.append(encoded)

            X_cat = np.column_stack(X_cat)
            start_idx = len(num_cols) if num_cols else 0
            cat_indices = list(range(start_idx, start_idx + len(cat_cols)))
            parts.append(X_cat)

        X_combined = np.hstack(parts) if len(parts) > 1 else parts[0]

        return X_combined, cat_indices


class FAISSConverterSimilarity(BaseEstimator, TransformerMixin):
    """
    Compute KNN-based similarity to known converters using FAISS.

    Creates features measuring distance to converted companies,
    which captures non-linear relationships that trees may miss.

    Parameters
    ----------
    n_neighbors : int, default=20
        Number of nearest neighbors to consider.
    embedding_cols : list, optional
        Columns to use for computing embeddings. If None, uses all numeric columns.
    random_state : int, default=42
        Random state for reproducibility.
    """

    def __init__(self, n_neighbors=20, embedding_cols=None, random_state=42):
        self.n_neighbors = n_neighbors
        self.embedding_cols = embedding_cols
        self.random_state = random_state

        self.index_ = None
        self.scaler_ = None
        self._fitted_cols = None

    def fit(self, X, y):
        """Build FAISS index from positive samples (converters)."""
        if not HAVE_FAISS:
            log.warning("FAISS not available. Returning dummy features.")
            return self

        X = self._ensure_dataframe(X)
        y = np.asarray(y)

        # Identify columns to use
        self._fitted_cols = self.embedding_cols or self._detect_numeric(X)
        log.info(f"FAISSConverterSimilarity: Using {len(self._fitted_cols)} columns for embedding")

        if not self._fitted_cols:
            log.warning("No numeric columns found for FAISS embedding.")
            return self

        # Prepare embedding data
        X_embed = X[self._fitted_cols].fillna(0).values.astype('float32')

        # Scale for better distance computation
        self.scaler_ = StandardScaler()
        X_embed = self.scaler_.fit_transform(X_embed).astype('float32')

        # Get positive samples only
        positive_mask = y == 1
        X_positive = X_embed[positive_mask]

        if len(X_positive) < self.n_neighbors:
            log.warning(f"Only {len(X_positive)} positive samples. Need at least {self.n_neighbors}.")
            return self

        # Build FAISS index with IVF for efficiency
        d = X_positive.shape[1]
        n_positives = len(X_positive)

        # Choose number of centroids for IVF (rule of thumb: sqrt(n) to 4*sqrt(n))
        n_centroids = min(100, max(10, int(np.sqrt(n_positives))))

        # Create IVF index for approximate nearest neighbor search
        quantizer = faiss.IndexFlatL2(d)
        self.index_ = faiss.IndexIVFFlat(quantizer, d, n_centroids)

        # Train and add vectors
        self.index_.train(X_positive)
        self.index_.add(X_positive)

        # Set search parameters for balance of speed/accuracy
        self.index_.nprobe = min(10, n_centroids)

        log.info(f"FAISSConverterSimilarity: Indexed {n_positives} positive samples with {n_centroids} centroids")

        return self

    def transform(self, X):
        """Compute distance-based features to known converters."""
        X = self._ensure_dataframe(X)

        if not HAVE_FAISS or self.index_ is None or not self._fitted_cols:
            # Return dummy features
            return pd.DataFrame({
                'mean_dist_converters': np.zeros(len(X)),
                'min_dist_converter': np.zeros(len(X)),
                'max_dist_converter': np.zeros(len(X))
            }, index=X.index)

        # Prepare embedding
        X_embed = X[self._fitted_cols].fillna(0).values.astype('float32')
        X_embed = self.scaler_.transform(X_embed).astype('float32')

        # Search for nearest neighbors
        distances, _ = self.index_.search(X_embed, self.n_neighbors)

        # Compute features from distances
        # Note: FAISS returns squared L2 distances
        distances = np.sqrt(np.maximum(distances, 0))  # Convert to actual distances

        return pd.DataFrame({
            'mean_dist_converters': distances.mean(axis=1),
            'min_dist_converter': distances.min(axis=1),
            'max_dist_converter': distances.max(axis=1),
            'std_dist_converters': distances.std(axis=1)
        }, index=X.index)

    def _ensure_dataframe(self, X):
        if not isinstance(X, pd.DataFrame):
            return pd.DataFrame(X)
        return X

    def _detect_numeric(self, X):
        """Auto-detect numeric columns."""
        return [
            col for col in X.columns
            if X[col].dtype in ['float64', 'float32', 'int64', 'int32']
        ]


class LookalikeFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Combined transformer for all lookalike features.

    Combines cluster conversion rates and FAISS KNN similarity
    into a single transformer for easy integration.

    Parameters
    ----------
    use_cluster_features : bool, default=True
        Whether to include cluster conversion rate features.
    use_faiss_features : bool, default=True
        Whether to include FAISS KNN similarity features.
    n_clusters : int, default=20
        Number of clusters for K-Prototypes.
    n_neighbors : int, default=20
        Number of neighbors for FAISS KNN.
    categorical_cols : list, optional
        Categorical columns for clustering.
    numeric_cols : list, optional
        Numeric columns for clustering and FAISS.
    random_state : int, default=42
        Random state for reproducibility.
    """

    def __init__(
        self,
        use_cluster_features=True,
        use_faiss_features=True,
        n_clusters=20,
        n_neighbors=20,
        categorical_cols=None,
        numeric_cols=None,
        random_state=42
    ):
        self.use_cluster_features = use_cluster_features
        self.use_faiss_features = use_faiss_features
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.categorical_cols = categorical_cols
        self.numeric_cols = numeric_cols
        self.random_state = random_state

        self.cluster_transformer_ = None
        self.faiss_transformer_ = None
        self._feature_names = []

    def fit(self, X, y):
        """Fit all lookalike feature transformers."""
        log.info("Fitting LookalikeFeatureTransformer...")

        if self.use_cluster_features and HAVE_KPROTOTYPES:
            self.cluster_transformer_ = ClusterConversionRateFeatures(
                n_clusters=self.n_clusters,
                categorical_cols=self.categorical_cols,
                numeric_cols=self.numeric_cols,
                random_state=self.random_state
            )
            self.cluster_transformer_.fit(X, y)
            self._feature_names.extend(['cluster_conversion_rate', 'cluster_id'])

        if self.use_faiss_features and HAVE_FAISS:
            self.faiss_transformer_ = FAISSConverterSimilarity(
                n_neighbors=self.n_neighbors,
                embedding_cols=self.numeric_cols,
                random_state=self.random_state
            )
            self.faiss_transformer_.fit(X, y)
            self._feature_names.extend([
                'mean_dist_converters', 'min_dist_converter',
                'max_dist_converter', 'std_dist_converters'
            ])

        log.info(f"LookalikeFeatureTransformer: Created {len(self._feature_names)} features")
        return self

    def transform(self, X):
        """Transform data to add lookalike features."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        features = []

        if self.cluster_transformer_ is not None:
            cluster_features = self.cluster_transformer_.transform(X)
            features.append(cluster_features)

        if self.faiss_transformer_ is not None:
            faiss_features = self.faiss_transformer_.transform(X)
            features.append(faiss_features)

        if not features:
            # Return empty DataFrame with correct index
            return pd.DataFrame(index=X.index)

        result = pd.concat(features, axis=1)
        return result

    def get_feature_names_out(self, input_features=None):
        """Return output feature names."""
        return np.array(self._feature_names)


def add_lookalike_features(
    X_train, y_train, X_val=None, X_test=None, X_current=None,
    n_clusters=20, n_neighbors=20,
    categorical_cols=None, numeric_cols=None,
    random_state=42
):
    """
    Convenience function to add lookalike features to train/val/test/current data.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : array-like
        Training labels.
    X_val, X_test, X_current : pd.DataFrame, optional
        Optional additional datasets to transform.
    n_clusters, n_neighbors : int
        Parameters for cluster and FAISS features.
    categorical_cols, numeric_cols : list, optional
        Column specifications.
    random_state : int
        Random state.

    Returns
    -------
    dict : Dictionary with transformed DataFrames for each input.
    """
    transformer = LookalikeFeatureTransformer(
        n_clusters=n_clusters,
        n_neighbors=n_neighbors,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        random_state=random_state
    )

    # Fit on training data
    transformer.fit(X_train, y_train)

    # Transform all datasets
    result = {
        'train': pd.concat([X_train.reset_index(drop=True),
                           transformer.transform(X_train).reset_index(drop=True)], axis=1)
    }

    if X_val is not None:
        result['val'] = pd.concat([X_val.reset_index(drop=True),
                                   transformer.transform(X_val).reset_index(drop=True)], axis=1)

    if X_test is not None:
        result['test'] = pd.concat([X_test.reset_index(drop=True),
                                    transformer.transform(X_test).reset_index(drop=True)], axis=1)

    if X_current is not None:
        result['current'] = pd.concat([X_current.reset_index(drop=True),
                                       transformer.transform(X_current).reset_index(drop=True)], axis=1)

    return result


if __name__ == "__main__":
    # Example usage
    print("Lookalike Feature Engineering Module")
    print(f"K-Prototypes available: {HAVE_KPROTOTYPES}")
    print(f"FAISS available: {HAVE_FAISS}")

    # Create sample data for testing
    np.random.seed(42)
    n_samples = 1000

    X = pd.DataFrame({
        'numeric1': np.random.randn(n_samples),
        'numeric2': np.random.randn(n_samples) * 10,
        'cat1': np.random.choice(['A', 'B', 'C'], n_samples),
        'cat2': np.random.choice(['X', 'Y'], n_samples)
    })
    y = np.random.binomial(1, 0.1, n_samples)  # 10% positive rate

    transformer = LookalikeFeatureTransformer(
        n_clusters=5,
        n_neighbors=10,
        categorical_cols=['cat1', 'cat2'],
        numeric_cols=['numeric1', 'numeric2']
    )

    transformer.fit(X, y)
    features = transformer.transform(X)
    print(f"\nGenerated features: {list(features.columns)}")
    print(features.head())
