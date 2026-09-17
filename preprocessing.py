"""
Preprocessing module for SRP project.
Implements various distribution transformation techniques and comparison framework.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler,
    QuantileTransformer, PowerTransformer, Normalizer
)
from sklearn.impute import KNNImputer
from scipy import stats
from scipy.stats import skew, kurtosis
from typing import Dict, Tuple, Optional, List


class Preprocessor:
    """
    Unified preprocessor supporting multiple transformation techniques.
    """
    
    def __init__(self, method: str = 'standard', **kwargs):
        """
        Initialize preprocessor with specified method.
        
        Args:
            method: One of 'standard', 'robust', 'minmax', 'quantile', 'power', 'normalizer',
                'rankgauss', 'knn_impute'
            **kwargs: Additional arguments passed to the scaler
        """
        self.method = method
        self.scaler = self._create_scaler(method, **kwargs)
        self._fitted = False
        
    def _create_scaler(self, method: str, **kwargs):
        """Create the appropriate scaler based on method."""
        if method == 'standard':
            return StandardScaler()
        elif method == 'robust':
            return RobustScaler()
        elif method == 'minmax':
            return MinMaxScaler()
        elif method == 'quantile':
            output_dist = kwargs.get('output_distribution', 'normal')
            n_quantiles = kwargs.get('n_quantiles', 1000)
            random_state = kwargs.get('random_state', 42)
            return QuantileTransformer(output_distribution=output_dist, 
                                       n_quantiles=n_quantiles, 
                                       random_state=random_state)
        elif method == 'power':
            method_type = kwargs.get('method', 'yeo-johnson')  # 'yeo-johnson' or 'box-cox'
            return PowerTransformer(method=method_type)
        elif method == 'normalizer':
            norm_type = kwargs.get('norm', 'l2')
            return Normalizer(norm=norm_type)
        elif method == 'rankgauss':
            return RankGaussTransformer(**kwargs)
        elif method == 'knn_impute':
            # Not a scaler — fills missing values from neighbor averages. Doesn't rescale
            # the distribution, so it's usually chained before one of the scalers above.
            n_neighbors = kwargs.get('n_neighbors', 5)
            return KNNImputer(n_neighbors=n_neighbors)
        else:
            raise ValueError(f"Unknown preprocessing method: {method}")
    
    def fit(self, X: np.ndarray) -> 'Preprocessor':
        """Fit the preprocessor on data."""
        self.scaler.fit(X)
        self._fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted preprocessor."""
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        return self.scaler.transform(X)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        X_transformed = self.scaler.fit_transform(X)
        self._fitted = True
        return X_transformed


class RankGaussTransformer:
    """
    Rank-Gauss (Normalizer) transformation.
    Maps data to standard normal using empirical CDF.

    fit() stores the *training* data's sorted values and their inverse-normal-CDF targets;
    transform() interpolates any new data (train, val, test, or a single row) into that same
    fixed mapping via np.interp, clipping to the train-time range for unseen extremes. This
    fixes the earlier version, where transform() recomputed ranks on whatever was passed —
    train and val/test would each get mapped through a *different* empirical distribution.
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self._fitted = False
        self.sorted_train_ = None
        self.gauss_targets_ = None

    def fit(self, X: np.ndarray) -> 'RankGaussTransformer':
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        self.sorted_train_ = np.sort(X, axis=0)
        ranks = (np.arange(n) + 0.5) / n
        ranks = np.clip(ranks, self.eps, 1 - self.eps)
        self.gauss_targets_ = stats.norm.ppf(ranks)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply rank-gauss transformation using the mapping learned at fit time."""
        if not self._fitted:
            raise RuntimeError("RankGaussTransformer must be fitted before transform")
        X = np.asarray(X, dtype=np.float64)
        out = np.empty_like(X)
        for j in range(X.shape[1]):
            out[:, j] = np.interp(
                X[:, j], self.sorted_train_[:, j], self.gauss_targets_,
                left=self.gauss_targets_[0], right=self.gauss_targets_[-1],
            )
        return out.astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def compute_distribution_stats(X: np.ndarray, name: str = "") -> Dict:
    """
    Compute statistics about the distribution of features.
    
    Returns dict with skewness, kurtosis, and normality test results.
    """
    stats_dict = {'name': name}
    
    # Per-feature statistics
    skewness = skew(X, axis=0)
    kurt = kurtosis(X, axis=0)
    
    stats_dict['mean_skewness'] = np.mean(np.abs(skewness))
    stats_dict['mean_kurtosis'] = np.mean(np.abs(kurt))
    
    # Shapiro-Wilk test for normality (on a sample if too many features)
    if X.shape[1] <= 5000:
        sample = X[:, :min(5000, X.shape[1])]
    else:
        sample = X[:, np.random.choice(X.shape[1], 5000, replace=False)]
    
    shapiro_stat, shapiro_p = stats.shapiro(sample.flatten()[:5000])
    stats_dict['shapiro_stat'] = shapiro_stat
    stats_dict['shapiro_p'] = shapiro_p
    stats_dict['is_normal'] = shapiro_p > 0.05
    
    return stats_dict


def compare_preprocessors(X: np.ndarray, methods: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compare multiple preprocessing methods on the same data.
    
    Returns DataFrame with statistics for each method.
    """
    if methods is None:
        methods = ['standard', 'robust', 'quantile', 'power', 'rankgauss']
    
    results = []
    for method in methods:
        try:
            preprocessor = Preprocessor(method)
            X_transformed = preprocessor.fit_transform(X)
            stats_dict = compute_distribution_stats(X_transformed, method)
            results.append(stats_dict)
        except Exception as e:
            print(f"Error with {method}: {e}")
    
    return pd.DataFrame(results)


def plot_preprocessing_comparison(X_original: np.ndarray, X_transformed: np.ndarray, 
                                   method_name: str, feature_idx: int = 0):
    """
    Plot comparison of original vs transformed distribution for a feature.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Original distribution
    axes[0].hist(X_original[:, feature_idx], bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_title(f'Original - Feature {feature_idx}')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Count')
    
    # Transformed distribution
    axes[1].hist(X_transformed[:, feature_idx], bins=50, alpha=0.7, edgecolor='black')
    axes[1].set_title(f'{method_name} - Feature {feature_idx}')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.show()


def plot_all_preprocessors(X: np.ndarray, methods: Optional[List[str]] = None, 
                            feature_idx: int = 0, ncols: int = 4):
    """
    Plot distributions for all preprocessing methods side by side.
    """
    if methods is None:
        methods = ['standard', 'robust', 'quantile', 'power', 'rankgauss']
    
    n_methods = len(methods)
    nrows = (n_methods + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
    axes = np.array(axes).flatten()
    
    # Plot original
    axes[0].hist(X[:, feature_idx], bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_title('Original')
    
    for i, method in enumerate(methods):
        try:
            preprocessor = Preprocessor(method)
            X_transformed = preprocessor.fit_transform(X)
            axes[i+1].hist(X_transformed[:, feature_idx], bins=50, alpha=0.7, edgecolor='black')
            axes[i+1].set_title(method)
        except Exception as e:
            axes[i+1].text(0.5, 0.5, f'Error:\n{e}', ha='center', va='center')
            axes[i+1].set_title(method)
    
    # Hide unused axes
    for j in range(len(methods)+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def evaluate_preprocessing_impact(X: np.ndarray, y: np.ndarray, methods: Optional[List[str]] = None,
                                   model_class=None, **model_kwargs) -> pd.DataFrame:
    """
    Evaluate preprocessing methods based on model performance.
    
    If model_class is provided, trains a simple model and returns performance metrics.
    Otherwise, returns distribution statistics.
    """
    if methods is None:
        methods = ['standard', 'robust', 'quantile', 'power', 'rankgauss']
    
    results = []
    
    for method in methods:
        try:
            preprocessor = Preprocessor(method)
            X_transformed = preprocessor.fit_transform(X)
            
            stats_dict = compute_distribution_stats(X_transformed, method)
            
            if model_class is not None:
                # Train and evaluate model
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import mean_squared_error, accuracy_score
                
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X_transformed, y, test_size=0.2, random_state=42
                )
                
                model = model_class(**model_kwargs)
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_val)
                
                if len(np.unique(y)) > 10:  # Classification
                    score = accuracy_score(y_val, y_pred)
                    stats_dict['accuracy'] = score
                else:  # Regression
                    score = -mean_squared_error(y_val, y_pred)
                    stats_dict['neg_mse'] = score
            
            results.append(stats_dict)
        except Exception as e:
            print(f"Error with {method}: {e}")
            results.append({'name': method, 'error': str(e)})
    
    return pd.DataFrame(results)


class TabularPreprocessor:
    """
    Numeric columns -> impute, then scale with a configurable technique.
    Low-cardinality categorical columns -> one-hot. Fit on train only.

    numeric_scaler: one of 'rankgauss', 'quantile', 'standard', 'minmax', 'robust', 'power',
        'normalizer', or 'none' (skip scaling entirely).
    numeric_imputer: 'mean' (per-column mean, fast) or 'knn' (KNNImputer — fills each missing
        value from its nearest neighbors' values on the other features, fit on train only).
    """

    def __init__(self, numeric_scaler: str = "rankgauss", numeric_imputer: str = "mean",
                 max_onehot_card: int = 25, **scaler_kwargs):
        self.numeric_scaler = numeric_scaler
        self.numeric_imputer = numeric_imputer
        self.max_onehot_card = max_onehot_card
        self.scaler_kwargs = scaler_kwargs

    def fit(self, df: pd.DataFrame) -> 'TabularPreprocessor':
        self.num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols_all = [c for c in df.columns if c not in self.num_cols]
        self.cat_cols = [c for c in cat_cols_all if df[c].nunique() <= self.max_onehot_card]
        self.dropped_cat_cols_ = [c for c in cat_cols_all if c not in self.cat_cols]

        self.cat_modes_ = {c: df[c].mode().iloc[0] for c in self.cat_cols}

        self.imputer_ = None
        self.num_means_ = None
        self.scaler = None
        if self.num_cols:
            Xnum_raw = df[self.num_cols].values.astype(np.float64)
            if self.numeric_imputer == "knn":
                self.imputer_ = KNNImputer(n_neighbors=self.scaler_kwargs.get('n_neighbors', 5))
                Xnum = self.imputer_.fit_transform(Xnum_raw)
            else:
                self.num_means_ = df[self.num_cols].mean()
                Xnum = df[self.num_cols].fillna(self.num_means_).values.astype(np.float64)

            if self.numeric_scaler != "none":
                self.scaler = Preprocessor(self.numeric_scaler, **self.scaler_kwargs)
                self.scaler.fit(Xnum)

        if self.cat_cols:
            cat_df = df[self.cat_cols].astype(str)
            for c in self.cat_cols:
                cat_df[c] = cat_df[c].fillna(str(self.cat_modes_[c]))
            self.dummy_columns_ = pd.get_dummies(cat_df).columns
        else:
            self.dummy_columns_ = []
        return self

    def _transform_numeric(self, df: pd.DataFrame) -> np.ndarray:
        if not self.num_cols:
            return np.zeros((len(df), 0), dtype=np.float32)
        Xnum_raw = df[self.num_cols].values.astype(np.float64)
        if self.numeric_imputer == "knn":
            Xnum = self.imputer_.transform(Xnum_raw)
        else:
            Xnum = df[self.num_cols].fillna(self.num_means_).values.astype(np.float64)
        if self.scaler is not None:
            Xnum = self.scaler.transform(Xnum)
        return np.asarray(Xnum, dtype=np.float32)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        Xnum = self._transform_numeric(df)
        if self.cat_cols:
            cat_df = df[self.cat_cols].astype(str)
            for c in self.cat_cols:
                cat_df[c] = cat_df[c].fillna(str(self.cat_modes_[c]))
            dummies = pd.get_dummies(cat_df).reindex(columns=self.dummy_columns_, fill_value=0)
            Xcat = dummies.values.astype(np.float32)
            return np.concatenate([Xnum, Xcat], axis=1)
        return Xnum

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)


# Quick test function
if __name__ == "__main__":
    from sklearn.datasets import fetch_california_housing
    
    # Load sample data
    data = fetch_california_housing()
    X, y = data.data, data.target
    
    print("Comparing preprocessing methods...")
    results = compare_preprocessors(X)
    print(results)
    
    # Plot comparison
    plot_all_preprocessors(X, feature_idx=0)