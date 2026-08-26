"""Shared model classes for RetailSync AI forecasting baselines."""

import numpy as np
import pandas as pd


class BaselineMeanPredictor:
    """A baseline predictor that returns the historical mean for every prediction.

    Implements the scikit-learn estimator interface (fit, predict,
    feature_importances_) so it can be saved alongside tree-based models
    and called uniformly by downstream code.
    """

    def __init__(self, n_features=0):
        self.train_mean_ = 0.0
        self._n_features = n_features

    def fit(self, X, y):
        self.train_mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.full(n, self.train_mean_)

    @property
    def feature_importances_(self):
        return np.zeros(self._n_features)


class NaivePredictor:
    """Predicts using the last known value (demand_lag_1d column)."""

    def __init__(self, lag_col="demand_lag_1d", n_features=0):
        self.lag_col = lag_col
        self._n_features = n_features

    def fit(self, X, y):
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.lag_col].values
        return np.zeros(X.shape[0])

    @property
    def feature_importances_(self):
        return np.zeros(self._n_features)


class MovingAveragePredictor:
    """Predicts using the 7-day rolling mean demand."""

    def __init__(self, rolling_col="demand_rolling_mean_7d", n_features=0):
        self.rolling_col = rolling_col
        self._n_features = n_features

    def fit(self, X, y):
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.rolling_col].values
        return np.zeros(X.shape[0])

    @property
    def feature_importances_(self):
        return np.zeros(self._n_features)
