"""SHAP-based explainability engine for the RetailSync AI demand forecaster.

This module is intentionally decoupled from model *training*. It consumes a
trained model package (the same dict saved by ``src/forecasting``) and computes
global and local SHAP explanations. It never retrains the model.

Supported models (gracefully detected):
    * Tree-based regressors -> ``shap.TreeExplainer``
      (RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting,
       XGBoost, LightGBM)
    * Linear regressors -> ``shap.LinearExplainer``
      (LinearRegression, Ridge, Lasso, SGDRegressor, ...)
    * Everything else -> raises ``UnsupportedModelError`` (no crash).

Performance controls:
    * A background sample is cached once and reused for every SHAP call.
    * Global explanations are computed on a bounded, sampled subset of rows.
    * Local explanations are computed for a single instance only.

All errors are surfaced through the exceptions in ``.exceptions`` so callers
can decide how to degrade gracefully (e.g. hide a dashboard panel).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .exceptions import (
    FeatureContributionError,
    MissingFeaturesError,
    ModelLoadError,
    UnsupportedModelError,
)

try:
    import shap

    _SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover - environment without shap
    shap = None  # type: ignore
    _SHAP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class Contribution:
    """A single (feature, shap_value, feature_value) triple."""

    feature: str
    shap_value: float
    feature_value: float

    @property
    def magnitude(self) -> float:
        return abs(self.shap_value)


@dataclass
class LocalExplanation:
    feature_names: list[str]
    shap_values: np.ndarray
    expected_value: float
    predicted_value: float
    feature_values: dict[str, float] = field(default_factory=dict)
    feature_medians: dict[str, float] = field(default_factory=dict)

    @property
    def direction(self) -> str:
        """``increase`` when the prediction sits above the model baseline."""
        return "increase" if self.predicted_value >= self.expected_value else "decrease"

    @property
    def net_effect(self) -> float:
        return float(np.sum(self.shap_values))

    def contributions(self, top_n: int | None = None) -> list[Contribution]:
        items = [
            Contribution(f, float(s), self.feature_values.get(f, float("nan")))
            for f, s in zip(self.feature_names, self.shap_values)
        ]
        items.sort(key=lambda c: c.magnitude, reverse=True)
        return items[:top_n] if top_n else items

    def top_positive(self, top_n: int | None = None) -> list[Contribution]:
        items = [
            Contribution(f, float(s), self.feature_values.get(f, float("nan")))
            for f, s in zip(self.feature_names, self.shap_values)
            if s > 0
        ]
        items.sort(key=lambda c: c.shap_value, reverse=True)
        return items[:top_n] if top_n else items

    def top_negative(self, top_n: int | None = None) -> list[Contribution]:
        items = [
            Contribution(f, float(s), self.feature_values.get(f, float("nan")))
            for f, s in zip(self.feature_names, self.shap_values)
            if s < 0
        ]
        items.sort(key=lambda c: c.shap_value)
        return items[:top_n] if top_n else items


@dataclass
class GlobalExplanation:
    feature_names: list[str]
    mean_abs_shap: np.ndarray
    sample_size: int

    @property
    def ranking(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "mean_abs_shap": np.asarray(self.mean_abs_shap, dtype=float),
            }
        )
        df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", np.arange(1, len(df) + 1))
        return df

    def top(self, n: int = 15) -> pd.DataFrame:
        return self.ranking.head(n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Explainer factory
# ---------------------------------------------------------------------------
def detect_explainer_kind(model: Any) -> str:
    """Return ``'tree'``, ``'linear'`` or ``'kernel'`` for a supported model.

    Raises ``UnsupportedModelError`` when SHAP cannot explain the model.
    """
    if not _SHAP_AVAILABLE:
        raise UnsupportedModelError(
            "The shap library is not installed, so model explanations are unavailable."
        )

    cls_name = type(model).__name__

    tree_models = {
        "RandomForestRegressor",
        "RandomForestClassifier",
        "ExtraTreesRegressor",
        "ExtraTreesClassifier",
        "GradientBoostingRegressor",
        "GradientBoostingClassifier",
        "HistGradientBoostingRegressor",
        "HistGradientBoostingClassifier",
        "XGBRegressor",
        "XGBClassifier",
        "LGBMRegressor",
        "LGBMClassifier",
    }
    linear_models = {
        "LinearRegression",
        "Ridge",
        "RidgeCV",
        "Lasso",
        "LassoCV",
        "ElasticNet",
        "SGDRegressor",
        "LinearSVR",
        "LogisticRegression",
        "SGDClassifier",
    }

    if cls_name in tree_models:
        return "tree"
    if cls_name in linear_models:
        return "linear"
    if "SVR" in cls_name or cls_name in {"SVC", "KNeighborsRegressor", "DecisionTreeRegressor"}:
        # Kernel/instance explainers are supported but expensive; allow them.
        return "kernel"
    raise UnsupportedModelError(
        f"Model type '{cls_name}' is not supported by the explainability layer."
    )


def build_explainer(model: Any, background_sample: pd.DataFrame | None = None):
    """Construct a SHAP explainer appropriate for ``model``.

    ``background_sample`` is required for linear/kernel explainers (for the
    interventional feature distribution) and optional for tree models.
    """
    kind = detect_explainer_kind(model)

    if kind == "tree":
        try:
            return shap.TreeExplainer(model), kind
        except Exception as exc:  # pragma: no cover
            raise UnsupportedModelError(f"Could not build a TreeExplainer: {exc}") from exc

    if background_sample is None or len(background_sample) == 0:
        raise MissingFeaturesError(
            "A background sample is required to explain this model type."
        )

    if kind == "linear":
        try:
            masker = shap.maskers.Independent(background_sample)
            return shap.LinearExplainer(model, masker), kind
        except Exception as exc:  # pragma: no cover
            raise UnsupportedModelError(f"Could not build a LinearExplainer: {exc}") from exc

    # kernel
    try:
        return shap.KernelExplainer(model.predict, background_sample), kind
    except Exception as exc:  # pragma: no cover
        raise UnsupportedModelError(f"Could not build a KernelExplainer: {exc}") from exc


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class ExplainabilityEngine:
    """Compute global and local SHAP explanations for a trained model package."""

    def __init__(
        self,
        model_package: dict,
        background_data: pd.DataFrame | None = None,
        background_sample_size: int = 100,
        global_sample_size: int = 200,
        random_state: int = 42,
    ) -> None:
        if not isinstance(model_package, dict) or "model" not in model_package:
            raise ModelLoadError("model_package must be a dict containing a 'model' key.")
        if "feature_cols" not in model_package:
            raise ModelLoadError("model_package is missing 'feature_cols'.")

        self.model = model_package["model"]
        self.feature_cols: list[str] = list(model_package["feature_cols"])
        self.model_name: str = model_package.get("model_name", type(self.model).__name__)
        self.model_package = model_package
        self.background_sample_size = background_sample_size
        self.global_sample_size = global_sample_size
        self.random_state = random_state

        self._rng = np.random.default_rng(random_state)
        self._explainer = None
        self._explainer_kind: str | None = None
        self._background_sample: pd.DataFrame | None = None
        self._global_cache: dict[tuple[int, int], np.ndarray] = {}

        if background_data is not None:
            self.set_background(background_data)

    # -- construction helpers ------------------------------------------------
    @classmethod
    def from_model_path(
        cls,
        model_path: str = "models/demand_forecaster.pkl",
        features_path: str = "data/processed/features_daily.csv",
        **kwargs,
    ) -> ExplainabilityEngine:
        if not os.path.exists(model_path):
            raise ModelLoadError(f"Model file not found: {model_path}")
        try:
            pkg = joblib.load(model_path)
        except Exception as exc:
            raise ModelLoadError(f"Failed to load model package: {exc}") from exc

        background = None
        if os.path.exists(features_path):
            background = pd.read_csv(features_path, parse_dates=["date"])

        return cls(pkg, background_data=background, **kwargs)

    def set_background(self, background_data: pd.DataFrame) -> None:
        missing = [c for c in self.feature_cols if c not in background_data.columns]
        if missing:
            raise MissingFeaturesError(
                f"Background data is missing {len(missing)} model features "
                f"(e.g. {missing[:3]})."
            )
        self.background_data = background_data[self.feature_cols].copy()
        self._background_sample = None  # invalidate cached sample
        self._global_cache.clear()

    # -- sampling / caching ------------------------------------------------
    def _get_background_sample(self) -> pd.DataFrame:
        if self._background_sample is None:
            df = self.background_data
            if df is None or len(df) == 0:
                raise MissingFeaturesError("No background data available for SHAP.")
            n = min(self.background_sample_size, len(df))
            idx = self._rng.choice(len(df), size=n, replace=False)
            self._background_sample = df.iloc[idx].reset_index(drop=True)
        return self._background_sample

    def _get_explainer(self):
        if self._explainer is None:
            sample = self._get_background_sample()
            self._explainer, self._explainer_kind = build_explainer(self.model, sample)
        return self._explainer

    @property
    def explainer_kind(self) -> str:
        if self._explainer_kind is None:
            # Detect without materialising a full explainer.
            self._explainer_kind = detect_explainer_kind(self.model)
        return self._explainer_kind

    @property
    def expected_value(self) -> float:
        explainer = self._get_explainer()
        ev = explainer.expected_value
        if isinstance(ev, (np.ndarray, list)):
            return float(np.asarray(ev).flatten()[0])
        return float(ev)

    # -- global -------------------------------------------------------------
    def global_importance(self, sample_size: int | None = None) -> GlobalExplanation:
        """Mean |SHAP| per feature across a bounded sample of the background."""
        size = sample_size or self.global_sample_size
        cache_key = (size, id(self.background_data))

        if cache_key not in self._global_cache:
            df = self.background_data
            if df is None or len(df) == 0:
                raise MissingFeaturesError("No background data available for global SHAP.")
            n = min(size, len(df))
            idx = self._rng.choice(len(df), size=n, replace=False)
            sample = df.iloc[idx].reset_index(drop=True)

            try:
                explainer = self._get_explainer()
                values = explainer.shap_values(sample.values)
            except Exception as exc:
                raise FeatureContributionError(
                    f"SHAP failed while computing global values: {exc}"
                ) from exc

            values = self._normalise_shap_values(values, n)
            self._global_cache[cache_key] = np.abs(values).mean(axis=0)

        mean_abs = np.asarray(self._global_cache[cache_key], dtype=float)
        return GlobalExplanation(
            feature_names=self.feature_cols,
            mean_abs_shap=mean_abs,
            sample_size=min(size, len(self.background_data) if self.background_data is not None else 0),
        )

    # -- local --------------------------------------------------------------
    def explain_instance(
        self,
        instance: pd.DataFrame | dict,
        context_features: pd.DataFrame | None = None,
    ) -> LocalExplanation:
        """Compute SHAP values for a single row / instance."""
        if isinstance(instance, dict):
            row = pd.DataFrame([instance])
        else:
            row = instance.iloc[[0]].reset_index(drop=True) if len(instance) > 1 else instance.copy()

        missing = [c for c in self.feature_cols if c not in row.columns]
        if missing:
            raise MissingFeaturesError(
                f"Instance is missing {len(missing)} model features (e.g. {missing[:3]})."
            )

        x = row[self.feature_cols]
        x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        try:
            explainer = self._get_explainer()
            values = explainer.shap_values(x.values)
        except Exception as exc:
            raise FeatureContributionError(
                f"SHAP failed while explaining this instance: {exc}"
            ) from exc

        values = self._normalise_shap_values(values, 1)
        shap_row = values[0]

        try:
            predicted = float(self.model.predict(x)[0])
        except (ValueError, TypeError):
            predicted = float(self.expected_value + np.sum(shap_row))

        base = self.expected_value
        feature_values = {c: float(x.iloc[0][c]) for c in self.feature_cols}
        medians = self._feature_medians()

        return LocalExplanation(
            feature_names=self.feature_cols,
            shap_values=shap_row,
            expected_value=base,
            predicted_value=predicted,
            feature_values=feature_values,
            feature_medians=medians,
        )

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _normalise_shap_values(values: Any, n_samples: int) -> np.ndarray:
        """Coerce SHAP output into a 2D (n_samples, n_features) float array."""
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 3:
            # Multi-output (e.g. multiclass) -> keep first output.
            arr = arr[:, :, 0]
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[0] != n_samples and arr.shape[1] == n_samples:
            arr = arr.T
        return arr

    def _feature_medians(self) -> dict[str, float]:
        if self.background_data is None:
            return {}
        return {
            c: float(self.background_data[c].median()) for c in self.feature_cols
        }

    def is_supported(self) -> bool:
        try:
            detect_explainer_kind(self.model)
            return True
        except UnsupportedModelError:
            return False


def load_explainability_engine(
    model_path: str = "models/demand_forecaster.pkl",
    features_path: str = "data/processed/features_daily.csv",
    **kwargs,
) -> ExplainabilityEngine:
    """Convenience entry point used by the dashboard and tests."""
    return ExplainabilityEngine.from_model_path(model_path, features_path, **kwargs)
