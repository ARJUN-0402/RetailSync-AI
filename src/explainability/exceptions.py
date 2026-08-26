"""Custom exceptions for the RetailSync AI explainability layer.

These are intentionally lightweight so that callers (the dashboard, tests,
and notebooks) can catch explainability-specific failures without leaking SHAP
implementation details.
"""

from __future__ import annotations


class ExplainabilityError(Exception):
    """Base class for all explainability errors."""


class ModelLoadError(ExplainabilityError):
    """Raised when a model package cannot be located or deserialised."""


class UnsupportedModelError(ExplainabilityError):
    """Raised when SHAP cannot explain the supplied model type."""


class MissingFeaturesError(ExplainabilityError):
    """Raised when the input data is missing required model features."""


class FeatureContributionError(ExplainabilityError):
    """Raised when SHAP fails to compute values at runtime (e.g. bad input)."""
