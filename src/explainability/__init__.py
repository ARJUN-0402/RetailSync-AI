"""Model explainability layer for RetailSync AI.

A modular, training-agnostic SHAP wrapper that provides global and local
explanations for the demand forecaster, plus natural-language and Plotly
visualisation helpers. See ``docs/model_explainability.md`` for usage.
"""

from __future__ import annotations

from .exceptions import (
    ExplainabilityError,
    FeatureContributionError,
    MissingFeaturesError,
    ModelLoadError,
    UnsupportedModelError,
)
from .explanation import build_explanation, explain_forecast
from .shap_explainer import (
    Contribution,
    ExplainabilityEngine,
    GlobalExplanation,
    LocalExplanation,
    detect_explainer_kind,
    load_explainability_engine,
)
from .visualizations import (
    driver_bars_chart,
    global_importance_chart,
    local_waterfall_chart,
    shap_summary_chart,
)

__all__ = [
    "Contribution",
    "ExplainabilityEngine",
    "ExplainabilityError",
    "FeatureContributionError",
    "GlobalExplanation",
    "LocalExplanation",
    "MissingFeaturesError",
    "ModelLoadError",
    "UnsupportedModelError",
    "build_explanation",
    "detect_explainer_kind",
    "driver_bars_chart",
    "explain_forecast",
    "global_importance_chart",
    "load_explainability_engine",
    "local_waterfall_chart",
    "shap_summary_chart",
]
