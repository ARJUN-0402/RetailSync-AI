"""
Tests for the RetailSync AI SHAP explainability layer.

Run with:
    pytest tests/test_explainability.py -v
"""

import os

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from src.explainability import (
    ExplainabilityEngine,
    GlobalExplanation,
    LocalExplanation,
    MissingFeaturesError,
    UnsupportedModelError,
    build_explanation,
    detect_explainer_kind,
    load_explainability_engine,
)
from src.explainability.exceptions import ModelLoadError

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def _load_package():
    path = os.path.join(MODELS_DIR, "demand_forecaster.pkl")
    assert os.path.exists(path), f"Missing {path}"
    return joblib.load(path)


def _load_features():
    path = os.path.join(PROCESSED_DIR, "features_daily.csv")
    assert os.path.exists(path), f"Missing {path}"
    return pd.read_csv(path, parse_dates=["date"])


def _engine():
    return load_explainability_engine(
        model_path=os.path.join(MODELS_DIR, "demand_forecaster.pkl"),
        features_path=os.path.join(PROCESSED_DIR, "features_daily.csv"),
    )


# ---------------------------------------------------------------------------
# Explainer creation / model detection
# ---------------------------------------------------------------------------
def test_detect_explainer_tree_model():
    pkg = _load_package()
    assert detect_explainer_kind(pkg["model"]) == "tree"


def test_detect_explainer_xgboost():
    model = XGBRegressor(n_estimators=5)
    model.fit(np.random.randn(20, 4), np.random.randn(20))
    assert detect_explainer_kind(model) == "tree"


def test_detect_explainer_linear():
    model = LinearRegression()
    model.fit(np.random.randn(20, 4), np.random.randn(20))
    assert detect_explainer_kind(model) == "linear"


def test_unsupported_model_raises():
    class TotallyCustomModel:
        def predict(self, X):
            return np.zeros(len(X))

    # A class name SHAP cannot explain must raise UnsupportedModelError.
    import pytest

    with pytest.raises(UnsupportedModelError):
        detect_explainer_kind(TotallyCustomModel())


def test_engine_from_real_model_is_supported():
    engine = _engine()
    assert engine.is_supported()
    assert engine.explainer_kind == "tree"
    assert np.isfinite(engine.expected_value)


def test_engine_rejects_bad_package():
    import pytest

    with pytest.raises(ModelLoadError):
        ExplainabilityEngine({"feature_cols": []})  # no "model" key


# ---------------------------------------------------------------------------
# Feature contribution extraction (local)
# ---------------------------------------------------------------------------
def test_explain_instance_returns_contributions():
    engine = _engine()
    features = _load_features()
    row = features[engine.feature_cols].iloc[[5000]]
    expl: LocalExplanation = engine.explain_instance(row)

    assert len(expl.shap_values) == len(engine.feature_cols)
    # SHAP additivity: expected_value + sum(shap) ~= prediction.
    recon = expl.expected_value + expl.net_effect
    assert abs(recon - expl.predicted_value) < max(1.0, abs(expl.predicted_value) * 0.05 + 1e-3)
    assert expl.direction in {"increase", "decrease"}


def test_local_top_positive_negative():
    engine = _engine()
    features = _load_features()
    row = features[engine.feature_cols].iloc[[5000]]
    expl = engine.explain_instance(row)

    pos = expl.top_positive()
    neg = expl.top_negative()
    if pos:
        assert all(c.shap_value > 0 for c in pos)
    if neg:
        assert all(c.shap_value < 0 for c in neg)
    # contributions sorted by magnitude descending
    contribs = expl.contributions()
    mags = [c.magnitude for c in contribs]
    assert mags == sorted(mags, reverse=True)


def test_missing_features_raises():
    import pytest

    engine = _engine()
    bad = pd.DataFrame({"demand_lag_1d": [1.0]})  # missing most features
    with pytest.raises(MissingFeaturesError):
        engine.explain_instance(bad)


def test_explain_instance_dict_input():
    engine = _engine()
    features = _load_features()
    row = features[engine.feature_cols].iloc[0].to_dict()
    expl = engine.explain_instance(row)
    assert isinstance(expl, LocalExplanation)
    assert len(expl.shap_values) == len(engine.feature_cols)


# ---------------------------------------------------------------------------
# Global importance
# ---------------------------------------------------------------------------
def test_global_importance_ranking():
    engine = _engine()
    g: GlobalExplanation = engine.global_importance(sample_size=120)

    assert len(g.mean_abs_shap) == len(engine.feature_cols)
    ranking = g.ranking
    assert (ranking["rank"].tolist() == list(range(1, len(ranking) + 1)))
    assert ranking["mean_abs_shap"].is_monotonic_decreasing
    # Top feature should have the largest mean |SHAP|.
    assert ranking.iloc[0]["mean_abs_shap"] == ranking["mean_abs_shap"].max()


def test_global_importance_cached_and_reproducible():
    engine = _engine()
    g1 = engine.global_importance(sample_size=100)
    g2 = engine.global_importance(sample_size=100)
    np.testing.assert_allclose(g1.mean_abs_shap, g2.mean_abs_shap, rtol=1e-9)


# ---------------------------------------------------------------------------
# Natural-language explanation
# ---------------------------------------------------------------------------
def test_explanation_generation_is_dynamic():
    engine = _engine()
    features = _load_features()
    row = features[engine.feature_cols].iloc[[5000]]
    expl = engine.explain_instance(row)
    text = build_explanation(expl, context={"product_id": "P001", "store_id": "S001"})

    assert isinstance(text, str)
    assert len(text) > 0
    # The explanation must reference the actual direction, not a fixed string.
    assert expl.direction in text.lower() or "expected to" in text.lower()
    # It must mention at least one real feature (dynamic labels, not hard-coded).
    from src.explainability.feature_descriptions import feature_label

    assert any(feature_label(feat) in text for feat in engine.feature_cols)


def test_explanation_high_vs_low_differs():
    engine = _engine()
    features = _load_features()
    high_row = features[engine.feature_cols].iloc[[5000]]
    low_idx = int(np.argmin(engine.model.predict(features[engine.feature_cols].fillna(0).values)))
    low_row = features[engine.feature_cols].iloc[[low_idx]]
    t_high = build_explanation(engine.explain_instance(high_row))
    t_low = build_explanation(engine.explain_instance(low_row))
    # Different instances should generally yield different explanations.
    assert t_high != t_low


# ---------------------------------------------------------------------------
# Linear-model path (sanity)
# ---------------------------------------------------------------------------
def test_linear_explainer_works():
    features = _load_features()
    cols = _load_package()["feature_cols"][:10]
    Xdf = features[cols].fillna(0).head(200)
    yser = features["target_demand_1d"].fillna(0).head(200).values
    lin = LinearRegression()
    lin.fit(Xdf.values, yser)
    engine = ExplainabilityEngine(
        {"model": lin, "feature_cols": cols, "model_name": "LinearRegression"},
        background_data=Xdf,
    )
    assert engine.explainer_kind == "linear"
    expl = engine.explain_instance(Xdf.iloc[[0]])
    assert isinstance(expl, LocalExplanation)
