"""Streamlit UI for the Model Explainability layer.

This module keeps all explainability dashboard code in one place so it stays
decoupled from the rest of the app. It is imported by ``dashboard/app.py`` and
exposes two entry points:

* :func:`render_explainability_page` - the dedicated "Model Explainability" page.
* :func:`render_why_forecast`    - the lightweight "Why this forecast?" panel that
  is embedded inside the existing Demand Forecast page.

Both degrade gracefully when SHAP is unavailable or the model is unsupported.
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from dashboard.components.ui import (
    COLORS,
    inject_global_css,
    render_alert,
    render_kpi_card,
    render_section_header,
)
from src.explainability import (
    ExplainabilityEngine,
    UnsupportedModelError,
    build_explanation,
    driver_bars_chart,
    global_importance_chart,
    local_waterfall_chart,
    shap_summary_chart,
)

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner="Loading explainability engine...")
def get_engine(model_name: str, sample_size: int) -> ExplainabilityEngine:
    """Build and cache a SHAP engine for the demand forecaster."""

    import joblib

    model_path = "models/demand_forecaster.pkl"
    pkg = joblib.load(model_path)
    engine = ExplainabilityEngine(pkg, background_sample_size=sample_size)
    return engine


def _model_package(models: dict) -> dict | None:
    pkg = models.get("demand_forecaster")
    if isinstance(pkg, dict) and "model" in pkg:
        return pkg
    return None


def render_explainability_page(models: dict, data: dict) -> None:
    inject_global_css()

    st.markdown(
        f"""
        <div class="brand-header">Model Explainability</div>
        <div class="brand-subtitle">Understand what drives the demand forecasts (SHAP)</div>
        """,
        unsafe_allow_html=True,
    )

    pkg = _model_package(models)
    if pkg is None:
        render_alert(
            message="The demand forecaster model is not available, so explanations cannot be computed. "
            "Run the forecasting pipeline to train and save `models/demand_forecaster.pkl`.",
            severity="warning",
            title="Model Not Available",
        )
        return

    features = data.get("features")
    if features is None or len(features) == 0:
        render_alert(
            message="Feature data is missing; cannot build the SHAP background set.",
            severity="error",
            title="Missing Data",
        )
        return

    model_name = pkg.get("model_name", type(pkg["model"]).__name__)
    col_a, col_b = st.columns(2)
    with col_a:
        render_kpi_card(
            label="Selected Model",
            value=model_name,
            icon="🧠",
            help_text="Name of the loaded forecasting model.",
        )
    with col_b:
        try:
            engine = get_engine(model_name, 100)
            engine.set_background(features)
            kind = engine.explainer_kind
            render_kpi_card(
                label="SHAP Explainer",
                value=kind.upper(),
                icon="🔍",
                help_text="Type of SHAP explainer being used.",
            )
        except UnsupportedModelError as exc:
            render_alert(
                message=f"Model not supported by SHAP: {exc}",
                severity="error",
                title="Unsupported Model",
            )
            return
        except Exception as exc:
            render_alert(
                message=f"Could not initialise explainer: {exc}",
                severity="error",
                title="Explainer Error",
            )
            return

    render_alert(
        message="SHAP attributes a prediction to each input feature. "
        "**Global** views show which features matter across the dataset; "
        "**local** views show why a single forecast is what it is.",
        severity="info",
        title="How SHAP Works",
    )

    tab_global, tab_local = st.tabs(["🌍 Global Explainability", "🔎 Local Explainability"])

    with tab_global:
        _render_global(engine, features)

    with tab_local:
        _render_local(engine, features)


def _render_global(engine: ExplainabilityEngine, features: pd.DataFrame) -> None:
    sample_size = st.slider(
        "Global sample size (rows used for SHAP)",
        min_value=50,
        max_value=500,
        value=150,
        step=50,
        help="More rows = smoother importance but slower. Results are cached per size.",
    )

    try:
        global_exp = engine.global_importance(sample_size=sample_size)
    except Exception as exc:
        render_alert(
            message=f"Could not compute global importance: {exc}",
            severity="error",
            title="Computation Error",
        )
        return

    render_section_header("Global Feature Importance (mean |SHAP|)", subtitle="Features ranked by average absolute SHAP value")

    top = global_exp.top(10)
    c1, c2 = st.columns([3, 1])
    with c1:
        st.plotly_chart(global_importance_chart(global_exp, top_n=20), use_container_width=True)
    with c2:
        st.markdown("**Top drivers**")
        for _, r in top.iterrows():
            st.markdown(f"- `{r['feature']}` — {r['mean_abs_shap']:.3f}")
        st.caption(f"Computed over {global_exp.sample_size} background rows.")

    render_section_header("SHAP Summary (beeswarm)", subtitle="Distribution of SHAP values across features")
    with st.spinner("Computing SHAP summary..."):
        try:
            explainer = engine._get_explainer()
            vals = explainer.shap_values(engine._get_background_sample().values)
            fig = shap_summary_chart(global_exp, vals, engine._get_background_sample(), max_features=15, sample=200)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            render_alert(
                message=f"Could not render SHAP summary: {exc}",
                severity="warning",
                title="Rendering Error",
            )


def _render_local(engine: ExplainabilityEngine, features: pd.DataFrame) -> None:
    products = features["product_id"].unique().tolist()
    stores = features["store_id"].unique().tolist()
    selected_product = st.selectbox("Product", products, key="exp_product")
    selected_store = st.selectbox("Store", stores, key="exp_store")

    sub = features[
        (features["product_id"] == selected_product) & (features["store_id"] == selected_store)
    ].sort_values("date")
    if sub.empty:
        render_alert(
            message="No historical rows for this product-store combination.",
            severity="info",
            title="No Data",
        )
        return

    dates = sub["date"].dt.date.tolist()
    selected_date = st.selectbox("Date to explain", dates, index=len(dates) - 1, key="exp_date")
    row = sub[sub["date"].dt.date == selected_date]

    if st.button("Explain this forecast", key="exp_btn"):
        with st.spinner("Computing local SHAP values..."):
            try:
                local = engine.explain_instance(row[engine.feature_cols])
            except Exception as exc:
                render_alert(
                    message=f"Could not explain this instance: {exc}",
                    severity="error",
                    title="Explanation Error",
                )
                return

        render_section_header("Prediction Breakdown", subtitle="Baseline, prediction, and net feature effect")

        m1, m2, m3 = st.columns(3)
        with m1:
            render_kpi_card(
                label="Baseline (typical demand)",
                value=f"{local.expected_value:.2f}",
                icon="📊",
            )
        with m2:
            render_kpi_card(
                label="Predicted demand",
                value=f"{local.predicted_value:.2f}",
                icon="🔮",
            )
        with m3:
            render_kpi_card(
                label="Net feature effect",
                value=f"{local.net_effect:+.2f}",
                delta=f"{'increase' if local.direction == 'increase' else 'decrease'}",
                icon="⚡",
            )

        st.plotly_chart(local_waterfall_chart(local), use_container_width=True)

        render_section_header("Top Positive & Negative Drivers", subtitle="Features that increased or decreased the forecast")
        st.plotly_chart(driver_bars_chart(local), use_container_width=True)

        cpos, cneg = st.columns(2)
        with cpos:
            st.markdown("**Top positive drivers (increase demand)**")
            for c in local.top_positive(5):
                st.markdown(f"- `{c.feature}` → +{c.shap_value:.2f}")
        with cneg:
            st.markdown("**Top negative drivers (decrease demand)**")
            for c in local.top_negative(5):
                st.markdown(f"- `{c.feature}` → {c.shap_value:.2f}")

        render_section_header("Human-readable explanation", subtitle="Natural language interpretation of the forecast")
        text = build_explanation(
            local,
            context={"product_id": selected_product, "store_id": selected_store},
            top_n=3,
        )
        render_alert(message=text, severity="success", title="Explanation")


def render_why_forecast(models: dict, data: dict, product: str, store: str) -> None:
    """Lightweight explainability panel embedded in the Demand Forecast page."""
    pkg = _model_package(models)
    features = data.get("features")
    if pkg is None or features is None:
        return

    model_name = pkg.get("model_name", type(pkg["model"]).__name__)

    render_section_header("Why this forecast?", subtitle="Quick explanation of the current forecast")

    try:
        engine = get_engine(model_name, 100)
        engine.set_background(features)
    except UnsupportedModelError:
        render_alert(
            message="SHAP explanations are not available for this model type.",
            severity="info",
            title="Unsupported",
        )
        return
    except Exception:
        return

    sub = features[(features["product_id"] == product) & (features["store_id"] == store)].sort_values("date")
    if sub.empty:
        return
    latest = sub[engine.feature_cols].iloc[[-1]]

    try:
        local = engine.explain_instance(latest)
    except Exception:
        render_alert(
            message="Could not compute an explanation for this forecast.",
            severity="info",
            title="Unavailable",
        )
        return

    text = build_explanation(
        local, context={"product_id": product, "store_id": store}, top_n=3
    )
    render_alert(message=text, severity="info", title="Forecast Explanation")

    with st.expander("See the top drivers"):
        st.plotly_chart(driver_bars_chart(local), use_container_width=True)
