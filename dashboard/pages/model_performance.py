"""Model Performance page for RetailSync AI."""

from __future__ import annotations

import logging

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.ui import (
    render_alert,
    render_data_table,
    render_empty_state,
    render_kpi_row,
    render_section_header,
)
from src.explainability import global_importance_chart

logger = logging.getLogger(__name__)


def render_model_performance_page(data: dict, models: dict) -> None:
    """Render the model performance page."""
    st.markdown(
        """
        <div class="brand-header">Model Performance</div>
        <div class="brand-subtitle">Demand forecasting model evaluation, metrics, and comparison</div>
        """,
        unsafe_allow_html=True,
    )

    features = data.get("features")
    model_pkg = models.get("demand_forecaster")

    # Model Status
    render_section_header("Model Status", subtitle="Loaded model information")

    if model_pkg and isinstance(model_pkg, dict) and "model" in model_pkg:
        model_name = model_pkg.get("model_name", type(model_pkg["model"]).__name__)
        feature_cols = model_pkg.get("feature_cols", [])
        render_kpi_row(
            [
                {
                    "label": "Selected Model",
                    "value": model_name,
                    "icon": "🧠",
                    "help_text": "Name of the loaded forecasting model.",
                },
                {
                    "label": "Feature Count",
                    "value": str(len(feature_cols)),
                    "icon": "📊",
                    "help_text": "Number of features used by the model.",
                },
            ],
            columns=2,
        )
    else:
        render_alert(
            message="No demand forecaster model is loaded. Train and save models/demand_forecaster.pkl to see performance metrics.",
            severity="warning",
            title="Model Not Loaded",
        )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # Accuracy Metrics
    render_section_header("Accuracy Metrics", subtitle="Backtested performance on held-out data")

    if features is not None and not features.empty and model_pkg:
        from src.business_metrics.kpi import compute_forecast_accuracy

        try:
            with st.spinner("Computing forecast accuracy..."):
                forecast_accuracy = compute_forecast_accuracy(features, model_pkg)
        except Exception as exc:
            logger.error("Accuracy computation failed: %s", exc)
            forecast_accuracy = {}

        overall = forecast_accuracy.get("overall", {})
        if overall:
            render_kpi_row(
                [
                    {
                        "label": "MAE",
                        "value": f"{overall.get('mae', 0):.4f}",
                        "icon": "📏",
                        "help_text": "Mean Absolute Error. Lower is better.",
                    },
                    {
                        "label": "RMSE",
                        "value": f"{overall.get('rmse', 0):.4f}",
                        "icon": "📐",
                        "help_text": "Root Mean Squared Error. Lower is better.",
                    },
                    {
                        "label": "sMAPE",
                        "value": f"{overall.get('smape', 0):.2f}%",
                        "icon": "🎯",
                        "help_text": "Symmetric Mean Absolute Percentage Error. Lower is better.",
                    },
                    {
                        "label": "Bias",
                        "value": f"{overall.get('bias', 0):.4f}",
                        "icon": "⚖️",
                        "help_text": "Average prediction bias (positive = over-forecasting).",
                    },
                ],
                columns=4,
            )

            st.markdown(f"**Test Rows:** {overall.get('test_rows', 'N/A')}")

            # By product
            by_product = forecast_accuracy.get("by_product")
            if by_product is not None and not by_product.empty:
                render_section_header("Accuracy by Product", subtitle="Top 10 products by MAE (worst first)")
                display = by_product.head(10)[
                    ["product_id", "mae", "rmse", "smape", "samples"]
                ].copy()
                display.columns = ["Product", "MAE", "RMSE", "sMAPE %", "Samples"]
                render_data_table(
                    display,
                    title="Product Accuracy",
                )

                fig = px.bar(
                    display,
                    x="Product",
                    y="MAE",
                    title="MAE by Product (Top 10)",
                    template="plotly_dark",
                    color="MAE",
                    height=400,
                )
                st.plotly_chart(fig, width="stretch")

            # By store
            by_store = forecast_accuracy.get("by_store")
            if by_store is not None and not by_store.empty:
                render_section_header("Accuracy by Store", subtitle="Top 10 stores by MAE (worst first)")
                display = by_store.head(10)[
                    ["store_id", "mae", "rmse", "smape", "samples"]
                ].copy()
                display.columns = ["Store", "MAE", "RMSE", "sMAPE %", "Samples"]
                render_data_table(
                    display,
                    title="Store Accuracy",
                )

                fig = px.bar(
                    display,
                    x="Store",
                    y="MAE",
                    title="MAE by Store (Top 10)",
                    template="plotly_dark",
                    color="MAE",
                    height=400,
                )
                st.plotly_chart(fig, width="stretch")

            # By category
            by_category = forecast_accuracy.get("by_category")
            if by_category is not None and not by_category.empty:
                render_section_header("Accuracy by Category", subtitle="MAE and sMAPE by product category")
                fig = px.bar(
                    by_category,
                    x="category",
                    y="mae",
                    color="smape",
                    title="Forecast MAE by Category",
                    template="plotly_dark",
                    height=400,
                )
                fig.update_layout(
                    xaxis_title="Category",
                    yaxis_title="MAE",
                )
                st.plotly_chart(fig, width="stretch")

                render_data_table(
                    by_category,
                    title="Category Accuracy",
                    download_label="Download Category Accuracy (CSV)",
                    download_filename="accuracy_by_category.csv",
                )
        else:
            render_alert(
                message="Could not compute accuracy metrics. Ensure the model package is valid and feature data is available.",
                severity="warning",
                title="No Metrics Available",
            )
    else:
        render_alert(
            message="No feature data or model available. Load data and train the model to see performance metrics.",
            severity="info",
            title="Insufficient Data",
        )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # Model Comparison
    render_section_header("Model Selection", subtitle="Best model selected by validation MAE")

    if model_pkg and isinstance(model_pkg, dict) and "model" in model_pkg:
        metrics = model_pkg.get("metrics", {})
        model_name = model_pkg.get("model_name", type(model_pkg["model"]).__name__)

        comparison_data = pd.DataFrame(
            {
                "Model": [model_name],
                "Test MAE": [round(metrics.get("mae", 0), 4)],
                "Test RMSE": [round(metrics.get("rmse", 0), 4)],
                "Test R²": [round(metrics.get("r2", 0), 4)],
                "Test sMAPE (%)": [round(metrics.get("smape", 0), 2)],
            }
        )

        render_alert(
            message=(
                f"**{model_name}** was selected as the best model by validation MAE "
                f"and achieves the following holdout test metrics:"
            ),
            severity="success",
            title="Selected Model",
        )

        render_data_table(
            comparison_data,
            title="Selected Model Test Metrics",
            download_label="Download Model Metrics (CSV)",
            download_filename="model_metrics.csv",
        )
    else:
        render_alert(
            message="No model loaded. Train the demand forecaster to see model selection results.",
            severity="info",
            title="Model Required",
        )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # Feature Importance
    render_section_header("Feature Importance", subtitle="Top features driving model predictions")

    if model_pkg and isinstance(model_pkg, dict) and "model" in model_pkg:
        from dashboard.explainability_page import get_engine

        model_name = model_pkg.get("model_name", type(model_pkg["model"]).__name__)
        try:
            engine = get_engine(model_name, 100)
            if features is not None and not features.empty:
                engine.set_background(features)
                with st.spinner("Computing feature importance..."):
                    global_exp = engine.global_importance(sample_size=150)

                st.plotly_chart(
                    global_importance_chart(global_exp, top_n=20),
                    width="stretch",
                )

                top = global_exp.top(10)
                render_data_table(
                    top,
                    title="Top 10 Features by Importance",
                    download_label="Download Feature Importance (CSV)",
                    download_filename="feature_importance.csv",
                )
            else:
                render_empty_state("No Feature Data", "Feature data is not available for SHAP computation.")
        except Exception as exc:
            logger.warning("Could not compute feature importance: %s", exc)
            render_alert(
                message="SHAP-based feature importance is not available for this model type.",
                severity="info",
                title="Feature Importance Unavailable",
            )
    else:
        render_alert(
            message="Load a trained demand forecaster model to see feature importance.",
            severity="info",
            title="Model Required",
        )
