"""Demand Forecast page for RetailSync AI."""

from __future__ import annotations

import logging

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components.ui import (
    COLORS,
    render_empty_state,
    render_kpi_row,
    render_section_header,
)
from dashboard.explainability_page import render_why_forecast

logger = logging.getLogger(__name__)


def render_demand_forecast_page(data: dict, models: dict) -> None:
    """Render the demand forecast page."""
    st.markdown(
        """
        <div class="brand-header">Demand Forecast</div>
        <div class="brand-subtitle">AI-powered demand predictions by product and store</div>
        """,
        unsafe_allow_html=True,
    )

    forecasts = data.get("forecasts")
    products = data.get("products")
    stores = data.get("stores")
    features = data.get("features")

    if forecasts is None or forecasts.empty:
        render_empty_state(
            title="No Forecast Data",
            message="Forecast data is not available. Please run the forecasting pipeline.",
        )
        return

    if products is None or products.empty or stores is None or stores.empty:
        render_empty_state(
            title="Missing Reference Data",
            message="Product or store master data is not available.",
        )
        return

    # Controls
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_product = st.selectbox(
            "Select Product",
            products["product_id"].tolist(),
            key="forecast_product",
        )
    with col2:
        selected_store = st.selectbox(
            "Select Store",
            stores["store_id"].tolist(),
            key="forecast_store",
        )
    with col3:
        forecast_horizon = st.slider(
            "Forecast Horizon (days)",
            min_value=1,
            max_value=14,
            value=14,
            key="forecast_horizon",
        )

    forecast_filtered = forecasts[
        (forecasts["product_id"] == selected_product)
        & (forecasts["store_id"] == selected_store)
    ].head(forecast_horizon)

    if forecast_filtered.empty:
        render_empty_state(
            title="No Forecast Available",
            message=f"No forecast data available for {selected_product} at {selected_store}.",
        )
        return

    st.markdown(f"### Forecast: {selected_product} at {selected_store}")

    # Forecast chart
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=forecast_filtered["date"],
            y=forecast_filtered["forecast_demand"],
            mode="lines+markers",
            name="Forecasted Demand",
            line={"color": COLORS["primary"], "width": 3},
            fill="tozeroy",
            fillcolor="rgba(59, 130, 246, 0.1)",
        )
    )
    fig.update_layout(
        title="Demand Forecast",
        xaxis_title="Date",
        yaxis_title="Quantity",
        template="plotly_dark",
        height=400,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Forecast stats
    total_demand = float(forecast_filtered["forecast_demand"].sum())
    total_revenue = float(forecast_filtered.get("forecast_revenue", pd.Series([0])).sum())
    avg_demand = float(forecast_filtered["forecast_demand"].mean())

    render_kpi_row(
        [
            {
                "label": "Total Forecasted Demand",
                "value": f"{total_demand:,.0f}",
                "icon": "📈",
                "help_text": f"Sum over {len(forecast_filtered)} days.",
            },
            {
                "label": "Total Forecasted Revenue",
                "value": f"${total_revenue:,.2f}",
                "icon": "💰",
                "help_text": "Estimated revenue based on forecasted demand.",
            },
            {
                "label": "Avg Daily Demand",
                "value": f"{avg_demand:,.1f}",
                "icon": "📊",
                "help_text": "Average daily forecasted demand.",
            },
            {
                "label": "Forecast Horizon",
                "value": f"{len(forecast_filtered)} days",
                "icon": "📅",
                "help_text": "Number of days in the forecast window.",
            },
        ],
        columns=4,
    )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # Historical vs Forecast
    render_section_header("Historical vs Forecast", subtitle="Recent demand history compared to forecast")

    if features is not None and not features.empty:
        hist = features[
            (features["product_id"] == selected_product)
            & (features["store_id"] == selected_store)
        ].tail(30)

        if not hist.empty and not forecast_filtered.empty:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=hist["date"],
                    y=hist["quantity_sold"],
                    mode="lines",
                    name="Historical Demand",
                    line={"color": COLORS["danger"], "width": 2},
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=forecast_filtered["date"],
                    y=forecast_filtered["forecast_demand"],
                    mode="lines+markers",
                    name="Forecasted Demand",
                    line={"color": COLORS["primary"], "dash": "dash", "width": 3},
                )
            )
            fig.update_layout(
                title="Historical vs Forecasted Demand",
                xaxis_title="Date",
                yaxis_title="Quantity",
                template="plotly_dark",
                height=400,
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            render_empty_state(
                title="No Historical Data",
                message="No historical data available for this product-store combination.",
            )
    else:
        render_empty_state(
            title="No Historical Data",
            message="Feature data is not available.",
        )

    # Why this forecast?
    if features is not None and not features.empty:
        render_why_forecast(models, data, selected_product, selected_store)

    # Download
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    render_section_header("Download Forecast Data")
    if not forecast_filtered.empty:
        csv = forecast_filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Forecast (CSV)",
            data=csv,
            file_name=f"forecast_{selected_product}_{selected_store}.csv",
            mime="text/csv",
        )
