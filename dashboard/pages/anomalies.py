"""Demand Anomalies page for RetailSync AI."""

from __future__ import annotations

import logging

import plotly.express as px
import streamlit as st

from dashboard.components.ui import (
    render_data_table,
    render_empty_state,
    render_kpi_row,
    render_section_header,
)

logger = logging.getLogger(__name__)


def render_anomalies_page(data: dict) -> None:
    """Render the demand anomalies page."""
    st.markdown(
        """
        <div class="brand-header">Demand Anomalies</div>
        <div class="brand-subtitle">Unusual demand patterns, spikes, and outlier detection</div>
        """,
        unsafe_allow_html=True,
    )

    anomalies = data.get("anomalies")
    features = data.get("features")
    products = data.get("products")

    if anomalies is None or anomalies.empty:
        render_empty_state(
            title="No Anomaly Data",
            message="Anomaly detection data is not available. Run the anomaly detection pipeline.",
        )
        return

    # Summary metrics
    total_anomalies = len(anomalies)
    total_features = len(features) if features is not None else 1
    anomaly_rate = total_anomalies / max(total_features, 1) * 100
    demand_spikes = int((anomalies["anomaly_type"] == "Demand Spike").sum())
    unusual_patterns = int((anomalies["anomaly_type"] == "Unusual Pattern").sum())

    render_kpi_row(
        [
            {
                "label": "Total Anomalies",
                "value": f"{total_anomalies:,}",
                "icon": "⚠️",
                "help_text": "Total anomalous demand events detected.",
            },
            {
                "label": "Anomaly Rate",
                "value": f"{anomaly_rate:.2f}%",
                "icon": "📊",
                "help_text": "Percentage of days with anomalous demand.",
            },
            {
                "label": "Demand Spikes",
                "value": f"{demand_spikes:,}",
                "icon": "🚀",
                "help_text": "Sudden, significant demand increases.",
            },
            {
                "label": "Unusual Patterns",
                "value": f"{unusual_patterns:,}",
                "icon": "🔍",
                "help_text": "Demand patterns that deviate from expected behavior.",
            },
        ],
        columns=4,
    )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # Anomaly timeline
    render_section_header("Anomaly Timeline", subtitle="Daily anomaly count over time")

    if "date" in anomalies.columns:
        anomaly_daily = (
            anomalies.groupby("date").size().reset_index(name="anomaly_count")
        )
        if not anomaly_daily.empty:
            fig = px.bar(
                anomaly_daily,
                x="date",
                y="anomaly_count",
                title="Daily Anomaly Count",
                template="plotly_dark",
                color="anomaly_count",
                color_continuous_scale="Reds",
                height=400,
            )
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Anomaly Count",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            render_empty_state("No Timeline Data", "No date information available for anomalies.")
    else:
        render_empty_state("No Timeline Data", "Date column not found in anomaly data.")

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # Top anomalous products
    render_section_header("Top Anomalous Products", subtitle="Products with the highest anomaly frequency")

    if "product_id" in anomalies.columns and products is not None and not products.empty:
        top_anomalous = (
            anomalies["product_id"].value_counts().head(10).reset_index()
        )
        top_anomalous.columns = ["product_id", "anomaly_count"]
        top_anomalous = top_anomalous.merge(
            products[["product_id", "category"]], on="product_id", how="left"
        )

        if not top_anomalous.empty:
            fig = px.bar(
                top_anomalous,
                x="anomaly_count",
                y="product_id",
                orientation="h",
                title="Top 10 Products by Anomaly Count",
                template="plotly_dark",
                color="category",
                height=400,
            )
            fig.update_layout(
                xaxis_title="Anomaly Count",
                yaxis_title="Product ID",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            render_empty_state("No Data", "No product-level anomaly data available.")
    else:
        render_empty_state("No Data", "Product or anomaly data is missing.")

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # Anomaly details
    render_section_header("Anomaly Details", subtitle="Detailed view of all detected anomalies")

    if "anomaly_type" in anomalies.columns:
        anomaly_type_filter = st.multiselect(
            "Filter by Anomaly Type",
            options=anomalies["anomaly_type"].unique().tolist(),
            default=anomalies["anomaly_type"].unique().tolist(),
            key="anomaly_type_filter",
        )
        filtered_anomalies = anomalies[
            anomalies["anomaly_type"].isin(anomaly_type_filter)
        ]
    else:
        filtered_anomalies = anomalies

    display_cols = [
        "date",
        "product_id",
        "store_id",
        "quantity_sold",
        "z_score",
        "anomaly_type",
    ]
    display_cols = [c for c in display_cols if c in filtered_anomalies.columns]

    if display_cols and not filtered_anomalies.empty:
        render_data_table(
            filtered_anomalies[display_cols].sort_values("date", ascending=False),
            title="All Anomalies",
            download_label="Download Anomalies (CSV)",
            download_filename="anomalies.csv",
        )
    else:
        render_empty_state("No Data", "No anomaly details available for the current selection.")
