"""Warehouse Intelligence page for RetailSync AI."""

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


def render_warehouse_page(data: dict) -> None:
    """Render the warehouse intelligence page."""
    st.markdown(
        """
        <div class="brand-header">Warehouse Intelligence</div>
        <div class="brand-subtitle">Capacity utilization, optimization, and distribution analysis</div>
        """,
        unsafe_allow_html=True,
    )

    wh_opt = data.get("wh_opt")
    _warehouses = data.get("warehouses")

    if wh_opt is None or wh_opt.empty:
        render_empty_state(
            title="No Warehouse Data",
            message="Warehouse optimization data is not available.",
        )
        return

    # Metrics
    total_warehouses = len(wh_opt)
    total_capacity = float(wh_opt["capacity_m3"].sum())
    avg_util = float(wh_opt["utilization_pct"].mean())
    total_occupied = float(wh_opt["occupied_volume_m3"].sum())

    render_kpi_row(
        [
            {
                "label": "Total Warehouses",
                "value": str(total_warehouses),
                "icon": "🏭",
                "help_text": "Number of warehouses in the network.",
            },
            {
                "label": "Total Capacity",
                "value": f"{total_capacity:,.0f} m³",
                "icon": "📦",
                "help_text": "Total cubic meter capacity across all warehouses.",
            },
            {
                "label": "Avg Utilization",
                "value": f"{avg_util:.1f}%",
                "icon": "📊",
                "help_text": "Average warehouse utilization percentage.",
            },
            {
                "label": "Total Occupied",
                "value": f"{total_occupied:,.0f} m³",
                "icon": "🏗️",
                "help_text": "Total occupied volume across all warehouses.",
            },
        ],
        columns=4,
    )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # Utilization chart
    render_section_header("Warehouse Utilization", subtitle="Utilization percentage by warehouse")

    fig = px.bar(
        wh_opt,
        x="warehouse_id",
        y="utilization_pct",
        title="Warehouse Utilization %",
        template="plotly_dark",
        color="utilization_pct",
        color_continuous_scale="RdYlGn_r",
        height=400,
    )
    fig.update_layout(
        xaxis_title="Warehouse ID",
        yaxis_title="Utilization %",
        showlegend=False,
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # Warehouse details
    render_section_header("Warehouse Details", subtitle="Detailed view of each warehouse")

    display_cols = [
        "warehouse_id",
        "warehouse_name",
        "city",
        "capacity_m3",
        "occupied_volume_m3",
        "utilization_pct",
        "capacity_risk",
        "recommendation",
    ]
    display_cols = [c for c in display_cols if c in wh_opt.columns]

    render_data_table(
        wh_opt[display_cols],
        title="Warehouse Details",
        download_label="Download Warehouse Data (CSV)",
        download_filename="warehouse_details.csv",
    )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # Warehouse summary charts
    render_section_header("Capacity & Risk Analysis", subtitle="Visual breakdown of warehouse metrics")

    if "capacity_risk" in wh_opt.columns:
        risk_counts = wh_opt["capacity_risk"].value_counts().reset_index()
        risk_counts.columns = ["risk", "count"]
        if not risk_counts.empty:
            fig = px.pie(
                risk_counts,
                values="count",
                names="risk",
                title="Capacity Risk Distribution",
                template="plotly_dark",
                height=400,
            )
            st.plotly_chart(fig, width="stretch")
