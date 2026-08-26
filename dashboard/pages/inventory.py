"""Inventory Intelligence and Alert Center page for RetailSync AI."""

from __future__ import annotations

import logging

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.ui import (
    COLORS,
    inject_global_css,
    render_alert,
    render_data_table,
    render_empty_state,
    render_kpi_row,
    render_pie_chart,
    render_section_header,
    render_filter_sidebar,
    apply_filters,
)

logger = logging.getLogger(__name__)


def render_inventory_page(data: dict, engine) -> None:
    """Render the inventory intelligence page with integrated alert center."""
    inject_global_css()

    st.markdown(
        """
        <div class="brand-header">Inventory Intelligence</div>
        <div class="brand-subtitle">Stockout risk, overstock analysis, and reorder recommendations</div>
        """,
        unsafe_allow_html=True,
    )

    inv_intel = data.get("inv_intel")
    inventory_alerts = data.get("inventory_alerts")
    products = data.get("products")
    stores = data.get("stores")
    forecasts = data.get("forecasts")
    suppliers = data.get("suppliers")

    if inv_intel is None or inv_intel.empty:
        render_empty_state(
            title="No Inventory Data",
            message="Inventory intelligence data is not available.",
        )
        return

    # Filters
    product_options = sorted(products["product_id"].unique().tolist()) if products is not None and not products.empty else []
    category_options = sorted(products["category"].unique().tolist()) if products is not None and "category" in products.columns else []
    store_options = sorted(stores["store_id"].unique().tolist()) if stores is not None and not stores.empty else []
    warehouse_options = sorted(inv_intel["warehouse_id"].dropna().unique().tolist()) if "warehouse_id" in inv_intel.columns else []

    filters = render_filter_sidebar(
        title="Inventory Filters",
        product_options=product_options,
        category_options=category_options,
        store_options=store_options,
        warehouse_options=warehouse_options,
    )

    inv_filtered = apply_filters(inv_intel, filters)

    # ============================================================
    # SUMMARY METRICS
    # ============================================================
    render_section_header("Inventory Summary", subtitle="Key risk indicators for filtered inventory")

    high_stockout = int((inv_filtered["stockout_risk"] == "HIGH").sum()) if not inv_filtered.empty else 0
    med_stockout = int((inv_filtered["stockout_risk"] == "MEDIUM").sum()) if not inv_filtered.empty else 0
    high_overstock = int((inv_filtered["overstock_risk"] == "HIGH").sum()) if not inv_filtered.empty else 0
    urgent_reorder = int((inv_filtered["reorder_urgency"] == "URGENT").sum()) if not inv_filtered.empty else 0

    render_kpi_row(
        [
            {
                "label": "Stockout HIGH",
                "value": str(high_stockout),
                "icon": "🔴",
                "help_text": "Items at high risk of stockout.",
            },
            {
                "label": "Stockout MEDIUM",
                "value": str(med_stockout),
                "icon": "🟡",
                "help_text": "Items at medium risk of stockout.",
            },
            {
                "label": "Overstock HIGH",
                "value": str(high_overstock),
                "icon": "🟠",
                "help_text": "Items with excess inventory.",
            },
            {
                "label": "Urgent Reorder",
                "value": str(urgent_reorder),
                "icon": "🚨",
                "help_text": "Items requiring immediate reorder.",
            },
        ],
        columns=4,
    )

    # ============================================================
    # RISK DISTRIBUTION
    # ============================================================
    render_section_header("Risk Distribution", subtitle="Breakdown of stockout and overstock risks")

    if not inv_filtered.empty:
        col1, col2 = st.columns(2)
        with col1:
            stockout_counts = inv_filtered["stockout_risk"].value_counts().reset_index()
            stockout_counts.columns = ["risk", "count"]
            render_pie_chart(
                stockout_counts,
                values="count",
                names="risk",
                title="Stockout Risk Distribution",
                color_map={"HIGH": COLORS["danger"], "MEDIUM": COLORS["warning"], "LOW": COLORS["success"]},
            )
        with col2:
            overstock_counts = inv_filtered["overstock_risk"].value_counts().reset_index()
            overstock_counts.columns = ["risk", "count"]
            render_pie_chart(
                overstock_counts,
                values="count",
                names="risk",
                title="Overstock Risk Distribution",
                color_map={"HIGH": COLORS["danger"], "MEDIUM": COLORS["warning"], "LOW": COLORS["success"]},
            )
    else:
        render_empty_state("No Data", "No inventory data matches the current filters.")

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ============================================================
    # CRITICAL ITEMS
    # ============================================================
    render_section_header("Critical Inventory Items", subtitle="Items requiring immediate action")

    if not inv_filtered.empty:
        critical = inv_filtered[
            (inv_filtered["stockout_risk"] == "HIGH")
            | (inv_filtered["overstock_risk"] == "HIGH")
            | (inv_filtered["reorder_urgency"] == "URGENT")
        ].sort_values("composite_risk_score", ascending=False)

        if not critical.empty:
            display_cols = [
                "product_id",
                "store_id",
                "stockout_risk",
                "overstock_risk",
                "reorder_urgency",
                "recommended_action",
            ]
            display_cols = [c for c in display_cols if c in critical.columns]
            render_data_table(
                critical[display_cols],
                title="Critical Items",
                download_label="Download Critical Items (CSV)",
                download_filename="critical_inventory_items.csv",
            )
        else:
            render_alert(
                message="No critical inventory items found for the current selection.",
                severity="success",
                title="All Clear",
            )
    else:
        render_empty_state("No Data", "No critical items match the current filters.")

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ============================================================
    # ALERT CENTER
    # ============================================================
    render_section_header("Alert Center", subtitle="Stockout risks, overstock risks, anomalies, and reorder alerts")

    alert_container = st.container()

    with alert_container:
        if inventory_alerts is not None and not inventory_alerts.empty:
            alert_filter = st.multiselect(
                "Filter by Alert Type",
                options=inventory_alerts["alert_type"].unique().tolist(),
                default=inventory_alerts["alert_type"].unique().tolist(),
                key="alert_type_filter",
            )
            filtered_alerts = inventory_alerts[
                inventory_alerts["alert_type"].isin(alert_filter)
            ]

            if not filtered_alerts.empty:
                # Severity counts
                sev_counts = filtered_alerts["severity"].value_counts() if "severity" in filtered_alerts.columns else {}
                sev_cols = st.columns(min(len(sev_counts), 4) or 1)
                severity_order = ["critical", "high", "medium", "low"]
                for idx, sev in enumerate([s for s in severity_order if s in sev_counts.index]):
                    with sev_cols[idx % 4]:
                        render_alert(
                            message=f"{sev_counts.get(sev, 0)} {sev} alert(s)",
                            severity=sev,
                            title=f"{sev.upper()} Alerts",
                        )

                # Alert table
                render_data_table(
                    filtered_alerts,
                    title="All Alerts",
                    download_label="Download Alerts (CSV)",
                    download_filename="inventory_alerts.csv",
                )
            else:
                render_empty_state("No Alerts", "No alerts match the selected filters.")
        else:
            render_empty_state("No Alerts", "No inventory alerts available.")

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ============================================================
    # REORDER RECOMMENDATIONS
    # ============================================================
    render_section_header("Reorder Recommendations", subtitle="Proactive replenishment suggestions")

    if not inv_filtered.empty:
        from src.business_metrics.reorder import generate_reorder_recommendations
        from business_metrics.config import BusinessConfig

        config = BusinessConfig()
        with st.spinner("Generating reorder recommendations..."):
            try:
                reorder_df = generate_reorder_recommendations(
                    inv_filtered,
                    products if products is not None else pd.DataFrame(),
                    forecasts if forecasts is not None else pd.DataFrame(),
                    suppliers if suppliers is not None else pd.DataFrame(),
                    config,
                )
            except Exception as exc:
                logger.error("Reorder generation failed: %s", exc)
                reorder_df = pd.DataFrame()

        if not reorder_df.empty:
            urgency_filter = st.multiselect(
                "Filter by Urgency",
                options=reorder_df["reorder_urgency_computed"].unique().tolist(),
                default=["CRITICAL", "URGENT", "SOON"],
                key="reorder_urgency_filter",
            )
            filtered_reorder = reorder_df[
                reorder_df["reorder_urgency_computed"].isin(urgency_filter)
            ]

            render_data_table(
                filtered_reorder,
                title="Reorder Recommendations",
                download_label="Download Reorder Recommendations (CSV)",
                download_filename="reorder_recommendations.csv",
            )

            if not filtered_reorder.empty:
                fig = px.scatter(
                    filtered_reorder,
                    x="expected_coverage_days",
                    y="recommended_quantity",
                    color="reorder_urgency_computed",
                    size="reorder_value",
                    hover_data=["product_id", "store_id"],
                    title="Reorder Recommendations: Coverage vs Quantity",
                    template="plotly_dark",
                    color_discrete_map={
                        "CRITICAL": COLORS["danger"],
                        "URGENT": "#f97316",
                        "SOON": COLORS["warning"],
                        "NORMAL": COLORS["success"],
                    },
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            render_alert(
                message="No reorder recommendations available for the current filters.",
                severity="info",
                title="No Recommendations",
            )
    else:
        render_empty_state("No Data", "No inventory data to generate reorder recommendations.")
