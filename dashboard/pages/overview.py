"""Executive Overview page for RetailSync AI."""

from __future__ import annotations

import logging

import streamlit as st

from dashboard.components.ui import (
    COLORS,
    inject_global_css,
    render_alert,
    render_data_table,
    render_empty_state,
    render_kpi_card,
    render_kpi_row,
    render_pie_chart,
    render_section_header,
)
from src.business_metrics.kpi import compute_forecast_accuracy

logger = logging.getLogger(__name__)


def render_overview_page(data: dict, models: dict, engine) -> None:
    """Render the executive overview dashboard."""
    inject_global_css()

    st.markdown(
        """
        <div class="brand-header">RetailSync AI</div>
        <div class="brand-subtitle">AI-Powered Retail Demand Forecasting & Supply Chain Intelligence</div>
        """,
        unsafe_allow_html=True,
    )

    # ============================================================
    # KEY BUSINESS KPIs
    # ============================================================
    render_section_header("Key Business Indicators", subtitle="At-a-glance operational health metrics")

    sales = data.get("sales")
    products = data.get("products")
    stores = data.get("stores")
    forecasts = data.get("forecasts")
    inv_intel = data.get("inv_intel")
    anomalies = data.get("anomalies")
    wh_opt = data.get("wh_opt")

    total_revenue = float(sales["revenue"].sum()) if sales is not None and not sales.empty else 0.0
    total_quantity = int(sales["quantity_sold"].sum()) if sales is not None and not sales.empty else 0
    total_products = int(products["product_id"].nunique()) if products is not None and not products.empty else 0
    total_stores = int(stores["store_id"].nunique()) if stores is not None and not stores.empty else 0
    forecast_demand = float(forecasts["forecast_demand"].sum()) if forecasts is not None and not forecasts.empty else 0.0
    high_stockout = int((inv_intel["stockout_risk"] == "HIGH").sum()) if inv_intel is not None and not inv_intel.empty else 0
    total_anomalies = int(len(anomalies)) if anomalies is not None else 0
    avg_util = float(wh_opt["utilization_pct"].mean()) if wh_opt is not None and not wh_opt.empty else 0.0

    render_kpi_row(
        [
            {
                "label": "Total Revenue",
                "value": f"${total_revenue:,.0f}",
                "icon": "💰",
                "help_text": "Sum of all recorded sales revenue.",
            },
            {
                "label": "Units Sold",
                "value": f"{total_quantity:,}",
                "icon": "📦",
                "help_text": "Total units sold across all stores.",
            },
            {
                "label": "Products",
                "value": f"{total_products:,}",
                "icon": "🏷️",
                "help_text": "Unique products in the catalog.",
            },
            {
                "label": "Stores",
                "value": f"{total_stores:,}",
                "icon": "🏪",
                "help_text": "Active retail stores.",
            },
            {
                "label": "14-Day Forecast",
                "value": f"{forecast_demand:,.0f} units",
                "icon": "🔮",
                "help_text": "Total forecasted demand for the next 14 days.",
            },
            {
                "label": "Stockout Risks (HIGH)",
                "value": f"{high_stockout:,}",
                "icon": "🚨",
                "help_text": "Items at high risk of stockout.",
            },
            {
                "label": "Anomalies Detected",
                "value": f"{total_anomalies:,}",
                "icon": "⚠️",
                "help_text": "Demand anomalies identified in recent data.",
            },
            {
                "label": "Avg Warehouse Utilization",
                "value": f"{avg_util:.1f}%",
                "icon": "🏭",
                "help_text": "Average warehouse capacity utilization.",
            },
        ],
        columns=4,
    )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ============================================================
    # FORECAST ACCURACY
    # ============================================================
    render_section_header("Forecast Accuracy", subtitle="Model performance on held-out test data")

    features = data.get("features")
    model_pkg = models.get("demand_forecaster")
    forecast_accuracy = {}
    if features is not None and not features.empty:
        try:
            forecast_accuracy = compute_forecast_accuracy(features, model_pkg)
        except Exception as exc:
            logger.warning("Could not compute forecast accuracy: %s", exc)

    overall = forecast_accuracy.get("overall", {})
    if overall:
        acc_cols = st.columns(4)
        with acc_cols[0]:
            render_kpi_card(
                label="Model",
                value=overall.get("model", "N/A"),
                icon="🧠",
                help_text=f"Test rows: {overall.get('test_rows', 'N/A')}",
            )
        with acc_cols[1]:
            render_kpi_card(
                label="MAE",
                value=f"{overall.get('mae', 0):.2f}",
                icon="📏",
                help_text="Mean Absolute Error (lower is better).",
            )
        with acc_cols[2]:
            render_kpi_card(
                label="RMSE",
                value=f"{overall.get('rmse', 0):.2f}",
                icon="📐",
                help_text="Root Mean Squared Error (lower is better).",
            )
        with acc_cols[3]:
            smape = overall.get("smape", 0)
            render_kpi_card(
                label="sMAPE",
                value=f"{smape:.2f}%",
                icon="🎯",
                help_text="Symmetric Mean Absolute Percentage Error (lower is better).",
            )
    else:
        render_alert(
            message="No trained model loaded. Run the forecasting pipeline to compute backtested accuracy metrics.",
            severity="info",
            title="Model Not Available",
        )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ============================================================
    # INVENTORY HEALTH
    # ============================================================
    render_section_header("Inventory Health", subtitle="Stockout, overstock, and reorder risk overview")

    if inv_intel is not None and not inv_intel.empty:
        high_stockout = int((inv_intel["stockout_risk"] == "HIGH").sum())
        med_stockout = int((inv_intel["stockout_risk"] == "MEDIUM").sum())
        high_overstock = int((inv_intel["overstock_risk"] == "HIGH").sum())
        urgent_reorder = int((inv_intel["reorder_urgency"] == "URGENT").sum())
        soon_reorder = int((inv_intel["reorder_urgency"] == "SOON").sum())

        inv_cols = st.columns(5)
        with inv_cols[0]:
            render_kpi_card(
                label="Stockout HIGH",
                value=str(high_stockout),
                icon="🔴",
                help_text="Items with HIGH stockout risk.",
            )
        with inv_cols[1]:
            render_kpi_card(
                label="Stockout MEDIUM",
                value=str(med_stockout),
                icon="🟡",
                help_text="Items with MEDIUM stockout risk.",
            )
        with inv_cols[2]:
            render_kpi_card(
                label="Overstock HIGH",
                value=str(high_overstock),
                icon="🟠",
                help_text="Items with HIGH overstock risk.",
            )
        with inv_cols[3]:
            render_kpi_card(
                label="Urgent Reorder",
                value=str(urgent_reorder),
                icon="🚨",
                help_text="Items requiring immediate reorder.",
            )
        with inv_cols[4]:
            render_kpi_card(
                label="Soon Reorder",
                value=str(soon_reorder),
                icon="📋",
                help_text="Items approaching reorder point.",
            )

        # Risk distribution charts
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            stockout_counts = inv_intel["stockout_risk"].value_counts().reset_index()
            stockout_counts.columns = ["risk", "count"]
            render_pie_chart(
                stockout_counts,
                values="count",
                names="risk",
                title="Stockout Risk Distribution",
                color_map={"HIGH": COLORS["danger"], "MEDIUM": COLORS["warning"], "LOW": COLORS["success"]},
            )
        with chart_col2:
            overstock_counts = inv_intel["overstock_risk"].value_counts().reset_index()
            overstock_counts.columns = ["risk", "count"]
            render_pie_chart(
                overstock_counts,
                values="count",
                names="risk",
                title="Overstock Risk Distribution",
                color_map={"HIGH": COLORS["danger"], "MEDIUM": COLORS["warning"], "LOW": COLORS["success"]},
            )
    else:
        render_empty_state(
            title="No Inventory Data",
            message="Inventory intelligence data is not available.",
        )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ============================================================
    # ACTIVE ALERTS
    # ============================================================
    render_section_header("Active Alerts", subtitle="Items requiring immediate attention")

    if inv_intel is not None and not inv_intel.empty:
        critical = inv_intel[
            (inv_intel["stockout_risk"] == "HIGH")
            | (inv_intel["overstock_risk"] == "HIGH")
            | (inv_intel["reorder_urgency"] == "URGENT")
        ].sort_values("composite_risk_score", ascending=False)

        if not critical.empty:
            alert_df = critical.head(15)[
                [
                    "product_id",
                    "store_id",
                    "stockout_risk",
                    "overstock_risk",
                    "reorder_urgency",
                    "recommended_action",
                ]
            ].copy()

            # Show severity counts
            sev_col1, sev_col2, sev_col3 = st.columns(3)
            with sev_col1:
                render_alert(
                    message=f"{len(critical)} items require immediate attention",
                    severity="critical",
                    title="Critical Alerts",
                )
            with sev_col2:
                spike_count = (
                    int((data["anomalies"]["anomaly_type"] == "Demand Spike").sum())
                    if data.get("anomalies") is not None
                    else 0
                )
                render_alert(
                    message=f"{spike_count} demand spikes detected recently",
                    severity="high",
                    title="Demand Spikes",
                )
            with sev_col3:
                render_alert(
                    message=f"{total_anomalies:,} anomalies detected in recent data",
                    severity="medium",
                    title="Total Anomalies",
                )

            render_data_table(
                alert_df,
                title="Top Priority Items",
                download_label="Download Critical Items (CSV)",
                download_filename="critical_items.csv",
            )
        else:
            render_alert(
                message="No critical inventory items detected. All systems operating within normal parameters.",
                severity="success",
                title="All Clear",
            )
    else:
        render_empty_state(
            title="No Alerts",
            message="No alert data available.",
        )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ============================================================
    # REORDER OPPORTUNITIES
    # ============================================================
    render_section_header("Reorder Opportunities", subtitle="Items where proactive replenishment can prevent stockouts")

    if inv_intel is not None and not inv_intel.empty:
        reorder_items = inv_intel[
            inv_intel["reorder_urgency"].isin(["URGENT", "SOON"])
        ].sort_values("composite_risk_score", ascending=False)

        if not reorder_items.empty:
            reorder_display = reorder_items.head(10)[
                [
                    "product_id",
                    "store_id",
                    "quantity_on_hand",
                    "reorder_urgency",
                    "recommended_action",
                    "stock_coverage_days",
                ]
            ].copy()
            render_data_table(
                reorder_display,
                title="Reorder Recommendations",
                download_label="Download Reorder Recommendations (CSV)",
                download_filename="reorder_recommendations.csv",
            )
        else:
            render_alert(
                message="No urgent reorder opportunities at this time.",
                severity="success",
                title="Inventory Healthy",
            )
    else:
        render_empty_state(
            title="No Reorder Data",
            message="Reorder data is not available.",
        )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ============================================================
    # QUICK NAVIGATION
    # ============================================================
    render_section_header("Quick Navigation", subtitle="Jump to detailed analytics")

    nav_cols = st.columns(4)
    nav_items = [
        ("🔮", "Demand Forecast", "🔮 Demand Forecast"),
        ("📦", "Inventory Intelligence", "📦 Inventory Intelligence"),
        ("⚠️", "Demand Anomalies", "⚠️ Demand Anomalies"),
        ("🧠", "Model Explainability", "🧠 Model Explainability"),
        ("🎯", "Segmentation", "🎯 Segmentation"),
        ("🏭", "Warehouse Intelligence", "🏭 Warehouse Intelligence"),
        ("📈", "Business Intelligence", "📈 Business Intelligence"),
        ("📊", "Model Performance", "📊 Model Performance"),
    ]

    for idx, (icon, label, page_name) in enumerate(nav_items):
        with nav_cols[idx % 4]:
            if st.button(
                f"{icon} {label}",
                key=f"nav_{idx}",
                use_container_width=True,
            ):
                st.session_state["nav_page"] = page_name
                st.rerun()

    # ============================================================
    # MODEL STATUS
    # ============================================================
    render_section_header("ML Models Status", subtitle="Pipeline component health check")

    model_items = [
        ("demand_forecaster", "Demand Forecaster"),
        ("product_clusterer", "Product Clusterer"),
        ("store_clusterer", "Store Clusterer"),
        ("warehouse_clusterer", "Warehouse Clusterer"),
    ]

    model_cols = st.columns(len(model_items))
    for idx, (key, name) in enumerate(model_items):
        with model_cols[idx]:
            if key in models:
                render_alert(
                    message=f"{name} is loaded and ready.",
                    severity="success",
                    title=f"✓ {name}",
                )
            else:
                render_alert(
                    message=f"{name} model file not found at models/{key}.pkl",
                    severity="warning",
                    title=f"✗ {name}",
                )
