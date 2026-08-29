"""Business Intelligence page for RetailSync AI."""

from __future__ import annotations

import logging

import pandas as pd
import plotly.express as px
import streamlit as st

from src.business_metrics.config import BusinessConfig
from src.business_metrics.kpi import (
    compute_executive_kpis,
    compute_forecast_accuracy,
    compute_inventory_carrying_cost,
    compute_overstock_value,
    compute_potential_revenue_protected,
    compute_stockout_cost,
)
from src.business_metrics.reorder import generate_reorder_recommendations
from dashboard.components.ui import (
    COLORS,
    render_alert,
    render_data_table,
    render_kpi_card,
    render_kpi_row,
    render_section_header,
)

logger = logging.getLogger(__name__)


def _apply_filters(data, selected_product, selected_category, selected_store, selected_warehouse, date_range):
    filtered = data.copy()

    if selected_product and selected_product != "All":
        filtered = filtered[filtered["product_id"] == selected_product]
    if selected_category and selected_category != "All":
        if "category" in filtered.columns:
            filtered = filtered[filtered["category"] == selected_category]
    if selected_store and selected_store != "All":
        if "store_id" in filtered.columns:
            filtered = filtered[filtered["store_id"] == selected_store]
    if selected_warehouse and selected_warehouse != "All":
        if "warehouse_id" in filtered.columns:
            filtered = filtered[filtered["warehouse_id"] == selected_warehouse]

    return filtered


def render_business_intelligence_page(engine, data: dict, models: dict):
    st.markdown(
        """
        <div class="brand-header">Business Intelligence</div>
        <div class="brand-subtitle">Financial KPIs, inventory economics, and reorder intelligence</div>
        """,
        unsafe_allow_html=True,
    )

    products = data.get("products")
    stores = data.get("stores")
    features = data.get("features")
    inv_intel = data.get("inv_intel")
    forecasts = data.get("forecasts")
    features = data.get("features")

    with st.sidebar.form("bi_filters"):
        product_options = ["All"] + sorted(products["product_id"].unique().tolist()) if products is not None and not products.empty else ["All"]
        category_options = ["All"] + sorted(products["category"].unique().tolist()) if products is not None and "category" in products.columns else ["All"]
        store_options = ["All"] + sorted(stores["store_id"].unique().tolist()) if stores is not None and not stores.empty else ["All"]
        warehouse_options = ["All"] + sorted(
            inv_intel["warehouse_id"].dropna().unique().tolist()
        ) if inv_intel is not None and "warehouse_id" in inv_intel.columns else ["All"]

        selected_product = st.selectbox("Product", product_options, key="bi_product")
        selected_category = st.selectbox("Category", category_options, key="bi_category")
        selected_store = st.selectbox("Store", store_options, key="bi_store")
        selected_warehouse = st.selectbox("Warehouse", warehouse_options, key="bi_warehouse")

        date_range = st.date_input(
            "Date Range",
            value=(
                features["date"].min().date() if features is not None and "date" in features.columns else None,
                features["date"].max().date() if features is not None and "date" in features.columns else None,
            ),
            key="bi_dates",
        )
        st.form_submit_button("Apply Filters", use_container_width=True)

    products_filtered = _apply_filters(
        products, selected_product, selected_category, None, None, None
    )
    inv_intel_filtered = _apply_filters(
        inv_intel, selected_product, selected_category, selected_store, selected_warehouse, date_range
    )
    forecasts_filtered = _apply_filters(
        forecasts, selected_product, selected_category, selected_store, None, None
    )

    config = BusinessConfig()

    # ============================================================
    # EXECUTIVE KPIs
    # ============================================================
    render_section_header("Executive KPIs", subtitle="Key financial and operational metrics for the selected filters")

    with st.spinner("Calculating KPIs..."):
        stockout_result = compute_stockout_cost(
            inv_intel_filtered, products_filtered, config
        )
        overstock_result = compute_overstock_value(
            inv_intel_filtered, products_filtered, config
        )
        carrying_result = compute_inventory_carrying_cost(
            inv_intel_filtered, products_filtered, config
        )
        revenue_protected = compute_potential_revenue_protected(
            stockout_result, overstock_result, config
        )
        forecast_accuracy = compute_forecast_accuracy(
            features, models.get("demand_forecaster"), config
        ) if features is not None and not features.empty else {}
        exec_kpis = compute_executive_kpis(
            inv_intel_filtered,
            products_filtered,
            forecasts_filtered,
            stockout_result,
            overstock_result,
            carrying_result,
            revenue_protected,
            forecast_accuracy,
            config,
        )

    render_kpi_row(
        [
            {
                "label": "Total Inventory Value",
                "value": f"${exec_kpis['total_inventory_value']:,.2f}",
                "icon": "💰",
                "help_text": "Sum of quantity_on_hand * cost_price for filtered items.",
            },
            {
                "label": "Est. Carrying Cost",
                "value": f"${exec_kpis['estimated_carrying_cost']:,.2f}",
                "icon": "📦",
                "help_text": f"{config.carrying_cost_pct:.0%} of inventory value per year.",
            },
            {
                "label": "Stockout Exposure",
                "value": f"${exec_kpis['stockout_exposure']:,.2f}",
                "icon": "🚨",
                "help_text": "Estimated cost of stockouts based on configurable assumptions.",
            },
            {
                "label": "Overstock Value",
                "value": f"${exec_kpis['overstock_value']:,.2f}",
                "icon": "📊",
                "help_text": "Value of excess inventory above max stock level.",
            },
            {
                "label": "Products Needing Reorder",
                "value": str(exec_kpis["products_requiring_reorder"]),
                "icon": "📋",
                "help_text": "Items with URGENT or SOON reorder urgency.",
            },
            {
                "label": "Potential Revenue Protected",
                "value": f"${exec_kpis['potential_revenue_protected']:,.2f}",
                "icon": "🛡️",
                "help_text": "ESTIMATE. Combined avoided stockout revenue and recovered overstock margin.",
            },
            {
                "label": "Forecast Accuracy (sMAPE)",
                "value": f"{exec_kpis['forecast_accuracy_smape']:.2f}%",
                "icon": "🎯",
                "help_text": f"Model: {exec_kpis['forecast_model']}",
            },
            {
                "label": "14-Day Forecasted Demand",
                "value": f"{exec_kpis['forecasted_demand_14d']:,.0f} units",
                "icon": "📈",
                "help_text": "Total forecasted demand for the next 14 days.",
            },
        ],
        columns=4,
    )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ============================================================
    # FORECAST ACCURACY BREAKDOWN
    # ============================================================
    render_section_header("Forecast Accuracy", subtitle="Model performance breakdown")

    col1, col2, col3 = st.columns(3)
    with col1:
        if forecast_accuracy.get("overall"):
            st.markdown("**Overall Test Metrics**")
            st.markdown(f"- **Model:** {forecast_accuracy['overall'].get('model', 'N/A')}")
            st.markdown(f"- **MAE:** {forecast_accuracy['overall'].get('mae', 'N/A')}")
            st.markdown(f"- **RMSE:** {forecast_accuracy['overall'].get('rmse', 'N/A')}")
            st.markdown(f"- **sMAPE:** {forecast_accuracy['overall'].get('smape', 'N/A')}%")
            st.markdown(f"- **Bias:** {forecast_accuracy['overall'].get('bias', 'N/A')}")
        else:
            st.info("No model loaded. Train the demand forecaster to see accuracy metrics.")

    with col2:
        by_product = forecast_accuracy.get("by_product")
        if by_product is not None and not by_product.empty:
            st.markdown("**Accuracy by Product (Top 10 by MAE)**")
            display = by_product.head(10)[
                ["product_id", "mae", "rmse", "smape", "samples"]
            ].copy()
            display.columns = ["Product", "MAE", "RMSE", "sMAPE %", "Samples"]
            st.dataframe(display, use_container_width=True)

    with col3:
        by_store = forecast_accuracy.get("by_store")
        if by_store is not None and not by_store.empty:
            st.markdown("**Accuracy by Store (Top 10 by MAE)**")
            display = by_store.head(10)[
                ["store_id", "mae", "rmse", "smape", "samples"]
            ].copy()
            display.columns = ["Store", "MAE", "RMSE", "sMAPE %", "Samples"]
            st.dataframe(display, use_container_width=True)

    by_category = forecast_accuracy.get("by_category")
    if by_category is not None and not by_category.empty:
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
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ============================================================
    # STOCKOUT ANALYSIS
    # ============================================================
    render_section_header("Stockout Analysis", subtitle="Estimated stockout impact and risk details")

    col1, col2, col3 = st.columns(3)
    with col1:
        render_kpi_card(
            label="Stockout Items",
            value=str(stockout_result["stockout_items_count"]),
            icon="🚨",
        )
        render_kpi_card(
            label="High Risk Items",
            value=str(stockout_result["high_risk_count"]),
            icon="🔴",
        )
    with col2:
        render_kpi_card(
            label="Est. Stockout Units",
            value=f"{stockout_result['estimated_stockout_units']:,.0f}",
            icon="📦",
        )
        render_kpi_card(
            label="Est. Lost Revenue",
            value=f"${stockout_result['estimated_lost_revenue']:,.2f}",
            icon="💸",
        )
    with col3:
        render_kpi_card(
            label="Est. Stockout Cost",
            value=f"${stockout_result['estimated_stockout_cost']:,.2f}",
            icon="💵",
        )

    st.caption("*These figures are estimates based on configurable assumptions.*")

    by_item = stockout_result.get("by_item")
    if by_item is not None and not by_item.empty:
        render_data_table(
            by_item,
            title="Stockout Details by Item",
        )

        fig = px.bar(
            by_item.head(20),
            x="product_id",
            y="stockout_cost",
            color="stockout_risk",
            title="Estimated Stockout Cost by Product",
            template="plotly_dark",
            height=400,
        )
        fig.update_layout(
            xaxis_title="Product ID",
            yaxis_title="Stockout Cost ($)",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ============================================================
    # OVERSTOCK ANALYSIS
    # ============================================================
    render_section_header("Overstock Analysis", subtitle="Excess inventory valuation and risk breakdown")

    col1, col2, col3 = st.columns(3)
    with col1:
        render_kpi_card(
            label="Overstock Items",
            value=str(overstock_result["overstock_items_count"]),
            icon="📦",
        )
        render_kpi_card(
            label="Excess Units",
            value=f"{overstock_result['excess_units']:,.0f}",
            icon="🏗️",
        )
    with col2:
        render_kpi_card(
            label="Overstock Inventory Value",
            value=f"${overstock_result['overstock_inventory_value']:,.2f}",
            icon="💰",
        )
        render_kpi_card(
            label="High Risk Items",
            value=str(overstock_result["high_risk_count"]),
            icon="🟠",
        )
    with col3:
        render_kpi_card(
            label="Medium Risk Items",
            value=str(overstock_result["medium_risk_count"]),
            icon="🟡",
        )

    by_item = overstock_result.get("by_item")
    if by_item is not None and not by_item.empty:
        render_data_table(
            by_item,
            title="Overstock Details by Item",
        )

        fig = px.bar(
            by_item.head(20),
            x="product_id",
            y="overstock_value",
            color="overstock_risk",
            title="Overstock Value by Product",
            template="plotly_dark",
            height=400,
        )
        fig.update_layout(
            xaxis_title="Product ID",
            yaxis_title="Overstock Value ($)",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ============================================================
    # REORDER RECOMMENDATIONS
    # ============================================================
    render_section_header("Reorder Recommendations", subtitle="Data-driven replenishment suggestions")

    with st.spinner("Generating reorder recommendations..."):
        reorder_df = generate_reorder_recommendations(
            inv_intel_filtered,
            products_filtered,
            forecasts_filtered,
            data["suppliers"],
            config,
        )

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

        st.markdown(f"**Showing {len(filtered_reorder)} recommendations**")

        display_cols = [
            "product_id",
            "store_id",
            "quantity_on_hand",
            "avg_daily_demand",
            "forecast_demand_14d",
            "lead_time_days",
            "safety_stock",
            "recommended_quantity",
            "reorder_value",
            "expected_coverage_days",
            "reorder_urgency_computed",
            "reorder_reasoning",
        ]
        display_cols = [c for c in display_cols if c in filtered_reorder.columns]
        render_data_table(
            filtered_reorder[display_cols],
            title="Reorder Recommendations",
            download_label="Download Reorder Recommendations (CSV)",
            download_filename="reorder_recommendations.csv",
        )

        fig = px.scatter(
            filtered_reorder,
            x="expected_coverage_days",
            y="recommended_quantity",
            color="reorder_urgency_computed",
            size="reorder_value",
            hover_data=["product_id", "store_id"],
            title="Reorder Recommendations: Coverage vs Quantity",
            template="plotly_dark",
            height=400,
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

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ============================================================
    # PRODUCT / STORE BREAKDOWN
    # ============================================================
    render_section_header("Product / Store Breakdown", subtitle="Inventory value and risk by category")

    if not inv_intel_filtered.empty and "category" in inv_intel_filtered.columns:
        cat_summary = (
            inv_intel_filtered.groupby("category")
            .agg(
                total_quantity=("quantity_on_hand", "sum"),
                avg_cost=("cost_price", "mean"),
                stockout_high=("stockout_risk", lambda x: (x == "HIGH").sum()),
                overstock_high=("overstock_risk", lambda x: (x == "HIGH").sum()),
            )
            .reset_index()
        )
        cat_summary["inventory_value"] = cat_summary["total_quantity"] * cat_summary["avg_cost"]

        fig = px.bar(
            cat_summary,
            x="category",
            y="inventory_value",
            color="category",
            title="Inventory Value by Category",
            template="plotly_dark",
            height=400,
        )
        fig.update_layout(
            xaxis_title="Category",
            yaxis_title="Inventory Value ($)",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        render_data_table(
            cat_summary,
            title="Category Summary",
        )

    # Download all executive KPIs
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    render_section_header("Download Data")

    exec_df = pd.DataFrame([exec_kpis])
    render_data_table(
        exec_df,
        title="Executive KPIs",
        download_label="Download Executive KPIs (CSV)",
        download_filename="executive_kpis.csv",
    )
