"""AI Analyst page for RetailSync AI."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from dashboard.components.ui import (
    COLORS,
    inject_global_css,
    render_alert,
    render_empty_state,
    render_kpi_card,
    render_section_header,
)

logger = logging.getLogger(__name__)


def _generate_insights(data: dict) -> list[dict]:
    """Generate automated insights from the current data."""
    insights = []
    sales = data.get("sales")
    products = data.get("products")
    stores = data.get("stores")
    forecasts = data.get("forecasts")
    inv_intel = data.get("inv_intel")
    anomalies = data.get("anomalies")
    wh_opt = data.get("wh_opt")

    if sales is not None and not sales.empty:
        total_revenue = float(sales["revenue"].sum())
        top_product = sales.groupby("product_id")["revenue"].sum().idxmax()
        top_store = sales.groupby("store_id")["revenue"].sum().idxmax()
        insights.append(
            {
                "title": "Revenue Overview",
                "severity": "info",
                "message": f"Total revenue is ${total_revenue:,.0f}. Top product: {top_product}. Top store: {top_store}.",
            }
        )

    if forecasts is not None and not forecasts.empty:
        total_forecast = float(forecasts["forecast_demand"].sum())
        insights.append(
            {
                "title": "Demand Forecast",
                "severity": "info",
                "message": f"The model forecasts {total_forecast:,.0f} units of demand over the next 14 days.",
            }
        )

    if inv_intel is not None and not inv_intel.empty:
        high_stockout = int((inv_intel["stockout_risk"] == "HIGH").sum())
        high_overstock = int((inv_intel["overstock_risk"] == "HIGH").sum())
        urgent = int((inv_intel["reorder_urgency"] == "URGENT").sum())

        if high_stockout > 0:
            insights.append(
                {
                    "title": "Stockout Risk Alert",
                    "severity": "critical",
                    "message": f"{high_stockout} items are at HIGH risk of stockout. Immediate action recommended.",
                }
            )
        if high_overstock > 0:
            insights.append(
                {
                    "title": "Overstock Risk",
                    "severity": "warning",
                    "message": f"{high_overstock} items have HIGH overstock risk. Consider promotions or redistribution.",
                }
            )
        if urgent > 0:
            insights.append(
                {
                    "title": "Urgent Reorders",
                    "severity": "critical",
                    "message": f"{urgent} items require URGENT reordering to prevent stockouts.",
                }
            )

    if anomalies is not None and not anomalies.empty:
        spike_count = int((anomalies["anomaly_type"] == "Demand Spike").sum())
        if spike_count > 0:
            insights.append(
                {
                    "title": "Demand Spikes Detected",
                    "severity": "high",
                    "message": f"{spike_count} demand spikes detected. Review inventory levels for affected products.",
                }
            )

    if wh_opt is not None and not wh_opt.empty:
        avg_util = float(wh_opt["utilization_pct"].mean())
        over_90 = int((wh_opt["utilization_pct"] > 90).sum())
        insights.append(
            {
                "title": "Warehouse Utilization",
                "severity": "info",
                "message": f"Average warehouse utilization is {avg_util:.1f}%. {over_90} warehouses are above 90% utilization.",
            }
        )

    if not insights:
        insights.append(
            {
                "title": "Data Summary",
                "severity": "info",
                "message": "All systems operating within normal parameters. No critical alerts at this time.",
            }
        )

    return insights


def render_ai_analyst_page(data: dict) -> None:
    """Render the AI Analyst page."""
    inject_global_css()

    st.markdown(
        f"""
        <div class="brand-header">AI Analyst</div>
        <div class="brand-subtitle">Automated insights and recommendations based on your data</div>
        """,
        unsafe_allow_html=True,
    )

    # Check for data
    if not data or all(v is None or v.empty for v in data.values() if isinstance(v, pd.DataFrame)):
        render_empty_state(
            title="No Data Available",
            message="Load data to generate AI-powered insights and recommendations.",
        )
        return

    # Generate insights
    insights = _generate_insights(data)

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # Display insights
    render_section_header("Automated Insights", subtitle="Data-driven observations and recommendations")

    for insight in insights:
        render_alert(
            message=insight["message"],
            severity=insight["severity"],
            title=insight["title"],
        )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # Interactive analysis
    render_section_header("Interactive Analysis", subtitle="Ask questions about your data")

    question = st.text_input(
        "Enter a question about your retail data:",
        placeholder="e.g., Which products have the highest stockout risk?",
        key="ai_question",
    )

    if question:
        answer = _answer_question(question, data)
        render_alert(
            message=answer,
            severity="info",
            title="AI Analyst Response",
        )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # Key metrics summary
    render_section_header("Executive Summary", subtitle="Quick reference for key business metrics")

    sales = data.get("sales")
    products = data.get("products")
    stores = data.get("stores")
    forecasts = data.get("forecasts")
    inv_intel = data.get("inv_intel")

    metrics = []
    if sales is not None and not sales.empty:
        metrics.append(
            {
                "label": "Total Revenue",
                "value": f"${sales['revenue'].sum():,.0f}",
                "icon": "💰",
            }
        )
    if products is not None and not products.empty:
        metrics.append(
            {
                "label": "Products",
                "value": str(products["product_id"].nunique()),
                "icon": "🏷️",
            }
        )
    if stores is not None and not stores.empty:
        metrics.append(
            {
                "label": "Stores",
                "value": str(stores["store_id"].nunique()),
                "icon": "🏪",
            }
        )
    if forecasts is not None and not forecasts.empty:
        metrics.append(
            {
                "label": "14-Day Forecast",
                "value": f"{forecasts['forecast_demand'].sum():,.0f}",
                "icon": "🔮",
            }
        )
    if inv_intel is not None and not inv_intel.empty:
        metrics.append(
            {
                "label": "Stockout Risks",
                "value": str(int((inv_intel["stockout_risk"] == "HIGH").sum())),
                "icon": "🚨",
            }
        )
        metrics.append(
            {
                "label": "Urgent Reorders",
                "value": str(int((inv_intel["reorder_urgency"] == "URGENT").sum())),
                "icon": "📋",
            }
        )

    if metrics:
        render_kpi_row(metrics, columns=3)


def _answer_question(question: str, data: dict) -> str:
    """Generate a simple answer to a user question based on the data.

    This is a lightweight rule-based response system. For production,
    this would integrate with an LLM or RAG system.
    """
    question_lower = question.lower()

    sales = data.get("sales")
    inv_intel = data.get("inv_intel")
    anomalies = data.get("anomalies")
    forecasts = data.get("forecasts")

    if "stockout" in question_lower or "out of stock" in question_lower:
        if inv_intel is not None and not inv_intel.empty:
            high = int((inv_intel["stockout_risk"] == "HIGH").sum())
            med = int((inv_intel["stockout_risk"] == "MEDIUM").sum())
            return f"There are {high} items at HIGH stockout risk and {med} at MEDIUM risk. Review the Inventory Intelligence page for details and reorder recommendations."
        return "Inventory data is not available."

    if "overstock" in question_lower or "excess" in question_lower:
        if inv_intel is not None and not inv_intel.empty:
            high = int((inv_intel["overstock_risk"] == "HIGH").sum())
            return f"There are {high} items at HIGH overstock risk. Consider running promotions or redistributing excess inventory."
        return "Inventory data is not available."

    if "reorder" in question_lower or "restock" in question_lower:
        if inv_intel is not None and not inv_intel.empty:
            urgent = int((inv_intel["reorder_urgency"] == "URGENT").sum())
            soon = int((inv_intel["reorder_urgency"] == "SOON").sum())
            return f"{urgent} items require URGENT reordering and {soon} items will need reordering soon. Check the Reorder Recommendations section."
        return "Inventory data is not available."

    if "anomal" in question_lower or "spike" in question_lower:
        if anomalies is not None and not anomalies.empty:
            total = len(anomalies)
            spikes = int((anomalies["anomaly_type"] == "Demand Spike").sum())
            return f"{total:,} anomalies detected, including {spikes:,} demand spikes. Visit the Demand Anomalies page for full details."
        return "Anomaly data is not available."

    if "forecast" in question_lower or "demand" in question_lower:
        if forecasts is not None and not forecasts.empty:
            total = float(forecasts["forecast_demand"].sum())
            return f"The 14-day forecast predicts {total:,.0f} total units of demand. Use the Demand Forecast page for product-level predictions."
        return "Forecast data is not available."

    if "revenue" in question_lower or "sales" in question_lower:
        if sales is not None and not sales.empty:
            total = float(sales["revenue"].sum())
            return f"Total recorded revenue is ${total:,.0f}. Check the Business Intelligence page for detailed financial KPIs."
        return "Sales data is not available."

    return (
        "I can help you analyze stockout risks, overstock situations, reorder needs, "
        "anomalies, forecasts, and revenue. Try asking a specific question about one of these topics."
    )
