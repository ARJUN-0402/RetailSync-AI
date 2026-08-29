"""AI Analyst page for RetailSync AI."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from dashboard.components.ui import (
    render_alert,
    render_empty_state,
    render_section_header,
)
from src.ai_analyst.config import AIAnalystConfig
from src.ai_analyst.orchestrator import ask
from src.ai_analyst.prompts import format_sources

logger = logging.getLogger(__name__)

SUGGESTED_QUESTIONS = [
    "Why are stockouts increasing?",
    "Which products should I reorder?",
    "Explain this anomaly.",
    "Which store has the highest inventory risk?",
    "Why is demand expected to increase next week?",
    "Which products are overstocked?",
    "What are the biggest inventory risks right now?",
    "What is the 14-day demand forecast?",
    "Which warehouses are near capacity?",
    "What do the executive KPIs show?",
]


def _init_session_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if len(st.session_state.chat_history) > 50:
        st.session_state.chat_history = st.session_state.chat_history[-50:]
    if len(st.session_state.chat_history) > 50:
        st.session_state.chat_history = st.session_state.chat_history[-50:]


def render_ai_analyst_page(data: dict) -> None:
    st.markdown(
        """
        <div class="brand-header">AI Analyst</div>
        <div class="brand-subtitle">Ask natural-language questions about your retail data</div>
        """,
        unsafe_allow_html=True,
    )

    config = AIAnalystConfig.from_env()

    if not data or all(v is None or v.empty for v in data.values() if isinstance(v, pd.DataFrame)):
        render_empty_state(
            title="No Data Available",
            message="Load data to use the AI Analyst.",
        )
        return

    if not config.is_configured:
        render_alert(
            message=(
                "AI Analyst is running in **offline mode** because no LLM API key is configured. "
                "Set `RETAILSYNC_AI_API_KEY` to enable full AI-powered answers. "
                "The dashboard and all data features remain fully functional."
            ),
            severity="info",
            title="Offline Mode",
        )

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    render_section_header("Suggested Questions", subtitle="Click a question to ask the AI Analyst")
    cols = st.columns(2)
    for idx, question in enumerate(SUGGESTED_QUESTIONS):
        with cols[idx % 2]:
            if st.button(question, key=f"suggest_{idx}", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": question})
                with st.spinner("Analyzing..."):
                    response = ask(question, config=config)
                answer = response.get("answer", "No answer generated.")
                sources = response.get("sources", [])
                error = response.get("error")
                if error:
                    answer = f"**Error:** {error}\n\n{answer}"
                source_text = format_sources(sources) if sources else ""
                if source_text:
                    answer += f"\n\n*{source_text}*"
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.rerun()

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    render_section_header("Chat", subtitle="Ask any question about your retail data")

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about stockouts, forecasts, inventory, anomalies..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                response = ask(prompt, config=config)
            answer = response.get("answer", "No answer generated.")
            sources = response.get("sources", [])
            error = response.get("error")
            if error:
                answer = f"**Error:** {error}\n\n{answer}"
            source_text = format_sources(sources) if sources else ""
            if source_text:
                answer += f"\n\n*{source_text}*"
            st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

    if st.session_state.chat_history:
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    render_section_header("Available Data", subtitle="Quick reference for the AI Analyst's data sources")
    metrics = []
    sales = data.get("sales")
    products = data.get("products")
    stores = data.get("stores")
    forecasts = data.get("forecasts")
    inv_intel = data.get("inv_intel")
    anomalies = data.get("anomalies")
    wh_opt = data.get("wh_opt")

    if sales is not None and not sales.empty:
        metrics.append({"label": "Total Revenue", "value": f"${sales['revenue'].sum():,.0f}", "icon": "💰"})
    if products is not None and not products.empty:
        metrics.append({"label": "Products", "value": str(products["product_id"].nunique()), "icon": "🏷️"})
    if stores is not None and not stores.empty:
        metrics.append({"label": "Stores", "value": str(stores["store_id"].nunique()), "icon": "🏪"})
    if forecasts is not None and not forecasts.empty:
        metrics.append({"label": "14-Day Forecast", "value": f"{forecasts['forecast_demand'].sum():,.0f}", "icon": "🔮"})
    if inv_intel is not None and not inv_intel.empty:
        metrics.append({"label": "Stockout Risks", "value": str(int((inv_intel["stockout_risk"] == "HIGH").sum())), "icon": "🚨"})
        metrics.append({"label": "Urgent Reorders", "value": str(int((inv_intel["reorder_urgency"] == "URGENT").sum())), "icon": "📋"})
        metrics.append({"label": "Overstock Risks", "value": str(int((inv_intel["overstock_risk"] == "HIGH").sum())), "icon": "📦"})
    if anomalies is not None and not anomalies.empty:
        metrics.append({"label": "Anomalies", "value": str(len(anomalies)), "icon": "⚠️"})
    if wh_opt is not None and not wh_opt.empty:
        metrics.append({"label": "Warehouses", "value": str(len(wh_opt)), "icon": "🏭"})

    if metrics:
        from dashboard.components.ui import render_kpi_row
        render_kpi_row(metrics, columns=3)
