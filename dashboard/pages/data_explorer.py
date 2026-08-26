"""Data Explorer page for RetailSync AI."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from dashboard.components.ui import (
    COLORS,
    inject_global_css,
    render_data_table,
    render_empty_state,
    render_section_header,
)

logger = logging.getLogger(__name__)


def render_data_explorer_page(data: dict) -> None:
    """Render the data explorer page."""
    inject_global_css()

    st.markdown(
        f"""
        <div class="brand-header">Data Explorer</div>
        <div class="brand-subtitle">Browse and download raw data tables</div>
        """,
        unsafe_allow_html=True,
    )

    table_option = st.selectbox(
        "Select Table",
        [
            "products",
            "stores",
            "suppliers",
            "warehouses",
            "sales",
            "inventory",
            "forecasts",
            "anomalies",
            "inventory_alerts",
        ],
        key="data_explorer_table",
    )

    df = data.get(table_option)
    if df is not None:
        render_data_table(
            df.head(1000) if table_option in ("sales", "inventory") else df,
            title=f"{table_option.replace('_', ' ').title()} Table",
            download_label=f"Download {table_option} (CSV)",
            download_filename=f"{table_option}.csv",
        )

        # Quick stats
        render_section_header("Quick Statistics", subtitle=f"Summary statistics for {table_option}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", f"{len(df):,}")
        with col2:
            st.metric("Columns", str(len(df.columns)))
        with col3:
            st.metric("Missing Values", str(int(df.isnull().sum().sum())))

        st.markdown(f"**Columns:** {', '.join(df.columns.tolist())}")
    else:
        render_empty_state(
            title="Table Not Found",
            message=f"The table '{table_option}' is not available in the current data.",
        )
