"""Reusable UI components for RetailSync AI dashboard.

This module provides a consistent design system for all dashboard pages,
including KPI cards, section headers, alerts, charts, tables, and filters.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

logger = logging.getLogger(__name__)


# ============================================================
# DESIGN TOKENS
# ============================================================

COLORS = {
    "primary": "#3b82f6",
    "primary_dark": "#1d4ed8",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "info": "#06b6d4",
    "surface": "#1e293b",
    "surface_alt": "#0f172a",
    "text": "#f8fafc",
    "text_muted": "#94a3b8",
    "border": "#334155",
    "gradient_start": "#1e293b",
    "gradient_end": "#0f172a",
}

SEVERITY_CONFIG = {
    "critical": {"color": "#ef4444", "bg": "rgba(239, 68, 68, 0.1)", "icon": "🔴"},
    "high": {"color": "#f97316", "bg": "rgba(249, 115, 22, 0.1)", "icon": "🟠"},
    "medium": {"color": "#f59e0b", "bg": "rgba(245, 158, 11, 0.1)", "icon": "🟡"},
    "low": {"color": "#10b981", "bg": "rgba(16, 185, 129, 0.1)", "icon": "🟢"},
    "info": {"color": "#3b82f6", "bg": "rgba(59, 130, 246, 0.1)", "icon": "ℹ️"},
    "success": {"color": "#10b981", "bg": "rgba(16, 185, 129, 0.1)", "icon": "✅"},
    "warning": {"color": "#f59e0b", "bg": "rgba(245, 158, 11, 0.1)", "icon": "⚠️"},
    "error": {"color": "#ef4444", "bg": "rgba(239, 68, 68, 0.1)", "icon": "❌"},
}

DEFAULT_CHART_TEMPLATE = "plotly_dark"


# ============================================================
# GLOBAL CSS
# ============================================================

def inject_global_css() -> None:
    """Inject the global design system CSS."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        :root {{
            --color-primary: {COLORS['primary']};
            --color-success: {COLORS['success']};
            --color-warning: {COLORS['warning']};
            --color-danger: {COLORS['danger']};
            --color-surface: {COLORS['surface']};
            --color-surface-alt: {COLORS['surface_alt']};
            --color-text: {COLORS['text']};
            --color-text-muted: {COLORS['text_muted']};
            --color-border: {COLORS['border']};
        }}

        * {{
            font-family: 'Inter', sans-serif;
        }}

        .stApp {{
            background-color: {COLORS['surface_alt']};
            color: {COLORS['text']};
        }}

        .stSidebar {{
            background-color: {COLORS['surface']} !important;
        }}

        /* Metric Cards */
        .kpi-card {{
            background: linear-gradient(135deg, {COLORS['surface']} 0%, {COLORS['surface_alt']} 100%);
            border: 1px solid {COLORS['border']};
            border-radius: 12px;
            padding: 20px;
            margin: 8px 0;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .kpi-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        }}
        .kpi-label {{
            font-size: 0.75rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: {COLORS['text_muted']};
            margin-bottom: 8px;
        }}
        .kpi-value {{
            font-size: 1.75rem;
            font-weight: 700;
            color: {COLORS['text']};
            line-height: 1.2;
        }}
        .kpi-delta {{
            font-size: 0.8rem;
            font-weight: 500;
            margin-top: 6px;
        }}
        .kpi-delta.positive {{
            color: {COLORS['success']};
        }}
        .kpi-delta.negative {{
            color: {COLORS['danger']};
        }}
        .kpi-delta.neutral {{
            color: {COLORS['text_muted']};
        }}

        /* Section Headers */
        .section-header {{
            font-size: 1.25rem;
            font-weight: 600;
            color: {COLORS['text']};
            margin-top: 1.5rem;
            margin-bottom: 0.75rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid {COLORS['border']};
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        /* Alert Boxes */
        .alert-box {{
            padding: 14px 18px;
            border-radius: 10px;
            margin: 10px 0;
            border-left: 4px solid;
            display: flex;
            align-items: flex-start;
            gap: 10px;
        }}
        .alert-box.critical {{
            background: rgba(239, 68, 68, 0.08);
            border-color: {COLORS['danger']};
        }}
        .alert-box.high {{
            background: rgba(249, 115, 22, 0.08);
            border-color: #f97316;
        }}
        .alert-box.medium {{
            background: rgba(245, 158, 11, 0.08);
            border-color: {COLORS['warning']};
        }}
        .alert-box.low {{
            background: rgba(16, 185, 129, 0.08);
            border-color: {COLORS['success']};
        }}
        .alert-box.info {{
            background: rgba(59, 130, 246, 0.08);
            border-color: {COLORS['primary']};
        }}
        .alert-box.success {{
            background: rgba(16, 185, 129, 0.08);
            border-color: {COLORS['success']};
        }}
        .alert-box.warning {{
            background: rgba(245, 158, 11, 0.08);
            border-color: {COLORS['warning']};
        }}
        .alert-box.error {{
            background: rgba(239, 68, 68, 0.08);
            border-color: {COLORS['danger']};
        }}
        .alert-title {{
            font-weight: 600;
            font-size: 0.9rem;
            margin-bottom: 4px;
        }}
        .alert-message {{
            font-size: 0.85rem;
            color: {COLORS['text_muted']};
            line-height: 1.5;
        }}

        /* Brand Header */
        .brand-header {{
            font-size: 2rem;
            font-weight: 700;
            color: {COLORS['primary']};
            margin-bottom: 0.25rem;
        }}
        .brand-subtitle {{
            font-size: 1rem;
            color: {COLORS['text_muted']};
            margin-bottom: 1.5rem;
        }}

        /* Badge */
        .badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }}
        .badge-critical {{
            background: rgba(239, 68, 68, 0.15);
            color: {COLORS['danger']};
        }}
        .badge-high {{
            background: rgba(249, 115, 22, 0.15);
            color: #f97316;
        }}
        .badge-medium {{
            background: rgba(245, 158, 11, 0.15);
            color: {COLORS['warning']};
        }}
        .badge-low {{
            background: rgba(16, 185, 129, 0.15);
            color: {COLORS['success']};
        }}

        /* Divider */
        .custom-divider {{
            border: none;
            border-top: 1px solid {COLORS['border']};
            margin: 1.5rem 0;
        }}

        /* Scrollbar */
        ::-webkit-scrollbar {{
            width: 6px;
            height: 6px;
        }}
        ::-webkit-scrollbar-track {{
            background: {COLORS['surface_alt']};
        }}
        ::-webkit-scrollbar-thumb {{
            background: {COLORS['border']};
            border-radius: 3px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# KPI CARDS
# ============================================================

def render_kpi_card(
    label: str,
    value: str,
    delta: str | None = None,
    delta_type: str = "neutral",
    help_text: str | None = None,
    icon: str | None = None,
    columns: int = 3,
) -> None:
    """Render a polished KPI card.

    Args:
        label: KPI label text.
        value: KPI value text.
        delta: Optional delta/trend text.
        delta_type: One of 'positive', 'negative', 'neutral'.
        help_text: Optional help text shown as tooltip.
        icon: Optional emoji icon.
        columns: Number of columns for the metric layout.
    """
    icon_html = f"<span style='font-size:1.2rem;margin-right:6px;'>{icon}</span>" if icon else ""
    delta_class = delta_type if delta_type in ("positive", "negative", "neutral") else "neutral"
    delta_html = ""
    if delta:
        delta_html = f'<div class="kpi-delta {delta_class}">{delta}</div>'

    card_html = f"""
    <div class="kpi-card">
        <div class="kpi-label">{icon_html}{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
    if help_text:
        st.caption(help_text)


def render_kpi_row(metrics: list[dict], columns: int = 4) -> None:
    """Render a row of KPI cards.

    Args:
        metrics: List of dicts with keys: label, value, delta, delta_type, help_text, icon.
        columns: Number of columns.
    """
    cols = st.columns(columns)
    for idx, metric in enumerate(metrics):
        with cols[idx % columns]:
            render_kpi_card(**metric)


# ============================================================
# SECTION HEADERS
# ============================================================

def render_section_header(
    title: str,
    subtitle: str | None = None,
    badge: str | None = None,
    badge_color: str = "low",
) -> None:
    """Render a consistent section header.

    Args:
        title: Section title.
        subtitle: Optional subtitle text.
        badge: Optional badge text.
        badge_color: Badge color variant.
    """
    badge_html = ""
    if badge:
        badge_html = f'<span class="badge badge-{badge_color}">{badge}</span>'

    subtitle_html = ""
    if subtitle:
        subtitle_html = f'<div style="font-size:0.85rem;color:{COLORS["text_muted"]};margin-top:2px;">{subtitle}</div>'

    st.markdown(
        f"""
        <div class="section-header">
            {title}
            {badge_html}
        </div>
        {subtitle_html}
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ALERT BOXES
# ============================================================

def render_alert(
    message: str,
    severity: str = "info",
    title: str | None = None,
) -> None:
    """Render an alert box with severity styling.

    Args:
        message: Alert message text.
        severity: One of 'critical', 'high', 'medium', 'low', 'info', 'success', 'warning', 'error'.
        title: Optional alert title.
    """
    severity = severity.lower()
    if severity not in SEVERITY_CONFIG:
        severity = "info"

    cfg = SEVERITY_CONFIG[severity]
    title_html = f'<div class="alert-title">{cfg["icon"]} {title}</div>' if title else ""
    st.markdown(
        f"""
        <div class="alert-box {severity}">
            <div style="flex:1;">
                {title_html}
                <div class="alert-message">{message}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_severity_badge(severity: str) -> str:
    """Return HTML for a severity badge.

    Args:
        severity: Severity level.

    Returns:
        HTML string for the badge.
    """
    severity = severity.lower()
    if severity not in SEVERITY_CONFIG:
        severity = "info"
    return f'<span class="badge badge-{severity}">{severity.upper()}</span>'


# ============================================================
# EMPTY & ERROR STATES
# ============================================================

def render_empty_state(
    title: str = "No Data Available",
    message: str = "There is no data to display for the current selection.",
    icon: str = "📭",
) -> None:
    """Render an empty state message.

    Args:
        title: Title text.
        message: Descriptive message.
        icon: Emoji icon.
    """
    st.markdown(
        f"""
        <div style="text-align:center;padding:3rem 1rem;color:{COLORS['text_muted']};">
            <div style="font-size:3rem;margin-bottom:1rem;">{icon}</div>
            <div style="font-size:1.1rem;font-weight:600;color:{COLORS['text']};">{title}</div>
            <div style="font-size:0.9rem;margin-top:0.5rem;">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_error_state(
    title: str = "Something went wrong",
    message: str = "An unexpected error occurred. Please try again later.",
    details: str | None = None,
) -> None:
    """Render a user-friendly error state.

    Args:
        title: Error title.
        message: User-friendly message.
        details: Optional technical details for debugging.
    """
    details_html = ""
    if details:
        details_html = f"""
        <details style="margin-top:10px;">
            <summary style="cursor:pointer;font-size:0.8rem;color:{COLORS['text_muted']};">Technical Details</summary>
            <pre style="font-size:0.75rem;background:{COLORS['surface_alt']};padding:10px;border-radius:6px;overflow-x:auto;">{details}</pre>
        </details>
        """
    st.markdown(
        f"""
        <div class="alert-box error">
            <div style="flex:1;">
                <div class="alert-title">❌ {title}</div>
                <div class="alert-message">{message}</div>
                {details_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    logger.error("%s: %s | Details: %s", title, message, details)


# ============================================================
# CHARTS
# ============================================================

def render_line_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str | None = None,
    color: str | None = None,
    line_width: int = 2,
    height: int = 400,
    markers: bool = False,
) -> None:
    """Render a consistent line chart.

    Args:
        df: DataFrame.
        x: X-axis column.
        y: Y-axis column.
        title: Chart title.
        color: Line color (hex).
        line_width: Line width.
        height: Chart height.
        markers: Whether to show markers.
    """
    if df.empty:
        render_empty_state("No Data", "No data available for this chart.")
        return

    fig = go.Figure()
    if color:
        fig.add_trace(
            go.Scatter(
                x=df[x],
                y=df[y],
                mode="lines+markers" if markers else "lines",
                name=title or y,
                line={"color": color, "width": line_width},
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=df[x],
                y=df[y],
                mode="lines+markers" if markers else "lines",
                name=title or y,
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title=x.replace("_", " ").title(),
        yaxis_title=y.replace("_", " ").title(),
        template=DEFAULT_CHART_TEMPLATE,
        height=height,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str | None = None,
    color: str | None = None,
    orientation: str = "v",
    height: int = 400,
    text_auto: bool = False,
) -> None:
    """Render a consistent bar chart.

    Args:
        df: DataFrame.
        x: X-axis column.
        y: Y-axis column.
        title: Chart title.
        color: Color column or hex color.
        orientation: 'v' or 'h'.
        height: Chart height.
        text_auto: Whether to show values on bars.
    """
    if df.empty:
        render_empty_state("No Data", "No data available for this chart.")
        return

    fig = px.bar(
        df,
        x=x,
        y=y,
        title=title,
        template=DEFAULT_CHART_TEMPLATE,
        color=color,
        orientation=orientation,
        height=height,
        text_auto=text_auto,
    )
    fig.update_layout(
        xaxis_title=x.replace("_", " ").title(),
        yaxis_title=y.replace("_", " ").title(),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_pie_chart(
    df: pd.DataFrame,
    values: str,
    names: str,
    title: str | None = None,
    color_map: dict | None = None,
    height: int = 400,
) -> None:
    """Render a consistent pie chart.

    Args:
        df: DataFrame.
        values: Values column.
        names: Names column.
        title: Chart title.
        color_map: Optional color mapping dict.
        height: Chart height.
    """
    if df.empty:
        render_empty_state("No Data", "No data available for this chart.")
        return

    fig = px.pie(
        df,
        values=values,
        names=names,
        title=title,
        template=DEFAULT_CHART_TEMPLATE,
        color=color_map,
        height=height,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)


def render_scatter_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str | None = None,
    color: str | None = None,
    size: str | None = None,
    hover_data: list | None = None,
    height: int = 400,
) -> None:
    """Render a consistent scatter chart.

    Args:
        df: DataFrame.
        x: X-axis column.
        y: Y-axis column.
        title: Chart title.
        color: Color column.
        size: Size column.
        hover_data: Additional hover data columns.
        height: Chart height.
    """
    if df.empty:
        render_empty_state("No Data", "No data available for this chart.")
        return

    fig = px.scatter(
        df,
        x=x,
        y=y,
        title=title,
        template=DEFAULT_CHART_TEMPLATE,
        color=color,
        size=size,
        hover_data=hover_data or [],
        height=height,
    )
    fig.update_layout(
        xaxis_title=x.replace("_", " ").title(),
        yaxis_title=y.replace("_", " ").title(),
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# DATA TABLES
# ============================================================

def render_data_table(
    df: pd.DataFrame,
    title: str | None = None,
    download_label: str | None = None,
    download_filename: str | None = None,
    height: int | None = None,
    column_config: dict | None = None,
) -> None:
    """Render a styled data table with optional download.

    Args:
        df: DataFrame to display.
        title: Optional section title.
        download_label: Optional download button label.
        download_filename: Optional download filename.
        height: Optional fixed height.
        column_config: Optional column configuration dict for st.dataframe.
    """
    if title:
        render_section_header(title)

    if df is None or df.empty:
        render_empty_state("No Data", "No records match the current filters.")
        return

    st.dataframe(
        df,
        use_container_width=True,
        height=height,
        column_config=column_config,
    )

    if download_label and download_filename:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=download_label,
            data=csv,
            file_name=download_filename,
            mime="text/csv",
        )


# ============================================================
# FILTERS
# ============================================================

def render_filter_sidebar(
    title: str = "Filters",
    product_options: list | None = None,
    category_options: list | None = None,
    store_options: list | None = None,
    warehouse_options: list | None = None,
    date_range: tuple | None = None,
) -> dict:
    """Render standardized filters in the sidebar.

    Args:
        title: Filter section title.
        product_options: List of product IDs.
        category_options: List of categories.
        store_options: List of store IDs.
        warehouse_options: List of warehouse IDs.
        date_range: Default date range tuple.

    Returns:
        Dict of selected filter values.
    """
    with st.sidebar.expander(title, expanded=True):
        selected_product = st.selectbox(
            "Product",
            ["All"] + (product_options or []),
            key="filter_product",
        )
        selected_category = st.selectbox(
            "Category",
            ["All"] + (category_options or []),
            key="filter_category",
        )
        selected_store = st.selectbox(
            "Store",
            ["All"] + (store_options or []),
            key="filter_store",
        )
        selected_warehouse = st.selectbox(
            "Warehouse",
            ["All"] + (warehouse_options or []),
            key="filter_warehouse",
        )

        if date_range:
            selected_dates = st.date_input(
                "Date Range",
                value=date_range,
                key="filter_dates",
            )
        else:
            selected_dates = None

    return {
        "product": selected_product if selected_product != "All" else None,
        "category": selected_category if selected_category != "All" else None,
        "store": selected_store if selected_store != "All" else None,
        "warehouse": selected_warehouse if selected_warehouse != "All" else None,
        "date_range": selected_dates,
    }


def apply_filters(
    df: pd.DataFrame,
    filters: dict,
    product_col: str = "product_id",
    category_col: str = "category",
    store_col: str = "store_id",
    warehouse_col: str = "warehouse_id",
    date_col: str = "date",
) -> pd.DataFrame:
    """Apply standard filters to a DataFrame.

    Args:
        df: DataFrame to filter.
        filters: Dict from render_filter_sidebar.
        product_col: Product column name.
        category_col: Category column name.
        store_col: Store column name.
        warehouse_col: Warehouse column name.
        date_col: Date column name.

    Returns:
        Filtered DataFrame.
    """
    if df is None or df.empty:
        return df

    filtered = df.copy()

    if filters.get("product") and product_col in filtered.columns:
        filtered = filtered[filtered[product_col] == filters["product"]]
    if filters.get("category") and category_col in filtered.columns:
        filtered = filtered[filtered[category_col] == filters["category"]]
    if filters.get("store") and store_col in filtered.columns:
        filtered = filtered[filtered[store_col] == filters["store"]]
    if filters.get("warehouse") and warehouse_col in filtered.columns:
        filtered = filtered[filtered[warehouse_col] == filters["warehouse"]]
    if filters.get("date_range") and date_col in filtered.columns:
        start, end = filters["date_range"]
        if start and end:
            filtered = filtered[
                (filtered[date_col].dt.date >= start)
                & (filtered[date_col].dt.date <= end)
            ]

    return filtered


# ============================================================
# DOWNLOAD UTILITIES
# ============================================================

def render_download_section(
    data: dict[str, pd.DataFrame],
    title: str = "Download Data",
) -> None:
    """Render a download section with multiple CSV download buttons.

    Args:
        data: Dict of {label: DataFrame}.
        title: Section title.
    """
    render_section_header(title)
    cols = st.columns(min(len(data), 4))
    for idx, (label, df) in enumerate(data.items()):
        with cols[idx % 4]:
            if df is not None and not df.empty:
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"Download {label}",
                    data=csv,
                    file_name=f"{label.lower().replace(' ', '_')}.csv",
                    mime="text/csv",
                    key=f"download_{label.lower().replace(' ', '_')}",
                )
            else:
                st.caption(f"No {label.lower()} data")


# ============================================================
# TABS
# ============================================================

def render_tab_content(tabs: list[dict], default_index: int = 0) -> None:
    """Render tabbed content.

    Args:
        tabs: List of dicts with keys: label, content (callable).
        default_index: Default active tab index.
    """
    if not tabs:
        return

    tab_labels = [t["label"] for t in tabs]
    selected = st.tabs(tab_labels)
    for idx, tab in enumerate(selected):
        with tab:
            if "content" in tabs[idx] and callable(tabs[idx]["content"]):
                tabs[idx]["content"]()


# ============================================================
# PROGRESS / LOADING
# ============================================================

def render_progress_bar(label: str, value: float, max_value: float = 100.0) -> None:
    """Render a progress bar.

    Args:
        label: Label text.
        value: Current value.
        max_value: Maximum value.
    """
    pct = min(value / max_value, 1.0)
    st.markdown(f"**{label}**: {value:.1f}%")
    st.progress(pct)


# ============================================================
# SIDEBAR BRANDING
# ============================================================

def render_sidebar_branding() -> None:
    """Render the sidebar branding block."""
    st.sidebar.markdown(
        f"""
        <div style="padding: 0.5rem 0; text-align: center;">
            <div style="font-size: 1.8rem;">🏪</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: {COLORS['primary']};">RetailSync AI</div>
            <div style="font-size: 0.75rem; color: {COLORS['text_muted']};">Enterprise Analytics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_footer() -> None:
    """Render the sidebar footer."""
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
        <div style="font-size: 0.75rem; color: {COLORS['text_muted']}; text-align: center;">
            RetailSync AI v2.0<br>
            © 2026 RetailSync AI
        </div>
        """,
        unsafe_allow_html=True,
    )
