"""Segmentation page for RetailSync AI."""

from __future__ import annotations

import logging

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.ui import (
    render_data_table,
    render_empty_state,
    render_pie_chart,
    render_section_header,
)

logger = logging.getLogger(__name__)


def render_segmentation_page(data: dict) -> None:
    """Render the segmentation page."""
    st.markdown(
        """
        <div class="brand-header">Segmentation</div>
        <div class="brand-subtitle">Product, store, and warehouse clustering analysis</div>
        """,
        unsafe_allow_html=True,
    )

    product_segments = data.get("product_segments")
    store_segments = data.get("store_segments")
    warehouse_segments = data.get("warehouse_segments")
    stores = data.get("stores")
    warehouses = data.get("warehouses")
    products = data.get("products")

    segment_type = st.radio(
        "Select Segmentation View",
        ["Products", "Stores", "Warehouses"],
        horizontal=True,
        key="segment_type",
    )

    if segment_type == "Products":
        render_product_segmentation(product_segments, products)
    elif segment_type == "Stores":
        render_store_segmentation(store_segments, stores)
    else:
        render_warehouse_segmentation(warehouse_segments, warehouses)


def render_product_segmentation(df: pd.DataFrame | None, products: pd.DataFrame | None) -> None:
    render_section_header("Product Segmentation", subtitle="K-Means clusters based on sales and demand variability")

    if df is None or df.empty:
        render_empty_state("No Data", "Product segmentation data is not available.")
        return

    label_counts = df["product_cluster_label"].value_counts().reset_index()
    label_counts.columns = ["cluster", "count"]

    col1, col2 = st.columns(2)
    with col1:
        render_pie_chart(
            label_counts,
            values="count",
            names="cluster",
            title="Product Cluster Distribution",
        )
    with col2:
        if "quantity_sold_sum" in df.columns and "demand_cv_28d_mean" in df.columns:
            fig = px.scatter(
                df,
                x="quantity_sold_sum",
                y="demand_cv_28d_mean",
                color="product_cluster_label",
                title="Product Segments (Revenue vs Variability)",
                template="plotly_dark",
                height=400,
            )
            fig.update_layout(
                xaxis_title="Total Quantity Sold",
                yaxis_title="Demand CV (28d Mean)",
            )
            st.plotly_chart(fig, width="stretch")
        else:
            render_empty_state("No Metrics", "Required columns missing for scatter plot.")

    render_data_table(
        df,
        title="Product Segment Details",
        download_label="Download Product Segments (CSV)",
        download_filename="product_segments.csv",
    )


def render_store_segmentation(df: pd.DataFrame | None, stores: pd.DataFrame | None) -> None:
    render_section_header("Store Segmentation", subtitle="Store clusters based on revenue and performance")

    if df is None or df.empty:
        render_empty_state("No Data", "Store segmentation data is not available.")
        return

    label_counts = df["store_cluster_label"].value_counts().reset_index()
    label_counts.columns = ["cluster", "count"]

    col1, col2 = st.columns(2)
    with col1:
        render_pie_chart(
            label_counts,
            values="count",
            names="cluster",
            title="Store Cluster Distribution",
        )
    with col2:
        if stores is not None and not stores.empty and "revenue_sum" in df.columns:
            store_segs = df.merge(stores[["store_id", "city", "state"]], on="store_id", how="left")
            fig = px.bar(
                store_segs,
                x="store_cluster_label",
                y="revenue_sum",
                color="store_cluster_label",
                title="Revenue by Store Cluster",
                template="plotly_dark",
                height=400,
            )
            fig.update_layout(
                xaxis_title="Store Cluster",
                yaxis_title="Total Revenue",
                showlegend=False,
            )
            st.plotly_chart(fig, width="stretch")
        else:
            render_empty_state("No Metrics", "Required data missing for bar chart.")

    render_data_table(
        df,
        title="Store Segment Details",
        download_label="Download Store Segments (CSV)",
        download_filename="store_segments.csv",
    )


def render_warehouse_segmentation(df: pd.DataFrame | None, warehouses: pd.DataFrame | None) -> None:
    render_section_header("Warehouse Segmentation", subtitle="Warehouse clusters based on utilization and capacity")

    if df is None or df.empty:
        render_empty_state("No Data", "Warehouse segmentation data is not available.")
        return

    label_counts = df["warehouse_cluster_label"].value_counts().reset_index()
    label_counts.columns = ["cluster", "count"]

    col1, col2 = st.columns(2)
    with col1:
        render_pie_chart(
            label_counts,
            values="count",
            names="cluster",
            title="Warehouse Cluster Distribution",
        )
    with col2:
        if warehouses is not None and not warehouses.empty and "revenue_sum" in df.columns:
            wh_segs = df.merge(warehouses[["warehouse_id", "city", "state"]], on="warehouse_id", how="left")
            fig = px.bar(
                wh_segs,
                x="warehouse_cluster_label",
                y="revenue_sum",
                color="warehouse_cluster_label",
                title="Revenue by Warehouse Cluster",
                template="plotly_dark",
                height=400,
            )
            fig.update_layout(
                xaxis_title="Warehouse Cluster",
                yaxis_title="Total Revenue",
                showlegend=False,
            )
            st.plotly_chart(fig, width="stretch")
        else:
            render_empty_state("No Metrics", "Required data missing for bar chart.")

    render_data_table(
        df,
        title="Warehouse Segment Details",
        download_label="Download Warehouse Segments (CSV)",
        download_filename="warehouse_segments.csv",
    )
