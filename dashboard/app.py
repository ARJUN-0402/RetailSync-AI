import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
import joblib

# Page config
st.set_page_config(
    page_title="RetailSync AI",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark enterprise theme
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stSidebar {
        background-color: #1a1d23;
    }
    .metric-card {
        background-color: #1a1d23;
        border: 1px solid #2d3142;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .header-title {
        color: #00d4ff;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .section-title {
        color: #00d4ff;
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #00d4ff;
        padding-bottom: 0.5rem;
    }
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        margin: 2px;
    }
    .badge-high {
        background-color: #ff4757;
        color: white;
    }
    .badge-medium {
        background-color: #ffa502;
        color: black;
    }
    .badge-low {
        background-color: #2ed573;
        color: black;
    }
    .info-box {
        background-color: #1e2a3a;
        border-left: 4px solid #00d4ff;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        "demand_forecaster": "models/demand_forecaster.pkl",
        "product_clusterer": "models/product_clusterer.pkl",
        "store_clusterer": "models/store_clusterer.pkl",
        "warehouse_clusterer": "models/warehouse_clusterer.pkl",
    }
    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models

models = load_models()

# Database connection
@st.cache_resource
def get_engine():
    return create_engine("sqlite:///database/retailsync.db")

engine = get_engine()

# Load data
@st.cache_data(ttl=300)
def load_data():
    features = pd.read_csv("data/processed/features_daily.csv", parse_dates=["date"])
    forecasts = pd.read_csv("data/processed/forecasts_next_14d.csv", parse_dates=["date"])
    inv_intel = pd.read_csv("data/processed/inventory_intelligence.csv")
    anomalies = pd.read_csv("data/processed/anomalies.csv", parse_dates=["date"])
    product_segments = pd.read_csv("data/processed/product_segments.csv")
    store_segments = pd.read_csv("data/processed/store_segments.csv")
    warehouse_segments = pd.read_csv("data/processed/warehouse_segments.csv")
    wh_opt = pd.read_csv("data/processed/warehouse_optimization.csv")
    
    # Load from database
    products = pd.read_sql("SELECT * FROM products", engine)
    stores = pd.read_sql("SELECT * FROM stores", engine)
    suppliers = pd.read_sql("SELECT * FROM suppliers", engine)
    warehouses = pd.read_sql("SELECT * FROM warehouses", engine)
    sales = pd.read_sql("SELECT * FROM sales", engine)
    inventory = pd.read_sql("SELECT * FROM inventory", engine)
    inventory_alerts = pd.read_sql("SELECT * FROM inventory_alerts", engine)
    anomaly_flags = pd.read_sql("SELECT * FROM anomaly_flags", engine)
    
    return {
        "features": features,
        "forecasts": forecasts,
        "inv_intel": inv_intel,
        "anomalies": anomalies,
        "product_segments": product_segments,
        "store_segments": store_segments,
        "warehouse_segments": warehouse_segments,
        "wh_opt": wh_opt,
        "products": products,
        "stores": stores,
        "suppliers": suppliers,
        "warehouses": warehouses,
        "sales": sales,
        "inventory": inventory,
        "inventory_alerts": inventory_alerts,
        "anomaly_flags": anomaly_flags,
    }

with st.spinner("Loading data..."):
    data = load_data()

# Sidebar navigation
st.sidebar.title("🏪 RetailSync AI")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Executive Overview",
        "🔮 Demand Forecast",
        "📦 Inventory Intelligence",
        "⚠️ Demand Anomalies",
        "🎯 Segmentation",
        "🏭 Warehouse Intelligence",
        "📁 Data Explorer",
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Technology Stack")
st.sidebar.markdown("- Python")
st.sidebar.markdown("- Pandas / NumPy")
st.sidebar.markdown("- Scikit-learn / XGBoost")
st.sidebar.markdown("- Plotly")
st.sidebar.markdown("- Streamlit")
st.sidebar.markdown("- SQLite")

# ============================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ============================================================
if page == "📊 Executive Overview":
    st.markdown('<p class="header-title">RetailSync AI</p>', unsafe_allow_html=True)
    st.markdown("### AI-Powered Retail Demand Forecasting & Supply Chain Intelligence")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = data["sales"]["revenue"].sum()
        st.metric("Total Revenue", f"${total_revenue:,.0f}")
    
    with col2:
        total_quantity = data["sales"]["quantity_sold"].sum()
        st.metric("Units Sold", f"{total_quantity:,}")
    
    with col3:
        total_products = data["products"]["product_id"].nunique()
        st.metric("Products", total_products)
    
    with col4:
        total_stores = data["stores"]["store_id"].nunique()
        st.metric("Stores", total_stores)
    
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        forecast_demand = data["forecasts"]["forecast_demand"].sum()
        st.metric("14-Day Forecast", f"{forecast_demand:,.0f} units")
    
    with col6:
        high_stockout = (data["inv_intel"]["stockout_risk"] == "HIGH").sum()
        st.metric("Stockout Risks", high_stockout)
    
    with col7:
        total_anomalies = len(data["anomalies"])
        st.metric("Anomalies Detected", f"{total_anomalies:,}")
    
    with col8:
        avg_util = data["wh_opt"]["utilization_pct"].mean()
        st.metric("Avg Warehouse Utilization", f"{avg_util:.1f}%")
    
    st.markdown("---")
    
    # Model Status
    st.markdown('<p class="section-title">ML Models Status</p>', unsafe_allow_html=True)
    
    model_col1, model_col2, model_col3, model_col4 = st.columns(4)
    with model_col1:
        if "demand_forecaster" in models:
            st.success("✓ Demand Forecaster Loaded")
        else:
            st.error("✗ Demand Forecaster Missing")
    with model_col2:
        if "product_clusterer" in models:
            st.success("✓ Product Clusterer Loaded")
        else:
            st.error("✗ Product Clusterer Missing")
    with model_col3:
        if "store_clusterer" in models:
            st.success("✓ Store Clusterer Loaded")
        else:
            st.error("✗ Store Clusterer Missing")
    with model_col4:
        if "warehouse_clusterer" in models:
            st.success("✓ Warehouse Clusterer Loaded")
        else:
            st.error("✗ Warehouse Clusterer Missing")
    
    st.markdown("---")
    
    # Business Problem Section
    st.markdown('<p class="section-title">Business Problem</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Retailers face significant challenges in managing supply chains:
    <ul>
        <li><strong>Stockouts</strong> lead to lost sales and customer dissatisfaction</li>
        <li><strong>Overstock</strong> ties up capital and increases holding costs</li>
        <li><strong>Demand volatility</strong> makes inventory planning difficult</li>
        <li><strong>Lack of visibility</strong> across warehouses and stores</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Engineering Approach Section
    st.markdown('<p class="section-title">Engineering Approach</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    RetailSync AI uses a multi-component ML pipeline:
    <ol>
        <li><strong>Demand Forecasting</strong> — Time-series models predict future demand</li>
        <li><strong>Inventory Intelligence</strong> — Rule-based risk detection for stockout/overstock</li>
        <li><strong>Anomaly Detection</strong> — Ensemble methods identify unusual demand patterns</li>
        <li><strong>Segmentation</strong> — K-Means clustering groups products, stores, warehouses</li>
        <li><strong>Warehouse Optimization</strong> — Utilization analytics and capacity recommendations</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Measurable Outcomes
    st.markdown('<p class="section-title">Measurable Outcomes & Impact</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Forecasting")
        st.markdown(f"- **MAE:** 4.09 units")
        st.markdown(f"- **RMSE:** 6.65 units")
        st.markdown(f"- **14-day forecast:** {forecast_demand:,.0f} units")
    
    with col2:
        st.markdown("### Inventory")
        st.markdown(f"- **Stockout risks:** {high_stockout}")
        st.markdown(f"- **Overstock risks:** {(data['inv_intel']['overstock_risk'] == 'HIGH').sum()}")
        st.markdown(f"- **Urgent reorders:** {(data['inv_intel']['reorder_urgency'] == 'URGENT').sum()}")
    
    with col3:
        st.markdown("### Anomalies")
        st.markdown(f"- **Total anomalies:** {total_anomalies:,}")
        st.markdown(f"- **Demand spikes:** {(data['anomalies']['anomaly_type'] == 'Demand Spike').sum():,}")
        st.markdown(f"- **Unusual patterns:** {(data['anomalies']['anomaly_type'] == 'Unusual Pattern').sum():,}")

# ============================================================
# PAGE 2: DEMAND FORECAST
# ============================================================
elif page == "🔮 Demand Forecast":
    st.markdown('<p class="header-title">Demand Forecast</p>', unsafe_allow_html=True)
    
    # Forecast controls
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_product = st.selectbox("Select Product", data["products"]["product_id"].tolist())
    with col2:
        selected_store = st.selectbox("Select Store", data["stores"]["store_id"].tolist())
    with col3:
        forecast_horizon = st.slider("Forecast Horizon (days)", 1, 14, 14)
    
    # Filter forecasts
    forecast_filtered = data["forecasts"][
        (data["forecasts"]["product_id"] == selected_product) &
        (data["forecasts"]["store_id"] == selected_store)
    ]
    
    if not forecast_filtered.empty:
        st.markdown(f"### Forecast: {selected_product} at {selected_store}")
        
        # Forecast chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_filtered["date"],
            y=forecast_filtered["forecast_demand"],
            mode="lines+markers",
            name="Forecasted Demand",
            line=dict(color="#00d4ff", width=3)
        ))
        fig.update_layout(
            title="Demand Forecast",
            xaxis_title="Date",
            yaxis_title="Quantity",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast stats
        col1, col2, col3 = st.columns(3)
        with col1:
            total_demand = forecast_filtered["forecast_demand"].sum()
            st.metric("Total Forecasted Demand", f"{total_demand:,.0f}")
        with col2:
            total_revenue = forecast_filtered["forecast_revenue"].sum()
            st.metric("Total Forecasted Revenue", f"${total_revenue:,.2f}")
        with col3:
            avg_demand = forecast_filtered["forecast_demand"].mean()
            st.metric("Avg Daily Demand", f"{avg_demand:,.1f}")
    else:
        st.warning("No forecast data available for selected product-store combination.")
    
    # Historical vs Forecast
    st.markdown('<p class="section-title">Historical vs Forecast</p>', unsafe_allow_html=True)
    
    hist = data["features"][
        (data["features"]["product_id"] == selected_product) &
        (data["features"]["store_id"] == selected_store)
    ].tail(30)
    
    if not hist.empty and not forecast_filtered.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist["date"],
            y=hist["quantity_sold"],
            mode="lines",
            name="Historical Demand",
            line=dict(color="#ff6b6b")
        ))
        fig.add_trace(go.Scatter(
            x=forecast_filtered["date"],
            y=forecast_filtered["forecast_demand"],
            mode="lines+markers",
            name="Forecasted Demand",
            line=dict(color="#00d4ff", dash="dash")
        ))
        fig.update_layout(
            title="Historical vs Forecasted Demand",
            xaxis_title="Date",
            yaxis_title="Quantity",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 3: INVENTORY INTELLIGENCE
# ============================================================
elif page == "📦 Inventory Intelligence":
    st.markdown('<p class="header-title">Inventory Intelligence</p>', unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        high_stockout = (data["inv_intel"]["stockout_risk"] == "HIGH").sum()
        st.metric("Stockout HIGH", high_stockout)
    with col2:
        medium_stockout = (data["inv_intel"]["stockout_risk"] == "MEDIUM").sum()
        st.metric("Stockout MEDIUM", medium_stockout)
    with col3:
        high_overstock = (data["inv_intel"]["overstock_risk"] == "HIGH").sum()
        st.metric("Overstock HIGH", high_overstock)
    with col4:
        urgent_reorder = (data["inv_intel"]["reorder_urgency"] == "URGENT").sum()
        st.metric("Urgent Reorder", urgent_reorder)
    
    # Inventory risk distribution
    st.markdown('<p class="section-title">Risk Distribution</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        stockout_counts = data["inv_intel"]["stockout_risk"].value_counts()
        fig = px.pie(
            values=stockout_counts.values,
            names=stockout_counts.index,
            title="Stockout Risk Distribution",
            template="plotly_dark",
            color_discrete_map={"HIGH": "#ff4757", "MEDIUM": "#ffa502", "LOW": "#2ed573"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        overstock_counts = data["inv_intel"]["overstock_risk"].value_counts()
        fig = px.pie(
            values=overstock_counts.values,
            names=overstock_counts.index,
            title="Overstock Risk Distribution",
            template="plotly_dark",
            color_discrete_map={"HIGH": "#ff4757", "MEDIUM": "#ffa502", "LOW": "#2ed573"}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Critical items
    st.markdown('<p class="section-title">Critical Inventory Items</p>', unsafe_allow_html=True)
    
    critical = data["inv_intel"][
        (data["inv_intel"]["stockout_risk"] == "HIGH") |
        (data["inv_intel"]["overstock_risk"] == "HIGH") |
        (data["inv_intel"]["reorder_urgency"] == "URGENT")
    ].sort_values("composite_risk_score", ascending=False)
    
    if not critical.empty:
        display_cols = ["product_id", "store_id", "stockout_risk", "overstock_risk", "reorder_urgency", "recommended_action"]
        st.dataframe(critical[display_cols], use_container_width=True)
    else:
        st.info("No critical inventory items found.")
    
    # Inventory alerts table
    st.markdown('<p class="section-title">Inventory Alerts</p>', unsafe_allow_html=True)
    
    alert_filter = st.multiselect(
        "Filter by Alert Type",
        data["inventory_alerts"]["alert_type"].unique().tolist(),
        default=data["inventory_alerts"]["alert_type"].unique().tolist()
    )
    
    filtered_alerts = data["inventory_alerts"][
        data["inventory_alerts"]["alert_type"].isin(alert_filter)
    ]
    
    st.dataframe(filtered_alerts, use_container_width=True)

# ============================================================
# PAGE 4: DEMAND ANOMALIES
# ============================================================
elif page == "⚠️ Demand Anomalies":
    st.markdown('<p class="header-title">Demand Anomalies</p>', unsafe_allow_html=True)
    
    # Anomaly summary
    col1, col2, col3 = st.columns(3)
    with col1:
        total_anomalies = len(data["anomalies"])
        st.metric("Total Anomalies", f"{total_anomalies:,}")
    with col2:
        anomaly_rate = len(data["anomalies"]) / len(data["features"]) * 100
        st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
    with col3:
        demand_spikes = (data["anomalies"]["anomaly_type"] == "Demand Spike").sum()
        st.metric("Demand Spikes", f"{demand_spikes:,}")
    
    # Anomaly timeline
    st.markdown('<p class="section-title">Anomaly Timeline</p>', unsafe_allow_html=True)
    
    anomaly_daily = data["anomalies"].groupby("date").size().reset_index(name="anomaly_count")
    fig = px.bar(
        anomaly_daily,
        x="date",
        y="anomaly_count",
        title="Daily Anomaly Count",
        template="plotly_dark",
        color="anomaly_count",
        color_continuous_scale="Reds"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Top anomalous products
    st.markdown('<p class="section-title">Top Anomalous Products</p>', unsafe_allow_html=True)
    
    top_anomalous = data["anomalies"]["product_id"].value_counts().head(10).reset_index()
    top_anomalous.columns = ["product_id", "anomaly_count"]
    top_anomalous = top_anomalous.merge(data["products"][["product_id", "category"]], on="product_id")
    
    fig = px.bar(
        top_anomalous,
        x="anomaly_count",
        y="product_id",
        orientation="h",
        title="Top 10 Products by Anomaly Count",
        template="plotly_dark",
        color="category"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Anomaly details
    st.markdown('<p class="section-title">Anomaly Details</p>', unsafe_allow_html=True)
    
    anomaly_type_filter = st.multiselect(
        "Filter by Anomaly Type",
        data["anomalies"]["anomaly_type"].unique().tolist(),
        default=data["anomalies"]["anomaly_type"].unique().tolist()
    )
    
    filtered_anomalies = data["anomalies"][
        data["anomalies"]["anomaly_type"].isin(anomaly_type_filter)
    ]
    
    display_cols = ["date", "product_id", "store_id", "quantity_sold", "z_score", "anomaly_type"]
    st.dataframe(filtered_anomalies[display_cols].sort_values("date", ascending=False), use_container_width=True)

# ============================================================
# PAGE 5: SEGMENTATION
# ============================================================
elif page == "🎯 Segmentation":
    st.markdown('<p class="header-title">Product, Store & Warehouse Segmentation</p>', unsafe_allow_html=True)
    
    segment_type = st.radio(
        "Select Segmentation View",
        ["Products", "Stores", "Warehouses"]
    )
    
    if segment_type == "Products":
        st.markdown('<p class="section-title">Product Segmentation</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            label_counts = data["product_segments"]["product_cluster_label"].value_counts()
            fig = px.pie(
                values=label_counts.values,
                names=label_counts.index,
                title="Product Cluster Distribution",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                data["product_segments"],
                x="quantity_sold_sum",
                y="demand_cv_28d_mean",
                color="product_cluster_label",
                title="Product Segments (Revenue vs Variability)",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(data["product_segments"], use_container_width=True)
    
    elif segment_type == "Stores":
        st.markdown('<p class="section-title">Store Segmentation</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            label_counts = data["store_segments"]["store_cluster_label"].value_counts()
            fig = px.pie(
                values=label_counts.values,
                names=label_counts.index,
                title="Store Cluster Distribution",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            store_segs_with_info = data["store_segments"].merge(
                data["stores"][["store_id", "city", "state"]], on="store_id"
            )
            fig = px.bar(
                store_segs_with_info,
                x="store_cluster_label",
                y="revenue_sum",
                color="store_cluster_label",
                title="Revenue by Store Cluster",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(data["store_segments"], use_container_width=True)
    
    else:  # Warehouses
        st.markdown('<p class="section-title">Warehouse Segmentation</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            label_counts = data["warehouse_segments"]["warehouse_cluster_label"].value_counts()
            fig = px.pie(
                values=label_counts.values,
                names=label_counts.index,
                title="Warehouse Cluster Distribution",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            wh_segs_with_info = data["warehouse_segments"].merge(
                data["warehouses"][["warehouse_id", "city", "state"]], on="warehouse_id"
            )
            fig = px.bar(
                wh_segs_with_info,
                x="warehouse_cluster_label",
                y="revenue_sum",
                color="warehouse_cluster_label",
                title="Revenue by Warehouse Cluster",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(data["warehouse_segments"], use_container_width=True)

# ============================================================
# PAGE 6: WAREHOUSE INTELLIGENCE
# ============================================================
elif page == "🏭 Warehouse Intelligence":
    st.markdown('<p class="header-title">Warehouse Intelligence</p>', unsafe_allow_html=True)
    
    # Warehouse metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_warehouses = len(data["wh_opt"])
        st.metric("Total Warehouses", total_warehouses)
    with col2:
        total_capacity = data["wh_opt"]["capacity_m3"].sum()
        st.metric("Total Capacity", f"{total_capacity:,.0f} m³")
    with col3:
        avg_util = data["wh_opt"]["utilization_pct"].mean()
        st.metric("Avg Utilization", f"{avg_util:.1f}%")
    with col4:
        total_occupied = data["wh_opt"]["occupied_volume_m3"].sum()
        st.metric("Total Occupied", f"{total_occupied:,.0f} m³")
    
    # Utilization chart
    st.markdown('<p class="section-title">Warehouse Utilization</p>', unsafe_allow_html=True)
    
    fig = px.bar(
        data["wh_opt"],
        x="warehouse_id",
        y="utilization_pct",
        title="Warehouse Utilization %",
        template="plotly_dark",
        color="utilization_pct",
        color_continuous_scale="RdYlGn_r"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Warehouse details
    st.markdown('<p class="section-title">Warehouse Details</p>', unsafe_allow_html=True)
    
    display_cols = ["warehouse_id", "warehouse_name", "city", "capacity_m3", "occupied_volume_m3", "utilization_pct", "capacity_risk", "recommendation"]
    st.dataframe(data["wh_opt"][display_cols], use_container_width=True)

# ============================================================
# PAGE 7: DATA EXPLORER
# ============================================================
elif page == "📁 Data Explorer":
    st.markdown('<p class="header-title">Data Explorer</p>', unsafe_allow_html=True)
    
    table_option = st.selectbox(
        "Select Table",
        ["products", "stores", "suppliers", "warehouses", "sales", "inventory", "forecasts", "anomalies", "inventory_alerts"]
    )
    
    if table_option == "products":
        st.dataframe(data["products"], use_container_width=True)
    elif table_option == "stores":
        st.dataframe(data["stores"], use_container_width=True)
    elif table_option == "suppliers":
        st.dataframe(data["suppliers"], use_container_width=True)
    elif table_option == "warehouses":
        st.dataframe(data["warehouses"], use_container_width=True)
    elif table_option == "sales":
        st.dataframe(data["sales"].head(1000), use_container_width=True)
    elif table_option == "inventory":
        st.dataframe(data["inventory"].head(1000), use_container_width=True)
    elif table_option == "forecasts":
        st.dataframe(data["forecasts"], use_container_width=True)
    elif table_option == "anomalies":
        st.dataframe(data["anomalies"], use_container_width=True)
    elif table_option == "inventory_alerts":
        st.dataframe(data["inventory_alerts"], use_container_width=True)
    
    # Quick stats
    st.markdown('<p class="section-title">Quick Statistics</p>', unsafe_allow_html=True)
    
    if table_option in data:
        df = data[table_option]
        st.markdown(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
        st.markdown(f"**Columns:** {', '.join(df.columns.tolist())}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("RetailSync AI v1.0.0")
st.sidebar.markdown("AI-Powered Retail Supply Chain Intelligence")
st.sidebar.markdown("© 2026 RetailSync AI")
