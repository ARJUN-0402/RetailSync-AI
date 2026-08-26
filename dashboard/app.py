import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "src"))

import joblib
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

from dashboard.business_intelligence import render_business_intelligence_page
from dashboard.components.ui import (
    COLORS,
    inject_global_css,
    render_sidebar_branding,
    render_sidebar_footer,
)
from dashboard.explainability_page import render_explainability_page
from dashboard.pages.anomalies import render_anomalies_page
from dashboard.pages.ai_analyst import render_ai_analyst_page
from dashboard.pages.data_explorer import render_data_explorer_page
from dashboard.pages.demand_forecast import render_demand_forecast_page
from dashboard.pages.inventory import render_inventory_page
from dashboard.pages.model_performance import render_model_performance_page
from dashboard.pages.overview import render_overview_page
from dashboard.pages.segmentation import render_segmentation_page
from dashboard.pages.warehouse import render_warehouse_page

# Page config
st.set_page_config(
    page_title="RetailSync AI",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()


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
            try:
                models[name] = joblib.load(path)
            except Exception as exc:
                st.warning(f"Could not load model {name}: {exc}")
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
    data = {}
    csv_files = {
        "features": "data/processed/features_daily.csv",
        "forecasts": "data/processed/forecasts_next_14d.csv",
        "inv_intel": "data/processed/inventory_intelligence.csv",
        "anomalies": "data/processed/anomalies.csv",
        "product_segments": "data/processed/product_segments.csv",
        "store_segments": "data/processed/store_segments.csv",
        "warehouse_segments": "data/processed/warehouse_segments.csv",
        "wh_opt": "data/processed/warehouse_optimization.csv",
    }
    for name, path in csv_files.items():
        if os.path.exists(path):
            try:
                if name == "features":
                    data[name] = pd.read_csv(path, parse_dates=["date"])
                elif name == "forecasts":
                    data[name] = pd.read_csv(path, parse_dates=["date"])
                elif name == "anomalies":
                    data[name] = pd.read_csv(path, parse_dates=["date"])
                else:
                    data[name] = pd.read_csv(path)
            except Exception as exc:
                st.warning(f"Could not load {path}: {exc}")
        else:
            data[name] = pd.DataFrame()

    # Load from database
    db_tables = {
        "products": "SELECT * FROM products",
        "stores": "SELECT * FROM stores",
        "suppliers": "SELECT * FROM suppliers",
        "warehouses": "SELECT * FROM warehouses",
        "sales": "SELECT * FROM sales",
        "inventory": "SELECT * FROM inventory",
        "inventory_alerts": "SELECT * FROM inventory_alerts",
        "anomaly_flags": "SELECT * FROM anomaly_flags",
    }
    for name, query in db_tables.items():
        try:
            data[name] = pd.read_sql(query, engine)
        except Exception as exc:
            st.warning(f"Could not load {name} from database: {exc}")
            data[name] = pd.DataFrame()

    return data


with st.spinner("Loading data..."):
    data = load_data()

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
render_sidebar_branding()

st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Navigation",
    [
        "📊 Executive Overview",
        "📈 Business Intelligence",
        "🔮 Demand Forecast",
        "📦 Inventory Intelligence",
        "⚠️ Demand Anomalies",
        "🎯 Segmentation",
        "🏭 Warehouse Intelligence",
        "📊 Model Performance",
        "🧠 Model Explainability",
        "🤖 AI Analyst",
        "📁 Data Explorer",
    ],
    key="main_navigation",
)

st.sidebar.markdown("---")

# Technology stack
st.sidebar.markdown("### Technology Stack")
st.sidebar.markdown("- Python")
st.sidebar.markdown("- Pandas / NumPy")
st.sidebar.markdown("- Scikit-learn / XGBoost")
st.sidebar.markdown("- Plotly")
st.sidebar.markdown("- Streamlit")
st.sidebar.markdown("- SQLite")
st.sidebar.markdown("- SHAP")

render_sidebar_footer()

# ============================================================
# PAGE ROUTING
# ============================================================
# Handle navigation state from overview page buttons
if "nav_page" in st.session_state:
    page = st.session_state["nav_page"]
    del st.session_state["nav_page"]

if page == "📊 Executive Overview":
    render_overview_page(data, models, engine)

elif page == "📈 Business Intelligence":
    render_business_intelligence_page(engine, data)

elif page == "🔮 Demand Forecast":
    render_demand_forecast_page(data, models)

elif page == "📦 Inventory Intelligence":
    render_inventory_page(data, engine)

elif page == "⚠️ Demand Anomalies":
    render_anomalies_page(data)

elif page == "🎯 Segmentation":
    render_segmentation_page(data)

elif page == "🏭 Warehouse Intelligence":
    render_warehouse_page(data)

elif page == "📊 Model Performance":
    render_model_performance_page(data, models)

elif page == "🧠 Model Explainability":
    render_explainability_page(models, data)

elif page == "🤖 AI Analyst":
    render_ai_analyst_page(data)

elif page == "📁 Data Explorer":
    render_data_explorer_page(data)
