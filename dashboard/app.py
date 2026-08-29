"""RetailSync AI - AI-Powered Retail Demand Forecasting & Supply Chain Intelligence Platform."""

import os

import joblib
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

from dashboard.business_intelligence import render_business_intelligence_page
from dashboard.components.ui import (
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
from src.config import settings
from src.health import get_health_status
from src.utils.logging import setup_logging

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

logger = setup_logging(__name__)

st.set_page_config(
    page_title="RetailSync AI",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()


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
        full_path = os.path.join(_project_root, path)
        if os.path.exists(full_path):
            try:
                models[name] = joblib.load(full_path)
            except Exception as exc:
                logger.warning("Could not load model %s: %s", name, exc)
                st.warning(f"Could not load model {name}: {exc}")
    return models


models = load_models()


@st.cache_resource
def get_engine():
    db_path = os.path.join(_project_root, settings.database.path)
    return create_engine(f"sqlite:///{db_path}")


engine = get_engine()


@st.cache_data(ttl=300)
def load_data():
    data = {}
    csv_files = {
        "features": ("data/processed/features_daily.csv", {"parse_dates": ["date"]}),
        "forecasts": ("data/processed/forecasts_next_14d.csv", {"parse_dates": ["date"]}),
        "inv_intel": ("data/processed/inventory_intelligence.csv", {}),
        "anomalies": ("data/processed/anomalies.csv", {"parse_dates": ["date"]}),
        "product_segments": ("data/processed/product_segments.csv", {}),
        "store_segments": ("data/processed/store_segments.csv", {}),
        "warehouse_segments": ("data/processed/warehouse_segments.csv", {}),
        "wh_opt": ("data/processed/warehouse_optimization.csv", {}),
    }
    for name, (path, kwargs) in csv_files.items():
        full_path = os.path.join(_project_root, path)
        if os.path.exists(full_path):
            try:
                data[name] = pd.read_csv(full_path, **kwargs)
            except Exception as exc:
                logger.warning("Could not load %s: %s", path, exc)
                st.warning(f"Could not load {path}: {exc}")
        else:
            data[name] = pd.DataFrame()

    db_queries = {
        "products": "SELECT product_id, product_name, category, subcategory, unit_price, cost_price, supplier_id FROM products",
        "stores": "SELECT store_id, store_name, city, state, store_type FROM stores",
        "suppliers": "SELECT supplier_id, supplier_name, country, lead_time_days, reliability_score FROM suppliers",
        "warehouses": "SELECT warehouse_id, warehouse_name, city, state, capacity_m3, supplier_id FROM warehouses",
        "sales": "SELECT date, product_id, store_id, quantity_sold, unit_price, revenue FROM sales",
        "inventory": "SELECT date, product_id, store_id, quantity_on_hand, reorder_point, max_stock_level, warehouse_id FROM inventory",
        "inventory_alerts": "SELECT * FROM inventory_alerts",
        "anomaly_flags": "SELECT * FROM anomaly_flags",
    }
    for name, query in db_queries.items():
        try:
            data[name] = pd.read_sql(query, engine)
        except Exception as exc:
            logger.warning("Could not load %s from database: %s", name, exc)
            st.warning(f"Could not load {name} from database: {exc}")
            data[name] = pd.DataFrame()

    if "date" in data.get("sales", pd.DataFrame()).columns:
        data["sales"]["date"] = pd.to_datetime(data["sales"]["date"])
    if "date" in data.get("inventory", pd.DataFrame()).columns:
        data["inventory"]["date"] = pd.to_datetime(data["inventory"]["date"])

    return data


with st.spinner("Loading data..."):
    data = load_data()

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

if st.sidebar.button("🔍 Health Check", width="stretch"):
    health = get_health_status()
    st.session_state["health_status"] = health

if "health_status" in st.session_state:
    health = st.session_state["health_status"]
    with st.sidebar.expander("System Health", expanded=True):
        status_color = {"healthy": "🟢", "degraded": "🟡", "unhealthy": "🔴"}.get(health["status"], "⚪")
        st.markdown(f"**{status_color} {health['status'].upper()}**")
        st.markdown(f"App: {health['app']} v{health['version']}")
        st.markdown(f"Env: {health['environment']}")
        for component, details in health.get("components", {}).items():
            comp_status = details.get("status", "unknown")
            comp_icon = {"healthy": "🟢", "degraded": "🟡", "unhealthy": "🔴"}.get(comp_status, "⚪")
            st.markdown(f"{comp_icon} {component}: {comp_status}")
            if details.get("error"):
                st.caption(f"Error: {details['error']}")
        if st.button("Close Health Check"):
            del st.session_state["health_status"]
            st.rerun()

st.sidebar.markdown("---")

st.sidebar.markdown("### Technology Stack")
st.sidebar.markdown("- Python")
st.sidebar.markdown("- Pandas / NumPy")
st.sidebar.markdown("- Scikit-learn / XGBoost")
st.sidebar.markdown("- Plotly")
st.sidebar.markdown("- Streamlit")
st.sidebar.markdown("- SQLite")
st.sidebar.markdown("- SHAP")

render_sidebar_footer()

if "nav_page" in st.session_state:
    page = st.session_state["nav_page"]
    del st.session_state["nav_page"]

try:
    if page == "📊 Executive Overview":
        render_overview_page(data, models, engine)
    elif page == "📈 Business Intelligence":
        render_business_intelligence_page(engine, data, models)
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
except Exception as exc:
    logger.error("Dashboard page error: %s", exc, exc_info=True)
    st.error(f"An unexpected error occurred: {exc}")
