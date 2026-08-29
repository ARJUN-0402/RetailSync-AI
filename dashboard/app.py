"""RetailSync AI - AI-Powered Retail Demand Forecasting & Supply Chain Intelligence Platform."""

from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import os  # noqa: E402, F401

import joblib  # noqa: E402, F401
import pandas as pd  # noqa: E402, F401
import streamlit as st  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402, F401

from dashboard.business_intelligence import render_business_intelligence_page  # noqa: E402
from dashboard.components.ui import (  # noqa: E402
    inject_global_css, 
    render_sidebar_branding, 
    render_sidebar_footer,
)  # noqa: E402
from dashboard.explainability_page import render_explainability_page  # noqa: E402
from dashboard.pages.anomalies import render_anomalies_page  # noqa: E402
from dashboard.pages.ai_analyst import render_ai_analyst_page  # noqa: E402
from dashboard.pages.data_explorer import render_data_explorer_page  # noqa: E402
from dashboard.pages.demand_forecast import render_demand_forecast_page  # noqa: E402
from dashboard.pages.inventory import render_inventory_page  # noqa: E402
from dashboard.pages.model_performance import render_model_performance_page  # noqa: E402
from dashboard.pages.overview import render_overview_page  # noqa: E402
from dashboard.pages.segmentation import render_segmentation_page  # noqa: E402
from dashboard.pages.warehouse import render_warehouse_page  # noqa: E402
from src.config import settings  # noqa: E402, F401
from src.health import get_health_status  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402


@st.cache_data(ttl=600)
def load_data():
    """Load all processed data and models into memory."""
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]

    # Load data CSVs
    data = {
        "features": pd.read_csv(_PROJECT_ROOT / "data" / "processed" / "features_daily.csv", parse_dates=["date"]),
        "forecasts": pd.read_csv(_PROJECT_ROOT / "data" / "processed" / "forecasts_next_14d.csv", parse_dates=["date"]),
        "inv_intel": pd.read_csv(_PROJECT_ROOT / "data" / "processed" / "inventory_intelligence.csv"),
        "anomalies": pd.read_csv(_PROJECT_ROOT / "data" / "processed" / "anomalies.csv", parse_dates=["date"]),
        "wh_opt": pd.read_csv(_PROJECT_ROOT / "data" / "processed" / "warehouse_optimization.csv"),
        "product_segments": pd.read_csv(_PROJECT_ROOT / "data" / "processed" / "product_segments.csv"),
        "store_segments": pd.read_csv(_PROJECT_ROOT / "data" / "processed" / "store_segments.csv"),
        "warehouse_segments": pd.read_csv(_PROJECT_ROOT / "data" / "processed" / "warehouse_segments.csv"),
        "products": pd.read_csv(_PROJECT_ROOT / "data" / "processed" / "products.csv"),
        "stores": pd.read_csv(_PROJECT_ROOT / "data" / "processed" / "stores.csv"),
        "suppliers": pd.read_csv(_PROJECT_ROOT / "data" / "processed" / "suppliers.csv"),
        "warehouses": pd.read_csv(_PROJECT_ROOT / "data" / "processed" / "warehouses.csv"),
    }

    # Load ML models
    models = {
        "demand_forecaster": joblib.load(str(_PROJECT_ROOT / "models" / "demand_forecaster.pkl")),
        "product_clusterer": joblib.load(str(_PROJECT_ROOT / "models" / "product_clusterer.pkl")),
        "store_clusterer": joblib.load(str(_PROJECT_ROOT / "models" / "store_clusterer.pkl")),
        "warehouse_clusterer": joblib.load(str(_PROJECT_ROOT / "models" / "warehouse_clusterer.pkl")),
    }

    # Create database engine
    engine = create_engine(f"sqlite:///{_PROJECT_ROOT / 'database' / 'retailsync.db'}")

    return data, models, engine


# Initialize session state for data/models/engine
if "data" not in st.session_state:
    st.session_state.data = load_data()[0]
if "models" not in st.session_state:
    st.session_state.models = load_data()[1]
if "engine" not in st.session_state:
    st.session_state.engine = load_data()[2]

# Initialize session state for current page
if "current_page" not in st.session_state:
    st.session_state.current_page = "Overview"


logger = setup_logging(__name__)

st.set_page_config(
    page_title="RetailSync AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()

# Initialize session state for current page
if "current_page" not in st.session_state:
    st.session_state.current_page = "Overview"

# load_data

# Sidebar navigation
render_sidebar_branding()

# Define navigation structure
NAVIGATION = {
    "Overview": ["Overview"],
    "ANALYTICS": [
        "Demand Forecast",
        "Inventory Intelligence",
        "Demand Anomalies",
        "Segmentation",
        "Warehouse Intelligence"
    ],
    "INTELLIGENCE": [
        "Model Performance",
        "Model Explainability",
        "Business Intelligence",
        "AI Analyst"
    ],
    "DATA": [
        "Data Explorer"
    ],
    "SYSTEM": [
        "Health Check"
    ]
}

# Render navigation buttons grouped by section
for group, pages in NAVIGATION.items():
    st.sidebar.markdown(f"### {group}")
    for page in pages:
        # Determine button type: primary for current page, secondary otherwise
        button_type = "primary" if st.session_state.current_page == page else "secondary"
        if st.sidebar.button(
            page, 
            use_container_width=True, 
            type=button_type,
            key=f"nav_{page}"  # Unique key to avoid conflicts
        ):
            st.session_state.current_page = page
            st.rerun()

st.sidebar.markdown("---")

# System status indicator (compact) - using simple text icons
health_status = get_health_status()
if health_status["status"] == "healthy":
    status_text = "✓ System Healthy"
elif health_status["status"] == "degraded":
    status_text = "! System Degraded"
else:
    status_text = "✗ System Unhealthy"
if st.sidebar.button(status_text, help="Click to see detailed health status"):
    st.session_state.show_health_details = not st.session_state.get("show_health_details", False)

if st.session_state.get("show_health_details", False):
    with st.sidebar.expander("System Health Details", expanded=True):
        health = get_health_status()
        if health["status"] == "healthy":
            status_display = "✓ Healthy"
        elif health["status"] == "degraded":
            status_display = "! Degraded"
        else:
            status_display = "✗ Unhealthy"
        st.markdown(f"**{status_display}**")
        st.markdown(f"App: {health['app']} v{health['version']}")
        st.markdown(f"Env: {health['environment']}")
        for component, details in health.get("components", {}).items():
            if details["status"] == "healthy":
                comp_display = "✓"
            elif details["status"] == "degraded":
                comp_display = "!"
            else:
                comp_display = "✗"
            st.markdown(f"{comp_display} {component}: {details['status']}")
            if details.get("error"):
                st.caption(f"Error: {details['error']}")
        if st.button("Close Health Check"):
            st.session_state.show_health_details = False
            st.rerun()

render_sidebar_footer()

# Main page routing using current_page session state
data = st.session_state.data
models = st.session_state.models
engine = st.session_state.engine

if st.session_state.current_page == "Overview":
    render_overview_page(data, models, engine)
elif st.session_state.current_page == "Demand Forecast":
    render_demand_forecast_page(data, models)
elif st.session_state.current_page == "Inventory Intelligence":
    render_inventory_page(data, engine)
elif st.session_state.current_page == "Demand Anomalies":
    render_anomalies_page(data)
elif st.session_state.current_page == "Segmentation":
    render_segmentation_page(data)
elif st.session_state.current_page == "Warehouse Intelligence":
    render_warehouse_page(data)
elif st.session_state.current_page == "Model Performance":
    render_model_performance_page(data, models)
elif st.session_state.current_page == "Model Explainability":
    render_explainability_page(models, data)
elif st.session_state.current_page == "Business Intelligence":
    render_business_intelligence_page(engine, data, models)
elif st.session_state.current_page == "AI Analyst":
    render_ai_analyst_page(data)
elif st.session_state.current_page == "Data Explorer":
    render_data_explorer_page(data)
elif st.session_state.current_page == "Health Check":
    # Render health check page in main area
    st.title("System Health")
    health = get_health_status()
    if health["status"] == "healthy":
        status_display = "✓ Healthy"
    elif health["status"] == "degraded":
        status_display = "! Degraded"
    else:
        status_display = "✗ Unhealthy"
    st.markdown(f"**{status_display}**")
    st.markdown(f"App: {health['app']} v{health['version']}")
    st.markdown(f"Env: {health['environment']}")
    for component, details in health.get("components", {}).items():
        if details["status"] == "healthy":
            comp_display = "✓"
        elif details["status"] == "degraded":
            comp_display = "!"
        else:
            comp_display = "✗"
        st.markdown(f"{comp_display} {component}: {details['status']}")
        if details.get("error"):
            st.caption(f"Error: {details['error']}")
else:
    # Fallback to Overview if somehow current_page is not set (should not happen)
    render_overview_page(data, models, engine)