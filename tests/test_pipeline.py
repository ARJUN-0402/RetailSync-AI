"""
Test suite for RetailSync AI pipeline.

Run with: python -m pytest tests/test_pipeline.py -v
Or: python tests/test_pipeline.py
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine, text

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Test configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DB_PATH = os.path.join(PROJECT_ROOT, "database", "retailsync.db")

class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def record(self, test_name, passed, message=""):
        if passed:
            self.passed += 1
            print(f"  [PASS] {test_name}")
        else:
            self.failed += 1
            self.errors.append(f"{test_name}: {message}")
            print(f"  [FAIL] {test_name}: {message}")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"TEST SUMMARY: {self.passed}/{total} passed")
        if self.errors:
            print(f"\nFAILED TESTS:")
            for error in self.errors:
                print(f"  - {error}")
        print(f"{'='*60}")
        return self.failed == 0

# ============================================================
# TEST SUITE
# ============================================================

def test_data_files_exist(results):
    """Test that all required data files exist."""
    print("\n=== Testing Data Files ===")
    
    required_files = [
        "data/processed/features_daily.csv",
        "data/processed/forecasts_next_14d.csv",
        "data/processed/inventory_intelligence.csv",
        "data/processed/anomalies.csv",
        "data/processed/product_segments.csv",
        "data/processed/store_segments.csv",
        "data/processed/warehouse_segments.csv",
        "data/processed/warehouse_optimization.csv",
        "models/demand_forecaster.pkl",
        "models/product_clusterer.pkl",
        "models/store_clusterer.pkl",
        "models/warehouse_clusterer.pkl",
    ]
    
    for file_path in required_files:
        full_path = os.path.join(PROJECT_ROOT, file_path)
        exists = os.path.exists(full_path)
        results.record(f"File exists: {file_path}", exists)
        
        if exists:
            size = os.path.getsize(full_path)
            results.record(f"File non-empty: {file_path}", size > 0, f"Size: {size} bytes")

def test_database_tables(results):
    """Test that database tables exist and have data."""
    print("\n=== Testing Database Tables ===")
    
    if not os.path.exists(DB_PATH):
        results.record("Database exists", False, f"DB not found at {DB_PATH}")
        return
    
    engine = create_engine(f"sqlite:///{DB_PATH}")
    
    with engine.connect() as conn:
        # Check tables exist
        tables = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()
        table_names = [r[0] for r in tables]
        
        required_tables = [
            "products", "stores", "suppliers", "warehouses",
            "sales", "inventory", "inventory_alerts",
            "anomaly_flags", "product_segments", "store_segments",
            "warehouse_segments", "warehouse_optimization"
        ]
        
        for table in required_tables:
            exists = table in table_names
            results.record(f"Table exists: {table}", exists)
            
            if exists:
                count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).fetchone()[0]
                results.record(f"Table has data: {table}", count > 0, f"Count: {count}")

def test_feature_engineering(results):
    """Test feature engineering output."""
    print("\n=== Testing Feature Engineering ===")
    
    features_path = os.path.join(PROCESSED_DIR, "features_daily.csv")
    if not os.path.exists(features_path):
        results.record("Features file exists", False)
        return
    
    df = pd.read_csv(features_path, parse_dates=["date"])
    
    # Basic checks
    results.record("Features not empty", len(df) > 0, f"Rows: {len(df)}")
    results.record("Has date column", "date" in df.columns)
    results.record("Has product_id column", "product_id" in df.columns)
    results.record("Has store_id column", "store_id" in df.columns)
    results.record("Has target columns", any("target" in c for c in df.columns))
    
    # Check for required feature categories
    has_lag = any("lag" in c for c in df.columns)
    has_rolling = any("rolling" in c for c in df.columns)
    has_time = any(c in df.columns for c in ["day_of_week", "month", "year"])
    
    results.record("Has lag features", has_lag)
    results.record("Has rolling features", has_rolling)
    results.record("Has time features", has_time)
    
    # Check no NaN in critical columns
    critical_cols = ["product_id", "store_id", "date", "target_demand_1d"]
    for col in critical_cols:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            results.record(f"No NaN in {col}", nan_count == 0, f"NaN count: {nan_count}")

def test_forecasting(results):
    """Test forecasting model and outputs."""
    print("\n=== Testing Forecasting ===")
    
    # Check model exists
    model_path = os.path.join(MODELS_DIR, "demand_forecaster.pkl")
    if not os.path.exists(model_path):
        results.record("Forecaster model exists", False)
        return
    
    model_package = joblib.load(model_path)
    results.record("Model has 'model' key", "model" in model_package)
    results.record("Model has 'feature_cols' key", "feature_cols" in model_package)
    results.record("Model has 'metrics' key", "metrics" in model_package)
    
    # Check forecasts
    forecasts_path = os.path.join(PROCESSED_DIR, "forecasts_next_14d.csv")
    if os.path.exists(forecasts_path):
        forecasts = pd.read_csv(forecasts_path, parse_dates=["date"])
        results.record("Forecasts not empty", len(forecasts) > 0, f"Rows: {len(forecasts)}")
        results.record("Has forecast_demand column", "forecast_demand" in forecasts.columns)
        results.record("Forecasts are non-negative", (forecasts["forecast_demand"] >= 0).all())

def test_inventory_intelligence(results):
    """Test inventory intelligence outputs."""
    print("\n=== Testing Inventory Intelligence ===")
    
    inv_path = os.path.join(PROCESSED_DIR, "inventory_intelligence.csv")
    if not os.path.exists(inv_path):
        results.record("Inventory intelligence file exists", False)
        return
    
    df = pd.read_csv(inv_path)
    
    results.record("Inventory data not empty", len(df) > 0, f"Rows: {len(df)}")
    results.record("Has stockout_risk column", "stockout_risk" in df.columns)
    results.record("Has overstock_risk column", "overstock_risk" in df.columns)
    results.record("Has reorder_urgency column", "reorder_urgency" in df.columns)
    results.record("Has composite_risk_score column", "composite_risk_score" in df.columns)
    
    # Check risk values are valid
    valid_stockout = df["stockout_risk"].isin(["HIGH", "MEDIUM", "LOW"]).all()
    results.record("Valid stockout risk values", valid_stockout)
    
    valid_overstock = df["overstock_risk"].isin(["HIGH", "MEDIUM", "LOW"]).all()
    results.record("Valid overstock risk values", valid_overstock)

def test_anomaly_detection(results):
    """Test anomaly detection outputs."""
    print("\n=== Testing Anomaly Detection ===")
    
    anomalies_path = os.path.join(PROCESSED_DIR, "anomalies.csv")
    if not os.path.exists(anomalies_path):
        results.record("Anomalies file exists", False)
        return
    
    df = pd.read_csv(anomalies_path, parse_dates=["date"])
    
    results.record("Anomalies not empty", len(df) > 0, f"Rows: {len(df)}")
    results.record("Has anomaly_type column", "anomaly_type" in df.columns)
    results.record("Has z_score column", "z_score" in df.columns)
    
    # Check anomaly types
    valid_types = ["Demand Spike", "Demand Drop", "Unusual Pattern"]
    has_valid_types = df["anomaly_type"].isin(valid_types).all()
    results.record("Valid anomaly types", has_valid_types)
    
    # Check database
    if os.path.exists(DB_PATH):
        engine = create_engine(f"sqlite:///{DB_PATH}")
        with engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM anomaly_flags")).fetchone()[0]
            results.record("Anomaly flags in DB", count > 0, f"Count: {count}")

def test_clustering(results):
    """Test clustering/segmentation outputs."""
    print("\n=== Testing Clustering ===")
    
    # Check product segments
    prod_path = os.path.join(PROCESSED_DIR, "product_segments.csv")
    if os.path.exists(prod_path):
        df = pd.read_csv(prod_path)
        results.record("Product segments not empty", len(df) > 0, f"Rows: {len(df)}")
        results.record("Has cluster column", "cluster" in df.columns or "product_cluster" in df.columns)
    
    # Check store segments
    store_path = os.path.join(PROCESSED_DIR, "store_segments.csv")
    if os.path.exists(store_path):
        df = pd.read_csv(store_path)
        results.record("Store segments not empty", len(df) > 0, f"Rows: {len(df)}")
    
    # Check warehouse segments
    wh_path = os.path.join(PROCESSED_DIR, "warehouse_segments.csv")
    if os.path.exists(wh_path):
        df = pd.read_csv(wh_path)
        results.record("Warehouse segments not empty", len(df) > 0, f"Rows: {len(df)}")
    
    # Check models
    for model_name in ["product_clusterer", "store_clusterer", "warehouse_clusterer"]:
        model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
        exists = os.path.exists(model_path)
        results.record(f"Clusterer model exists: {model_name}", exists)
        
        if exists:
            model = joblib.load(model_path)
            results.record(f"Model has 'model' key: {model_name}", "model" in model)

def test_pipeline_integration(results):
    """Test end-to-end pipeline integration."""
    print("\n=== Testing Pipeline Integration ===")
    
    # Check all outputs can be loaded together
    try:
        features = pd.read_csv(os.path.join(PROCESSED_DIR, "features_daily.csv"), parse_dates=["date"])
        forecasts = pd.read_csv(os.path.join(PROCESSED_DIR, "forecasts_next_14d.csv"), parse_dates=["date"])
        inv_intel = pd.read_csv(os.path.join(PROCESSED_DIR, "inventory_intelligence.csv"))
        anomalies = pd.read_csv(os.path.join(PROCESSED_DIR, "anomalies.csv"), parse_dates=["date"])
        
        results.record("All main outputs loadable", True)
        
        # Check date ranges
        features_date_range = (features["date"].min(), features["date"].max())
        forecasts_date_range = (forecasts["date"].min(), forecasts["date"].max())
        
        results.record("Features date range valid", features_date_range[0] < features_date_range[1])
        results.record("Forecasts are future dates", forecasts_date_range[0] >= features_date_range[1])
        
    except Exception as e:
        results.record("Pipeline integration", False, str(e))

def test_dashboard_artifacts(results):
    """Test dashboard can be built."""
    print("\n=== Testing Dashboard ===")
    
    dashboard_path = os.path.join(PROJECT_ROOT, "dashboard", "app.py")
    results.record("Dashboard app exists", os.path.exists(dashboard_path))
    
    if os.path.exists(dashboard_path):
        with open(dashboard_path, "r", encoding="utf-8") as f:
            code = f.read()
        
        results.record("Dashboard has imports", "import streamlit" in code)
        results.record("Dashboard has load_data", "load_data" in code)
        results.record("Dashboard has pages", "page =" in code or "radio" in code)

# ============================================================
# MAIN TEST RUNNER
# ============================================================

def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("RETAILSYNC AI - TEST SUITE")
    print("="*60)
    
    results = TestResults()
    
    # Run all test suites
    test_data_files_exist(results)
    test_database_tables(results)
    test_feature_engineering(results)
    test_forecasting(results)
    test_inventory_intelligence(results)
    test_anomaly_detection(results)
    test_clustering(results)
    test_pipeline_integration(results)
    test_dashboard_artifacts(results)
    
    # Print summary
    all_passed = results.summary()
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
