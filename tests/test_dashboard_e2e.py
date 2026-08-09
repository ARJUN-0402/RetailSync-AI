"""
End-to-end dashboard data test.

Exercises the dashboard data-loading path without starting Streamlit,
so we can verify live CSV/SQL/model wiring produces sane outputs.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine, text

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DB_PATH = os.path.join(PROJECT_ROOT, "database", "retailsync.db")


def main():
    print("=== END-TO-END DASHBOARD DATA TEST ===\n")

    # 1. Load core data files
    print("1. Loading processed data files...")
    data = {}
    files = {
        "features": ("features_daily.csv", {"parse_dates": ["date"]}),
        "forecasts": ("forecasts_next_14d.csv", {"parse_dates": ["date"]}),
        "inv_intel": ("inventory_intelligence.csv", {}),
        "anomalies": ("anomalies.csv", {"parse_dates": ["date"]}),
        "wh_opt": ("warehouse_optimization.csv", {}),
        "product_segments": ("product_segments.csv", {}),
        "store_segments": ("store_segments.csv", {}),
        "warehouse_segments": ("warehouse_segments.csv", {}),
    }

    for key, (name, kwargs) in files.items():
        path = os.path.join(DATA_DIR, name)
        if not os.path.exists(path):
            print(f"  [MISSING] {name}")
            return 1
        data[key] = pd.read_csv(path, **kwargs)
        print(f"  [OK] {name}: {data[key].shape}")

    # 2. Compute dashboard-style KPIs from live data
    print("\n2. Computing dashboard KPIs from live data...")
    features = data["features"]
    forecasts = data["forecasts"]
    inv_intel = data["inv_intel"]
    anomalies = data["anomalies"]
    wh_opt = data["wh_opt"]

    kpis = {
        "total_revenue": features["revenue"].sum(),
        "total_quantity": features["quantity_sold"].sum(),
        "total_products": features["product_id"].nunique(),
        "total_stores": features["store_id"].nunique(),
        "forecast_demand_14d": forecasts["forecast_demand"].sum(),
        "stockout_high": (inv_intel["stockout_risk"] == "HIGH").sum(),
        "stockout_medium": (inv_intel["stockout_risk"] == "MEDIUM").sum(),
        "overstock_high": (inv_intel["overstock_risk"] == "HIGH").sum(),
        "urgent_reorder": (inv_intel["reorder_urgency"] == "URGENT").sum(),
        "total_anomalies": len(anomalies),
        "avg_warehouse_utilization": wh_opt["utilization_pct"].mean(),
    }

    for k, v in kpis.items():
        if isinstance(v, float):
            print(f"  {k}: {v:,.2f}")
        else:
            print(f"  {k}: {v:,}")

    # 3. Validate KPI sanity
    print("\n3. Validating KPI sanity...")
    checks = {
        "revenue_positive": kpis["total_revenue"] > 0,
        "quantity_positive": kpis["total_quantity"] > 0,
        "products_count": kpis["total_products"] == 50,
        "stores_count": kpis["total_stores"] == 10,
        "forecast_positive": kpis["forecast_demand_14d"] > 0,
        "warehouse_util_in_range": 0 <= kpis["avg_warehouse_utilization"] <= 100,
        "anomalies_present": kpis["total_anomalies"] > 0,
    }

    for name, passed in checks.items():
        print(f"  [{'OK' if passed else 'FAIL'}] {name}")

    if not all(checks.values()):
        return 1

    # 4. Load ML models
    print("\n4. Loading ML models...")
    model_files = {
        "demand_forecaster": "demand_forecaster.pkl",
        "product_clusterer": "product_clusterer.pkl",
        "store_clusterer": "store_clusterer.pkl",
        "warehouse_clusterer": "warehouse_clusterer.pkl",
    }

    models = {}
    for name, filename in model_files.items():
        path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(path):
            print(f"  [MISSING] {filename}")
            return 1
        models[name] = joblib.load(path)
        print(f"  [OK] {filename}")

    # 5. Validate model structure
    print("\n5. Validating model structure...")
    for name, model in models.items():
        if not isinstance(model, dict):
            print(f"  [FAIL] {name} is not a dict")
            return 1
        print(f"  [OK] {name} keys: {list(model.keys())[:5]}")

    # 6. Load database tables used by dashboard
    print("\n6. Loading database tables used by dashboard...")
    if not os.path.exists(DB_PATH):
        print(f"  [MISSING] {DB_PATH}")
        return 1

    engine = create_engine(f"sqlite:///{DB_PATH}")
    with engine.connect() as conn:
        tables = pd.read_sql('SELECT name FROM sqlite_master WHERE type="table"', conn)
        table_names = set(tables["name"].tolist())

        required_db_tables = [
            "products",
            "stores",
            "sales",
            "inventory",
            "inventory_alerts",
            "anomaly_flags",
            "product_segments",
            "store_segments",
            "warehouse_segments",
            "warehouse_optimization",
        ]

        for table in required_db_tables:
            if table not in table_names:
                print(f"  [MISSING] {table}")
                return 1
            count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).fetchone()[0]
            print(f"  [OK] {table}: {count} rows")

    # 7. Simulate dashboard page data selections
    print("\n7. Simulating dashboard selections...")

    # Demand Forecast page: pick first product-store combo
    sample = features.dropna(subset=["product_id", "store_id"]).iloc[0]
    product_id = str(sample["product_id"])
    store_id = str(sample["store_id"])
    subset = features[
        (features["product_id"] == product_id) & (features["store_id"] == store_id)
    ].sort_values("date")
    print(f"  [OK] Demand Forecast sample rows: {len(subset)}")

    # Inventory Intelligence page: filter alerts
    high_alerts = inv_intel[
        (inv_intel["stockout_risk"] == "HIGH") | (inv_intel["overstock_risk"] == "HIGH")
    ]
    print(f"  [OK] High-risk alerts: {len(high_alerts)}")

    # Segmentation page: cluster counts
    product_clusters = data["product_segments"].get("cluster", data["product_segments"].get("product_cluster"))
    print(f"  [OK] Product cluster counts:\n{product_clusters.value_counts().to_dict()}")

    # Warehouse page: utilization summary
    print(f"  [OK] Warehouse utilization:\n{wh_opt[['warehouse_id','utilization_pct']].to_dict(orient='records')}")

    print("\n=== END-TO-END DASHBOARD TEST PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
