import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine, text

np.random.seed(42)

print("=== RETAILSYNC AI - WAREHOUSE OPTIMIZATION ===\n")

# Load data
print("Loading data...")
engine = create_engine("sqlite:///database/retailsync.db")
warehouses = pd.read_sql("SELECT * FROM warehouses", engine)
inventory = pd.read_sql("SELECT * FROM inventory", engine)
inventory["date"] = pd.to_datetime(inventory["date"])
products = pd.read_sql("SELECT * FROM products", engine)

print(f"Warehouses: {len(warehouses)}")
print(f"Inventory records: {len(inventory)}")
print(f"Products: {len(products)}")

# ============================================================
# 1. CALCULATE WAREHOUSE UTILIZATION
# ============================================================
print("\n=== 1. WAREHOUSE UTILIZATION ANALYSIS ===")

latest_date = inventory["date"].max()
latest_inventory = inventory[inventory["date"] == latest_date].copy()

print(f"Latest inventory date: {latest_date.date()}")

# Merge with product volume data
latest_inventory = latest_inventory.merge(
    products[["product_id", "volume_m3"]], on="product_id", how="left"
)

# Calculate occupied volume per warehouse
warehouse_util = latest_inventory.groupby("warehouse_id").agg(
    total_quantity=("quantity_on_hand", "sum"),
    distinct_products=("product_id", "nunique"),
    occupied_volume_m3=("quantity_on_hand", lambda x: (x * latest_inventory.loc[x.index, "volume_m3"]).sum()),
    avg_quantity_per_product=("quantity_on_hand", "mean"),
    max_quantity=("quantity_on_hand", "max"),
    min_quantity=("quantity_on_hand", "min"),
).reset_index()

# Merge with warehouse capacity
warehouse_util = warehouse_util.merge(
    warehouses[["warehouse_id", "warehouse_name", "city", "state", "capacity_m3"]],
    on="warehouse_id", how="left"
)

# Calculate utilization metrics
warehouse_util["utilization_pct"] = (warehouse_util["occupied_volume_m3"] / warehouse_util["capacity_m3"] * 100).round(2)
warehouse_util["available_capacity_m3"] = warehouse_util["capacity_m3"] - warehouse_util["occupied_volume_m3"]
warehouse_util["capacity_risk"] = "LOW"
warehouse_util["capacity_risk_reason"] = ""

# Classify utilization
# HIGH: > 80% utilization
# MEDIUM: 50-80% utilization
# LOW: < 50% utilization

high_util = warehouse_util["utilization_pct"] > 80
medium_util = (warehouse_util["utilization_pct"] >= 50) & (warehouse_util["utilization_pct"] <= 80)
low_util = warehouse_util["utilization_pct"] < 50

warehouse_util.loc[high_util, "capacity_risk"] = "HIGH"
warehouse_util.loc[high_util, "capacity_risk_reason"] = "Near or at capacity"
warehouse_util.loc[medium_util, "capacity_risk"] = "MEDIUM"
warehouse_util.loc[medium_util, "capacity_risk_reason"] = "Moderate utilization"
warehouse_util.loc[low_util, "capacity_risk"] = "LOW"
warehouse_util.loc[low_util, "capacity_risk_reason"] = "Underutilized"

print("\nWarehouse Utilization:")
print(warehouse_util[["warehouse_id", "warehouse_name", "capacity_m3", "occupied_volume_m3", "utilization_pct", "capacity_risk"]].to_string(index=False))

# ============================================================
# 2. WAREHOUSE PERFORMANCE METRICS
# ============================================================
print("\n=== 2. WAREHOUSE PERFORMANCE METRICS ===")

# Calculate historical utilization trends
warehouse_trends = inventory.groupby(["warehouse_id", "date"]).agg(
    daily_quantity=("quantity_on_hand", "sum"),
).reset_index()

warehouse_trends = warehouse_trends.merge(
    warehouses[["warehouse_id", "capacity_m3"]], on="warehouse_id", how="left"
)

# Merge with product volume
warehouse_trends = warehouse_trends.merge(
    inventory[["date", "warehouse_id", "product_id", "quantity_on_hand"]].merge(
        products[["product_id", "volume_m3"]], on="product_id", how="left"
    ),
    on=["date", "warehouse_id"], how="left"
)

# This is getting complex, let me simplify by calculating from latest inventory
# and adding turnover metrics

# Calculate inventory turnover (approximate)
# Turnover = total quantity sold / average inventory
sales_totals = pd.read_sql("""
    SELECT warehouse_id, SUM(quantity_sold) as total_sold
    FROM sales s
    JOIN inventory i ON s.product_id = i.product_id AND s.store_id = i.store_id
    WHERE i.date = (SELECT MAX(date) FROM inventory)
    GROUP BY warehouse_id
""", engine)

warehouse_util = warehouse_util.merge(sales_totals, on="warehouse_id", how="left")
warehouse_util["total_sold"] = warehouse_util["total_sold"].fillna(0)

# Approximate turnover ratio
warehouse_util["turnover_ratio"] = np.where(
    warehouse_util["total_quantity"] > 0,
    warehouse_util["total_sold"] / warehouse_util["total_quantity"],
    0
)

print("\nWarehouse Performance:")
print(warehouse_util[["warehouse_id", "total_quantity", "total_sold", "turnover_ratio", "distinct_products"]].to_string(index=False))

# ============================================================
# 3. WAREHOUSE SEGMENTATION (from clustering)
# ============================================================
print("\n=== 3. WAREHOUSE SEGMENTATION INTEGRATION ===")

# Load clustering results
warehouse_segments = pd.read_sql("SELECT * FROM warehouse_segments", engine)
warehouse_util = warehouse_util.merge(
    warehouse_segments[["warehouse_id", "cluster_label"]],
    on="warehouse_id", how="left"
)

print("\nWarehouse Clusters:")
print(warehouse_util[["warehouse_id", "cluster_label", "utilization_pct", "capacity_risk"]].to_string(index=False))

# ============================================================
# 4. OPTIMIZATION RECOMMENDATIONS
# ============================================================
print("\n=== 4. OPTIMIZATION RECOMMENDATIONS ===")

warehouse_util["recommendation"] = "Monitor"
warehouse_util["optimization_potential"] = 0.0

# High utilization - capacity risk
high_util_mask = warehouse_util["utilization_pct"] > 80
warehouse_util.loc[high_util_mask, "recommendation"] = "Expand capacity or redistribute inventory"
warehouse_util.loc[high_util_mask, "optimization_potential"] = (warehouse_util.loc[high_util_mask, "utilization_pct"] - 80) / 100

# Low utilization - underutilized
low_util_mask = warehouse_util["utilization_pct"] < 50
warehouse_util.loc[low_util_mask, "recommendation"] = "Consolidate inventory or reduce footprint"
warehouse_util.loc[low_util_mask, "optimization_potential"] = (50 - warehouse_util.loc[low_util_mask, "utilization_pct"]) / 100

# Medium utilization - balanced
medium_util_mask = ~high_util_mask & ~low_util_mask
warehouse_util.loc[medium_util_mask, "recommendation"] = "Maintain current operations"
warehouse_util.loc[medium_util_mask, "optimization_potential"] = 0.1

# Cluster-based recommendations
warehouse_util.loc[warehouse_util["cluster_label"] == "High-Utilization", "recommendation"] = "Consider expansion or overflow to other warehouses"
warehouse_util.loc[warehouse_util["cluster_label"] == "Underutilized", "recommendation"] = "Redirect inventory from high-utilization warehouses"
warehouse_util.loc[warehouse_util["cluster_label"] == "Overstocked", "recommendation"] = "Run promotions or reduce incoming orders"
warehouse_util.loc[warehouse_util["cluster_label"] == "Balanced", "recommendation"] = "Maintain current operations"

print("\nRecommendations:")
for _, row in warehouse_util.iterrows():
    print(f"  {row['warehouse_id']} ({row['warehouse_name']}): {row['recommendation']}")
    print(f"    Utilization: {row['utilization_pct']:.1f}% | Capacity Risk: {row['capacity_risk']}")

# ============================================================
# 5. SAVE RESULTS
# ============================================================
print("\n=== 5. SAVING RESULTS ===")

os.makedirs("data/processed", exist_ok=True)

warehouse_util.to_csv("data/processed/warehouse_optimization.csv", index=False)
print(f"Saved: data/processed/warehouse_optimization.csv ({len(warehouse_util)} warehouses)")

# Summary statistics
summary = {
    "total_warehouses": len(warehouse_util),
    "total_capacity_m3": warehouse_util["capacity_m3"].sum(),
    "total_occupied_m3": warehouse_util["occupied_volume_m3"].sum(),
    "avg_utilization_pct": warehouse_util["utilization_pct"].mean(),
    "high_utilization_count": high_util.sum(),
    "medium_utilization_count": medium_util.sum(),
    "low_utilization_count": low_util.sum(),
    "high_capacity_risk": (warehouse_util["capacity_risk"] == "HIGH").sum(),
    "medium_capacity_risk": (warehouse_util["capacity_risk"] == "MEDIUM").sum(),
    "low_capacity_risk": (warehouse_util["capacity_risk"] == "LOW").sum(),
    "balanced_warehouses": (warehouse_util["cluster_label"] == "Balanced").sum(),
    "overstocked_warehouses": (warehouse_util["cluster_label"] == "Overstocked").sum(),
    "underutilized_warehouses": (warehouse_util["cluster_label"] == "Underutilized").sum(),
    "high_utilization_warehouses": (warehouse_util["cluster_label"] == "High-Utilization").sum(),
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv("data/processed/warehouse_optimization_summary.csv", index=False)
print(f"Saved: data/processed/warehouse_optimization_summary.csv")

# ============================================================
# 6. UPDATE DATABASE
# ============================================================
print("\n=== 6. UPDATING DATABASE ===")

with engine.connect() as conn:
    # Create warehouse optimization table
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS warehouse_optimization (
            warehouse_id TEXT PRIMARY KEY,
            warehouse_name TEXT,
            city TEXT,
            state TEXT,
            capacity_m3 REAL,
            occupied_volume_m3 REAL,
            utilization_pct REAL,
            available_capacity_m3 REAL,
            capacity_risk TEXT,
            capacity_risk_reason TEXT,
            total_quantity INTEGER,
            distinct_products INTEGER,
            turnover_ratio REAL,
            cluster_label TEXT,
            recommendation TEXT,
            optimization_potential REAL,
            FOREIGN KEY (warehouse_id) REFERENCES warehouses(warehouse_id)
        )
    """))
    
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_wh_opt_id ON warehouse_optimization(warehouse_id)"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_wh_opt_risk ON warehouse_optimization(capacity_risk)"))
    conn.commit()
    
    # Clear existing data
    conn.execute(text("DELETE FROM warehouse_optimization"))
    conn.commit()
    
    # Load new data (select only columns that exist in schema)
    cols_for_db = ["warehouse_id", "warehouse_name", "city", "state", "capacity_m3",
                   "occupied_volume_m3", "utilization_pct", "available_capacity_m3",
                   "capacity_risk", "capacity_risk_reason", "total_quantity",
                   "distinct_products", "turnover_ratio", "cluster_label",
                   "recommendation", "optimization_potential"]
    warehouse_util[cols_for_db].to_sql("warehouse_optimization", con=engine, if_exists="append", index=False)
    
    count = conn.execute(text("SELECT COUNT(*) FROM warehouse_optimization")).fetchone()[0]
    print(f"Loaded {count} warehouse records into database")

# ============================================================
# 7. SUMMARY
# ============================================================
print("\n=== 7. WAREHOUSE OPTIMIZATION SUMMARY ===")

print(f"""
Total Warehouses: {summary['total_warehouses']}
Total Capacity: {summary['total_capacity_m3']:,.0f} m³
Total Occupied: {summary['total_occupied_m3']:,.0f} m³
Average Utilization: {summary['avg_utilization_pct']:.1f}%

Utilization Distribution:
  - High (>80%): {summary['high_utilization_count']} warehouses
  - Medium (50-80%): {summary['medium_utilization_count']} warehouses
  - Low (<50%): {summary['low_utilization_count']} warehouses

Capacity Risk:
  - HIGH: {summary['high_capacity_risk']}
  - MEDIUM: {summary['medium_capacity_risk']}
  - LOW: {summary['low_capacity_risk']}

Cluster Distribution:
  - Balanced: {summary['balanced_warehouses']}
  - Overstocked: {summary['overstocked_warehouses']}
  - Underutilized: {summary['underutilized_warehouses']}
  - High-Utilization: {summary['high_utilization_warehouses']}
""")

print("Warehouse optimization complete.")
