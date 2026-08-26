import os

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

np.random.seed(42)

print("=== RETAILSYNC AI - INVENTORY INTELLIGENCE ===\n")

# Load data
print("Loading data...")
features_df = pd.read_csv("data/processed/features_daily.csv", parse_dates=["date"])
engine = create_engine("sqlite:///database/retailsync.db")
inventory_df = pd.read_sql("SELECT * FROM inventory", engine)
inventory_df["date"] = pd.to_datetime(inventory_df["date"])

# Get latest inventory snapshot
latest_date = inventory_df["date"].max()
latest_inventory = inventory_df[inventory_df["date"] == latest_date].copy()
print(f"Latest inventory date: {latest_date.date()}")
print(f"Latest inventory records: {len(latest_inventory)}")

# ============================================================
# 1. STOCKOUT RISK DETECTION
# ============================================================
print("\n=== 1. STOCKOUT RISK DETECTION ===")

latest_inventory["stockout_risk"] = "LOW"
latest_inventory["stockout_reason"] = ""

# Rule-based stockout risk
# HIGH: quantity_on_hand <= 0 OR quantity_on_hand <= reorder_point * 0.5
# MEDIUM: quantity_on_hand <= reorder_point
# LOW: otherwise

high_stockout = latest_inventory["quantity_on_hand"] <= 0
medium_stockout = (latest_inventory["quantity_on_hand"] > 0) & (
    latest_inventory["quantity_on_hand"] <= latest_inventory["reorder_point"]
)
low_stockout = latest_inventory["quantity_on_hand"] > latest_inventory["reorder_point"]

latest_inventory.loc[high_stockout, "stockout_risk"] = "HIGH"
latest_inventory.loc[high_stockout, "stockout_reason"] = "Out of stock"
latest_inventory.loc[medium_stockout, "stockout_risk"] = "MEDIUM"
latest_inventory.loc[medium_stockout, "stockout_reason"] = "Below reorder point"
latest_inventory.loc[low_stockout, "stockout_risk"] = "LOW"
latest_inventory.loc[low_stockout, "stockout_reason"] = "Adequate stock"

# Add forecast-based risk
latest_inventory = latest_inventory.merge(
    features_df.groupby(["product_id", "store_id"])
    .agg(
        forecast_demand_7d=("target_demand_7d", "mean"),
        forecast_demand_14d=("target_demand_14d", "mean"),
        demand_cv_28d=("demand_cv_28d", "last"),
        stock_coverage_days=("stock_coverage_days", "last"),
    )
    .reset_index(),
    on=["product_id", "store_id"],
    how="left",
)

# Adjust risk based on forecast
forecast_high = (
    latest_inventory["forecast_demand_7d"] > latest_inventory["quantity_on_hand"]
)
latest_inventory.loc[
    forecast_high & (latest_inventory["stockout_risk"] == "LOW"), "stockout_risk"
] = "MEDIUM"
latest_inventory.loc[
    forecast_high & (latest_inventory["stockout_risk"] == "MEDIUM"), "stockout_risk"
] = "HIGH"
latest_inventory.loc[forecast_high, "stockout_reason"] = (
    latest_inventory.loc[forecast_high, "stockout_reason"] + " + High forecasted demand"
)

print("Stockout risk distribution:")
print(latest_inventory["stockout_risk"].value_counts())

# ============================================================
# 2. OVERSTOCK RISK DETECTION
# ============================================================
print("\n=== 2. OVERSTOCK RISK DETECTION ===")

latest_inventory["overstock_risk"] = "LOW"
latest_inventory["overstock_reason"] = ""

# Rule-based overstock risk
# HIGH: quantity_on_hand > max_stock_level * 1.5
# MEDIUM: quantity_on_hand > max_stock_level
# LOW: otherwise

high_overstock = (
    latest_inventory["quantity_on_hand"] > latest_inventory["max_stock_level"] * 1.5
)
medium_overstock = (
    latest_inventory["quantity_on_hand"] > latest_inventory["max_stock_level"]
) & (latest_inventory["quantity_on_hand"] <= latest_inventory["max_stock_level"] * 1.5)
low_overstock = (
    latest_inventory["quantity_on_hand"] <= latest_inventory["max_stock_level"]
)

latest_inventory.loc[high_overstock, "overstock_risk"] = "HIGH"
latest_inventory.loc[high_overstock, "overstock_reason"] = "Exceeds max stock by >50%"
latest_inventory.loc[medium_overstock, "overstock_risk"] = "MEDIUM"
latest_inventory.loc[medium_overstock, "overstock_reason"] = "Exceeds max stock level"
latest_inventory.loc[low_overstock, "overstock_risk"] = "LOW"
latest_inventory.loc[low_overstock, "overstock_reason"] = "Within stock limits"

# Adjust based on demand variability
high_cv = latest_inventory["demand_cv_28d"] > 3.0
medium_cv = (latest_inventory["demand_cv_28d"] > 2.0) & (
    latest_inventory["demand_cv_28d"] <= 3.0
)
latest_inventory.loc[
    high_cv & (latest_inventory["overstock_risk"] == "MEDIUM"), "overstock_risk"
] = "HIGH"
latest_inventory.loc[
    high_cv & (latest_inventory["overstock_risk"] == "LOW"), "overstock_risk"
] = "HIGH"
latest_inventory.loc[
    medium_cv & (latest_inventory["overstock_risk"] == "LOW"), "overstock_risk"
] = "MEDIUM"

print("Overstock risk distribution:")
print(latest_inventory["overstock_risk"].value_counts())

# ============================================================
# 3. DEAD STOCK DETECTION
# ============================================================
print("\n=== 3. DEAD STOCK DETECTION ===")

# Dead stock: high inventory + low/zero demand + low turnover
# Criteria:
# - quantity_on_hand > max_stock_level * 0.8 (high inventory)
# - forecast_demand_14d == 0 OR demand_cv_28d == 0 (no recent demand)
# - stock_coverage_days > 90 (excess coverage)

latest_inventory["dead_stock"] = False
latest_inventory["dead_stock_reason"] = ""

dead_condition = (
    (latest_inventory["quantity_on_hand"] > latest_inventory["max_stock_level"] * 0.8)
    & (
        (latest_inventory["forecast_demand_14d"] == 0)
        | (latest_inventory["demand_cv_28d"] == 0)
    )
    & (latest_inventory["stock_coverage_days"] > 90)
)

latest_inventory.loc[dead_condition, "dead_stock"] = True
latest_inventory.loc[dead_condition, "dead_stock_reason"] = (
    "High inventory with no demand"
)

# Also flag if quantity_on_hand > 0 but no sales in last 28 days
no_sales_28d = (
    features_df.groupby(["product_id", "store_id"])
    .agg(
        sales_last_28d=(
            "quantity_sold",
            lambda x: x.iloc[-28:].sum() if len(x) >= 28 else x.sum(),
        )
    )
    .reset_index()
)

latest_inventory = latest_inventory.merge(
    no_sales_28d, on=["product_id", "store_id"], how="left"
)
no_recent_sales = (latest_inventory["sales_last_28d"] == 0) & (
    latest_inventory["quantity_on_hand"] > 0
)
latest_inventory.loc[no_recent_sales, "dead_stock"] = True
latest_inventory.loc[no_recent_sales, "dead_stock_reason"] = "No sales in last 28 days"

dead_stock_count = latest_inventory["dead_stock"].sum()
print(
    f"Dead stock items: {dead_stock_count} ({dead_stock_count / len(latest_inventory) * 100:.1f}%)"
)

# ============================================================
# 4. INVENTORY COVERAGE & REORDER INSIGHTS
# ============================================================
print("\n=== 4. INVENTORY COVERAGE & REORDER INSIGHTS ===")

latest_inventory["stock_coverage_days"] = np.where(
    latest_inventory["forecast_demand_7d"] > 0,
    latest_inventory["quantity_on_hand"] / latest_inventory["forecast_demand_7d"],
    np.inf,
)
latest_inventory["stock_coverage_days"] = (
    latest_inventory["stock_coverage_days"].replace([np.inf, -np.inf], 999).fillna(999)
)

# Reorder urgency
latest_inventory["reorder_urgency"] = "NONE"
latest_inventory["reorder_urgency_reason"] = ""

urgent_reorder = (
    latest_inventory["quantity_on_hand"] <= latest_inventory["reorder_point"]
)
soon_reorder = (
    latest_inventory["quantity_on_hand"] > latest_inventory["reorder_point"]
) & (latest_inventory["stock_coverage_days"] <= 7)
monitor_reorder = (
    latest_inventory["quantity_on_hand"] > latest_inventory["reorder_point"]
) & (latest_inventory["stock_coverage_days"] <= 14)

latest_inventory.loc[urgent_reorder, "reorder_urgency"] = "URGENT"
latest_inventory.loc[urgent_reorder, "reorder_urgency_reason"] = "Below reorder point"
latest_inventory.loc[soon_reorder, "reorder_urgency"] = "SOON"
latest_inventory.loc[soon_reorder, "reorder_urgency_reason"] = "Coverage <= 7 days"
latest_inventory.loc[monitor_reorder, "reorder_urgency"] = "MONITOR"
latest_inventory.loc[monitor_reorder, "reorder_urgency_reason"] = "Coverage <= 14 days"

print("Reorder urgency distribution:")
print(latest_inventory["reorder_urgency"].value_counts())

# ============================================================
# 5. COMPOSITE INVENTORY SCORE
# ============================================================
print("\n=== 5. COMPOSITE INVENTORY SCORE ===")

# Create composite risk score (0-100)
latest_inventory["stockout_score"] = latest_inventory["stockout_risk"].map(
    {"HIGH": 100, "MEDIUM": 60, "LOW": 20}
)
latest_inventory["overstock_score"] = latest_inventory["overstock_risk"].map(
    {"HIGH": 100, "MEDIUM": 60, "LOW": 20}
)
latest_inventory["dead_stock_score"] = latest_inventory["dead_stock"].map(
    {True: 100, False: 0}
)
latest_inventory["reorder_score"] = latest_inventory["reorder_urgency"].map(
    {"URGENT": 100, "SOON": 70, "MONITOR": 40, "NONE": 0}
)

latest_inventory["composite_risk_score"] = (
    latest_inventory["stockout_score"] * 0.35
    + latest_inventory["overstock_score"] * 0.25
    + latest_inventory["dead_stock_score"] * 0.20
    + latest_inventory["reorder_score"] * 0.20
)

latest_inventory["composite_risk_level"] = pd.cut(
    latest_inventory["composite_risk_score"],
    bins=[0, 25, 50, 75, 100],
    labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
).astype(str)

print("Composite risk distribution:")
print(latest_inventory["composite_risk_level"].value_counts())

# ============================================================
# 6. RECOMMENDATIONS
# ============================================================
print("\n=== 6. RECOMMENDATIONS ===")

latest_inventory["recommended_action"] = "Monitor"

# Stockout recommendations
latest_inventory.loc[
    latest_inventory["stockout_risk"] == "HIGH", "recommended_action"
] = "Immediate restock required"
latest_inventory.loc[
    latest_inventory["stockout_risk"] == "MEDIUM", "recommended_action"
] = "Schedule restock soon"

# Overstock recommendations
latest_inventory.loc[
    latest_inventory["overstock_risk"] == "HIGH", "recommended_action"
] = "Run promotion or reduce orders"
latest_inventory.loc[
    latest_inventory["overstock_risk"] == "MEDIUM", "recommended_action"
] = "Review upcoming demand"

# Dead stock recommendations
latest_inventory.loc[latest_inventory["dead_stock"], "recommended_action"] = (
    "Consider clearance or return to supplier"
)

# Reorder recommendations
latest_inventory.loc[
    latest_inventory["reorder_urgency"] == "URGENT", "recommended_action"
] = "Place emergency order"
latest_inventory.loc[
    latest_inventory["reorder_urgency"] == "SOON", "recommended_action"
] = "Place order within 7 days"

print("Action distribution:")
print(latest_inventory["recommended_action"].value_counts().head(10))

# ============================================================
# 7. SAVE INVENTORY INTELLIGENCE RESULTS
# ============================================================
print("\n=== 7. SAVING RESULTS ===")

os.makedirs("data/processed", exist_ok=True)

# Full inventory intelligence
latest_inventory.to_csv("data/processed/inventory_intelligence.csv", index=False)
print(
    f"Saved: data/processed/inventory_intelligence.csv ({len(latest_inventory)} rows)"
)

# Summary statistics
summary = {
    "total_products_stores": len(latest_inventory),
    "stockout_high": (latest_inventory["stockout_risk"] == "HIGH").sum(),
    "stockout_medium": (latest_inventory["stockout_risk"] == "MEDIUM").sum(),
    "overstock_high": (latest_inventory["overstock_risk"] == "HIGH").sum(),
    "overstock_medium": (latest_inventory["overstock_risk"] == "MEDIUM").sum(),
    "dead_stock_count": latest_inventory["dead_stock"].sum(),
    "urgent_reorder": (latest_inventory["reorder_urgency"] == "URGENT").sum(),
    "soon_reorder": (latest_inventory["reorder_urgency"] == "SOON").sum(),
    "critical_risk": (latest_inventory["composite_risk_level"] == "CRITICAL").sum(),
    "high_risk": (latest_inventory["composite_risk_level"] == "HIGH").sum(),
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv("data/processed/inventory_intelligence_summary.csv", index=False)
print("Saved: data/processed/inventory_intelligence_summary.csv")

print("\n=== INVENTORY INTELLIGENCE SUMMARY ===")
for k, v in summary.items():
    print(f"  {k}: {v}")

# Top critical items
print("\n=== TOP 10 CRITICAL INVENTORY ITEMS ===")
critical_items = (
    latest_inventory[
        latest_inventory["composite_risk_level"].isin(["CRITICAL", "HIGH"])
    ]
    .sort_values("composite_risk_score", ascending=False)
    .head(10)
)

print(
    critical_items[
        [
            "product_id",
            "store_id",
            "stockout_risk",
            "overstock_risk",
            "dead_stock",
            "reorder_urgency",
            "composite_risk_score",
            "recommended_action",
        ]
    ].to_string(index=False)
)

print("\nInventory intelligence complete.")
