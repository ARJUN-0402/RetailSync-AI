import pandas as pd
from sqlalchemy import create_engine, text

print("=== LOADING INVENTORY ALERTS INTO DATABASE ===\n")

# Load inventory intelligence results
alerts_df = pd.read_csv("data/processed/inventory_intelligence.csv")

# Derive alert date from the latest inventory snapshot
alert_date = pd.to_datetime(alerts_df["date"].max()).strftime("%Y-%m-%d")
print(f"Alert date: {alert_date} (derived from latest inventory snapshot)")

# Filter to only alerts (non-LOW risk items)
alert_types = []
for _, row in alerts_df.iterrows():
    if row["stockout_risk"] in ["HIGH", "MEDIUM"]:
        alert_types.append(
            {
                "product_id": row["product_id"],
                "store_id": row["store_id"],
                "warehouse_id": row["warehouse_id"],
                "alert_date": alert_date,
                "alert_type": "Stockout Risk",
                "risk_level": row["stockout_risk"],
                "reason": row["stockout_reason"],
                "quantity_on_hand": row["quantity_on_hand"],
                "reorder_point": row["reorder_point"],
                "max_stock_level": row["max_stock_level"],
                "stock_coverage_days": row["stock_coverage_days"],
                "forecast_demand_7d": row["forecast_demand_7d"],
                "recommended_action": row["recommended_action"],
            }
        )

    if row["overstock_risk"] in ["HIGH", "MEDIUM"]:
        alert_types.append(
            {
                "product_id": row["product_id"],
                "store_id": row["store_id"],
                "warehouse_id": row["warehouse_id"],
                "alert_date": alert_date,
                "alert_type": "Overstock Risk",
                "risk_level": row["overstock_risk"],
                "reason": row["overstock_reason"],
                "quantity_on_hand": row["quantity_on_hand"],
                "reorder_point": row["reorder_point"],
                "max_stock_level": row["max_stock_level"],
                "stock_coverage_days": row["stock_coverage_days"],
                "forecast_demand_7d": row["forecast_demand_7d"],
                "recommended_action": row["recommended_action"],
            }
        )

    if row["dead_stock"]:
        alert_types.append(
            {
                "product_id": row["product_id"],
                "store_id": row["store_id"],
                "warehouse_id": row["warehouse_id"],
                "alert_date": alert_date,
                "alert_type": "Dead Stock",
                "risk_level": "HIGH",
                "reason": row["dead_stock_reason"],
                "quantity_on_hand": row["quantity_on_hand"],
                "reorder_point": row["reorder_point"],
                "max_stock_level": row["max_stock_level"],
                "stock_coverage_days": row["stock_coverage_days"],
                "forecast_demand_7d": row["forecast_demand_14d"],
                "recommended_action": row["recommended_action"],
            }
        )

    if row["reorder_urgency"] in ["URGENT", "SOON"]:
        alert_types.append(
            {
                "product_id": row["product_id"],
                "store_id": row["store_id"],
                "warehouse_id": row["warehouse_id"],
                "alert_date": alert_date,
                "alert_type": "Reorder Urgency",
                "risk_level": row["reorder_urgency"],
                "reason": row["reorder_urgency_reason"],
                "quantity_on_hand": row["quantity_on_hand"],
                "reorder_point": row["reorder_point"],
                "max_stock_level": row["max_stock_level"],
                "stock_coverage_days": row["stock_coverage_days"],
                "forecast_demand_7d": row["forecast_demand_7d"],
                "recommended_action": row["recommended_action"],
            }
        )

alerts_df = pd.DataFrame(alert_types)
print(f"Total alerts generated: {len(alerts_df)}")
print(f"Alert types: {alerts_df['alert_type'].value_counts().to_dict()}")
print(f"Risk levels: {alerts_df['risk_level'].value_counts().to_dict()}")

# Create database table and load alerts
engine = create_engine("sqlite:///database/retailsync.db")

with engine.connect() as conn:
    # Create table
    with open("database/alerts_schema.sql", "r", encoding="utf-8") as f:
        schema_sql = f.read()
    statements = [s.strip() for s in schema_sql.split(";") if s.strip()]
    for stmt in statements:
        conn.execute(text(stmt))
    conn.commit()
    print("Created inventory_alerts table")

    # Clear existing alerts
    conn.execute(text("DELETE FROM inventory_alerts"))
    conn.commit()
    print("Cleared existing alerts")

    # Load new alerts
    alerts_df.to_sql("inventory_alerts", con=engine, if_exists="append", index=False)
    print(f"Loaded {len(alerts_df)} alerts into database")

# Verify
with engine.connect() as conn:
    count = conn.execute(text("SELECT COUNT(*) FROM inventory_alerts")).fetchone()[0]
    print(f"Total alerts in database: {count}")

    # Sample alerts
    sample = conn.execute(
        text("""
        SELECT alert_type, risk_level, COUNT(*) as count
        FROM inventory_alerts
        GROUP BY alert_type, risk_level
        ORDER BY alert_type, risk_level
    """)
    ).fetchall()
    print("\nAlert summary:")
    for row in sample:
        print(f"  {row[0]} | {row[1]}: {row[2]}")

print("\nInventory alerts loaded successfully.")
