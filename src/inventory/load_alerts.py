"""Load inventory alerts into the database."""

from __future__ import annotations

import logging
import os

import pandas as pd
from sqlalchemy import create_engine, text

from src.config import settings
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def load() -> None:
    """Load inventory intelligence alerts into the database."""
    setup_logging(__name__)
    logger.info("=== Loading Inventory Alerts into Database ===")

    alerts_path = os.path.join(str(settings.paths.processed_data), "inventory_intelligence.csv")
    if not os.path.exists(alerts_path):
        raise FileNotFoundError(f"Inventory intelligence file not found: {alerts_path}")

    alerts_df = pd.read_csv(alerts_path)
    alert_date = pd.to_datetime(alerts_df["date"].max()).strftime("%Y-%m-%d")
    logger.info("Alert date: %s", alert_date)

    db_path = os.path.join(str(settings.paths.database), "retailsync.db")
    engine = create_engine(f"sqlite:///{db_path}")

    with engine.connect() as conn:
        conn.execute(text("DELETE FROM inventory_alerts WHERE alert_date = :date"), {"date": alert_date})
        conn.commit()

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
                    "reason": row["reorder_reason"],
                    "quantity_on_hand": row["quantity_on_hand"],
                    "reorder_point": row["reorder_point"],
                    "max_stock_level": row["max_stock_level"],
                    "stock_coverage_days": row["stock_coverage_days"],
                    "forecast_demand_7d": row["forecast_demand_7d"],
                    "recommended_action": row["recommended_action"],
                }
            )

    if alert_types:
        alerts_import_df = pd.DataFrame(alert_types)
        alerts_import_df.to_sql("inventory_alerts", con=engine, if_exists="append", index=False)
        logger.info("Loaded %d inventory alerts into database", len(alert_types))
    else:
        logger.info("No alerts to load")


def main() -> None:
    """Main entry point."""
    load()


if __name__ == "__main__":
    main()
