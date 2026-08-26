"""Warehouse optimization and utilization analysis for RetailSync AI."""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from src.config import settings
from src.exceptions import DataError, PipelineError
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

np.random.seed(42)


def analyze() -> pd.DataFrame:
    """Analyze warehouse utilization and return optimization dataframe."""
    setup_logging(__name__)
    logger.info("=== RetailSync AI - Warehouse Optimization ===")

    db_path = os.path.join(str(settings.paths.database), "retailsync.db")
    if not os.path.exists(db_path):
        raise DataError(f"Database not found: {db_path}")

    engine = create_engine(f"sqlite:///{db_path}")
    warehouses = pd.read_sql("SELECT * FROM warehouses", engine)
    inventory = pd.read_sql("SELECT * FROM inventory", engine)
    inventory["date"] = pd.to_datetime(inventory["date"])
    products = pd.read_sql("SELECT * FROM products", engine)

    logger.info("Warehouses: %d", len(warehouses))
    logger.info("Inventory records: %d", len(inventory))
    logger.info("Products: %d", len(products))

    latest_date = inventory["date"].max()
    latest_inventory = inventory[inventory["date"] == latest_date].copy()
    logger.info("Latest inventory date: %s", latest_date.date())

    latest_inventory = latest_inventory.merge(
        products[["product_id", "volume_m3"]], on="product_id", how="left"
    )

    warehouse_util = (
        latest_inventory.groupby("warehouse_id")
        .agg(
            total_quantity=("quantity_on_hand", "sum"),
            distinct_products=("product_id", "nunique"),
            occupied_volume_m3=(
                "quantity_on_hand",
                lambda x: (x * latest_inventory.loc[x.index, "volume_m3"]).sum(),
            ),
            avg_quantity_per_product=("quantity_on_hand", "mean"),
            max_quantity=("quantity_on_hand", "max"),
            min_quantity=("quantity_on_hand", "min"),
        )
        .reset_index()
    )

    warehouse_util = warehouse_util.merge(
        warehouses[["warehouse_id", "warehouse_name", "city", "state", "capacity_m3"]],
        on="warehouse_id",
        how="left",
    )

    warehouse_util["utilization_pct"] = (
        warehouse_util["occupied_volume_m3"] / warehouse_util["capacity_m3"] * 100
    ).round(2)
    warehouse_util["available_capacity_m3"] = (
        warehouse_util["capacity_m3"] - warehouse_util["occupied_volume_m3"]
    )
    warehouse_util["capacity_risk"] = "LOW"
    warehouse_util["capacity_risk_reason"] = ""

    high_util = warehouse_util["utilization_pct"] > 80
    medium_util = (warehouse_util["utilization_pct"] >= 50) & (
        warehouse_util["utilization_pct"] <= 80
    )
    low_util = warehouse_util["utilization_pct"] < 50

    warehouse_util.loc[high_util, "capacity_risk"] = "HIGH"
    warehouse_util.loc[high_util, "capacity_risk_reason"] = "Near or at capacity"
    warehouse_util.loc[medium_util, "capacity_risk"] = "MEDIUM"
    warehouse_util.loc[medium_util, "capacity_risk_reason"] = "Moderate utilization"
    warehouse_util.loc[low_util, "capacity_risk"] = "LOW"
    warehouse_util.loc[low_util, "capacity_risk_reason"] = "Plenty of capacity available"

    logger.info("Utilization distribution:\n%s", warehouse_util["capacity_risk"].value_counts().to_string())

    warehouse_util["inventory_turnover"] = (
        warehouse_util["total_quantity"] / (warehouse_util["avg_quantity_per_product"] + 1)
    ).round(2)

    warehouse_util["recommendation"] = warehouse_util.apply(
        lambda row: (
            "Expand capacity or redistribute inventory"
            if row["utilization_pct"] > 80
            else (
                "Consider consolidation"
                if row["utilization_pct"] < 50
                else "Maintain current operations"
            )
        ),
        axis=1,
    )

    output_path = os.path.join(str(settings.paths.processed_data), "warehouse_optimization.csv")
    warehouse_util.to_csv(output_path, index=False)
    logger.info("Saved warehouse optimization to %s", output_path)

    logger.info("Total warehouses: %d", len(warehouse_util))
    logger.info("Total capacity: %s m3", f"{warehouse_util['capacity_m3'].sum():,.0f}")
    logger.info("Total occupied: %s m3", f"{warehouse_util['occupied_volume_m3'].sum():,.0f}")
    logger.info("Average utilization: %.1f%%", warehouse_util["utilization_pct"].mean())
    logger.info("High utilization (>80%%): %d", high_util.sum())
    logger.info("Low utilization (<50%%): %d", low_util.sum())

    with engine.connect() as conn:
        conn.execute(text("DELETE FROM warehouse_optimization"))
        conn.commit()
    warehouse_util.to_sql("warehouse_optimization", con=engine, if_exists="append", index=False)
    logger.info("Loaded warehouse optimization into database")

    return warehouse_util


def main() -> None:
    """Main entry point."""
    try:
        analyze()
    except (DataError, PipelineError):
        raise
    except Exception as exc:
        raise PipelineError(f"Warehouse optimization failed: {exc}") from exc


if __name__ == "__main__":
    main()
