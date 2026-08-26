"""Inventory intelligence and risk detection for RetailSync AI."""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from src.config import settings
from src.exceptions import DataError, PipelineError
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

np.random.seed(42)


def detect_risks() -> pd.DataFrame:
    """Detect inventory risks and return enriched inventory dataframe."""
    setup_logging(__name__)
    logger.info("=== RetailSync AI - Inventory Intelligence ===")

    features_path = os.path.join(str(settings.paths.processed_data), "features_daily.csv")
    if not os.path.exists(features_path):
        raise DataError(f"Features file not found: {features_path}")

    features_df = pd.read_csv(features_path, parse_dates=["date"])

    db_path = os.path.join(str(settings.paths.database), "retailsync.db")
    if not os.path.exists(db_path):
        raise DataError(f"Database not found: {db_path}")

    engine = create_engine(f"sqlite:///{db_path}")
    inventory_df = pd.read_sql("SELECT * FROM inventory", engine)
    inventory_df["date"] = pd.to_datetime(inventory_df["date"])

    latest_date = inventory_df["date"].max()
    latest_inventory = inventory_df[inventory_df["date"] == latest_date].copy()
    logger.info("Latest inventory date: %s", latest_date.date())
    logger.info("Latest inventory records: %d", len(latest_inventory))

    latest_inventory["stockout_risk"] = "LOW"
    latest_inventory["stockout_reason"] = ""

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

    logger.info("Stockout risk distribution:\n%s", latest_inventory["stockout_risk"].value_counts().to_string())

    latest_inventory["overstock_risk"] = "LOW"
    latest_inventory["overstock_reason"] = ""

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

    logger.info("Overstock risk distribution:\n%s", latest_inventory["overstock_risk"].value_counts().to_string())

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
        "High inventory with no recent demand and excess coverage"
    )
    logger.info("Dead stock detected: %d", dead_condition.sum())

    latest_inventory["reorder_urgency"] = "NONE"
    latest_inventory["reorder_reason"] = ""

    urgent_reorder = latest_inventory["quantity_on_hand"] <= latest_inventory["reorder_point"] * 0.5
    soon_reorder = (
        (latest_inventory["quantity_on_hand"] > latest_inventory["reorder_point"] * 0.5)
        & (latest_inventory["quantity_on_hand"] <= latest_inventory["reorder_point"])
    )
    monitor_reorder = (
        (latest_inventory["quantity_on_hand"] > latest_inventory["reorder_point"])
        & (
            latest_inventory["quantity_on_hand"]
            <= latest_inventory["reorder_point"] * 1.5
        )
    )

    latest_inventory.loc[urgent_reorder, "reorder_urgency"] = "URGENT"
    latest_inventory.loc[urgent_reorder, "reorder_reason"] = "Stock critically low"
    latest_inventory.loc[soon_reorder, "reorder_urgency"] = "SOON"
    latest_inventory.loc[soon_reorder, "reorder_reason"] = "Approaching reorder point"
    latest_inventory.loc[monitor_reorder, "reorder_urgency"] = "MONITOR"
    latest_inventory.loc[monitor_reorder, "reorder_reason"] = "Near reorder threshold"
    latest_inventory.loc[~urgent_reorder & ~soon_reorder & ~monitor_reorder, "reorder_urgency"] = "NONE"
    latest_inventory.loc[~urgent_reorder & ~soon_reorder & ~monitor_reorder, "reorder_reason"] = (
        "Adequate stock"
    )

    logger.info("Reorder urgency distribution:\n%s", latest_inventory["reorder_urgency"].value_counts().to_string())

    latest_inventory["stockout_score"] = latest_inventory["stockout_risk"].map(
        {"HIGH": 100, "MEDIUM": 60, "LOW": 20}
    )
    latest_inventory["overstock_score"] = latest_inventory["overstock_risk"].map(
        {"HIGH": 100, "MEDIUM": 60, "LOW": 20}
    )
    latest_inventory["reorder_score"] = latest_inventory["reorder_urgency"].map(
        {"URGENT": 100, "SOON": 60, "MONITOR": 30, "NONE": 0}
    )
    latest_inventory["dead_stock_score"] = latest_inventory["dead_stock"].astype(int) * 100

    latest_inventory["composite_risk_score"] = (
        latest_inventory["stockout_score"] * 0.35
        + latest_inventory["overstock_score"] * 0.25
        + latest_inventory["reorder_score"] * 0.25
        + latest_inventory["dead_stock_score"] * 0.15
    ).round(2)

    def generate_recommendation(row):
        recommendations = []
        if row["stockout_risk"] == "HIGH":
            recommendations.append("Immediate reorder required")
        if row["overstock_risk"] == "HIGH":
            recommendations.append("Consider promotion or reduction")
        if row["reorder_urgency"] == "URGENT":
            recommendations.append("Expedite procurement")
        if row["dead_stock"]:
            recommendations.append("Mark for clearance or write-off")
        if not recommendations:
            recommendations.append("Monitor stock levels")
        return "; ".join(recommendations)

    latest_inventory["recommended_action"] = latest_inventory.apply(
        generate_recommendation, axis=1
    )

    output_path = os.path.join(str(settings.paths.processed_data), "inventory_intelligence.csv")
    latest_inventory.to_csv(output_path, index=False)
    logger.info("Saved inventory intelligence to %s", output_path)

    logger.info("High stockout: %d", (latest_inventory["stockout_risk"] == "HIGH").sum())
    logger.info("Medium stockout: %d", (latest_inventory["stockout_risk"] == "MEDIUM").sum())
    logger.info("High overstock: %d", (latest_inventory["overstock_risk"] == "HIGH").sum())
    logger.info("Urgent reorder: %d", (latest_inventory["reorder_urgency"] == "URGENT").sum())

    return latest_inventory


def main() -> None:
    """Main entry point."""
    try:
        detect_risks()
    except (DataError, PipelineError):
        raise
    except Exception as exc:
        raise PipelineError(f"Inventory intelligence failed: {exc}") from exc


if __name__ == "__main__":
    main()
