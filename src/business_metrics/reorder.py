"""Reorder recommendation engine for RetailSync AI.

Builds reusable recommendations using:
    - predicted demand
    - current stock
    - reorder point
    - safety stock
    - lead time
    - demand variability

Returns:
    - recommended reorder quantity
    - reorder urgency
    - reasoning
    - stockout risk
    - expected coverage
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .config import BusinessConfig
from .utils import safe_num, safe_div

logger = logging.getLogger(__name__)


def _compute_safety_stock(
    avg_demand: float,
    demand_std: float,
    lead_time_days: int,
    z_score: float,
) -> float:
    """Compute safety stock using the standard statistical formula.

    safety_stock = z * std_demand * sqrt(lead_time)

    If demand_std is zero, safety_stock is zero.
    """
    if demand_std <= 0 or lead_time_days <= 0:
        return 0.0
    return safe_num(z_score * demand_std * np.sqrt(lead_time_days), default=0.0)


def _compute_lead_time_demand(avg_demand: float, lead_time_days: int) -> float:
    """Compute expected demand during lead time."""
    if avg_demand <= 0 or lead_time_days <= 0:
        return 0.0
    return safe_num(avg_demand * lead_time_days, default=0.0)


def generate_reorder_recommendations(
    inv_intel_df: pd.DataFrame,
    products_df: pd.DataFrame,
    forecasts_df: pd.DataFrame,
    suppliers_df: pd.DataFrame,
    config: Optional[BusinessConfig] = None,
) -> pd.DataFrame:
    """Generate reorder recommendations for each product-store combination.

    Recommendation logic:
        1. Determine lead time (from suppliers or default).
        2. Compute lead-time demand from forecast_demand_14d.
        3. Compute safety stock from demand_cv_28d and rolling std.
        4. Compute reorder point = lead_time_demand + safety_stock.
        5. Recommended quantity = max(0, reorder_point - quantity_on_hand).
        6. Expected coverage = quantity_on_hand / avg_daily_demand (if > 0).
        7. Urgency based on coverage days.

    Args:
        inv_intel_df: Latest inventory intelligence snapshot.
        products_df: Product master.
        forecasts_df: 14-day forecasts with forecast_demand.
        suppliers_df: Supplier master with lead_time_days.
        config: BusinessConfig.

    Returns:
        DataFrame with reorder recommendations per product-store.
    """
    config = config or BusinessConfig()
    if inv_intel_df is None or inv_intel_df.empty:
        return pd.DataFrame()

    df = inv_intel_df.copy()
    required = ["product_id", "store_id", "quantity_on_hand"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()

    # Merge with products for cost and category info
    product_cols = ["product_id", "cost_price"]
    if "category" in products_df.columns:
        product_cols.append("category")
    if "cost_price" not in df.columns and products_df is not None:
        df = df.merge(
            products_df[product_cols],
            on="product_id",
            how="left",
        )
    df["cost_price"] = df.get("cost_price", pd.Series(dtype=float)).fillna(0)

    # Merge with suppliers for lead time
    if suppliers_df is not None and products_df is not None and "supplier_id" in products_df.columns:
        suppliers = products_df[["product_id", "supplier_id"]].merge(
            suppliers_df[["supplier_id", "lead_time_days"]], on="supplier_id", how="left"
        )
        df = df.merge(
            suppliers[["product_id", "lead_time_days"]], on="product_id", how="left"
        )
    if "lead_time_days" not in df.columns:
        df["lead_time_days"] = config.lead_time_default_days
    df["lead_time_days"] = df["lead_time_days"].fillna(config.lead_time_default_days).clip(lower=1)

    # Merge with forecasts for predicted demand
    if forecasts_df is not None and not forecasts_df.empty:
        forecast_agg = (
            forecasts_df.groupby(["product_id", "store_id"])
            .agg(
                forecast_demand_14d=("forecast_demand", "sum"),
                forecast_revenue_14d=("forecast_revenue", "sum"),
            )
            .reset_index()
        )
        # Drop conflicting columns from inv_intel before merge
        drop_cols = [c for c in ["forecast_demand_14d", "forecast_revenue_14d"] if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
        df = df.merge(forecast_agg, on=["product_id", "store_id"], how="left")
    else:
        df["forecast_demand_14d"] = 0.0
        df["forecast_revenue_14d"] = 0.0

    df["forecast_demand_14d"] = df["forecast_demand_14d"].fillna(0)
    df["avg_daily_demand"] = df["forecast_demand_14d"] / config.forecast_accuracy_period_days

    # Demand variability: use demand_cv_28d to estimate std
    if "demand_cv_28d" in df.columns:
        df["demand_std"] = df["demand_cv_28d"] * df["avg_daily_demand"]
    else:
        df["demand_std"] = 0.0
    df["demand_std"] = df["demand_std"].fillna(0).clip(lower=0)

    # Compute safety stock vectorized
    if "demand_std" not in df.columns:
        df["demand_std"] = 0.0
    df["demand_std"] = df["demand_std"].fillna(0).clip(lower=0)
    df["safety_stock"] = np.where(
        (df["demand_std"] > 0) & (df["lead_time_days"] > 0),
        config.safety_stock_z_score * df["demand_std"] * np.sqrt(df["lead_time_days"]),
        0.0,
    )

    # Compute lead-time demand vectorized
    df["lead_time_demand"] = np.where(
        (df["avg_daily_demand"] > 0) & (df["lead_time_days"] > 0),
        df["avg_daily_demand"] * df["lead_time_days"],
        0.0,
    )

    # Reorder point = lead_time_demand + safety_stock
    df["reorder_point_computed"] = df["lead_time_demand"] + df["safety_stock"]

    # Recommended quantity = max(0, reorder_point_computed - quantity_on_hand)
    df["recommended_quantity"] = (
        df["reorder_point_computed"] - df["quantity_on_hand"]
    ).clip(lower=0)

    # Expected coverage days vectorized
    df["expected_coverage_days"] = np.where(
        df["avg_daily_demand"] > 0,
        df["quantity_on_hand"] / df["avg_daily_demand"],
        999.0,
    )
    df["expected_coverage_days"] = df["expected_coverage_days"].fillna(999.0)

    # Reorder urgency vectorized
    coverage = df["expected_coverage_days"].fillna(999.0)
    urgency_conditions = [
        df["quantity_on_hand"] <= 0,
        coverage <= 3,
        coverage <= 7,
        coverage <= 14,
    ]
    urgency_choices = ["CRITICAL", "URGENT", "SOON", "MONITOR"]
    df["reorder_urgency_computed"] = np.select(urgency_conditions, urgency_choices, default="NONE")

    # Reasoning vectorized
    reasons = []
    if df["quantity_on_hand"].le(0).any():
        reasons.append(np.where(df["quantity_on_hand"] <= 0, "Out of stock", ""))
    if coverage.le(3).any():
        reasons.append(np.where(coverage <= 3, "Coverage below 3 days", ""))
    if coverage.le(7).any():
        reasons.append(np.where(coverage <= 7, "Coverage below 7 days", ""))
    if (df["safety_stock"] > df["quantity_on_hand"]).any():
        reasons.append(np.where(df["safety_stock"] > df["quantity_on_hand"], "Below safety stock", ""))
    if (df["lead_time_demand"] > df["quantity_on_hand"]).any():
        reasons.append(np.where(df["lead_time_demand"] > df["quantity_on_hand"], "Insufficient for lead time", ""))

    df["reorder_reasoning"] = "Adequate stock"
    for r in reasons:
        df["reorder_reasoning"] = np.where(
            r != "",
            np.where(df["reorder_reasoning"] == "Adequate stock", r, df["reorder_reasoning"] + "; " + r),
            df["reorder_reasoning"],
        )

    # Stockout risk vectorized
    stockout_conditions = [
        df["quantity_on_hand"] <= 0,
        df["quantity_on_hand"] <= df["reorder_point"],
        df["forecast_demand_14d"] > df["quantity_on_hand"],
    ]
    stockout_choices = ["HIGH", "HIGH", "MEDIUM"]
    df["stockout_risk_computed"] = np.select(stockout_conditions, stockout_choices, default="LOW")

    # Order value
    df["reorder_value"] = df["recommended_quantity"] * df["cost_price"]

    # Select output columns
    out_cols = [
        "product_id",
        "store_id",
        "quantity_on_hand",
        "avg_daily_demand",
        "forecast_demand_14d",
        "lead_time_days",
        "safety_stock",
        "lead_time_demand",
        "reorder_point_computed",
        "recommended_quantity",
        "reorder_value",
        "expected_coverage_days",
        "reorder_urgency_computed",
        "stockout_risk_computed",
        "reorder_reasoning",
    ]
    # Only keep columns that exist
    out_cols = [c for c in out_cols if c in df.columns]
    return df[out_cols].sort_values(
        ["reorder_urgency_computed", "recommended_quantity"], ascending=[True, False]
    )
