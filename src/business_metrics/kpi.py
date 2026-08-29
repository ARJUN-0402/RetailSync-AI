"""Core KPI calculation functions for RetailSync AI.

Each function accepts raw dataframes and a BusinessConfig, and returns
structured results that can be consumed by the dashboard or tests.

Design principles:
- No arbitrary numbers presented as facts.
- Clearly distinguish actual values from estimated values.
- Business assumptions must be visible and configurable.
- Calculations must be reproducible.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .config import BusinessConfig
from .utils import (
    compute_smape_batch,
    safe_num,
)

logger = logging.getLogger(__name__)

_FORECAST_ACCURACY_CACHE: dict[str, dict] = {}


def _forecast_accuracy_cache_key(features_df, model_package) -> str:
    features_hash = pd.util.hash_pandas_object(features_df).sum() if features_df is not None else 0
    model_hash = id(model_package) if model_package is not None else 0
    return f"{features_hash}:{model_hash}"


# ============================================================
# 1. FORECAST ACCURACY
# ============================================================

def compute_forecast_accuracy(
    features_df: pd.DataFrame,
    model_package: Optional[dict] = None,
    config: Optional[BusinessConfig] = None,
) -> dict:
    """Compute forecast accuracy metrics.

    Metrics:
    - MAE, RMSE, sMAPE, bias at the aggregate level.
    - Per-product, per-store, per-category breakdowns.

    The function evaluates the loaded model against the test set
    (dates > 2025-06-09) to avoid data leakage. If no model is
    provided, it returns only what can be computed from existing
    forecast files.

    Args:
        features_df: Engineered daily features with target columns.
        model_package: Dict containing model, feature_cols, model_name.
        config: BusinessConfig (unused here but kept for API consistency).

    Returns:
        Dict with overall metrics and breakdowns.
    """
    config = config or BusinessConfig()
    cache_key = _forecast_accuracy_cache_key(features_df, model_package)
    if cache_key in _FORECAST_ACCURACY_CACHE:
        return _FORECAST_ACCURACY_CACHE[cache_key]

    result = {
        "overall": {},
        "by_product": pd.DataFrame(),
        "by_store": pd.DataFrame(),
        "by_category": pd.DataFrame(),
        "model_name": "N/A",
        "test_rows": 0,
        "note": "",
    }

    if features_df is None or features_df.empty:
        result["note"] = "No feature data available."
        return result

    required = ["date", "product_id", "store_id", "category", "target_demand_1d"]
    missing = [c for c in required if c not in features_df.columns]
    if missing:
        result["note"] = f"Missing required columns: {missing}"
        return result

    test_end = pd.Timestamp("2025-06-09")
    test_df = features_df[features_df["date"] > test_end].copy()
    if test_df.empty:
        result["note"] = "No test data available after 2025-06-09."
        return result

    result["test_rows"] = len(test_df)

    # If model is provided, compute predictions
    if model_package and "model" in model_package and "feature_cols" in model_package:
        model = model_package["model"]
        feature_cols = model_package["feature_cols"]
        model_name = model_package.get("model_name", "Unknown")

        available_cols = [c for c in feature_cols if c in test_df.columns]
        if not available_cols:
            result["note"] = "No matching feature columns in test set."
            return result

        X_test = test_df[available_cols].copy()
        X_test = X_test.select_dtypes(include=[np.number]).fillna(0)

        try:
            preds = model.predict(X_test.values)
            preds = np.maximum(preds, 0)  # demand cannot be negative
        except Exception as exc:
            result["note"] = f"Model prediction failed: {exc}"
            return result

        y_true = test_df["target_demand_1d"].values.astype(float)

        # Overall metrics
        mae = float(mean_absolute_error(y_true, preds))
        rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
        smape_val = float(compute_smape_batch(y_true, preds))
        bias = float(np.mean(preds - y_true))

        result["overall"] = {
            "model": model_name,
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "smape": round(smape_val, 2),
            "bias": round(bias, 4),
            "test_rows": len(test_df),
        }
        result["model_name"] = model_name

        test_df = test_df.reset_index(drop=True)
        test_df["prediction"] = preds
        test_df["error"] = preds - y_true
        test_df["abs_error"] = np.abs(test_df["error"])
    else:
        # No model: note that only file-based metrics are available
        result["note"] = (
            "No model provided. Load a model_package to compute "
            "backtested accuracy metrics."
        )
        return result

    # Breakdown by product
    by_product = (
        test_df.groupby("product_id")
        .agg(
            mae=("abs_error", "mean"),
            rmse=("error", lambda x: float(np.sqrt(np.mean(x ** 2)))),
            smape=(
                "target_demand_1d",
                lambda x: float(
                    compute_smape_batch(
                        x.values,
                        test_df.loc[x.index, "prediction"].values,
                    )
                ),
            ),
            bias=("error", "mean"),
            actual_sum=("target_demand_1d", "sum"),
            predicted_sum=("prediction", "sum"),
            samples=("target_demand_1d", "count"),
        )
        .reset_index()
    )
    by_product = by_product.sort_values("mae", ascending=False)
    result["by_product"] = by_product

    # Breakdown by store
    by_store = (
        test_df.groupby("store_id")
        .agg(
            mae=("abs_error", "mean"),
            rmse=("error", lambda x: float(np.sqrt(np.mean(x ** 2)))),
            smape=(
                "target_demand_1d",
                lambda x: float(
                    compute_smape_batch(
                        x.values,
                        test_df.loc[x.index, "prediction"].values,
                    )
                ),
            ),
            bias=("error", "mean"),
            actual_sum=("target_demand_1d", "sum"),
            predicted_sum=("prediction", "sum"),
            samples=("target_demand_1d", "count"),
        )
        .reset_index()
    )
    by_store = by_store.sort_values("mae", ascending=False)
    result["by_store"] = by_store

    # Breakdown by category
    by_category = (
        test_df.groupby("category")
        .agg(
            mae=("abs_error", "mean"),
            rmse=("error", lambda x: float(np.sqrt(np.mean(x ** 2)))),
            smape=(
                "target_demand_1d",
                lambda x: float(
                    compute_smape_batch(
                        x.values,
                        test_df.loc[x.index, "prediction"].values,
                    )
                ),
            ),
            bias=("error", "mean"),
            actual_sum=("target_demand_1d", "sum"),
            predicted_sum=("prediction", "sum"),
            samples=("target_demand_1d", "count"),
        )
        .reset_index()
    )
    by_category = by_category.sort_values("mae", ascending=False)
    result["by_category"] = by_category

    return result


# ============================================================
# 2. INVENTORY CARRYING COST
# ============================================================

def compute_inventory_carrying_cost(
    inv_intel_df: pd.DataFrame,
    products_df: pd.DataFrame,
    config: Optional[BusinessConfig] = None,
    period_days: int = 365,
) -> dict:
    """Compute inventory carrying cost.

    Formula:
        carrying_cost = total_inventory_value * (carrying_cost_pct / 365) * period_days

    Where:
        total_inventory_value = sum(quantity_on_hand * cost_price)

    Assumptions:
        - carrying_cost_pct defaults to 25% (industry standard range 20-30%).
        - cost_price is used as the unit cost proxy.
        - period_days defaults to 365 (annual).

    Args:
        inv_intel_df: Latest inventory intelligence snapshot.
        products_df: Product master with cost_price.
        config: BusinessConfig with carrying_cost_pct.
        period_days: Time period in days for the cost calculation.

    Returns:
        Dict with carrying cost details.
    """
    config = config or BusinessConfig()
    result = {
        "total_inventory_units": 0,
        "total_inventory_value": 0.0,
        "carrying_cost_pct": config.carrying_cost_pct,
        "period_days": period_days,
        "estimated_carrying_cost": 0.0,
        "by_product": pd.DataFrame(),
        "assumptions": {
            "carrying_cost_pct": config.carrying_cost_pct,
            "description": (
                f"{config.carrying_cost_pct:.0%} of inventory value per year, "
                "industry reference range 20-30%."
            ),
        },
    }

    if inv_intel_df is None or inv_intel_df.empty:
        return result
    if products_df is None or products_df.empty:
        return result

    df = inv_intel_df.copy()
    if "quantity_on_hand" not in df.columns:
        return result

    # Merge with products to get cost_price
    if "cost_price" not in df.columns:
        if "product_id" in products_df.columns and "cost_price" in products_df.columns:
            df = df.merge(
                products_df[["product_id", "cost_price"]], on="product_id", how="left"
            )
    else:
        df["cost_price"] = df["cost_price"]

    df["cost_price"] = df["cost_price"].fillna(0)
    df["quantity_on_hand"] = df["quantity_on_hand"].fillna(0)
    df["quantity_on_hand"] = df["quantity_on_hand"].clip(lower=0)

    df["inventory_value"] = df["quantity_on_hand"] * df["cost_price"]
    total_value = float(df["inventory_value"].sum())
    total_units = int(df["quantity_on_hand"].sum())

    carrying_cost = total_value * (config.carrying_cost_pct / 365.0) * float(period_days)

    result["total_inventory_units"] = total_units
    result["total_inventory_value"] = round(total_value, 2)
    result["estimated_carrying_cost"] = round(carrying_cost, 2)

    by_product = (
        df.groupby("product_id")
        .agg(
            quantity_on_hand=("quantity_on_hand", "sum"),
            unit_cost=("cost_price", "mean"),
            inventory_value=("inventory_value", "sum"),
            carrying_cost=(
                "inventory_value",
                lambda x: float(x.sum() * (config.carrying_cost_pct / 365.0) * period_days),
            ),
        )
        .reset_index()
        .sort_values("inventory_value", ascending=False)
    )
    result["by_product"] = by_product

    return result


# ============================================================
# 3. STOCKOUT COST
# ============================================================

def compute_stockout_cost(
    inv_intel_df: pd.DataFrame,
    products_df: pd.DataFrame,
    config: Optional[BusinessConfig] = None,
) -> dict:
    """Estimate stockout cost.

    Assumptions:
        - HIGH stockout risk: quantity_on_hand <= 0. Estimated stockout units
          = abs(quantity_on_hand) * stockout_high_risk_multiplier.
        - MEDIUM stockout risk: estimated stockout units = stockout_medium_risk_units.
        - Lost revenue = stockout_units * unit_price.
        - Stockout cost = lost_revenue * stockout_cost_rate.

    This is an ESTIMATE. Actual stockout units require point-of-sale
    data with out-of-stock flags, which is not available in this dataset.

    Args:
        inv_intel_df: Latest inventory intelligence snapshot.
        products_df: Product master with unit_price.
        config: BusinessConfig with stockout assumptions.

    Returns:
        Dict with stockout cost details.
    """
    config = config or BusinessConfig()
    result = {
        "stockout_items_count": 0,
        "estimated_stockout_units": 0,
        "estimated_lost_revenue": 0.0,
        "estimated_stockout_cost": 0.0,
        "high_risk_count": 0,
        "medium_risk_count": 0,
        "by_item": pd.DataFrame(),
        "assumptions": {
            "stockout_cost_rate": config.stockout_cost_rate,
            "high_risk_multiplier": config.stockout_high_risk_multiplier,
            "medium_risk_units": config.stockout_medium_risk_units,
            "description": (
                f"High-risk stockouts: abs(qty_on_hand) * {config.stockout_high_risk_multiplier} "
                f"estimated units. Medium-risk: {config.stockout_medium_risk_units} flat units. "
                "Unit price used as proxy for lost revenue."
            ),
        },
    }

    if inv_intel_df is None or inv_intel_df.empty:
        return result
    if products_df is None or products_df.empty:
        return result

    df = inv_intel_df.copy()
    if "stockout_risk" not in df.columns or "quantity_on_hand" not in df.columns:
        return result

    if "unit_price" not in df.columns:
        if "product_id" in products_df.columns and "unit_price" in products_df.columns:
            df = df.merge(
                products_df[["product_id", "unit_price"]], on="product_id", how="left"
            )

    df["unit_price"] = df["unit_price"].fillna(0)
    df["quantity_on_hand"] = df["quantity_on_hand"].fillna(0)

    df["stockout_units"] = 0.0
    high_mask = df["stockout_risk"] == "HIGH"
    medium_mask = df["stockout_risk"] == "MEDIUM"

    df.loc[high_mask, "stockout_units"] = (
        df.loc[high_mask, "quantity_on_hand"].abs()
        * config.stockout_high_risk_multiplier
    )
    df.loc[medium_mask, "stockout_units"] = config.stockout_medium_risk_units

    df["lost_revenue"] = df["stockout_units"] * df["unit_price"]
    df["stockout_cost"] = df["lost_revenue"] * config.stockout_cost_rate

    result["high_risk_count"] = int(high_mask.sum())
    result["medium_risk_count"] = int(medium_mask.sum())
    result["stockout_items_count"] = int((df["stockout_units"] > 0).sum())
    result["estimated_stockout_units"] = float(df["stockout_units"].sum())
    result["estimated_lost_revenue"] = float(df["lost_revenue"].sum())
    result["estimated_stockout_cost"] = float(df["stockout_cost"].sum())

    by_item = (
        df[df["stockout_units"] > 0]
        .groupby(["product_id", "stockout_risk"])
        .agg(
            stockout_units=("stockout_units", "sum"),
            lost_revenue=("lost_revenue", "sum"),
            stockout_cost=("stockout_cost", "sum"),
            unit_price=("unit_price", "mean"),
        )
        .reset_index()
        .sort_values("stockout_cost", ascending=False)
    )
    result["by_item"] = by_item

    return result


# ============================================================
# 4. OVERSTOCK VALUE
# ============================================================

def compute_overstock_value(
    inv_intel_df: pd.DataFrame,
    products_df: pd.DataFrame,
    config: Optional[BusinessConfig] = None,
) -> dict:
    """Compute overstock (excess inventory) value.

    Formula:
        excess_units = max(0, quantity_on_hand - max_stock_level * overstock_excess_threshold)
        overstock_value = excess_units * cost_price

    Assumptions:
        - overstock_excess_threshold defaults to 1.0 (anything above max is excess).
        - cost_price is used as unit value proxy.

    Args:
        inv_intel_df: Latest inventory intelligence snapshot.
        products_df: Product master with cost_price.
        config: BusinessConfig with overstock threshold.

    Returns:
        Dict with overstock value details.
    """
    config = config or BusinessConfig()
    result = {
        "overstock_items_count": 0,
        "excess_units": 0,
        "overstock_inventory_value": 0.0,
        "high_risk_count": 0,
        "medium_risk_count": 0,
        "by_item": pd.DataFrame(),
        "assumptions": {
            "overstock_excess_threshold": config.overstock_excess_threshold,
            "description": (
                f"Inventory above max_stock_level * {config.overstock_excess_threshold} "
                "is counted as excess."
            ),
        },
    }

    if inv_intel_df is None or inv_intel_df.empty:
        return result
    if products_df is None or products_df.empty:
        return result

    df = inv_intel_df.copy()
    required = ["quantity_on_hand", "max_stock_level", "overstock_risk"]
    if not all(c in df.columns for c in required):
        return result

    if "cost_price" not in df.columns:
        if "product_id" in products_df.columns and "cost_price" in products_df.columns:
            df = df.merge(
                products_df[["product_id", "cost_price"]], on="product_id", how="left"
            )

    df["cost_price"] = df["cost_price"].fillna(0)
    df["quantity_on_hand"] = df["quantity_on_hand"].fillna(0)
    df["max_stock_level"] = df["max_stock_level"].fillna(0)

    threshold = config.overstock_excess_threshold
    df["excess_units"] = (df["quantity_on_hand"] - df["max_stock_level"] * threshold).clip(
        lower=0
    )
    df["overstock_value"] = df["excess_units"] * df["cost_price"]

    result["excess_units"] = int(df["excess_units"].sum())
    result["overstock_inventory_value"] = float(df["overstock_value"].sum())
    result["high_risk_count"] = int((df["overstock_risk"] == "HIGH").sum())
    result["medium_risk_count"] = int((df["overstock_risk"] == "MEDIUM").sum())
    result["overstock_items_count"] = int((df["excess_units"] > 0).sum())

    by_item = (
        df[df["excess_units"] > 0]
        .groupby(["product_id", "overstock_risk"])
        .agg(
            excess_units=("excess_units", "sum"),
            overstock_value=("overstock_value", "sum"),
            quantity_on_hand=("quantity_on_hand", "sum"),
            max_stock_level=("max_stock_level", "mean"),
            unit_cost=("cost_price", "mean"),
        )
        .reset_index()
        .sort_values("overstock_value", ascending=False)
    )
    result["by_item"] = by_item

    return result


# ============================================================
# 5. POTENTIAL REVENUE PROTECTED
# ============================================================

def compute_potential_revenue_protected(
    stockout_result: dict,
    overstock_result: dict,
    config: Optional[BusinessConfig] = None,
) -> dict:
    """Estimate potential revenue protected through better inventory management.

    This is an ESTIMATE. It combines:
        - Avoided stockout revenue (from stockout_result)
        - Recovered overstock value (from overstock_result, using margin)

    The estimate is scaled by revenue_protected_confidence to reflect
    uncertainty.

    Args:
        stockout_result: Output from compute_stockout_cost.
        overstock_result: Output from compute_overstock_value.
        config: BusinessConfig.

    Returns:
        Dict with revenue protected estimate and assumptions.
    """
    config = config or BusinessConfig()
    avoided_stockout_revenue = safe_num(
        stockout_result.get("estimated_lost_revenue", 0.0)
    )
    overstock_value = safe_num(
        overstock_result.get("overstock_inventory_value", 0.0)
    )

    # Assume a 30% margin on recovered overstock
    recovered_margin = overstock_value * 0.30
    gross_estimate = avoided_stockout_revenue + recovered_margin
    adjusted_estimate = gross_estimate * config.revenue_protected_confidence

    return {
        "estimated_revenue_protected": round(adjusted_estimate, 2),
        "gross_estimate": round(gross_estimate, 2),
        "avoided_stockout_revenue": round(avoided_stockout_revenue, 2),
        "recovered_overstock_margin": round(recovered_margin, 2),
        "confidence_level": config.revenue_protected_confidence,
        "assumptions": {
            "revenue_protected_confidence": config.revenue_protected_confidence,
            "description": (
                "Estimate = (avoided_stockout_revenue + overstock_value * 30%) "
                f"* {config.revenue_protected_confidence:.0%} confidence. "
                "30% margin is a typical retail assumption. "
                "This figure is illustrative, not audited."
            ),
        },
    }


# ============================================================
# 6. EXECUTIVE KPIs
# ============================================================

def compute_executive_kpis(
    inv_intel_df: pd.DataFrame,
    products_df: pd.DataFrame,
    forecasts_df: pd.DataFrame,
    stockout_result: dict,
    overstock_result: dict,
    carrying_cost_result: dict,
    revenue_protected: dict,
    forecast_accuracy: dict,
    config: Optional[BusinessConfig] = None,
) -> dict:
    """Compute top-level executive KPIs.

    Args:
        inv_intel_df: Latest inventory intelligence snapshot.
        products_df: Product master.
        forecasts_df: 14-day forecasts.
        stockout_result: Output from compute_stockout_cost.
        overstock_result: Output from compute_overstock_value.
        carrying_cost_result: Output from compute_inventory_carrying_cost.
        revenue_protected: Output from compute_potential_revenue_protected.
        forecast_accuracy: Output from compute_forecast_accuracy.
        config: BusinessConfig.

    Returns:
        Dict of executive KPIs.
    """
    config = config or BusinessConfig()

    total_inventory_value = carrying_cost_result.get("total_inventory_value", 0.0)
    estimated_carrying_cost = carrying_cost_result.get("estimated_carrying_cost", 0.0)
    stockout_exposure = stockout_result.get("estimated_stockout_cost", 0.0)
    overstock_value = overstock_result.get("overstock_inventory_value", 0.0)
    potential_revenue_protected = revenue_protected.get("estimated_revenue_protected", 0.0)

    products_requiring_reorder = 0
    if inv_intel_df is not None and not inv_intel_df.empty:
        if "reorder_urgency" in inv_intel_df.columns:
            products_requiring_reorder = int(
                inv_intel_df["reorder_urgency"].isin(["URGENT", "SOON"]).sum()
            )

    forecast_accuracy_value = 0.0
    forecast_model = "N/A"
    overall = forecast_accuracy.get("overall", {})
    if overall:
        forecast_accuracy_value = overall.get("smape", 0.0)
        forecast_model = overall.get("model", "N/A")

    # 14-day forecasted demand
    forecasted_demand = 0.0
    if forecasts_df is not None and not forecasts_df.empty and "forecast_demand" in forecasts_df.columns:
        forecasted_demand = float(forecasts_df["forecast_demand"].sum())

    return {
        "total_inventory_value": round(total_inventory_value, 2),
        "estimated_carrying_cost": round(estimated_carrying_cost, 2),
        "stockout_exposure": round(stockout_exposure, 2),
        "overstock_value": round(overstock_value, 2),
        "products_requiring_reorder": products_requiring_reorder,
        "potential_revenue_protected": round(potential_revenue_protected, 2),
        "forecast_accuracy_smape": round(forecast_accuracy_value, 2),
        "forecast_model": forecast_model,
        "forecasted_demand_14d": round(forecasted_demand, 2),
        "assumptions": {
            "carrying_cost_pct": config.carrying_cost_pct,
            "stockout_cost_rate": config.stockout_cost_rate,
        },
    }
