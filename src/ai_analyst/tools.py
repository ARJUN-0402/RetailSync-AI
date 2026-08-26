"""Data access tools for the AI analyst layer.

Each tool is a safe, read-only function that retrieves grounded data
from the existing RetailSync AI modules. No tool modifies the database.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine

from .exceptions import ToolExecutionError

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATABASE_PATH = os.path.join(_PROJECT_ROOT, "database", "retailsync.db")
PROCESSED_DIR = os.path.join(_PROJECT_ROOT, "data", "processed")


def _get_engine():
    return create_engine(f"sqlite:///{DATABASE_PATH}")


def _load_csv(name: str) -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, name)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        if name in ("features_daily.csv", "forecasts_next_14d.csv", "anomalies.csv"):
            return pd.read_csv(path, parse_dates=["date"])
        return pd.read_csv(path)
    except Exception as exc:
        logger.warning("Failed to load %s: %s", name, exc)
        return pd.DataFrame()


def _get_inventory_alerts() -> pd.DataFrame:
    engine = _get_engine()
    try:
        return pd.read_sql("SELECT * FROM inventory_alerts", engine)
    except Exception:
        return pd.DataFrame()


def _get_anomaly_flags() -> pd.DataFrame:
    engine = _get_engine()
    try:
        return pd.read_sql("SELECT * FROM anomaly_flags", engine)
    except Exception:
        return pd.DataFrame()


def _get_warehouse_optimization() -> pd.DataFrame:
    engine = _get_engine()
    try:
        return pd.read_sql("SELECT * FROM warehouse_optimization", engine)
    except Exception:
        return pd.DataFrame()


# ============================================================
# TOOL DEFINITIONS
# ============================================================

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_sales_trends",
        "description": "Get recent sales trends (revenue and quantity) by date, product, or store.",
        "parameters": {
            "product_id": "Optional product ID filter.",
            "store_id": "Optional store ID filter.",
            "days": "Number of recent days to include (default 30).",
        },
    },
    {
        "name": "get_forecasts",
        "description": "Get 14-day demand forecasts for products and stores.",
        "parameters": {
            "product_id": "Optional product ID filter.",
            "store_id": "Optional store ID filter.",
            "category": "Optional category filter.",
        },
    },
    {
        "name": "get_inventory_snapshot",
        "description": "Get the latest inventory snapshot with stock levels, reorder points, and max stock.",
        "parameters": {
            "product_id": "Optional product ID filter.",
            "store_id": "Optional store ID filter.",
            "warehouse_id": "Optional warehouse ID filter.",
        },
    },
    {
        "name": "get_stockout_risks",
        "description": "Get items at HIGH or MEDIUM stockout risk from inventory intelligence.",
        "parameters": {
            "risk_level": "Filter by HIGH or MEDIUM (optional).",
            "product_id": "Optional product ID filter.",
            "store_id": "Optional store ID filter.",
        },
    },
    {
        "name": "get_overstock_risks",
        "description": "Get items at HIGH or MEDIUM overstock risk from inventory intelligence.",
        "parameters": {
            "risk_level": "Filter by HIGH or MEDIUM (optional).",
            "product_id": "Optional product ID filter.",
            "store_id": "Optional store ID filter.",
        },
    },
    {
        "name": "get_anomalies",
        "description": "Get recent demand anomalies (spikes, drops, unusual patterns).",
        "parameters": {
            "anomaly_type": "Filter by Demand Spike, Demand Drop, or Unusual Pattern.",
            "product_id": "Optional product ID filter.",
            "store_id": "Optional store ID filter.",
            "limit": "Max records to return (default 100).",
        },
    },
    {
        "name": "get_product_segments",
        "description": "Get product segmentation results (cluster labels like High-Volume, Slow-Moving).",
        "parameters": {
            "cluster_label": "Optional cluster label filter.",
            "product_id": "Optional product ID filter.",
        },
    },
    {
        "name": "get_store_segments",
        "description": "Get store segmentation results (cluster labels like High-Performance, Low-Performance).",
        "parameters": {
            "cluster_label": "Optional cluster label filter.",
            "store_id": "Optional store ID filter.",
        },
    },
    {
        "name": "get_warehouse_performance",
        "description": "Get warehouse utilization and capacity risk metrics.",
        "parameters": {
            "warehouse_id": "Optional warehouse ID filter.",
            "capacity_risk": "Optional filter by HIGH, MEDIUM, or LOW.",
        },
    },
    {
        "name": "get_executive_kpis",
        "description": "Get top-level executive KPIs: inventory value, carrying cost, stockout exposure, overstock value, reorder counts, forecast accuracy, and 14-day forecasted demand.",
        "parameters": {},
    },
    {
        "name": "get_reorder_recommendations",
        "description": "Get reorder recommendations with quantities, urgency, and reasoning.",
        "parameters": {
            "urgency": "Filter by CRITICAL, URGENT, SOON, MONITOR, or NONE.",
            "product_id": "Optional product ID filter.",
            "store_id": "Optional store ID filter.",
        },
    },
    {
        "name": "get_forecast_explanation",
        "description": "Get a SHAP-based natural-language explanation for a specific product-store forecast, explaining why demand is expected to increase or decrease.",
        "parameters": {
            "product_id": "Product ID.",
            "store_id": "Store ID.",
        },
    },
]


# ============================================================
# TOOL IMPLEMENTATIONS
# ============================================================

def get_sales_trends(product_id: str | None = None, store_id: str | None = None, days: int = 30) -> dict[str, Any]:
    sales = _load_csv("sales.csv")
    if sales.empty:
        return {"error": "Sales data not available."}
    sales["date"] = pd.to_datetime(sales["date"])
    cutoff = sales["date"].max() - pd.Timedelta(days=days)
    df = sales[sales["date"] >= cutoff].copy()
    if product_id:
        df = df[df["product_id"] == product_id]
    if store_id:
        df = df[df["store_id"] == store_id]
    if df.empty:
        return {"error": "No sales data matches the filters."}
    result = (
        df.groupby("date")
        .agg(total_quantity=("quantity_sold", "sum"), total_revenue=("revenue", "sum"))
        .reset_index()
        .sort_values("date")
    )
    return {
        "data": result.to_dict(orient="records"),
        "summary": {
            "total_revenue": float(result["total_revenue"].sum()),
            "total_quantity": int(result["total_quantity"].sum()),
            "days": len(result),
        },
    }


def get_forecasts(product_id: str | None = None, store_id: str | None = None, category: str | None = None) -> dict[str, Any]:
    forecasts = _load_csv("forecasts_next_14d.csv")
    if forecasts.empty:
        return {"error": "Forecast data not available."}
    df = forecasts.copy()
    if product_id:
        df = df[df["product_id"] == product_id]
    if store_id:
        df = df[df["store_id"] == store_id]
    if category:
        df = df[df["category"] == category]
    if df.empty:
        return {"error": "No forecasts match the filters."}
    agg = (
        df.groupby(["product_id", "store_id", "category"])
        .agg(
            forecast_demand_14d=("forecast_demand", "sum"),
            forecast_revenue_14d=("forecast_revenue", "sum"),
            avg_daily_demand=("forecast_demand", "mean"),
        )
        .reset_index()
        .sort_values("forecast_revenue_14d", ascending=False)
    )
    return {
        "data": agg.to_dict(orient="records"),
        "summary": {
            "total_forecast_demand": float(agg["forecast_demand_14d"].sum()),
            "total_forecast_revenue": float(agg["forecast_revenue_14d"].sum()),
            "product_store_combinations": len(agg),
        },
    }


def get_inventory_snapshot(
    product_id: str | None = None, store_id: str | None = None, warehouse_id: str | None = None
) -> dict[str, Any]:
    inv = _load_csv("inventory_intelligence.csv")
    if inv.empty:
        inv = _load_csv("inventory.csv")
        if inv.empty:
            return {"error": "Inventory data not available."}
    df = inv.copy()
    if product_id:
        df = df[df["product_id"] == product_id]
    if store_id:
        df = df[df["store_id"] == store_id]
    if warehouse_id and "warehouse_id" in df.columns:
        df = df[df["warehouse_id"] == warehouse_id]
    if df.empty:
        return {"error": "No inventory records match the filters."}
    cols = [
        "product_id", "store_id", "warehouse_id", "quantity_on_hand",
        "reorder_point", "max_stock_level", "stockout_risk", "overstock_risk",
        "reorder_urgency", "composite_risk_level", "recommended_action",
    ]
    cols = [c for c in cols if c in df.columns]
    return {
        "data": df[cols].to_dict(orient="records"),
        "summary": {
            "total_records": len(df),
            "high_stockout": int((df.get("stockout_risk", pd.Series()) == "HIGH").sum()) if "stockout_risk" in df.columns else 0,
            "high_overstock": int((df.get("overstock_risk", pd.Series()) == "HIGH").sum()) if "overstock_risk" in df.columns else 0,
            "urgent_reorder": int((df.get("reorder_urgency", pd.Series()) == "URGENT").sum()) if "reorder_urgency" in df.columns else 0,
        },
    }


def get_stockout_risks(
    risk_level: str | None = None, product_id: str | None = None, store_id: str | None = None
) -> dict[str, Any]:
    alerts = _get_inventory_alerts()
    if alerts.empty:
        inv = _load_csv("inventory_intelligence.csv")
        if inv.empty:
            return {"error": "Inventory alerts data not available."}
        alerts = inv[inv["stockout_risk"].isin(["HIGH", "MEDIUM"])].copy()
        if alerts.empty:
            return {"error": "No stockout risks found."}
        alerts["alert_type"] = "Stockout Risk"
        alerts["risk_level"] = alerts["stockout_risk"]
        alerts["reason"] = alerts.get("stockout_reason", "")
    else:
        alerts = alerts[alerts["alert_type"] == "Stockout Risk"]
    if risk_level:
        alerts = alerts[alerts["risk_level"] == risk_level]
    if product_id:
        alerts = alerts[alerts["product_id"] == product_id]
    if store_id:
        alerts = alerts[alerts["store_id"] == store_id]
    if alerts.empty:
        return {"error": "No stockout risks match the filters."}
    cols = ["product_id", "store_id", "warehouse_id", "risk_level", "reason", "quantity_on_hand", "reorder_point", "stock_coverage_days", "recommended_action"]
    cols = [c for c in cols if c in alerts.columns]
    return {
        "data": alerts[cols].to_dict(orient="records"),
        "summary": {
            "total_items": len(alerts),
            "high_risk": int((alerts["risk_level"] == "HIGH").sum()),
            "medium_risk": int((alerts["risk_level"] == "MEDIUM").sum()),
        },
    }


def get_overstock_risks(
    risk_level: str | None = None, product_id: str | None = None, store_id: str | None = None
) -> dict[str, Any]:
    alerts = _get_inventory_alerts()
    if alerts.empty:
        inv = _load_csv("inventory_intelligence.csv")
        if inv.empty:
            return {"error": "Inventory alerts data not available."}
        alerts = inv[inv["overstock_risk"].isin(["HIGH", "MEDIUM"])].copy()
        if alerts.empty:
            return {"error": "No overstock risks found."}
        alerts["alert_type"] = "Overstock Risk"
        alerts["risk_level"] = alerts["overstock_risk"]
        alerts["reason"] = alerts.get("overstock_reason", "")
    else:
        alerts = alerts[alerts["alert_type"] == "Overstock Risk"]
    if risk_level:
        alerts = alerts[alerts["risk_level"] == risk_level]
    if product_id:
        alerts = alerts[alerts["product_id"] == product_id]
    if store_id:
        alerts = alerts[alerts["store_id"] == store_id]
    if alerts.empty:
        return {"error": "No overstock risks match the filters."}
    cols = ["product_id", "store_id", "warehouse_id", "risk_level", "reason", "quantity_on_hand", "max_stock_level", "stock_coverage_days", "recommended_action"]
    cols = [c for c in cols if c in alerts.columns]
    return {
        "data": alerts[cols].to_dict(orient="records"),
        "summary": {
            "total_items": len(alerts),
            "high_risk": int((alerts["risk_level"] == "HIGH").sum()),
            "medium_risk": int((alerts["risk_level"] == "MEDIUM").sum()),
        },
    }


def get_anomalies(
    anomaly_type: str | None = None,
    product_id: str | None = None,
    store_id: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    anomalies = _load_csv("anomalies.csv")
    if anomalies.empty:
        flags = _get_anomaly_flags()
        if flags.empty:
            return {"error": "Anomaly data not available."}
        anomalies = flags.copy()
        anomalies["anomaly_type"] = anomalies.get("anomaly_type", "Unusual Pattern")
    df = anomalies.copy()
    if anomaly_type:
        df = df[df["anomaly_type"] == anomaly_type]
    if product_id:
        df = df[df["product_id"] == product_id]
    if store_id:
        df = df[df["store_id"] == store_id]
    if df.empty:
        return {"error": "No anomalies match the filters."}
    df = df.sort_values("date", ascending=False).head(limit)
    cols = ["date", "product_id", "store_id", "quantity_sold", "z_score", "anomaly_type", "category", "store_type", "city"]
    cols = [c for c in cols if c in df.columns]
    return {
        "data": df[cols].to_dict(orient="records"),
        "summary": {
            "total_returned": len(df),
            "spikes": int((df["anomaly_type"] == "Demand Spike").sum()) if "anomaly_type" in df.columns else 0,
            "drops": int((df["anomaly_type"] == "Significant Drop").sum()) if "anomaly_type" in df.columns else 0,
            "unusual": int((df["anomaly_type"] == "Unusual Pattern").sum()) if "anomaly_type" in df.columns else 0,
        },
    }


def get_product_segments(cluster_label: str | None = None, product_id: str | None = None) -> dict[str, Any]:
    df = _load_csv("product_segments.csv")
    if df.empty:
        return {"error": "Product segments not available."}
    if cluster_label:
        df = df[df["product_cluster_label"] == cluster_label]
    if product_id:
        df = df[df["product_id"] == product_id]
    if df.empty:
        return {"error": "No product segments match the filters."}
    cols = ["product_id", "cluster", "cluster_label", "revenue_sum", "demand_cv_28d_mean"]
    cols = [c for c in cols if c in df.columns]
    return {
        "data": df[cols].to_dict(orient="records"),
        "summary": {
            "total_products": len(df),
            "labels": df["cluster_label"].value_counts().to_dict() if "cluster_label" in df.columns else {},
        },
    }


def get_store_segments(cluster_label: str | None = None, store_id: str | None = None) -> dict[str, Any]:
    df = _load_csv("store_segments.csv")
    if df.empty:
        return {"error": "Store segments not available."}
    if cluster_label:
        df = df[df["store_cluster_label"] == cluster_label]
    if store_id:
        df = df[df["store_id"] == store_id]
    if df.empty:
        return {"error": "No store segments match the filters."}
    cols = ["store_id", "cluster", "cluster_label", "revenue_sum", "demand_cv_28d_mean"]
    cols = [c for c in cols if c in df.columns]
    return {
        "data": df[cols].to_dict(orient="records"),
        "summary": {
            "total_stores": len(df),
            "labels": df["cluster_cluster_label"].value_counts().to_dict() if "cluster_label" in df.columns else {},
        },
    }


def get_warehouse_performance(
    warehouse_id: str | None = None, capacity_risk: str | None = None
) -> dict[str, Any]:
    df = _get_warehouse_optimization()
    if df.empty:
        df = _load_csv("warehouse_optimization.csv")
        if df.empty:
            return {"error": "Warehouse data not available."}
    if warehouse_id:
        df = df[df["warehouse_id"] == warehouse_id]
    if capacity_risk:
        df = df[df["capacity_risk"] == capacity_risk]
    if df.empty:
        return {"error": "No warehouses match the filters."}
    cols = [
        "warehouse_id", "warehouse_name", "city", "state", "capacity_m3",
        "occupied_volume_m3", "utilization_pct", "capacity_risk",
        "turnover_ratio", "cluster_label", "recommendation",
    ]
    cols = [c for c in cols if c in df.columns]
    return {
        "data": df[cols].to_dict(orient="records"),
        "summary": {
            "total_warehouses": len(df),
            "avg_utilization_pct": float(df["utilization_pct"].mean()) if "utilization_pct" in df.columns else 0.0,
            "high_capacity_risk": int((df["capacity_risk"] == "HIGH").sum()) if "capacity_risk" in df.columns else 0,
        },
    }


def get_executive_kpis() -> dict[str, Any]:
    try:
        from src.business_metrics.config import BusinessConfig
        from src.business_metrics.kpi import (
            compute_executive_kpis,
            compute_forecast_accuracy,
            compute_inventory_carrying_cost,
            compute_overstock_value,
            compute_potential_revenue_protected,
            compute_stockout_cost,
        )
        from src.business_metrics.reorder import generate_reorder_recommendations

        inv_intel = _load_csv("inventory_intelligence.csv")
        products = pd.read_sql("SELECT * FROM products", _get_engine())
        forecasts = _load_csv("forecasts_next_14d.csv")
        suppliers = pd.read_sql("SELECT * FROM suppliers", _get_engine())

        config = BusinessConfig()
        stockout = compute_stockout_cost(inv_intel, products, config) if not inv_intel.empty else {}
        overstock = compute_overstock_value(inv_intel, products, config) if not inv_intel.empty else {}
        carrying = compute_inventory_carrying_cost(inv_intel, products, config) if not inv_intel.empty else {}
        revenue = compute_potential_revenue_protected(stockout, overstock, config)
        _reorder_df = generate_reorder_recommendations(inv_intel, products, forecasts, suppliers, config) if not inv_intel.empty else pd.DataFrame()

        features = _load_csv("features_daily.csv")
        model_pkg = None
        model_path = os.path.join(_PROJECT_ROOT, "models", "demand_forecaster.pkl")
        if os.path.exists(model_path):
            try:
                import joblib
                model_pkg = joblib.load(model_path)
            except Exception:
                pass
        forecast_acc = compute_forecast_accuracy(features, model_pkg, config) if not features.empty else {}

        kpis = compute_executive_kpis(
            inv_intel, products, forecasts, stockout, overstock, carrying, revenue, forecast_acc, config
        )
        return {"data": kpis}
    except Exception as exc:
        logger.warning("Failed to compute executive KPIs: %s", exc)
        return {"error": f"Could not compute executive KPIs: {exc}"}


def get_reorder_recommendations(
    urgency: str | None = None, product_id: str | None = None, store_id: str | None = None
) -> dict[str, Any]:
    try:
        from src.business_metrics.config import BusinessConfig
        from src.business_metrics.reorder import generate_reorder_recommendations

        inv_intel = _load_csv("inventory_intelligence.csv")
        products = pd.read_sql("SELECT * FROM products", _get_engine())
        forecasts = _load_csv("forecasts_next_14d.csv")
        suppliers = pd.read_sql("SELECT * FROM suppliers", _get_engine())
        config = BusinessConfig()
        df = generate_reorder_recommendations(inv_intel, products, forecasts, suppliers, config)
        if df.empty:
            return {"error": "Reorder recommendations not available."}
        if urgency:
            df = df[df["reorder_urgency_computed"] == urgency]
        if product_id:
            df = df[df["product_id"] == product_id]
        if store_id:
            df = df[df["store_id"] == store_id]
        if df.empty:
            return {"error": "No reorder recommendations match the filters."}
        cols = [
            "product_id", "store_id", "quantity_on_hand", "avg_daily_demand",
            "forecast_demand_14d", "lead_time_days", "safety_stock",
            "reorder_point_computed", "recommended_quantity", "reorder_value",
            "expected_coverage_days", "reorder_urgency_computed", "stockout_risk_computed",
            "reorder_reasoning",
        ]
        cols = [c for c in cols if c in df.columns]
        return {
            "data": df[cols].to_dict(orient="records"),
            "summary": {
                "total_recommendations": len(df),
                "critical": int((df["reorder_urgency_computed"] == "CRITICAL").sum()),
                "urgent": int((df["reorder_urgency_computed"] == "URGENT").sum()),
                "soon": int((df["reorder_urgency_computed"] == "SOON").sum()),
            },
        }
    except Exception as exc:
        logger.warning("Failed to generate reorder recommendations: %s", exc)
        return {"error": f"Could not generate reorder recommendations: {exc}"}


def get_forecast_explanation(product_id: str, store_id: str) -> dict[str, Any]:
    try:
        from src.explainability import build_explanation, load_explainability_engine

        features = _load_csv("features_daily.csv")
        if features.empty:
            return {"error": "Feature data not available for explanation."}
        engine = load_explainability_engine()
        engine.set_background(features)
        sub = features[(features["product_id"] == product_id) & (features["store_id"] == store_id)].sort_values("date")
        if sub.empty:
            return {"error": f"No data found for {product_id} at {store_id}."}
        latest = sub[engine.feature_cols].iloc[[-1]]
        local = engine.explain_instance(latest)
        text = build_explanation(local, context={"product_id": product_id, "store_id": store_id}, top_n=3)
        return {
            "data": {
                "explanation": text,
                "predicted_value": round(local.predicted_value, 2),
                "expected_value": round(local.expected_value, 2),
                "net_effect": round(local.net_effect, 2),
                "top_positive": [
                    {"feature": c.feature, "shap_value": round(c.shap_value, 4), "feature_value": round(c.feature_value, 4)}
                    for c in local.top_positive(3)
                ],
                "top_negative": [
                    {"feature": c.feature, "shap_value": round(c.shap_value, 4), "feature_value": round(c.feature_value, 4)}
                    for c in local.top_negative(3)
                ],
            }
        }
    except Exception as exc:
        logger.warning("Failed to generate forecast explanation: %s", exc)
        return {"error": f"Could not generate forecast explanation: {exc}"}


# ============================================================
# TOOL REGISTRY
# ============================================================

TOOL_REGISTRY: dict[str, callable] = {
    "get_sales_trends": get_sales_trends,
    "get_forecasts": get_forecasts,
    "get_inventory_snapshot": get_inventory_snapshot,
    "get_stockout_risks": get_stockout_risks,
    "get_overstock_risks": get_overstock_risks,
    "get_anomalies": get_anomalies,
    "get_product_segments": get_product_segments,
    "get_store_segments": get_store_segments,
    "get_warehouse_performance": get_warehouse_performance,
    "get_executive_kpis": get_executive_kpis,
    "get_reorder_recommendations": get_reorder_recommendations,
    "get_forecast_explanation": get_forecast_explanation,
}


def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a tool by name with the given arguments."""
    if name not in TOOL_REGISTRY:
        raise ToolExecutionError(f"Unknown tool: {name}")
    func = TOOL_REGISTRY[name]
    validated_args: dict[str, Any] = {}
    for key, value in arguments.items():
        if key in ("product_id", "store_id", "warehouse_id") and value is not None:
            validated_args[key] = str(value).strip()
        elif key == "days" and value is not None:
            validated_args[key] = max(1, min(int(value), 365))
        elif key == "limit" and value is not None:
            validated_args[key] = max(1, min(int(value), 1000))
        elif key == "risk_level" and value is not None:
            validated_args[key] = str(value).strip().upper()
        elif key == "urgency" and value is not None:
            validated_args[key] = str(value).strip().upper()
        elif key == "category" and value is not None:
            validated_args[key] = str(value).strip()
        elif key == "anomaly_type" and value is not None:
            validated_args[key] = str(value).strip()
        elif key == "cluster_label" and value is not None:
            validated_args[key] = str(value).strip()
        elif key == "capacity_risk" and value is not None:
            validated_args[key] = str(value).strip().upper()
        else:
            validated_args[key] = value
    try:
        result = func(**validated_args)
        return result
    except ToolExecutionError:
        raise
    except Exception as exc:
        raise ToolExecutionError(f"Tool {name} failed: {exc}") from exc
