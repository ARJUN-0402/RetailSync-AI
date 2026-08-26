"""Demand forecast generation for RetailSync AI."""

from __future__ import annotations

import logging
import os
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd

from src.config import settings
from src.exceptions import ModelError, DataError
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

HOLIDAYS = {
    (1, 1): 1.25,
    (1, 15): 1.08,
    (2, 14): 1.35,
    (5, 27): 1.15,
    (7, 4): 1.25,
    (9, 2): 1.12,
    (11, 11): 1.15,
    (11, 28): 1.55,
    (11, 29): 1.75,
    (12, 24): 1.35,
    (12, 25): 1.45,
}


def generate_forecasts() -> pd.DataFrame:
    """Generate 14-day demand forecasts for all product-store combinations."""
    setup_logging(__name__)
    logger.info("=== RetailSync AI - Demand Forecast Generation ===")

    model_path = os.path.join(str(settings.paths.models), "demand_forecaster.pkl")
    if not os.path.exists(model_path):
        raise ModelError(f"Model file not found: {model_path}")

    model_package = joblib.load(model_path)
    model = model_package["model"]
    feature_cols = model_package["feature_cols"]
    model_name = model_package["model_name"]

    logger.info("Loaded model: %s", model_name)
    logger.info("Features: %d", len(feature_cols))

    features_path = os.path.join(str(settings.paths.processed_data), "features_daily.csv")
    if not os.path.exists(features_path):
        raise DataError(f"Features file not found: {features_path}")

    df = pd.read_csv(features_path, parse_dates=["date"])
    latest_date = df["date"].max()
    start_date = df["date"].min()
    logger.info("Latest data date: %s", latest_date.date())
    logger.info("Date range: %s to %s", start_date.date(), latest_date.date())

    product_stores = df[["product_id", "store_id"]].drop_duplicates().reset_index(drop=True)
    forecast_dates = pd.date_range(
        start=latest_date + timedelta(days=1), periods=14, freq="D"
    )

    forecasts = []

    for _, ps_row in product_stores.iterrows():
        product_id = ps_row["product_id"]
        store_id = ps_row["store_id"]

        hist = df[
            (df["product_id"] == product_id) & (df["store_id"] == store_id)
        ].sort_values("date")

        if len(hist) == 0:
            continue

        latest_row = hist.iloc[-1].copy()

        for forecast_date in forecast_dates:
            forecast_row = latest_row.copy()
            forecast_row["date"] = forecast_date
            forecast_row["day_of_week"] = forecast_date.dayofweek
            forecast_row["day_of_month"] = forecast_date.day
            forecast_row["week_of_year"] = int(forecast_date.isocalendar().week)
            forecast_row["month"] = forecast_date.month
            forecast_row["quarter"] = forecast_date.quarter
            forecast_row["year"] = forecast_date.year
            forecast_row["day_of_year"] = int(forecast_date.dayofyear)
            forecast_row["is_weekend"] = 1 if forecast_date.dayofweek >= 5 else 0
            forecast_row["is_month_start"] = 1 if forecast_date.day <= 5 else 0
            forecast_row["is_month_end"] = 1 if forecast_date.day >= 25 else 0

            holiday_mult = HOLIDAYS.get((forecast_date.month, forecast_date.day), 1.0)
            forecast_row["holiday_multiplier"] = holiday_mult
            forecast_row["is_holiday"] = 1 if holiday_mult > 1.0 else 0

            forecast_row["demand_lag_1d"] = latest_row.get("demand_lag_1d", 0)
            forecast_row["demand_lag_7d"] = latest_row.get("demand_lag_7d", 0)
            forecast_row["demand_lag_14d"] = latest_row.get("demand_lag_14d", 0)
            forecast_row["demand_lag_28d"] = latest_row.get("demand_lag_28d", 0)

            forecast_row["revenue_lag_1d"] = latest_row.get("revenue_lag_1d", 0)
            forecast_row["revenue_lag_7d"] = latest_row.get("revenue_lag_7d", 0)
            forecast_row["revenue_lag_14d"] = latest_row.get("revenue_lag_14d", 0)
            forecast_row["revenue_lag_28d"] = latest_row.get("revenue_lag_28d", 0)

            for w in [7, 14, 28]:
                forecast_row[f"demand_rolling_mean_{w}d"] = latest_row.get(f"demand_rolling_mean_{w}d", 0)
                forecast_row[f"demand_rolling_median_{w}d"] = latest_row.get(f"demand_rolling_median_{w}d", 0)
                forecast_row[f"demand_rolling_std_{w}d"] = latest_row.get(f"demand_rolling_std_{w}d", 0)
                forecast_row[f"demand_rolling_max_{w}d"] = latest_row.get(f"demand_rolling_max_{w}d", 0)
                forecast_row[f"demand_rolling_min_{w}d"] = latest_row.get(f"demand_rolling_min_{w}d", 0)

            forecast_row["demand_expanding_mean"] = latest_row.get("demand_expanding_mean", 0)
            forecast_row["demand_expanding_std"] = latest_row.get("demand_expanding_std", 0)

            forecast_row["month_sin"] = np.sin(2 * np.pi * forecast_date.month / 12)
            forecast_row["month_cos"] = np.cos(2 * np.pi * forecast_date.month / 12)
            forecast_row["dow_sin"] = np.sin(2 * np.pi * forecast_date.dayofweek / 7)
            forecast_row["dow_cos"] = np.cos(2 * np.pi * forecast_date.dayofweek / 7)

            forecast_row["days_since_start"] = (forecast_date - start_date).days
            forecast_row["trend_linear"] = forecast_row["days_since_start"]

            forecast_row["price_vs_cost"] = latest_row.get("price_vs_cost", 0)
            forecast_row["price_margin_pct"] = latest_row.get("price_margin_pct", 0)
            forecast_row["effective_price"] = latest_row.get("effective_price", 0)
            forecast_row["discount_amount"] = latest_row.get("discount_amount", 0)

            forecast_row["promotion_last_7d"] = latest_row.get("promotion_last_7d", 0)
            forecast_row["promotion_last_14d"] = latest_row.get("promotion_last_14d", 0)

            forecast_row["store_avg_demand"] = latest_row.get("store_avg_demand", 0)
            forecast_row["product_avg_demand"] = latest_row.get("product_avg_demand", 0)
            forecast_row["category_avg_demand"] = latest_row.get("category_avg_demand", 0)
            forecast_row["store_type_avg_demand"] = latest_row.get("store_type_avg_demand", 0)

            forecast_row["stock_coverage_days"] = latest_row.get("stock_coverage_days", 0)
            forecast_row["stock_vs_reorder"] = latest_row.get("stock_vs_reorder", 0)
            forecast_row["stock_vs_max"] = latest_row.get("stock_vs_max", 0)
            forecast_row["stock_to_max_ratio"] = latest_row.get("stock_to_max_ratio", 0)

            forecast_row["demand_cv_28d"] = latest_row.get("demand_cv_28d", 0)
            forecast_row["zero_demand_pct_28d"] = latest_row.get("zero_demand_pct_28d", 0)

            available_cols = [c for c in feature_cols if c in forecast_row.index]
            X = forecast_row[available_cols].values.reshape(1, -1)
            if X.shape[1] == 0:
                continue

            pred = float(model.predict(X)[0])
            pred = max(0.0, pred)

            forecasts.append(
                {
                    "date": forecast_date,
                    "product_id": product_id,
                    "store_id": store_id,
                    "forecast_demand": round(pred, 2),
                    "model_name": model_name,
                    "forecast_revenue": round(pred * latest_row.get("unit_price", 0), 2),
                }
            )

    logger.info("Generated %d forecast rows", len(forecasts))
    return pd.DataFrame(forecasts)


def main() -> None:
    """Main entry point."""
    setup_logging(__name__)
    forecasts_df = generate_forecasts()
    output_path = os.path.join(str(settings.paths.processed_data), "forecasts_next_14d.csv")
    forecasts_df.to_csv(output_path, index=False)
    logger.info("Saved forecasts to %s", output_path)
    logger.info(
        "Total forecasted demand: %s units",
        f"{forecasts_df['forecast_demand'].sum():,.0f}",
    )
    logger.info(
        "Total forecasted revenue: $%s",
        f"{forecasts_df['forecast_revenue'].sum():,.2f}",
    )


if __name__ == "__main__":
    main()
