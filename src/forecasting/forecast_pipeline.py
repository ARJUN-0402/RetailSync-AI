import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

print("=== DEMAND FORECAST GENERATION ===\n")

# Load model
model_package = joblib.load("models/demand_forecaster.pkl")
model = model_package["model"]
feature_cols = model_package["feature_cols"]
model_name = model_package["model_name"]

print(f"Loaded model: {model_name}")
print(f"Features: {len(feature_cols)}")

# Load latest feature data
df = pd.read_csv("data/processed/features_daily.csv", parse_dates=["date"])
latest_date = df["date"].max()
print(f"Latest data date: {latest_date.date()}")

# ============================================================
# GENERATE 14-DAY FORECASTS FOR ALL PRODUCT-STORE COMBINATIONS
# ============================================================
print("\nGenerating 14-day forecasts...")

# Get unique product-store combinations
product_stores = df[["product_id", "store_id"]].drop_duplicates().reset_index(drop=True)
forecast_dates = pd.date_range(start=latest_date + timedelta(days=1), periods=14, freq="D")

forecasts = []

for _, ps_row in product_stores.iterrows():
    product_id = ps_row["product_id"]
    store_id = ps_row["store_id"]
    
    # Get historical data for this product-store
    hist = df[(df["product_id"] == product_id) & (df["store_id"] == store_id)].sort_values("date")
    
    if len(hist) == 0:
        continue
    
    # Get the most recent row as base for forecasting
    latest_row = hist.iloc[-1].copy()
    
    for forecast_date in forecast_dates:
        # Update date-based features
        forecast_row = latest_row.copy()
        forecast_row["date"] = forecast_date
        forecast_row["day_of_week"] = forecast_date.dayofweek
        forecast_row["day_of_month"] = forecast_date.day
        forecast_row["month"] = forecast_date.month
        forecast_row["quarter"] = forecast_date.quarter
        forecast_row["year"] = forecast_date.year
        forecast_row["is_weekend"] = 1 if forecast_date.dayofweek >= 5 else 0
        forecast_row["is_month_start"] = 1 if forecast_date.day <= 5 else 0
        forecast_row["is_month_end"] = 1 if forecast_date.day >= 25 else 0
        forecast_row["month_sin"] = np.sin(2 * np.pi * forecast_date.month / 12)
        forecast_row["month_cos"] = np.cos(2 * np.pi * forecast_date.month / 12)
        forecast_row["dow_sin"] = np.sin(2 * np.pi * forecast_date.dayofweek / 7)
        forecast_row["dow_cos"] = np.cos(2 * np.pi * forecast_date.dayofweek / 7)
        
        # Use last known values for other features
        # In a production system, these would be updated iteratively
        
        # Prepare features for prediction
        X_forecast = forecast_row[feature_cols].values.reshape(1, -1)
        
        # Generate prediction
        pred_demand = model.predict(X_forecast)[0]
        pred_demand = max(0, pred_demand)  # Ensure non-negative
        
        # Get unit price for revenue calculation
        unit_price = forecast_row["unit_price"]
        pred_revenue = pred_demand * unit_price
        
        forecasts.append({
            "date": forecast_date,
            "product_id": product_id,
            "store_id": store_id,
            "forecast_demand": round(pred_demand, 2),
            "forecast_revenue": round(pred_revenue, 2),
            "unit_price": unit_price,
            "category": forecast_row["category"],
            "store_type": forecast_row["store_type"],
            "model": model_name,
        })

forecasts_df = pd.DataFrame(forecasts)
print(f"Generated {len(forecasts_df)} forecasts")

# ============================================================
# AGGREGATE FORECASTS
# ============================================================
print("\n=== FORECAST SUMMARY ===")

# Daily totals
daily_totals = forecasts_df.groupby("date").agg(
    total_demand=("forecast_demand", "sum"),
    total_revenue=("forecast_revenue", "sum"),
    products_forecast=("product_id", "nunique"),
    stores_forecast=("store_id", "nunique"),
).reset_index()

print("\nDaily Forecast Totals (next 14 days):")
print(daily_totals.to_string(index=False))

# Product-level totals
product_totals = forecasts_df.groupby("product_id").agg(
    total_demand=("forecast_demand", "sum"),
    total_revenue=("forecast_revenue", "sum"),
).reset_index().sort_values("total_revenue", ascending=False)

print("\nTop 10 Products by Forecasted Revenue:")
print(product_totals.head(10).to_string(index=False))

# Store-level totals
store_totals = forecasts_df.groupby("store_id").agg(
    total_demand=("forecast_demand", "sum"),
    total_revenue=("forecast_revenue", "sum"),
).reset_index().sort_values("total_revenue", ascending=False)

print("\nStore-level Forecast Totals:")
print(store_totals.to_string(index=False))

# Category-level totals
category_totals = forecasts_df.groupby("category").agg(
    total_demand=("forecast_demand", "sum"),
    total_revenue=("forecast_revenue", "sum"),
).reset_index().sort_values("total_revenue", ascending=False)

print("\nCategory-level Forecast Totals:")
print(category_totals.to_string(index=False))

# ============================================================
# SAVE FORECASTS
# ============================================================
print("\n=== SAVING FORECASTS ===")

forecasts_df.to_csv("data/processed/forecasts_next_14d.csv", index=False)
print(f"Saved: data/processed/forecasts_next_14d.csv")

daily_totals.to_csv("data/processed/forecast_daily_totals.csv", index=False)
print(f"Saved: data/processed/forecast_daily_totals.csv")

product_totals.to_csv("data/processed/forecast_product_totals.csv", index=False)
print(f"Saved: data/processed/forecast_product_totals.csv")

# ============================================================
# FORECAST QUALITY FLAGS
# ============================================================
print("\n=== FORECAST QUALITY FLAGS ===")

# Flag products with zero forecasted demand
zero_demand = forecasts_df[forecasts_df["forecast_demand"] == 0]
print(f"Product-store combinations with zero forecast: {len(zero_demand)} ({len(zero_demand)/len(forecasts_df)*100:.1f}%)")

# Flag low forecast (potential stockout risk)
low_threshold = 2.0
low_forecast = forecasts_df[forecasts_df["forecast_demand"] < low_threshold]
print(f"Product-store combinations with low forecast (<{low_threshold}): {len(low_forecast)} ({len(low_forecast)/len(forecasts_df)*100:.1f}%)")

print("\nForecast generation complete.")
