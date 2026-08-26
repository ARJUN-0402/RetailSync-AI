import os
import time

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

np.random.seed(42)
start_time = time.time()

PROCESSED_DIR = "data/processed"
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

engine = create_engine("sqlite:///database/retailsync.db")

print("Loading data...")
products = pd.read_sql("SELECT * FROM products", engine)
stores = pd.read_sql("SELECT * FROM stores", engine)
sales = pd.read_sql("SELECT * FROM sales", engine)
inventory = pd.read_sql("SELECT * FROM inventory", engine)
suppliers = pd.read_sql("SELECT * FROM suppliers", engine)

sales["date"] = pd.to_datetime(sales["date"])
inventory["date"] = pd.to_datetime(inventory["date"])

all_products_stores = (
    sales[["product_id", "store_id"]].drop_duplicates().reset_index(drop=True)
)
full_dates = pd.date_range(start=sales["date"].min(), end=sales["date"].max(), freq="D")

print(
    f"Dates: {len(full_dates)}, Product-Stores: {len(all_products_stores)}, Total rows: {len(full_dates) * len(all_products_stores):,}"
)

# ============================================================
# 1. CREATE DAILY PRODUCT-STORE LEVEL DATASET
# ============================================================
print("Creating daily product-store level dataset...")

daily_base = (
    all_products_stores.assign(key=1)
    .merge(pd.DataFrame({"date": full_dates, "key": 1}), on="key")
    .drop("key", axis=1)
)

daily_sales = sales.groupby(["date", "product_id", "store_id"], as_index=False).agg(
    quantity_sold=("quantity_sold", "sum"),
    revenue=("revenue", "sum"),
    unit_price=("unit_price", "mean"),
    promotion=("promotion", "max"),
    discount_pct=("discount_pct", "mean"),
)

daily_df = daily_base.merge(
    daily_sales, on=["date", "product_id", "store_id"], how="left"
)
daily_df["quantity_sold"] = daily_df["quantity_sold"].fillna(0).astype(int)
daily_df["revenue"] = daily_df["revenue"].fillna(0.0)

unit_price_map = sales.groupby("product_id")["unit_price"].first().to_dict()
daily_df["unit_price"] = daily_df["product_id"].map(unit_price_map)
daily_df["promotion"] = daily_df["promotion"].fillna(0).astype(int)
daily_df["discount_pct"] = daily_df["discount_pct"].fillna(0.0)

daily_df = daily_df.merge(
    products[
        [
            "product_id",
            "category",
            "subcategory",
            "cost_price",
            "supplier_id",
            "weight_kg",
            "volume_m3",
        ]
    ],
    on="product_id",
    how="left",
)
daily_df = daily_df.merge(
    stores[["store_id", "store_type", "city", "state"]], on="store_id", how="left"
)
daily_df = daily_df.merge(
    suppliers[["supplier_id", "lead_time_days", "reliability_score"]],
    on="supplier_id",
    how="left",
)

daily_df = daily_df.sort_values(["product_id", "store_id", "date"]).reset_index(
    drop=True
)

print(f"Daily dataset shape: {daily_df.shape}")

# ============================================================
# 2. LAG FEATURES
# ============================================================
print("Creating lag features...")

grouped_qty = daily_df.groupby(["product_id", "store_id"])["quantity_sold"]
grouped_rev = daily_df.groupby(["product_id", "store_id"])["revenue"]

for lag in [1, 7, 14, 28]:
    daily_df[f"demand_lag_{lag}d"] = grouped_qty.shift(lag)
    daily_df[f"revenue_lag_{lag}d"] = grouped_rev.shift(lag)

# ============================================================
# 3. ROLLING FEATURES
# ============================================================
print("Creating rolling features...")

qty_shifted = grouped_qty.shift(1)

for window in [7, 14, 28]:
    rolling = qty_shifted.groupby(
        [daily_df["product_id"], daily_df["store_id"]], group_keys=False
    )
    daily_df[f"demand_rolling_mean_{window}d"] = (
        rolling.rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )
    daily_df[f"demand_rolling_median_{window}d"] = (
        rolling.rolling(window=window, min_periods=1)
        .median()
        .reset_index(level=[0, 1], drop=True)
    )
    daily_df[f"demand_rolling_std_{window}d"] = (
        rolling.rolling(window=window, min_periods=1)
        .std()
        .reset_index(level=[0, 1], drop=True)
        .fillna(0)
    )
    daily_df[f"demand_rolling_max_{window}d"] = (
        rolling.rolling(window=window, min_periods=1)
        .max()
        .reset_index(level=[0, 1], drop=True)
    )
    daily_df[f"demand_rolling_min_{window}d"] = (
        rolling.rolling(window=window, min_periods=1)
        .min()
        .reset_index(level=[0, 1], drop=True)
    )

daily_df["demand_expanding_mean"] = (
    qty_shifted.groupby(
        [daily_df["product_id"], daily_df["store_id"]], group_keys=False
    )
    .expanding(min_periods=1)
    .mean()
    .reset_index(level=[0, 1], drop=True)
)
daily_df["demand_expanding_std"] = (
    qty_shifted.groupby(
        [daily_df["product_id"], daily_df["store_id"]], group_keys=False
    )
    .expanding(min_periods=1)
    .std()
    .reset_index(level=[0, 1], drop=True)
    .fillna(0)
)

# ============================================================
# 4. TIME-BASED FEATURES
# ============================================================
print("Creating time-based features...")

daily_df["day_of_week"] = daily_df["date"].dt.dayofweek
daily_df["day_of_month"] = daily_df["date"].dt.day
daily_df["week_of_year"] = daily_df["date"].dt.isocalendar().week.astype(int)
daily_df["month"] = daily_df["date"].dt.month
daily_df["quarter"] = daily_df["date"].dt.quarter
daily_df["year"] = daily_df["date"].dt.year
daily_df["day_of_year"] = daily_df["date"].dt.dayofyear
daily_df["is_weekend"] = (daily_df["day_of_week"] >= 5).astype(int)
daily_df["is_month_start"] = (daily_df["day_of_month"] <= 5).astype(int)
daily_df["is_month_end"] = (daily_df["day_of_month"] >= 25).astype(int)

# Cyclical encoding
daily_df["month_sin"] = np.sin(2 * np.pi * daily_df["month"] / 12)
daily_df["month_cos"] = np.cos(2 * np.pi * daily_df["month"] / 12)
daily_df["dow_sin"] = np.sin(2 * np.pi * daily_df["day_of_week"] / 7)
daily_df["dow_cos"] = np.cos(2 * np.pi * daily_df["day_of_week"] / 7)

# Trend feature (days since start)
daily_df["days_since_start"] = (daily_df["date"] - daily_df["date"].min()).dt.days
daily_df["trend_linear"] = daily_df["days_since_start"]

# Holiday indicator
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

daily_df["is_holiday"] = (
    daily_df.apply(lambda row: 1 if (row["date"].month, row["date"].day) in HOLIDAYS else 0, axis=1)
)
daily_df["holiday_multiplier"] = (
    daily_df.apply(lambda row: HOLIDAYS.get((row["date"].month, row["date"].day), 1.0), axis=1)
)

# ============================================================
# 5. PRICE & PROMOTION FEATURES
# ============================================================
print("Creating price and promotion features...")

daily_df["price_vs_cost"] = daily_df["unit_price"] - daily_df["cost_price"]
daily_df["price_margin_pct"] = (
    daily_df["price_vs_cost"] / daily_df["cost_price"].replace(0, np.nan)
).fillna(0)
daily_df["effective_price"] = daily_df["unit_price"] * (1 - daily_df["discount_pct"])
daily_df["discount_amount"] = daily_df["unit_price"] * daily_df["discount_pct"]

promo_shifted = daily_df.groupby(["product_id", "store_id"])["promotion"].shift(1)
daily_df["promotion_last_7d"] = (
    promo_shifted.groupby(
        [daily_df["product_id"], daily_df["store_id"]], group_keys=False
    )
    .rolling(window=7, min_periods=1)
    .sum()
    .reset_index(level=[0, 1], drop=True)
)
daily_df["promotion_last_14d"] = (
    promo_shifted.groupby(
        [daily_df["product_id"], daily_df["store_id"]], group_keys=False
    )
    .rolling(window=14, min_periods=1)
    .sum()
    .reset_index(level=[0, 1], drop=True)
)

# ============================================================
# 6. STORE / PRODUCT AGGREGATE FEATURES
# ============================================================
print("Creating aggregate features...")

daily_df = daily_df.sort_values(["date", "store_id", "product_id"]).reset_index(drop=True)

store_daily = (
    daily_df.groupby(["date", "store_id"], as_index=False)["quantity_sold"]
    .mean()
    .rename(columns={"quantity_sold": "store_avg_demand"})
)
store_daily["store_avg_demand"] = store_daily.groupby("store_id")["store_avg_demand"].shift(1).fillna(0)
daily_df = daily_df.merge(store_daily, on=["date", "store_id"], how="left")

product_daily = (
    daily_df.groupby(["date", "product_id"], as_index=False)["quantity_sold"]
    .mean()
    .rename(columns={"quantity_sold": "product_avg_demand"})
)
product_daily["product_avg_demand"] = product_daily.groupby("product_id")["product_avg_demand"].shift(1).fillna(0)
daily_df = daily_df.merge(product_daily, on=["date", "product_id"], how="left")

cat_daily = (
    daily_df.groupby(["date", "category"], as_index=False)["quantity_sold"]
    .mean()
    .rename(columns={"quantity_sold": "category_avg_demand"})
)
cat_daily["category_avg_demand"] = cat_daily.groupby("category")["category_avg_demand"].shift(1).fillna(0)
daily_df = daily_df.merge(cat_daily, on=["date", "category"], how="left")

store_type_daily = (
    daily_df.groupby(["date", "store_type"], as_index=False)["quantity_sold"]
    .mean()
    .rename(columns={"quantity_sold": "store_type_avg_demand"})
)
store_type_daily["store_type_avg_demand"] = store_type_daily.groupby("store_type")["store_type_avg_demand"].shift(1).fillna(0)
daily_df = daily_df.merge(store_type_daily, on=["date", "store_type"], how="left")

daily_df = daily_df.sort_values(["product_id", "store_id", "date"]).reset_index(drop=True)

# ============================================================
# 7. INVENTORY FEATURES
# ============================================================
print("Creating inventory features...")

inv_sorted = inventory[
    [
        "date",
        "product_id",
        "store_id",
        "quantity_on_hand",
        "reorder_point",
        "max_stock_level",
        "warehouse_id",
    ]
].sort_values(["product_id", "store_id", "date"])
inv_filled = inv_sorted.copy()
for col in ["quantity_on_hand", "reorder_point", "max_stock_level", "warehouse_id"]:
    inv_filled[col] = inv_filled.groupby(["product_id", "store_id"])[col].ffill()

daily_df = daily_df.merge(inv_filled, on=["date", "product_id", "store_id"], how="left")

for col in ["quantity_on_hand", "reorder_point", "max_stock_level"]:
    daily_df[col] = daily_df.groupby(["product_id", "store_id"])[col].ffill().fillna(0)
daily_df["warehouse_id"] = (
    daily_df.groupby(["product_id", "store_id"])["warehouse_id"]
    .ffill()
    .fillna("Unknown")
)

daily_df["stock_coverage_days"] = np.where(
    daily_df["demand_rolling_mean_7d"] > 0,
    daily_df["quantity_on_hand"] / daily_df["demand_rolling_mean_7d"],
    0,
)

daily_df["stock_vs_reorder"] = daily_df["quantity_on_hand"] - daily_df["reorder_point"]
daily_df["stock_vs_max"] = daily_df["quantity_on_hand"] - daily_df["max_stock_level"]
daily_df["stock_to_max_ratio"] = np.where(
    daily_df["max_stock_level"] > 0,
    daily_df["quantity_on_hand"] / daily_df["max_stock_level"],
    0,
)

# ============================================================
# 8. TARGET VARIABLES
# ============================================================
print("Creating target variables...")

for horizon in [1, 7, 14]:
    daily_df[f"target_demand_{horizon}d"] = daily_df.groupby(
        ["product_id", "store_id"]
    )["quantity_sold"].shift(-horizon)
    daily_df[f"target_revenue_{horizon}d"] = daily_df.groupby(
        ["product_id", "store_id"]
    )["revenue"].shift(-horizon)

# ============================================================
# 9. DEMAND VARIABILITY FEATURES
# ============================================================
print("Creating demand variability features...")

qty_for_cv = qty_shifted.groupby(
    [daily_df["product_id"], daily_df["store_id"]], group_keys=False
)
rolling_mean_28 = (
    qty_for_cv.rolling(window=28, min_periods=7)
    .mean()
    .reset_index(level=[0, 1], drop=True)
)
rolling_std_28 = (
    qty_for_cv.rolling(window=28, min_periods=7)
    .std()
    .reset_index(level=[0, 1], drop=True)
)
daily_df["demand_cv_28d"] = np.where(
    rolling_mean_28 > 0, rolling_std_28 / rolling_mean_28, 0
)
daily_df["demand_cv_28d"] = daily_df["demand_cv_28d"].fillna(0)

zero_mask = (qty_shifted == 0).astype(float)
daily_df["zero_demand_pct_28d"] = (
    zero_mask.groupby([daily_df["product_id"], daily_df["store_id"]], group_keys=False)
    .rolling(window=28, min_periods=7)
    .mean()
    .reset_index(level=[0, 1], drop=True)
    .fillna(0)
)

# ============================================================
# 10. FINAL CLEANUP
# ============================================================
print("Cleaning up features...")

daily_df = daily_df.replace([np.inf, -np.inf], 0)

numeric_cols = daily_df.select_dtypes(include=[np.number]).columns
daily_df[numeric_cols] = daily_df[numeric_cols].fillna(0)

daily_df = daily_df.dropna(
    subset=["target_demand_1d", "target_demand_7d", "target_demand_14d"]
).reset_index(drop=True)

print(f"Final feature dataset shape: {daily_df.shape}")
print(f"Date range: {daily_df['date'].min().date()} to {daily_df['date'].max().date()}")

# ============================================================
# 11. SAVE FEATURES
# ============================================================
print("Saving features...")

daily_df.to_csv(os.path.join(OUTPUT_DIR, "features_daily.csv"), index=False)
print(f"Saved: {os.path.join(OUTPUT_DIR, 'features_daily.csv')}")

feature_metadata = pd.DataFrame(
    {
        "feature_name": daily_df.columns,
        "dtype": [str(daily_df[c].dtype) for c in daily_df.columns],
        "non_null_count": [daily_df[c].notna().sum() for c in daily_df.columns],
        "unique_count": [daily_df[c].nunique() for c in daily_df.columns],
    }
)
feature_metadata.to_csv(os.path.join(OUTPUT_DIR, "feature_metadata.csv"), index=False)
print(f"Saved: {os.path.join(OUTPUT_DIR, 'feature_metadata.csv')}")

print("\n=== FEATURE SUMMARY ===")
print(f"Total features: {len(daily_df.columns)}")
print(f"Rows: {len(daily_df)}")
print(f"Products: {daily_df['product_id'].nunique()}")
print(f"Stores: {daily_df['store_id'].nunique()}")
print(f"Date range: {daily_df['date'].min().date()} to {daily_df['date'].max().date()}")

print("\nFeature categories:")
print(f"  Lag features: {sum(1 for c in daily_df.columns if 'lag' in c)}")
print(f"  Rolling features: {sum(1 for c in daily_df.columns if 'rolling' in c)}")
print(f"  Expanding features: {sum(1 for c in daily_df.columns if 'expanding' in c)}")
print(
    f"  Time features: {sum(1 for c in daily_df.columns if c in ['day_of_week', 'day_of_month', 'week_of_year', 'month', 'quarter', 'year', 'day_of_year', 'is_weekend', 'is_month_start', 'is_month_end', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos', 'days_since_start', 'trend_linear', 'is_holiday', 'holiday_multiplier'])}"
)
print(f"  Price features: {sum(1 for c in daily_df.columns if 'price' in c)}")
print(f"  Promotion features: {sum(1 for c in daily_df.columns if 'promotion' in c)}")
print(
    f"  Inventory features: {sum(1 for c in daily_df.columns if any(x in c for x in ['stock', 'inventory', 'reorder', 'coverage']))}"
)
print(
    f"  Demand variability: {sum(1 for c in daily_df.columns if 'cv' in c or 'zero' in c)}"
)
print(
    f"  Aggregate features: {sum(1 for c in daily_df.columns if 'avg_demand' in c)}"
)
print(f"  Target variables: {sum(1 for c in daily_df.columns if 'target' in c)}")

elapsed = time.time() - start_time
print(f"\nFeature engineering completed in {elapsed:.1f} seconds.")
