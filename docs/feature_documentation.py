import pandas as pd

df = pd.read_csv("data/processed/features_daily.csv", parse_dates=["date"])
meta = pd.read_csv("data/processed/feature_metadata.csv")

print("=== FEATURE DOCUMENTATION ===\n")

feature_groups = {
    "Identifiers": ["product_id", "store_id", "date"],
    "Static Attributes": ["category", "subcategory", "store_type", "city", "state", "supplier_id", "warehouse_id"],
    "Product Attributes": ["unit_price", "cost_price", "weight_kg", "volume_m3"],
    "Supplier Attributes": ["lead_time_days", "reliability_score"],
    "Lag Features": [c for c in df.columns if "lag" in c],
    "Rolling Features": [c for c in df.columns if "rolling" in c],
    "Expanding Features": [c for c in df.columns if "expanding" in c],
    "Time Features": ["day_of_week", "day_of_month", "month", "quarter", "year", "is_weekend", "is_month_start", "is_month_end", "month_sin", "month_cos", "dow_sin", "dow_cos"],
    "Price Features": [c for c in df.columns if "price" in c and c not in ["unit_price", "cost_price"]],
    "Promotion Features": [c for c in df.columns if "promotion" in c],
    "Inventory Features": ["quantity_on_hand", "reorder_point", "max_stock_level", "stock_coverage_days", "stock_vs_reorder", "stock_vs_max", "stock_to_max_ratio"],
    "Demand Variability": ["demand_cv_28d", "zero_demand_pct_28d"],
    "Aggregate Features": ["category_avg_demand", "store_type_avg_demand"],
    "Target Variables": [c for c in df.columns if "target" in c],
}

for group_name, features in feature_groups.items():
    print(f"\n{group_name}:")
    for feat in features:
        if feat in df.columns:
            dtype = str(df[feat].dtype)
            print(f"  {feat}: {dtype}")

print("\n=== FEATURE STATISTICS ===")
print(f"Total features: {len(df.columns)}")
print(f"Total rows: {len(df)}")
print(f"Products: {df['product_id'].nunique()}")
print(f"Stores: {df['store_id'].nunique()}")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

print("\n=== TARGET VARIABLE DISTRIBUTION ===")
for target in ["target_demand_1d", "target_demand_7d", "target_demand_14d"]:
    if target in df.columns:
        print(f"\n{target}:")
        print(f"  Mean: {df[target].mean():.2f}")
        print(f"  Std: {df[target].std():.2f}")
        print(f"  Min: {df[target].min():.2f}")
        print(f"  Max: {df[target].max():.2f}")
        print(f"  Zero count: {(df[target] == 0).sum()} ({(df[target] == 0).mean() * 100:.1f}%)")

print("\n=== DATA LEAKAGE PREVENTION ====")
print("- All lag features use .shift() to access past values only")
print("- All rolling features use .shift(1) before rolling to exclude current day")
print("- All expanding features use .shift(1) before expanding")
print("- Target variables use .shift(-horizon) to access future values")
print("- No future data is used in any feature calculation")
print("- Features are computed within each product-store group independently")
