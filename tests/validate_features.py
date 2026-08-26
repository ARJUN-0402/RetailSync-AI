import pandas as pd

print("=== DATA LEAKAGE VALIDATION ===\n")

df = pd.read_csv("data/processed/features_daily.csv", parse_dates=["date"])
df = df.sort_values(["product_id", "store_id", "date"]).reset_index(drop=True)

# Use a subset for faster validation
df_sample = df.head(50000).copy()

leakage_checks = []

# 1. Lag features: verify lag value corresponds to past data
print("1. Lag feature validation...")
sample = (
    df_sample[["date", "product_id", "store_id", "quantity_sold", "demand_lag_1d"]]
    .dropna()
    .head(1000)
)
lag1_match = 0
for idx, row in sample.iterrows():
    past = df_sample[
        (df_sample["date"] == row["date"] - pd.Timedelta(days=1))
        & (df_sample["product_id"] == row["product_id"])
        & (df_sample["store_id"] == row["store_id"])
    ]
    if (
        not past.empty
        and abs(past.iloc[0]["quantity_sold"] - row["demand_lag_1d"]) < 1e-9
    ):
        lag1_match += 1
lag1_accuracy = lag1_match / len(sample) * 100 if len(sample) > 0 else 0
print(f"   demand_lag_1d accuracy: {lag1_accuracy:.1f}%")
leakage_checks.append(("demand_lag_1d accuracy", lag1_accuracy, 95))

# 2. Rolling features: verify they use only past data
print("2. Rolling feature validation...")
sample2 = (
    df_sample[
        ["date", "product_id", "store_id", "quantity_sold", "demand_rolling_mean_7d"]
    ]
    .dropna()
    .head(1000)
)
rolling_match = 0
for idx, row in sample2.iterrows():
    hist = df_sample[
        (df_sample["date"] < row["date"])
        & (df_sample["product_id"] == row["product_id"])
        & (df_sample["store_id"] == row["store_id"])
        & (df_sample["date"] >= row["date"] - pd.Timedelta(days=7))
    ]
    if not hist.empty:
        expected = hist["quantity_sold"].mean()
        if abs(expected - row["demand_rolling_mean_7d"]) < 1e-6:
            rolling_match += 1
rolling_accuracy = rolling_match / len(sample2) * 100 if len(sample2) > 0 else 0
print(f"   demand_rolling_mean_7d accuracy: {rolling_accuracy:.1f}%")
leakage_checks.append(("demand_rolling_mean_7d accuracy", rolling_accuracy, 95))

# 3. Target variables: verify they are future values
print("3. Target variable validation...")
sample3 = (
    df_sample[["date", "product_id", "store_id", "quantity_sold", "target_demand_1d"]]
    .dropna()
    .head(1000)
)
target_match = 0
for idx, row in sample3.iterrows():
    future = df_sample[
        (df_sample["date"] == row["date"] + pd.Timedelta(days=1))
        & (df_sample["product_id"] == row["product_id"])
        & (df_sample["store_id"] == row["store_id"])
    ]
    if (
        not future.empty
        and abs(future.iloc[0]["quantity_sold"] - row["target_demand_1d"]) < 1e-9
    ):
        target_match += 1
target_accuracy = target_match / len(sample3) * 100 if len(sample3) > 0 else 0
print(f"   target_demand_1d accuracy: {target_accuracy:.1f}%")
leakage_checks.append(("target_demand_1d accuracy", target_accuracy, 95))

# 4. No future information in features
print("4. Future information check...")
feature_cols = [
    c
    for c in df.columns
    if c
    not in [
        "date",
        "product_id",
        "store_id",
        "category",
        "subcategory",
        "store_type",
        "city",
        "state",
        "supplier_id",
        "warehouse_id",
    ]
]
suspicious = []
for col in feature_cols:
    if "target" in col:
        continue
    corr_past = df_sample[col].corr(df_sample["quantity_sold"])
    corr_future = df_sample[col].corr(df_sample["target_demand_1d"])
    if (
        pd.notna(corr_past)
        and pd.notna(corr_future)
        and abs(corr_future) > abs(corr_past) + 0.1
    ):
        suspicious.append((col, corr_past, corr_future))

if suspicious:
    print(f"   WARNING: {len(suspicious)} features show suspicious future correlation:")
    for col, past, future in suspicious[:5]:
        print(f"     {col}: past={past:.3f}, future={future:.3f}")
else:
    print("   No suspicious future correlations detected.")
leakage_checks.append(("No suspicious future correlations", len(suspicious), 0))

# 5. Row count sanity check
print("5. Row count sanity check...")
expected_rows = 365000
actual_rows = len(df)
row_check = actual_rows == expected_rows
print(f"   Expected rows: {expected_rows}, Actual: {actual_rows}")
leakage_checks.append(("Row count matches", actual_rows, expected_rows))

# 6. No NaN in critical features
print("6. Critical feature NaN check...")
critical_features = [
    "demand_lag_1d",
    "demand_rolling_mean_7d",
    "demand_rolling_std_7d",
    "stock_coverage_days",
    "target_demand_1d",
]
nan_counts = {}
for feat in critical_features:
    nan_count = df[feat].isna().sum()
    nan_counts[feat] = nan_count
    print(f"   {feat}: {nan_count} NaN values")
leakage_checks.append(("No NaN in critical features", sum(nan_counts.values()), 0))

# Summary
print("\n=== LEAKAGE VALIDATION SUMMARY ===")
all_passed = True
for check_name, value, threshold in leakage_checks:
    if isinstance(threshold, str):
        passed = str(value) == threshold
    else:
        passed = (
            value <= threshold
            if check_name
            in ["No suspicious future correlations", "No NaN in critical features"]
            else value >= threshold
        )
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_passed = False
    print(f"  {status}: {check_name} = {value} (threshold: {threshold})")

print(
    f"\nOverall: {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED - review needed'}"
)
