import pandas as pd
import numpy as np
import os
import time
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

print("=== RETAILSYNC AI - DEMAND ANOMALY DETECTION ===\n")

# Load data
print("Loading data...")
df = pd.read_csv("data/processed/features_daily.csv", parse_dates=["date"])
print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

# Sort for time-based operations
df = df.sort_values(["product_id", "store_id", "date"]).reset_index(drop=True)

# ============================================================
# 1. ROLLING Z-SCORE METHOD
# ============================================================
print("\n=== 1. ROLLING Z-SCORE METHOD ===")

start_time = time.time()

# Calculate rolling statistics per product-store
df["rolling_mean_30d"] = df.groupby(["product_id", "store_id"])["quantity_sold"].transform(
    lambda x: x.shift(1).rolling(window=30, min_periods=7).mean()
)
df["rolling_std_30d"] = df.groupby(["product_id", "store_id"])["quantity_sold"].transform(
    lambda x: x.shift(1).rolling(window=30, min_periods=7).std()
).fillna(0)

# Calculate Z-score
df["z_score"] = np.where(
    df["rolling_std_30d"] > 0,
    (df["quantity_sold"] - df["rolling_mean_30d"]) / df["rolling_std_30d"],
    0
)

# Classify anomalies
df["anomaly_zscore"] = "Normal"
df.loc[df["z_score"] > 2, "anomaly_zscore"] = "Unusual"
df.loc[df["z_score"] > 3, "anomaly_zscore"] = "Significant Spike"
df.loc[df["z_score"] < -2, "anomaly_zscore"] = "Unusual (Low)"
df.loc[df["z_score"] < -3, "anomaly_zscore"] = "Significant Drop"

zscore_time = time.time() - start_time
print(f"Rolling Z-score completed in {zscore_time:.1f}s")
print(f"Z-score distribution:")
print(df["anomaly_zscore"].value_counts())

# ============================================================
# 2. IQR METHOD
# ============================================================
print("\n=== 2. IQR METHOD ===")

start_time = time.time()

# Calculate IQR per product-store
q1 = df.groupby(["product_id", "store_id"])["quantity_sold"].transform(
    lambda x: x.shift(1).quantile(0.25)
)
q3 = df.groupby(["product_id", "store_id"])["quantity_sold"].transform(
    lambda x: x.shift(1).quantile(0.75)
)
iqr = q3 - q1

df["iqr_lower"] = q1 - 1.5 * iqr
df["iqr_upper"] = q3 + 1.5 * iqr

df["anomaly_iqr"] = "Normal"
df.loc[df["quantity_sold"] > df["iqr_upper"], "anomaly_iqr"] = "Significant Spike"
df.loc[df["quantity_sold"] < df["iqr_lower"], "anomaly_iqr"] = "Significant Drop"

iqr_time = time.time() - start_time
print(f"IQR method completed in {iqr_time:.1f}s")
print(f"IQR anomaly distribution:")
print(df["anomaly_iqr"].value_counts())

# ============================================================
# 3. ISOLATION FOREST METHOD
# ============================================================
print("\n=== 3. ISOLATION FOREST METHOD ===")

start_time = time.time()

# Prepare features for Isolation Forest
# Use product-store level features
iso_features = df.groupby(["product_id", "store_id"]).agg({
    "quantity_sold": ["mean", "std", "max", "min"],
    "revenue": ["mean", "std"],
    "demand_cv_28d": "last",
    "demand_rolling_mean_7d": "last",
    "stock_coverage_days": "last",
}).reset_index()

# Flatten column names
iso_features.columns = ["product_id", "store_id"] + [f"{col[0]}_{col[1]}" for col in iso_features.columns[2:]]

# Fill NaN
iso_features = iso_features.fillna(0)

# Train Isolation Forest
scaler = StandardScaler()
X_iso = scaler.fit_transform(iso_features.iloc[:, 2:])

iso_forest = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
iso_forest.fit(X_iso)

# Predict anomalies
iso_predictions = iso_forest.predict(X_iso)
iso_scores = iso_forest.decision_function(X_iso)

iso_features["anomaly_isolation"] = "Normal"
iso_features.loc[iso_predictions == -1, "anomaly_isolation"] = "Anomaly"

# Merge back to main dataframe
df = df.merge(
    iso_features[["product_id", "store_id", "anomaly_isolation"]],
    on=["product_id", "store_id"],
    how="left"
)

iso_time = time.time() - start_time
print(f"Isolation Forest completed in {iso_time:.1f}s")
print(f"Isolation Forest anomaly distribution:")
print(df["anomaly_isolation"].value_counts())

# ============================================================
# 4. ENSEMBLE ANOMALY DETECTION
# ============================================================
print("\n=== 4. ENSEMBLE ANOMALY DETECTION ===")

# Combine methods: flag as anomaly if 2+ methods agree
def ensemble_anomaly(row):
    votes = 0
    if row["anomaly_zscore"] in ["Unusual", "Significant Spike", "Significant Drop"]:
        votes += 1
    if row["anomaly_iqr"] in ["Significant Spike", "Significant Drop"]:
        votes += 1
    if row["anomaly_isolation"] == "Anomaly":
        votes += 1
    
    if votes >= 2:
        return "Anomaly"
    elif votes == 1:
        return "Suspicious"
    else:
        return "Normal"

df["anomaly_ensemble"] = df.apply(ensemble_anomaly, axis=1)

print(f"Ensemble anomaly distribution:")
print(df["anomaly_ensemble"].value_counts())

# ============================================================
# 5. DETAILED ANOMALY ANALYSIS
# ============================================================
print("\n=== 5. DETAILED ANOMALY ANALYSIS ===")

anomalies = df[df["anomaly_ensemble"] == "Anomaly"].copy()
print(f"Total anomalous records: {len(anomalies)} ({len(anomalies)/len(df)*100:.2f}%)")

# Anomaly breakdown by type
anomalies["anomaly_type"] = "Unknown"
anomalies.loc[anomalies["anomaly_zscore"] == "Significant Spike", "anomaly_type"] = "Demand Spike"
anomalies.loc[anomalies["anomaly_zscore"] == "Significant Drop", "anomaly_type"] = "Demand Drop"
anomalies.loc[anomalies["anomaly_iqr"] == "Significant Spike", "anomaly_type"] = "Demand Spike"
anomalies.loc[anomalies["anomaly_iqr"] == "Significant Drop", "anomaly_type"] = "Demand Drop"
anomalies.loc[anomalies["anomaly_isolation"] == "Anomaly", "anomaly_type"] = "Unusual Pattern"

print(f"\nAnomaly types:")
print(anomalies["anomaly_type"].value_counts())

# Get category and store info from features
product_info = df[["product_id", "category", "subcategory"]].drop_duplicates()
store_info = df[["store_id", "store_type", "city", "state"]].drop_duplicates()

# Anomalies by category
anomalies_with_cat = anomalies.merge(product_info, on="product_id", how="left")
if "category" in anomalies_with_cat.columns:
    print(f"\nAnomalies by category:")
    print(anomalies_with_cat["category"].value_counts())
else:
    print("Warning: category column not available")

# Anomalies by store type
anomalies_with_store = anomalies.merge(store_info, on="store_id", how="left")
if "store_type" in anomalies_with_store.columns:
    print(f"\nAnomalies by store type:")
    print(anomalies_with_store["store_type"].value_counts())
else:
    print("Warning: store_type column not available")

# Top anomalous product-store combinations
anomaly_freq = anomalies.groupby(["product_id", "store_id"]).size().reset_index(name="anomaly_count")
top_anomalies = anomaly_freq.merge(product_info, on="product_id", how="left").merge(
    store_info[["store_id", "store_type", "city"]], on="store_id", how="left"
).sort_values("anomaly_count", ascending=False)

print(f"\nTop 10 most anomalous product-store combinations:")
print(top_anomalies.head(10)[["product_id", "store_id", "category", "store_type", "city", "anomaly_count"]].to_string(index=False))

# ============================================================
# 6. SAVE ANOMALY RESULTS
# ============================================================
print("\n=== 6. SAVING ANOMALY RESULTS ===")

os.makedirs("data/processed", exist_ok=True)

# Save full dataset with anomaly flags
df.to_csv("data/processed/features_with_anomalies.csv", index=False)
print(f"Saved: data/processed/features_with_anomalies.csv ({len(df)} rows)")

# Save anomalies only
anomalies.to_csv("data/processed/anomalies.csv", index=False)
print(f"Saved: data/processed/anomalies.csv ({len(anomalies)} rows)")

# Save anomaly summary
anomaly_summary = {
    "total_records": len(df),
    "total_anomalies": len(anomalies),
    "anomaly_pct": len(anomalies) / len(df) * 100,
    "zscore_anomalies": (df["anomaly_zscore"] != "Normal").sum(),
    "iqr_anomalies": (df["anomaly_iqr"] != "Normal").sum(),
    "isolation_anomalies": (df["anomaly_isolation"] == "Anomaly").sum(),
    "ensemble_anomalies": (df["anomaly_ensemble"] == "Anomaly").sum(),
    "demand_spikes": (anomalies["anomaly_type"] == "Demand Spike").sum(),
    "demand_drops": (anomalies["anomaly_type"] == "Demand Drop").sum(),
    "unusual_patterns": (anomalies["anomaly_type"] == "Unusual Pattern").sum(),
    "top_anomalous_product": top_anomalies.iloc[0]["product_id"] if len(top_anomalies) > 0 else "N/A",
    "top_anomalous_store": top_anomalies.iloc[0]["store_id"] if len(top_anomalies) > 0 else "N/A",
}

anomaly_summary_df = pd.DataFrame([anomaly_summary])
anomaly_summary_df.to_csv("data/processed/anomaly_summary.csv", index=False)
print(f"Saved: data/processed/anomaly_summary.csv")

# ============================================================
# 7. UPDATE DATABASE
# ============================================================
print("\n=== 7. UPDATING DATABASE ===")

from sqlalchemy import create_engine, text

engine = create_engine("sqlite:///database/retailsync.db")

with engine.connect() as conn:
    # Create anomaly flags table
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS anomaly_flags (
            anomaly_id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            product_id TEXT NOT NULL,
            store_id TEXT NOT NULL,
            quantity_sold REAL,
            anomaly_type TEXT NOT NULL,
            z_score REAL,
            method_agreement INTEGER,
            category TEXT,
            store_type TEXT,
            city TEXT,
            FOREIGN KEY (product_id) REFERENCES products(product_id),
            FOREIGN KEY (store_id) REFERENCES stores(store_id)
        )
    """))
    
    # Create indexes
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_anomaly_date ON anomaly_flags(date)"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_anomaly_product ON anomaly_flags(product_id)"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_anomaly_store ON anomaly_flags(store_id)"))
    conn.execute(text("CREATE INDEX IF NOT EXISTS idx_anomaly_type ON anomaly_flags(anomaly_type)"))
    conn.commit()
    
    # Clear existing anomalies
    conn.execute(text("DELETE FROM anomaly_flags"))
    conn.commit()
    
    # Insert anomalies
    cols_to_insert = ["date", "product_id", "store_id", "quantity_sold", "anomaly_type", "z_score"]
    available_cols = [c for c in cols_to_insert if c in anomalies.columns]
    insert_df = anomalies[available_cols].copy()
    
    # Add category and store info if available
    if "category" in anomalies.columns:
        insert_df["category"] = anomalies["category"]
    if "store_type" in anomalies.columns:
        insert_df["store_type"] = anomalies["store_type"]
    if "city" in anomalies.columns:
        insert_df["city"] = anomalies["city"]
    
    # Calculate method agreement (how many methods flagged this as anomaly)
    insert_df["method_agreement"] = (
        (anomalies["anomaly_zscore"] != "Normal").astype(int) +
        (anomalies["anomaly_iqr"] != "Normal").astype(int) +
        (anomalies["anomaly_isolation"] == "Anomaly").astype(int)
    )
    
    insert_df.to_sql(
        "anomaly_flags",
        con=engine,
        if_exists="append",
        index=False
    )
    
    count = conn.execute(text("SELECT COUNT(*) FROM anomaly_flags")).fetchone()[0]
    print(f"Loaded {count} anomalies into database")

# ============================================================
# 8. METHOD COMPARISON
# ============================================================
print("\n=== 8. METHOD COMPARISON ===")

print(f"""
Method               | Anomalies | % of Total | Time (s)
---------------------|-----------|------------|----------
Rolling Z-Score      | {(df['anomaly_zscore'] != 'Normal').sum():>9} | {(df['anomaly_zscore'] != 'Normal').mean()*100:>9.2f}% | {zscore_time:>8.1f}
IQR Method           | {(df['anomaly_iqr'] != 'Normal').sum():>9} | {(df['anomaly_iqr'] != 'Normal').mean()*100:>9.2f}% | {iqr_time:>8.1f}
Isolation Forest     | {(df['anomaly_isolation'] == 'Anomaly').sum():>9} | {(df['anomaly_isolation'] == 'Anomaly').mean()*100:>9.2f}% | {iso_time:>8.1f}
Ensemble (Selected)  | {(df['anomaly_ensemble'] == 'Anomaly').sum():>9} | {(df['anomaly_ensemble'] == 'Anomaly').mean()*100:>9.2f}% | {'N/A':>8}
""")

print("Selected method: Ensemble (requires 2+ method agreement)")
print("Rationale: Reduces false positives by combining multiple detection strategies")

# ============================================================
# 9. SAMPLE ANOMALIES
# ============================================================
print("\n=== 9. SAMPLE ANOMALIES ===")

sample_anomalies = anomalies.head(10)
for _, row in sample_anomalies.iterrows():
    print(f"  {row['date'].date()} | {row['product_id']} | {row['store_id']} | "
          f"Qty: {row['quantity_sold']:.0f} | Z: {row['z_score']:.2f} | "
          f"Type: {row['anomaly_type']}")

print("\nAnomaly detection complete.")
