import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore")

np.random.seed(42)

print("=== RETAILSYNC AI - CLUSTERING & SEGMENTATION ===\n")

# Load data
print("Loading data...")
df = pd.read_csv("data/processed/features_daily.csv", parse_dates=["date"])
print(f"Dataset shape: {df.shape}")

os.makedirs("models", exist_ok=True)
os.makedirs("docs", exist_ok=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================


def find_optimal_k(X_scaled, k_range=range(2, 11)):
    """Find optimal K using elbow method and silhouette scores."""
    inertias = []
    silhouettes = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    # Select K with best silhouette score
    best_k = list(k_range)[np.argmax(silhouettes)]
    return best_k, inertias, silhouettes


def plot_elbow(k_range, inertias, silhouettes, title, output_path):
    """Plot elbow curve and silhouette scores."""
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(k_range, inertias, "bo-", linewidth=2, markersize=8)
    ax1.set_xlabel("Number of Clusters (K)")
    ax1.set_ylabel("Inertia")
    ax1.set_title(f"{title} - Elbow Method")
    ax1.grid(True, alpha=0.3)

    ax2.plot(k_range, silhouettes, "ro-", linewidth=2, markersize=8)
    ax2.set_xlabel("Number of Clusters (K)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title(f"{title} - Silhouette Scores")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def interpret_clusters(df, cluster_col, feature_cols, entity_name):
    """Interpret cluster characteristics."""
    print(f"\n{entity_name} Cluster Interpretation:")

    cluster_stats = df.groupby(cluster_col)[feature_cols].mean()
    cluster_counts = df[cluster_col].value_counts().sort_index()

    for cluster_id in sorted(df[cluster_col].unique()):
        count = cluster_counts.get(cluster_id, 0)
        print(f"\n  Cluster {cluster_id} (n={count}):")

        # Get top 3 distinguishing features
        cluster_means = cluster_stats.loc[cluster_id]
        overall_means = df[feature_cols].mean()

        # Calculate z-scores for interpretation
        z_scores = (cluster_means - overall_means) / df[feature_cols].std()
        top_features = z_scores.abs().nlargest(3)

        for feat in top_features.index:
            z = z_scores[feat]
            direction = "high" if z > 0 else "low"
            print(f"    - {feat}: {direction} (z={z:.2f})")

    return cluster_stats


# ============================================================
# 1. PRODUCT SEGMENTATION
# ============================================================
print("\n=== 1. PRODUCT SEGMENTATION ===")

# Aggregate features to product level
product_features = (
    df.groupby("product_id")
    .agg(
        {
            "quantity_sold": ["sum", "mean", "std"],
            "revenue": ["sum", "mean"],
            "unit_price": "mean",
            "cost_price": "mean",
            "demand_cv_28d": "mean",
            "zero_demand_pct_28d": "mean",
            "stock_coverage_days": "mean",
            "demand_rolling_mean_7d": "mean",
            "demand_rolling_std_7d": "mean",
        }
    )
    .reset_index()
)

# Flatten column names
product_features.columns = ["product_id"] + [
    f"{col[0]}_{col[1]}" for col in product_features.columns[1:]
]

# Add static attributes
product_static = df[
    ["product_id", "category", "subcategory", "weight_kg", "volume_m3"]
].drop_duplicates()
product_features = product_features.merge(product_static, on="product_id", how="left")

# Add inventory metrics
product_inv = (
    df.groupby("product_id")
    .agg(
        {
            "quantity_on_hand": "mean",
            "reorder_point": "mean",
            "max_stock_level": "mean",
            "stock_vs_reorder": "mean",
        }
    )
    .reset_index()
)
product_inv.columns = ["product_id"] + [f"inv_{col}" for col in product_inv.columns[1:]]
product_features = product_features.merge(product_inv, on="product_id", how="left")

# Select features for clustering
product_cluster_features = [
    "quantity_sold_sum",
    "quantity_sold_mean",
    "quantity_sold_std",
    "revenue_sum",
    "revenue_mean",
    "demand_cv_28d_mean",
    "zero_demand_pct_28d_mean",
    "stock_coverage_days_mean",
    "inv_quantity_on_hand",
    "inv_reorder_point",
    "inv_max_stock_level",
    "unit_price_mean",
    "cost_price_mean",
]

# Handle NaN and infinite values
product_features[product_cluster_features] = product_features[
    product_cluster_features
].fillna(0)
product_features[product_cluster_features] = product_features[
    product_cluster_features
].replace([np.inf, -np.inf], 0)

# Scale features
scaler_prod = StandardScaler()
X_prod = scaler_prod.fit_transform(product_features[product_cluster_features])

# Find optimal K
print("Finding optimal K for products...")
best_k_prod, inertias_prod, silhouettes_prod = find_optimal_k(X_prod, range(2, 8))
print(f"  Optimal K: {best_k_prod} (Silhouette: {max(silhouettes_prod):.3f})")

plot_elbow(
    range(2, 8),
    inertias_prod,
    silhouettes_prod,
    "Product Segmentation",
    "docs/product_cluster_elbow.png",
)

# Train final model
kmeans_prod = KMeans(n_clusters=best_k_prod, random_state=42, n_init=10, max_iter=300)
product_features["product_cluster"] = kmeans_prod.fit_predict(X_prod)

# Evaluate
silhouette_prod = silhouette_score(X_prod, product_features["product_cluster"])
print(f"  Final Silhouette Score: {silhouette_prod:.3f}")
print("  Cluster distribution:")
print(product_features["product_cluster"].value_counts().sort_index())

# Save model
joblib.dump(
    {"model": kmeans_prod, "scaler": scaler_prod, "features": product_cluster_features},
    "models/product_clusterer.pkl",
)
print("  Saved: models/product_clusterer.pkl")

# ============================================================
# 2. STORE SEGMENTATION
# ============================================================
print("\n=== 2. STORE SEGMENTATION ===")

# Aggregate features to store level
store_features = (
    df.groupby("store_id")
    .agg(
        {
            "quantity_sold": ["sum", "mean", "std"],
            "revenue": ["sum", "mean"],
            "unit_price": "mean",
            "demand_cv_28d": "mean",
            "zero_demand_pct_28d": "mean",
            "stock_coverage_days": "mean",
            "demand_rolling_mean_7d": "mean",
        }
    )
    .reset_index()
)

store_features.columns = ["store_id"] + [
    f"{col[0]}_{col[1]}" for col in store_features.columns[1:]
]

# Add static attributes
store_static = df[["store_id", "store_type", "city", "state"]].drop_duplicates()
store_features = store_features.merge(store_static, on="store_id", how="left")

# Add inventory metrics
store_inv = (
    df.groupby("store_id")
    .agg(
        {
            "quantity_on_hand": "mean",
            "reorder_point": "mean",
            "max_stock_level": "mean",
        }
    )
    .reset_index()
)
store_inv.columns = ["store_id"] + [f"inv_{col}" for col in store_inv.columns[1:]]
store_features = store_features.merge(store_inv, on="store_id", how="left")

# Select features
store_cluster_features = [
    "quantity_sold_sum",
    "quantity_sold_mean",
    "quantity_sold_std",
    "revenue_sum",
    "revenue_mean",
    "demand_cv_28d_mean",
    "zero_demand_pct_28d_mean",
    "stock_coverage_days_mean",
    "inv_quantity_on_hand",
    "inv_reorder_point",
    "inv_max_stock_level",
    "unit_price_mean",
]

# Handle NaN
store_features[store_cluster_features] = store_features[store_cluster_features].fillna(
    0
)
store_features[store_cluster_features] = store_features[store_cluster_features].replace(
    [np.inf, -np.inf], 0
)

# Scale
scaler_store = StandardScaler()
X_store = scaler_store.fit_transform(store_features[store_cluster_features])

# Find optimal K
print("Finding optimal K for stores...")
best_k_store, inertias_store, silhouettes_store = find_optimal_k(X_store, range(2, 6))
print(f"  Optimal K: {best_k_store} (Silhouette: {max(silhouettes_store):.3f})")

plot_elbow(
    range(2, 6),
    inertias_store,
    silhouettes_store,
    "Store Segmentation",
    "docs/store_cluster_elbow.png",
)

# Train
kmeans_store = KMeans(n_clusters=best_k_store, random_state=42, n_init=10, max_iter=300)
store_features["store_cluster"] = kmeans_store.fit_predict(X_store)

silhouette_store = silhouette_score(X_store, store_features["store_cluster"])
print(f"  Final Silhouette Score: {silhouette_store:.3f}")
print("  Cluster distribution:")
print(store_features["store_cluster"].value_counts().sort_index())

joblib.dump(
    {"model": kmeans_store, "scaler": scaler_store, "features": store_cluster_features},
    "models/store_clusterer.pkl",
)
print("  Saved: models/store_clusterer.pkl")

# ============================================================
# 3. WAREHOUSE SEGMENTATION
# ============================================================
print("\n=== 3. WAREHOUSE SEGMENTATION ===")

# Aggregate features to warehouse level
warehouse_features = (
    df.groupby("warehouse_id")
    .agg(
        {
            "quantity_sold": ["sum", "mean", "std"],
            "revenue": ["sum", "mean"],
            "demand_cv_28d": "mean",
            "stock_coverage_days": "mean",
            "quantity_on_hand": "mean",
            "reorder_point": "mean",
            "max_stock_level": "mean",
        }
    )
    .reset_index()
)

warehouse_features.columns = ["warehouse_id"] + [
    f"{col[0]}_{col[1]}" for col in warehouse_features.columns[1:]
]

# Add warehouse static data
warehouse_static = df[["warehouse_id", "supplier_id"]].drop_duplicates(
    subset=["warehouse_id"]
)
warehouse_features = warehouse_features.merge(
    warehouse_static, on="warehouse_id", how="left"
)

# Select features
warehouse_cluster_features = [
    "quantity_sold_sum",
    "quantity_sold_mean",
    "quantity_sold_std",
    "revenue_sum",
    "revenue_mean",
    "demand_cv_28d_mean",
    "stock_coverage_days_mean",
    "quantity_on_hand_mean",
    "reorder_point_mean",
    "max_stock_level_mean",
]

# Handle NaN
warehouse_features[warehouse_cluster_features] = warehouse_features[
    warehouse_cluster_features
].fillna(0)
warehouse_features[warehouse_cluster_features] = warehouse_features[
    warehouse_cluster_features
].replace([np.inf, -np.inf], 0)

# Scale
scaler_wh = StandardScaler()
X_wh = scaler_wh.fit_transform(warehouse_features[warehouse_cluster_features])

# Find optimal K (small dataset, limited range)
print("Finding optimal K for warehouses...")
best_k_wh, inertias_wh, silhouettes_wh = find_optimal_k(X_wh, range(2, 5))
print(f"  Optimal K: {best_k_wh} (Silhouette: {max(silhouettes_wh):.3f})")

plot_elbow(
    range(2, 5),
    inertias_wh,
    silhouettes_wh,
    "Warehouse Segmentation",
    "docs/warehouse_cluster_elbow.png",
)

# Train
kmeans_wh = KMeans(n_clusters=best_k_wh, random_state=42, n_init=10, max_iter=300)
warehouse_features["warehouse_cluster"] = kmeans_wh.fit_predict(X_wh)

silhouette_wh = silhouette_score(X_wh, warehouse_features["warehouse_cluster"])
print(f"  Final Silhouette Score: {silhouette_wh:.3f}")
print("  Cluster distribution:")
print(warehouse_features["warehouse_cluster"].value_counts().sort_index())

joblib.dump(
    {"model": kmeans_wh, "scaler": scaler_wh, "features": warehouse_cluster_features},
    "models/warehouse_clusterer.pkl",
)
print("  Saved: models/warehouse_clusterer.pkl")

# ============================================================
# 4. INTERPRET CLUSTERS
# ============================================================
print("\n=== 4. CLUSTER INTERPRETATION ===")

# Product clusters
print("\n--- Product Clusters ---")
product_cluster_stats = interpret_clusters(
    product_features, "product_cluster", product_cluster_features, "Product"
)

# Store clusters
print("\n--- Store Clusters ---")
store_cluster_stats = interpret_clusters(
    store_features, "store_cluster", store_cluster_features, "Store"
)

# Warehouse clusters
print("\n--- Warehouse Clusters ---")
warehouse_cluster_stats = interpret_clusters(
    warehouse_features, "warehouse_cluster", warehouse_cluster_features, "Warehouse"
)

# ============================================================
# 5. BUSINESS LABELS
# ============================================================
print("\n=== 5. ASSIGNING BUSINESS LABELS ===")


# Product labels based on cluster characteristics
def label_product_cluster(row):
    if (
        row["revenue_sum"] > product_features["revenue_sum"].quantile(0.75)
        and row["demand_cv_28d_mean"] < product_features["demand_cv_28d_mean"].median()
    ):
        return "High-Volume / Stable"
    elif (
        row["revenue_sum"] > product_features["revenue_sum"].quantile(0.75)
        and row["demand_cv_28d_mean"] >= product_features["demand_cv_28d_mean"].median()
    ):
        return "High-Volume / Volatile"
    elif (
        row["revenue_sum"] <= product_features["revenue_sum"].quantile(0.25)
        and row["demand_cv_28d_mean"] >= product_features["demand_cv_28d_mean"].median()
    ):
        return "Low-Volume / Volatile"
    elif row["zero_demand_pct_28d_mean"] > 0.8:
        return "Slow-Moving"
    else:
        return "Medium-Volume / Moderate"


product_features["product_cluster_label"] = product_features.apply(
    label_product_cluster, axis=1
)
print("\nProduct Cluster Labels:")
print(product_features["product_cluster_label"].value_counts())


# Store labels
def label_store_cluster(row):
    if row["revenue_sum"] > store_features["revenue_sum"].quantile(0.75):
        return "High-Performance"
    elif row["revenue_sum"] <= store_features["revenue_sum"].quantile(0.25):
        return "Low-Performance"
    elif row["demand_cv_28d_mean"] > store_features["demand_cv_28d_mean"].median():
        return "High-Variability"
    else:
        return "Stable Performance"


store_features["store_cluster_label"] = store_features.apply(
    label_store_cluster, axis=1
)
print("\nStore Cluster Labels:")
print(store_features["store_cluster_label"].value_counts())


# Warehouse labels
def label_warehouse_cluster(row):
    if row["revenue_sum"] > warehouse_features["revenue_sum"].quantile(0.75):
        return "High-Utilization"
    elif (
        row["stock_coverage_days_mean"]
        > warehouse_features["stock_coverage_days_mean"].median()
    ):
        return "Overstocked"
    elif row["stock_coverage_days_mean"] < warehouse_features[
        "stock_coverage_days_mean"
    ].quantile(0.25):
        return "Underutilized"
    else:
        return "Balanced"


warehouse_features["warehouse_cluster_label"] = warehouse_features.apply(
    label_warehouse_cluster, axis=1
)
print("\nWarehouse Cluster Labels:")
print(warehouse_features["warehouse_cluster_label"].value_counts())

# ============================================================
# 6. SAVE RESULTS
# ============================================================
print("\n=== 6. SAVING SEGMENTATION RESULTS ===")

product_features.to_csv("data/processed/product_segments.csv", index=False)
print(f"Saved: data/processed/product_segments.csv ({len(product_features)} products)")

store_features.to_csv("data/processed/store_segments.csv", index=False)
print(f"Saved: data/processed/store_segments.csv ({len(store_features)} stores)")

warehouse_features.to_csv("data/processed/warehouse_segments.csv", index=False)
print(
    f"Saved: data/processed/warehouse_segments.csv ({len(warehouse_features)} warehouses)"
)

# Merge clusters back to main features
df_with_segments = (
    df.merge(
        product_features[["product_id", "product_cluster", "product_cluster_label"]],
        on="product_id",
        how="left",
    )
    .merge(
        store_features[["store_id", "store_cluster", "store_cluster_label"]],
        on="store_id",
        how="left",
    )
    .merge(
        warehouse_features[
            ["warehouse_id", "warehouse_cluster", "warehouse_cluster_label"]
        ],
        on="warehouse_id",
        how="left",
    )
)

df_with_segments.to_csv("data/processed/features_with_segments.csv", index=False)
print(
    f"Saved: data/processed/features_with_segments.csv ({len(df_with_segments)} rows)"
)

# ============================================================
# 7. UPDATE DATABASE
# ============================================================
print("\n=== 7. UPDATING DATABASE ===")

engine = create_engine("sqlite:///database/retailsync.db")

with engine.connect() as conn:
    # Drop existing tables to recreate with correct schema
    conn.execute(text("DROP TABLE IF EXISTS product_segments"))
    conn.execute(text("DROP TABLE IF EXISTS store_segments"))
    conn.execute(text("DROP TABLE IF EXISTS warehouse_segments"))
    conn.commit()

    # Create product segments table
    conn.execute(
        text("""
        CREATE TABLE IF NOT EXISTS product_segments (
            product_id TEXT PRIMARY KEY,
            cluster INTEGER NOT NULL,
            cluster_label TEXT NOT NULL,
            revenue_sum REAL,
            demand_cv_28d_mean REAL,
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        )
    """)
    )

    # Create store segments table
    conn.execute(
        text("""
        CREATE TABLE IF NOT EXISTS store_segments (
            store_id TEXT PRIMARY KEY,
            cluster INTEGER NOT NULL,
            cluster_label TEXT NOT NULL,
            revenue_sum REAL,
            demand_cv_28d_mean REAL,
            FOREIGN KEY (store_id) REFERENCES stores(store_id)
        )
    """)
    )

    # Create warehouse segments table
    conn.execute(
        text("""
        CREATE TABLE IF NOT EXISTS warehouse_segments (
            warehouse_id TEXT PRIMARY KEY,
            cluster INTEGER NOT NULL,
            cluster_label TEXT NOT NULL,
            revenue_sum REAL,
            stock_coverage_days_mean REAL,
            FOREIGN KEY (warehouse_id) REFERENCES warehouses(warehouse_id)
        )
    """)
    )

    conn.commit()

    # Clear existing segments
    conn.execute(text("DELETE FROM product_segments"))
    conn.execute(text("DELETE FROM store_segments"))
    conn.execute(text("DELETE FROM warehouse_segments"))
    conn.commit()

    # Load new segments
    prod_insert = product_features[
        [
            "product_id",
            "product_cluster",
            "product_cluster_label",
            "revenue_sum",
            "demand_cv_28d_mean",
        ]
    ].rename(
        columns={"product_cluster": "cluster", "product_cluster_label": "cluster_label"}
    )
    store_insert = store_features[
        [
            "store_id",
            "store_cluster",
            "store_cluster_label",
            "revenue_sum",
            "demand_cv_28d_mean",
        ]
    ].rename(
        columns={"store_cluster": "cluster", "store_cluster_label": "cluster_label"}
    )
    wh_insert = (
        warehouse_features[
            [
                "warehouse_id",
                "warehouse_cluster",
                "warehouse_cluster_label",
                "revenue_sum",
                "stock_coverage_days_mean",
            ]
        ]
        .rename(
            columns={
                "warehouse_cluster": "cluster",
                "warehouse_cluster_label": "cluster_label",
            }
        )
        .drop_duplicates(subset=["warehouse_id"])
    )

    prod_insert.to_sql("product_segments", con=engine, if_exists="append", index=False)
    store_insert.to_sql("store_segments", con=engine, if_exists="append", index=False)
    wh_insert.to_sql("warehouse_segments", con=engine, if_exists="append", index=False)

    print("Loaded segmentation results into database")

# Verify
with engine.connect() as conn:
    prod_count = conn.execute(text("SELECT COUNT(*) FROM product_segments")).fetchone()[
        0
    ]
    store_count = conn.execute(text("SELECT COUNT(*) FROM store_segments")).fetchone()[
        0
    ]
    wh_count = conn.execute(text("SELECT COUNT(*) FROM warehouse_segments")).fetchone()[
        0
    ]
    print(f"Product segments: {prod_count}")
    print(f"Store segments: {store_count}")
    print(f"Warehouse segments: {wh_count}")

# ============================================================
# 8. SUMMARY
# ============================================================
print("\n=== 8. SEGMENTATION SUMMARY ===")

print(f"""
Product Segmentation:
  - Optimal clusters: {best_k_prod}
  - Silhouette score: {silhouette_prod:.3f}
  - Labels: {product_features["product_cluster_label"].value_counts().to_dict()}

Store Segmentation:
  - Optimal clusters: {best_k_store}
  - Silhouette score: {silhouette_store:.3f}
  - Labels: {store_features["store_cluster_label"].value_counts().to_dict()}

Warehouse Segmentation:
  - Optimal clusters: {best_k_wh}
  - Silhouette score: {silhouette_wh:.3f}
  - Labels: {warehouse_features["warehouse_cluster_label"].value_counts().to_dict()}
""")

print("Clustering & Segmentation complete.")
