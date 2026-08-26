"""Clustering and segmentation for RetailSync AI."""

from __future__ import annotations

import logging
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine

from src.config import settings
from src.exceptions import DataError, ModelError, PipelineError
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

np.random.seed(42)


def find_optimal_k(X_scaled, k_range=range(2, 11)):
    """Find optimal K using elbow method and silhouette scores."""
    inertias = []
    silhouettes = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
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
    logger.info("Saved: %s", output_path)


def interpret_clusters(df, cluster_col, feature_cols, entity_name):
    """Interpret cluster characteristics."""
    logger.info("\n%s Cluster Interpretation:", entity_name)
    cluster_counts = df[cluster_col].value_counts().sort_index()
    for cluster_id in sorted(df[cluster_col].unique()):
        count = cluster_counts.get(cluster_id, 0)
        logger.info("  Cluster %d (n=%d):", cluster_id, count)


def segment_products(features_df: pd.DataFrame) -> pd.DataFrame:
    """Segment products into clusters."""
    logger.info("=== Product Segmentation ===")

    product_features = (
        features_df.groupby("product_id")
        .agg(
            total_revenue=("revenue", "sum"),
            avg_demand=("quantity_sold", "mean"),
            demand_cv=("demand_cv_28d", "mean"),
            zero_demand_pct=("zero_demand_pct_28d", "mean"),
        )
        .reset_index()
    )

    feature_cols = ["total_revenue", "avg_demand", "demand_cv", "zero_demand_pct"]
    X = product_features[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k, _, silhouettes = find_optimal_k(X_scaled, k_range=range(2, 6))
    logger.info("Optimal K for products: %d (silhouette=%.3f)", best_k, max(silhouettes))

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=300)
    product_features["cluster"] = kmeans.fit_predict(X_scaled)

    interpret_clusters(product_features, "cluster", feature_cols, "Product")

    label_map = {}
    for cluster_id in sorted(product_features["cluster"].unique()):
        subset = product_features[product_features["cluster"] == cluster_id]
        avg_rev = subset["total_revenue"].mean()
        avg_cv = subset["demand_cv"].mean()
        if avg_rev > product_features["total_revenue"].median() and avg_cv < 1.0:
            label_map[cluster_id] = "High-Performance"
        elif avg_rev <= product_features["total_revenue"].median() and avg_cv > 1.5:
            label_map[cluster_id] = "Low-Volume / Volatile"
        elif avg_cv <= 0.5:
            label_map[cluster_id] = "High-Volume / Stable"
        else:
            label_map[cluster_id] = "Medium-Volume / Moderate"

    product_features["cluster_label"] = product_features["cluster"].map(label_map)
    logger.info("Product labels:\n%s", product_features["cluster_label"].value_counts().to_string())

    model_path = os.path.join(str(settings.paths.models), "product_clusterer.pkl")
    joblib.dump(
        {"model": kmeans, "scaler": scaler, "features": feature_cols, "k": best_k},
        model_path,
    )
    logger.info("Saved product clusterer to %s", model_path)
    return product_features


def segment_stores(features_df: pd.DataFrame) -> pd.DataFrame:
    """Segment stores into clusters."""
    logger.info("=== Store Segmentation ===")

    store_features = (
        features_df.groupby("store_id")
        .agg(
            total_revenue=("revenue", "sum"),
            avg_demand=("quantity_sold", "mean"),
            demand_cv=("demand_cv_28d", "mean"),
            zero_demand_pct=("zero_demand_pct_28d", "mean"),
        )
        .reset_index()
    )

    feature_cols = ["total_revenue", "avg_demand", "demand_cv", "zero_demand_pct"]
    X = store_features[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k, _, silhouettes = find_optimal_k(X_scaled, k_range=range(2, 6))
    logger.info("Optimal K for stores: %d (silhouette=%.3f)", best_k, max(silhouettes))

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=300)
    store_features["cluster"] = kmeans.fit_predict(X_scaled)

    interpret_clusters(store_features, "cluster", feature_cols, "Store")

    label_map = {}
    for cluster_id in sorted(store_features["cluster"].unique()):
        subset = store_features[store_features["cluster"] == cluster_id]
        avg_rev = subset["total_revenue"].mean()
        avg_cv = subset["demand_cv"].mean()
        if avg_rev > store_features["total_revenue"].median() and avg_cv < 1.0:
            label_map[cluster_id] = "High-Performance"
        elif avg_rev <= store_features["total_revenue"].median() and avg_cv > 1.5:
            label_map[cluster_id] = "Low-Performance"
        elif avg_cv > 1.0:
            label_map[cluster_id] = "High-Variability"
        else:
            label_map[cluster_id] = "Stable Performance"

    store_features["cluster_label"] = store_features["cluster"].map(label_map)
    logger.info("Store labels:\n%s", store_features["cluster_label"].value_counts().to_string())

    model_path = os.path.join(str(settings.paths.models), "store_clusterer.pkl")
    joblib.dump(
        {"model": kmeans, "scaler": scaler, "features": feature_cols, "k": best_k},
        model_path,
    )
    logger.info("Saved store clusterer to %s", model_path)
    return store_features


def segment_warehouses(features_df: pd.DataFrame, inventory_df: pd.DataFrame) -> pd.DataFrame:
    """Segment warehouses into clusters."""
    logger.info("=== Warehouse Segmentation ===")

    wh_features = inventory_df.groupby("warehouse_id").agg(
        total_quantity=("quantity_on_hand", "sum"),
        avg_stock_coverage=("stock_coverage_days", "mean"),
    ).reset_index()

    warehouse_capacity = pd.read_sql("SELECT * FROM warehouses", create_engine(f"sqlite:///{settings.database.path}"))
    wh_features = wh_features.merge(
        warehouse_capacity[["warehouse_id", "capacity_m3"]], on="warehouse_id", how="left"
    )
    wh_features["utilization_pct"] = (
        wh_features["total_quantity"] / wh_features["capacity_m3"] * 100
    ).fillna(0)
    wh_features["turnover"] = wh_features["total_quantity"] / (wh_features["avg_stock_coverage"] + 1)

    feature_cols = ["utilization_pct", "avg_stock_coverage", "turnover"]
    X = wh_features[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k, _, silhouettes = find_optimal_k(X_scaled, k_range=range(2, 6))
    logger.info("Optimal K for warehouses: %d (silhouette=%.3f)", best_k, max(silhouettes))

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10, max_iter=300)
    wh_features["cluster"] = kmeans.fit_predict(X_scaled)

    interpret_clusters(wh_features, "cluster", feature_cols, "Warehouse")

    label_map = {}
    for cluster_id in sorted(wh_features["cluster"].unique()):
        subset = wh_features[wh_features["cluster"] == cluster_id]
        avg_util = subset["utilization_pct"].mean()
        avg_cov = subset["avg_stock_coverage"].mean()
        if avg_util > 60 and avg_cov < 30:
            label_map[cluster_id] = "High-Utilization"
        elif avg_util < 30 and avg_cov > 60:
            label_map[cluster_id] = "Underutilized"
        elif avg_util > 50:
            label_map[cluster_id] = "Overstocked"
        else:
            label_map[cluster_id] = "Balanced"

    wh_features["cluster_label"] = wh_features["cluster"].map(label_map)
    logger.info("Warehouse labels:\n%s", wh_features["cluster_label"].value_counts().to_string())

    model_path = os.path.join(str(settings.paths.models), "warehouse_clusterer.pkl")
    joblib.dump(
        {"model": kmeans, "scaler": scaler, "features": feature_cols, "k": best_k},
        model_path,
    )
    logger.info("Saved warehouse clusterer to %s", model_path)
    return wh_features


def main() -> None:
    """Main entry point."""
    setup_logging(__name__)
    logger.info("=== RetailSync AI - Clustering & Segmentation ===")

    settings.paths.ensure_dirs()

    features_path = os.path.join(str(settings.paths.processed_data), "features_daily.csv")
    if not os.path.exists(features_path):
        raise DataError(f"Features file not found: {features_path}")

    df = pd.read_csv(features_path, parse_dates=["date"])
    logger.info("Dataset shape: %s", df.shape)

    db_path = os.path.join(str(settings.paths.database), "retailsync.db")
    engine = create_engine(f"sqlite:///{db_path}")
    inventory_df = pd.read_sql("SELECT * FROM inventory", engine)
    inventory_df["date"] = pd.to_datetime(inventory_df["date"])

    product_segments = segment_products(df)
    store_segments = segment_stores(df)
    warehouse_segments = segment_warehouses(df, inventory_df)

    product_segments.to_csv(
        os.path.join(str(settings.paths.processed_data), "product_segments.csv"), index=False
    )
    store_segments.to_csv(
        os.path.join(str(settings.paths.processed_data), "store_segments.csv"), index=False
    )
    warehouse_segments.to_csv(
        os.path.join(str(settings.paths.processed_data), "warehouse_segments.csv"), index=False
    )

    logger.info("Segmentation complete.")


if __name__ == "__main__":
    try:
        main()
    except (DataError, ModelError, PipelineError):
        raise
    except Exception as exc:
        raise PipelineError(f"Segmentation failed: {exc}") from exc
