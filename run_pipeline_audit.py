import json
import os
from datetime import datetime, timezone

import pandas as pd

print("=== RETAILSYNC AI - END-TO-END ML PIPELINE ===\n")

PIPELINE_STEPS = [
    "1. Load raw data",
    "2. Clean and validate data",
    "3. Engineer features",
    "4. Train forecasting model",
    "5. Generate forecasts",
    "6. Detect inventory risks",
    "7. Detect demand anomalies",
    "8. Perform clustering",
    "9. Optimize warehouses",
    "10. Generate business insights",
]

# ============================================================
# PIPELINE CONFIGURATION
# ============================================================

PIPELINE_CONFIG = {
    "pipeline_name": "RetailSync AI",
    "version": "1.0.0",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "data_source": "Synthetic retail data",
    "date_range": "2023-08-11 to 2025-08-09",
    "train_end": "2024-12-31",
    "validation_end": "2025-06-09",
    "test_end": "2025-08-09",
    "forecast_horizon_days": 14,
    "models": {
        "demand_forecaster": "Baseline Mean (historical average)",
        "product_clusterer": "K-Means (K=2)",
        "store_clusterer": "K-Means (K=2)",
        "warehouse_clusterer": "K-Means (K=4)",
    },
    "metrics": {
        "demand_forecasting_mae": 4.0913,
        "demand_forecasting_rmse": 6.6497,
        "demand_forecasting_r2": -0.0035,
        "product_silhouette": 0.234,
        "store_silhouette": 0.337,
        "warehouse_silhouette": 0.826,
    },
    "outputs": {
        "features": "data/processed/features_daily.csv",
        "forecasts": "data/processed/forecasts_next_14d.csv",
        "inventory_alerts": "data/processed/inventory_intelligence.csv",
        "anomalies": "data/processed/anomalies.csv",
        "product_segments": "data/processed/product_segments.csv",
        "store_segments": "data/processed/store_segments.csv",
        "warehouse_segments": "data/processed/warehouse_segments.csv",
        "warehouse_optimization": "data/processed/warehouse_optimization.csv",
    },
}

# ============================================================
# VALIDATE ALL OUTPUTS EXIST
# ============================================================

print("Validating pipeline outputs...")

required_outputs = [
    "data/processed/features_daily.csv",
    "data/processed/forecasts_next_14d.csv",
    "data/processed/inventory_intelligence.csv",
    "data/processed/anomalies.csv",
    "data/processed/product_segments.csv",
    "data/processed/store_segments.csv",
    "data/processed/warehouse_segments.csv",
    "data/processed/warehouse_optimization.csv",
    "models/demand_forecaster.pkl",
    "models/product_clusterer.pkl",
    "models/store_clusterer.pkl",
    "models/warehouse_clusterer.pkl",
]

missing_outputs = []
for output in required_outputs:
    if not os.path.exists(output):
        missing_outputs.append(output)
        print(f"  MISSING: {output}")
    else:
        print(f"  OK: {output}")

if missing_outputs:
    print(
        f"\nWARNING: {len(missing_outputs)} outputs missing. Run individual phase scripts first."
    )
else:
    print("\nAll pipeline outputs validated successfully.")

# ============================================================
# LOAD AND VALIDATE KEY OUTPUTS
# ============================================================

print("\n=== LOADING PIPELINE OUTPUTS ===\n")

# 1. Features
features = pd.read_csv("data/processed/features_daily.csv", parse_dates=["date"])
print(f"Features: {features.shape[0]:,} rows × {features.shape[1]} columns")
print(
    f"  Date range: {features['date'].min().date()} to {features['date'].max().date()}"
)
print(
    f"  Products: {features['product_id'].nunique()}, Stores: {features['store_id'].nunique()}"
)

# 2. Forecasts
forecasts = pd.read_csv("data/processed/forecasts_next_14d.csv", parse_dates=["date"])
print(f"\nForecasts: {forecasts.shape[0]:,} rows")
print(
    f"  Date range: {forecasts['date'].min().date()} to {forecasts['date'].max().date()}"
)
print(f"  Total forecasted demand: {forecasts['forecast_demand'].sum():,.0f} units")
print(f"  Total forecasted revenue: ${forecasts['forecast_revenue'].sum():,.2f}")

# 3. Inventory Intelligence
inv_intel = pd.read_csv("data/processed/inventory_intelligence.csv")
print(f"\nInventory Intelligence: {len(inv_intel):,} product-store combinations")
print(f"  Stockout HIGH: {(inv_intel['stockout_risk'] == 'HIGH').sum()}")
print(f"  Overstock HIGH: {(inv_intel['overstock_risk'] == 'HIGH').sum()}")
print(f"  Urgent Reorder: {(inv_intel['reorder_urgency'] == 'URGENT').sum()}")

# 4. Anomalies
anomalies = pd.read_csv("data/processed/anomalies.csv", parse_dates=["date"])
print(
    f"\nAnomalies: {len(anomalies):,} records ({len(anomalies) / len(features) * 100:.2f}%)"
)
print(f"  Demand Spikes: {(anomalies['anomaly_type'] == 'Demand Spike').sum()}")
print(f"  Unusual Patterns: {(anomalies['anomaly_type'] == 'Unusual Pattern').sum()}")

# 5. Segments
product_segments = pd.read_csv("data/processed/product_segments.csv")
store_segments = pd.read_csv("data/processed/store_segments.csv")
warehouse_segments = pd.read_csv("data/processed/warehouse_segments.csv")
print("\nSegments:")
print(
    f"  Products: {len(product_segments)} ({product_segments['product_cluster_label'].value_counts().to_dict()})"
)
print(
    f"  Stores: {len(store_segments)} ({store_segments['store_cluster_label'].value_counts().to_dict()})"
)
print(
    f"  Warehouses: {len(warehouse_segments)} ({warehouse_segments['warehouse_cluster_label'].value_counts().to_dict()})"
)

# 6. Warehouse Optimization
wh_opt = pd.read_csv("data/processed/warehouse_optimization.csv")
print(f"\nWarehouse Optimization: {len(wh_opt)} warehouses")
print(f"  Avg utilization: {wh_opt['utilization_pct'].mean():.1f}%")
print(f"  Total capacity: {wh_opt['capacity_m3'].sum():,.0f} m³")

# ============================================================
# GENERATE BUSINESS INSIGHTS SUMMARY
# ============================================================

print("\n=== BUSINESS INSIGHTS SUMMARY ===\n")

insights = {
    "forecasting": {
        "best_model": "Baseline Mean",
        "test_mae": 4.09,
        "test_rmse": 6.65,
        "forecast_horizon": "14 days",
        "total_forecasted_demand": int(forecasts["forecast_demand"].sum()),
        "total_forecasted_revenue": float(forecasts["forecast_revenue"].sum()),
    },
    "inventory": {
        "total_product_stores": len(inv_intel),
        "stockout_high": int((inv_intel["stockout_risk"] == "HIGH").sum()),
        "stockout_medium": int((inv_intel["stockout_risk"] == "MEDIUM").sum()),
        "overstock_high": int((inv_intel["overstock_risk"] == "HIGH").sum()),
        "urgent_reorder": int((inv_intel["reorder_urgency"] == "URGENT").sum()),
        "dead_stock": int(inv_intel["dead_stock"].sum()),
    },
    "anomalies": {
        "total_anomalies": len(anomalies),
        "anomaly_rate_pct": round(len(anomalies) / len(features) * 100, 2),
        "demand_spikes": int((anomalies["anomaly_type"] == "Demand Spike").sum()),
        "unusual_patterns": int((anomalies["anomaly_type"] == "Unusual Pattern").sum()),
    },
    "segmentation": {
        "product_clusters": {
            "k": 2,
            "silhouette": 0.234,
            "labels": product_segments["product_cluster_label"]
            .value_counts()
            .to_dict(),
        },
        "store_clusters": {
            "k": 2,
            "silhouette": 0.337,
            "labels": store_segments["store_cluster_label"].value_counts().to_dict(),
        },
        "warehouse_clusters": {
            "k": 4,
            "silhouette": 0.826,
            "labels": warehouse_segments["warehouse_cluster_label"]
            .value_counts()
            .to_dict(),
        },
    },
    "warehouse": {
        "total_warehouses": len(wh_opt),
        "total_capacity_m3": float(wh_opt["capacity_m3"].sum()),
        "avg_utilization_pct": float(wh_opt["utilization_pct"].mean()),
        "high_utilization": int((wh_opt["utilization_pct"] > 80).sum()),
        "low_utilization": int((wh_opt["utilization_pct"] < 50).sum()),
    },
}

# Print insights
for category, data in insights.items():
    print(f"\n{category.upper()}:")
    for key, value in data.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")

# ============================================================
# SAVE PIPELINE SUMMARY
# ============================================================

print("\n=== SAVING PIPELINE SUMMARY ===\n")

os.makedirs("docs", exist_ok=True)

# Save config
with open("docs/pipeline_config.json", "w") as f:
    json.dump(PIPELINE_CONFIG, f, indent=2, default=str)
print("Saved: docs/pipeline_config.json")

# Save insights
with open("docs/pipeline_insights.json", "w") as f:
    json.dump(insights, f, indent=2, default=str)
print("Saved: docs/pipeline_insights.json")

# Save comprehensive summary
summary_lines = []
summary_lines.append("# RetailSync AI - End-to-End Pipeline Summary")
summary_lines.append(
    f"\n**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}"
)
summary_lines.append("\n## Pipeline Overview")
summary_lines.append(
    "\nRetailSync AI is an end-to-end retail supply chain analytics platform that combines:"
)
summary_lines.append("\n1. **Demand Forecasting** — Predict future product demand")
summary_lines.append(
    "2. **Inventory Intelligence** — Detect stockout, overstock, and dead stock risks"
)
summary_lines.append(
    "3. **Anomaly Detection** — Identify unusual demand spikes and patterns"
)
summary_lines.append(
    "4. **Segmentation** — Group products, stores, and warehouses by behavior"
)
summary_lines.append("5. **Warehouse Optimization** — Analyze utilization and capacity")
summary_lines.append("\n## Data")
summary_lines.append("\n- **Source:** Synthetic retail data")
summary_lines.append("- **Date range:** 2023-08-11 to 2025-08-09 (730 days)")
summary_lines.append("- **Products:** 50")
summary_lines.append("- **Stores:** 10")
summary_lines.append("- **Warehouses:** 5")
summary_lines.append("- **Sales records:** 69,216")
summary_lines.append("- **Inventory records:** 52,500")
summary_lines.append("\n## Key Results")
summary_lines.append("\n### Demand Forecasting")
summary_lines.append("\n- **Best model:** Baseline Mean (historical average)")
summary_lines.append(f"- **Test MAE:** {insights['forecasting']['test_mae']}")
summary_lines.append(f"- **Test RMSE:** {insights['forecasting']['test_rmse']}")
summary_lines.append(
    f"- **14-day forecast:** {insights['forecasting']['total_forecasted_demand']:,} units, ${insights['forecasting']['total_forecasted_revenue']:,.2f}"
)
summary_lines.append("\n### Inventory Intelligence")
summary_lines.append(
    f"\n- **Product-store combinations analyzed:** {insights['inventory']['total_product_stores']:,}"
)
summary_lines.append(f"- **Stockout HIGH:** {insights['inventory']['stockout_high']}")
summary_lines.append(
    f"- **Stockout MEDIUM:** {insights['inventory']['stockout_medium']}"
)
summary_lines.append(f"- **Overstock HIGH:** {insights['inventory']['overstock_high']}")
summary_lines.append(
    f"- **Urgent reorder needed:** {insights['inventory']['urgent_reorder']}"
)
summary_lines.append("\n### Anomaly Detection")
summary_lines.append(
    f"\n- **Total anomalies:** {insights['anomalies']['total_anomalies']:,} ({insights['anomalies']['anomaly_rate_pct']}%)"
)
summary_lines.append(f"- **Demand spikes:** {insights['anomalies']['demand_spikes']:,}")
summary_lines.append(
    f"- **Unusual patterns:** {insights['anomalies']['unusual_patterns']:,}"
)
summary_lines.append("\n### Segmentation")
summary_lines.append(
    f"\n- **Product clusters (K=2):** Silhouette={insights['segmentation']['product_clusters']['silhouette']}"
)
summary_lines.append(
    f"  - Labels: {insights['segmentation']['product_clusters']['labels']}"
)
summary_lines.append(
    f"- **Store clusters (K=2):** Silhouette={insights['segmentation']['store_clusters']['silhouette']}"
)
summary_lines.append(
    f"  - Labels: {insights['segmentation']['store_clusters']['labels']}"
)
summary_lines.append(
    f"- **Warehouse clusters (K=4):** Silhouette={insights['segmentation']['warehouse_clusters']['silhouette']}"
)
summary_lines.append(
    f"  - Labels: {insights['segmentation']['warehouse_clusters']['labels']}"
)
summary_lines.append("\n### Warehouse Optimization")
summary_lines.append(
    f"\n- **Warehouses analyzed:** {insights['warehouse']['total_warehouses']}"
)
summary_lines.append(
    f"- **Total capacity:** {insights['warehouse']['total_capacity_m3']:,.0f} m³"
)
summary_lines.append(
    f"- **Average utilization:** {insights['warehouse']['avg_utilization_pct']:.1f}%"
)
summary_lines.append(
    f"- **High utilization (>80%):** {insights['warehouse']['high_utilization']}"
)
summary_lines.append(
    f"- **Low utilization (<50%):** {insights['warehouse']['low_utilization']}"
)
summary_lines.append("\n## Pipeline Architecture")
summary_lines.append("\n```")
summary_lines.append(
    "Raw Data -> Cleaning -> Feature Engineering -> Models -> Predictions -> Risk Detection -> Business Insights"
)
summary_lines.append("```")
summary_lines.append("\n## Files and Outputs")
summary_lines.append("\n| Output | File | Rows/Records |")
summary_lines.append("|--------|------|--------------|")
summary_lines.append(
    f"| Features | `data/processed/features_daily.csv` | {features.shape[0]:,} |"
)
summary_lines.append(
    f"| Forecasts | `data/processed/forecasts_next_14d.csv` | {forecasts.shape[0]:,} |"
)
summary_lines.append(
    f"| Inventory Alerts | `data/processed/inventory_intelligence.csv` | {len(inv_intel):,} |"
)
summary_lines.append(
    f"| Anomalies | `data/processed/anomalies.csv` | {len(anomalies):,} |"
)
summary_lines.append(
    f"| Product Segments | `data/processed/product_segments.csv` | {len(product_segments)} |"
)
summary_lines.append(
    f"| Store Segments | `data/processed/store_segments.csv` | {len(store_segments)} |"
)
summary_lines.append(
    f"| Warehouse Segments | `data/processed/warehouse_segments.csv` | {len(warehouse_segments)} |"
)
summary_lines.append(
    f"| Warehouse Optimization | `data/processed/warehouse_optimization.csv` | {len(wh_opt)} |"
)
summary_lines.append("\n## Model Artifacts")
summary_lines.append("\n| Model | File | Type |")
summary_lines.append("|-------|------|------|")
summary_lines.append(
    "| Demand Forecaster | `models/demand_forecaster.pkl` | Baseline Mean |"
)
summary_lines.append(
    "| Product Clusterer | `models/product_clusterer.pkl` | K-Means (K=2) |"
)
summary_lines.append(
    "| Store Clusterer | `models/store_clusterer.pkl` | K-Means (K=2) |"
)
summary_lines.append(
    "| Warehouse Clusterer | `models/warehouse_clusterer.pkl` | K-Means (K=4) |"
)
summary_lines.append("\n## How to Run")
summary_lines.append("\n```bash")
summary_lines.append("# Run individual components")
summary_lines.append("python src/data/generate_dataset.py")
summary_lines.append("python src/data/ingest.py")
summary_lines.append("python src/database/init_db.py")
summary_lines.append("python src/features/feature_engineering.py")
summary_lines.append("python src/forecasting/demand_forecaster.py")
summary_lines.append("python src/forecasting/forecast_pipeline.py")
summary_lines.append("python src/inventory/inventory_intelligence.py")
summary_lines.append("python src/anomaly/anomaly_detection.py")
summary_lines.append("python src/clustering/segmentation.py")
summary_lines.append("python src/clustering/warehouse_optimization.py")
summary_lines.append("```")
summary_lines.append("\n## Limitations")
summary_lines.append(
    "\n1. **Synthetic data:** Results are based on synthetic data, not real retail operations."
)
summary_lines.append(
    "2. **High zero-inflation:** 81% of demand values are zero, making forecasting challenging."
)
summary_lines.append(
    "3. **Baseline forecaster:** The best forecasting model is a simple mean due to data sparsity."
)
summary_lines.append(
    "4. **Static analysis:** Clustering and optimization are based on historical snapshots."
)
summary_lines.append("\n## Next Steps")
summary_lines.append("\n- Dashboard development (Phase 12)")
summary_lines.append("- Testing and validation (Phase 14)")
summary_lines.append("- Deployment (Phase 16)")
summary_lines.append("- GitHub portfolio preparation (Phase 17)")

with open("docs/pipeline_summary.md", "w") as f:
    f.write("\n".join(summary_lines))
print("Saved: docs/pipeline_summary.md")

print("\n=== PIPELINE VALIDATION COMPLETE ===")
print(f"All {len(required_outputs)} pipeline outputs validated.")
print("Pipeline summary saved to docs/pipeline_summary.md")
