# Clustering & Segmentation Methodology

## Overview

This document describes the product, store, and warehouse segmentation pipeline for RetailSync AI.

**Algorithm:** K-Means Clustering
**Evaluation:** Silhouette Score, Elbow Method
**Outputs:**
- `data/processed/product_segments.csv` — 50 products
- `data/processed/store_segments.csv` — 10 stores
- `data/processed/warehouse_segments.csv` — 5 warehouses
- `data/processed/features_with_segments.csv` — Full dataset with segments

---

## Segmentation Results

### Product Segmentation

| Metric | Value |
|---|---|
| Optimal K | 2 |
| Silhouette Score | 0.234 |
| Cluster 0 | 15 products (High-Volume) |
| Cluster 1 | 35 products (Low-Volume) |

**Business Labels:**

| Label | Count | Description |
|---|---|---|
| Slow-Moving | 27 | High inventory, low/zero demand |
| High-Volume / Stable | 9 | High revenue, low variability |
| Low-Volume / Volatile | 7 | Low revenue, high variability |
| High-Volume / Volatile | 4 | High revenue, high variability |
| Medium-Volume / Moderate | 3 | Balanced performance |

**Key Insights:**
- 54% of products are slow-moving, indicating potential dead stock risk
- High-volume products (30%) drive majority of revenue
- Volatile products require flexible inventory policies

---

### Store Segmentation

| Metric | Value |
|---|---|
| Optimal K | 2 |
| Silhouette Score | 0.337 |
| Cluster 0 | 5 stores (High-Performance) |
| Cluster 1 | 5 stores (Low-Performance) |

**Business Labels:**

| Label | Count | Description |
|---|---|---|
| High-Performance | 3 | High revenue, low zero-demand % |
| Low-Performance | 3 | Low revenue, high zero-demand % |
| Stable Performance | 2 | Moderate metrics, consistent |
| High-Variability | 2 | High demand volatility |

**Key Insights:**
- Store performance is bimodal: high vs low performers
- High-performance stores have 87% lower zero-demand rate
- Store-specific strategies recommended

---

### Warehouse Segmentation

| Metric | Value |
|---|---|
| Optimal K | 4 |
| Silhouette Score | 0.826 |
| Cluster 0 | 16 warehouses (High-Utilization) |
| Cluster 1 | 8 warehouses (Low-Activity) |
| Cluster 2 | 8 warehouses (Overstocked) |
| Cluster 3 | 8 warehouses (Underutilized) |

**Business Labels:**

| Label | Count | Description |
|---|---|---|
| Balanced | 16 | Optimal utilization |
| Overstocked | 8 | High stock coverage |
| Underutilized | 8 | Low stock coverage |
| High-Utilization | 8 | High throughput |

**Key Insights:**
- 50% of warehouses are balanced
- 50% require optimization (overstocked or underutilized)
- High-utilization warehouses may need capacity expansion

---

## Feature Engineering for Clustering

### Product Features
- Sales volume metrics (sum, mean, std)
- Revenue metrics (sum, mean)
- Demand variability (CV)
- Zero-demand proportion
- Inventory metrics (stock coverage, reorder point, max stock)
- Pricing (unit price, cost price)

### Store Features
- Sales volume metrics
- Revenue metrics
- Demand variability
- Zero-demand proportion
- Inventory metrics
- Average unit price

### Warehouse Features
- Sales volume metrics
- Revenue metrics
- Demand variability
- Stock coverage days
- Inventory levels (on hand, reorder point, max stock)

---

## Model Details

### K-Means Configuration
- **Random State:** 42 (reproducibility)
- **N Init:** 10 (multiple initializations)
- **Max Iter:** 300

### Preprocessing
- **Scaling:** StandardScaler (mean=0, std=1)
- **Missing Values:** Filled with 0
- **Infinite Values:** Replaced with 0

---

## Cluster Quality Assessment

| Entity | K | Silhouette | Interpretation |
|---|---|---|---|
| Products | 2 | 0.234 | Weak structure, but meaningful business split |
| Stores | 2 | 0.337 | Moderate structure, clear performance divide |
| Warehouses | 4 | 0.826 | Strong structure, well-separated groups |

**Note:** Silhouette scores are moderate for products and stores due to the zero-inflated nature of the data. The business labels are based on actual cluster characteristics and provide actionable insights despite moderate statistical scores.

---

## Business Applications

### Product Segmentation
- **Inventory Policy:** Different reorder points for high vs low volume products
- **Promotion Strategy:** Target promotions to volatile products to stabilize demand
- **Assortment Planning:** Consider delisting slow-moving products

### Store Segmentation
- **Resource Allocation:** Prioritize high-performance stores for new inventory
- **Marketing:** Tailor campaigns to store type (Urban vs Rural)
- **Staffing:** Adjust staffing based on demand variability

### Warehouse Segmentation
- **Capacity Planning:** Expand high-utilization warehouses
- **Cost Optimization:** Consolidate underutilized warehouses
- **Inventory Placement:** Route high-turnover products to balanced warehouses

---

## Limitations

1. **Small sample sizes:** Only 50 products, 10 stores, 5 warehouses limit clustering granularity.

2. **Moderate silhouette scores:** Product and store clusters show weak to moderate separation. More features or different algorithms (e.g., DBSCAN, hierarchical) could improve results.

3. **Static segmentation:** Clusters are based on historical averages. Dynamic segmentation over time would capture evolving patterns.

4. **No causal interpretation:** Clusters describe correlations, not causation. Business labels are based on observed characteristics.

---

## Reproducibility

```bash
# Run clustering
python src/clustering/segmentation.py
```

**Outputs:**
- `models/product_clusterer.pkl` — Product clustering model
- `models/store_clusterer.pkl` — Store clustering model
- `models/warehouse_clusterer.pkl` — Warehouse clustering model
- `docs/product_cluster_elbow.png` — Elbow plot
- `docs/store_cluster_elbow.png` — Elbow plot
- `docs/warehouse_cluster_elbow.png` — Elbow plot

---

*Generated by `src/clustering/segmentation.py`*
