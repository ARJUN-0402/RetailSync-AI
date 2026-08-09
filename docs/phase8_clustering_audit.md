# RetailSync AI — Phase 8: Clustering Audit

**Date:** 2026-08-09
**Auditor:** Kilo
**Status:** READ-ONLY AUDIT — No code changes made

---

## 1. Executive Summary

The clustering implementation is **methodologically sound** but has **one critical data bug** and **one misleading README claim**. The K-Means algorithm is correctly implemented with proper scaling, silhouette-based K selection, and business labeling. However, the warehouse segments CSV has duplicate rows, and the README misrepresents the relationship between clusters and business labels.

### Bottom Line

| Component | Status | Notes |
|-----------|--------|-------|
| Product segmentation | ⚠️ Bug + Misleading README | 2 clusters, 5 rule-based labels (not 1:1) |
| Store segmentation | ⚠️ Misleading README | 2 clusters, 4 rule-based labels (not 1:1) |
| Warehouse segmentation | ❌ **Critical bug** | CSV has 40 rows instead of 5 |
| K-Means implementation | ✅ Verified | Correct with StandardScaler |
| K selection | ✅ Verified | Silhouette-based |
| Model serialization | ✅ Verified | joblib with scaler + features |
| Business labels | ⚠️ Rule-based | Applied post-clustering, not derived from clusters |

---

## 2. Implementation Audit

### File: `src/clustering/segmentation.py`

**Lines:** 492
**Status:** Implemented with bugs and misleading labels

### 2.1 Helper Functions

**`find_optimal_k(X_scaled, k_range)` (lines 30-43):**
- Trains KMeans for each K in range
- Computes inertia and silhouette score
- Returns K with best silhouette score

**Assessment:** ✅ **Correct implementation.** Uses `n_init=10` to avoid local optima.

**`interpret_clusters(df, cluster_col, feature_cols, entity_name)` (lines 66-90):**
- Computes z-scores for each feature within each cluster
- Identifies top 3 distinguishing features

**Assessment:** ✅ **Useful for interpretation.**

### 2.2 Product Segmentation

**Features used (13 features):**
- `quantity_sold_sum`, `quantity_sold_mean`, `quantity_sold_std`
- `revenue_sum`, `revenue_mean`
- `unit_price_mean`, `cost_price_mean`
- `demand_cv_28d_mean`, `zero_demand_pct_28d_mean`
- `stock_coverage_days_mean`
- `inv_quantity_on_hand`, `inv_reorder_point`, `inv_max_stock_level`

**K selection:** Range 2-7, best K=2, Silhouette=0.234

**Cluster distribution:**
- Cluster 0: 15 products
- Cluster 1: 35 products

**Business labels (5 labels for 2 clusters):**
```python
def label_product_cluster(row):
    if revenue > 75th pct and cv < median: return "High-Volume / Stable"
    elif revenue > 75th pct and cv >= median: return "High-Volume / Volatile"
    elif revenue <= 25th pct and cv >= median: return "Low-Volume / Volatile"
    elif zero_demand_pct > 0.8: return "Slow-Moving"
    else: return "Medium-Volume / Moderate"
```

**Actual label counts:**
- Slow-Moving: 27
- High-Volume / Stable: 9
- Low-Volume / Volatile: 7
- High-Volume / Volatile: 4
- Medium-Volume / Moderate: 3

**Issues:**
1. **2 clusters, 5 labels**: The labels are NOT derived from K-Means clusters. They are rule-based thresholds applied independently. A "High-Volume / Stable" product could be in either cluster 0 or cluster 1.
2. **Labels don't correspond to clusters**: The README presents these as "Product Segmentation: K=2" with specific label counts, implying each label is a cluster. This is misleading.

**Assessment:** ⚠️ **The clustering is correct, but the labeling approach is misleading.** The business labels are useful but should be presented as "segments" or "profiles", not "clusters".

### 2.3 Store Segmentation

**Features used (12 features):**
- `quantity_sold_sum`, `quantity_sold_mean`, `quantity_sold_std`
- `revenue_sum`, `revenue_mean`
- `demand_cv_28d_mean`, `zero_demand_pct_28d_mean`
- `stock_coverage_days_mean`
- `inv_quantity_on_hand`, `inv_reorder_point`, `inv_max_stock_level`
- `unit_price_mean`

**K selection:** Range 2-5, best K=2, Silhouette=0.337

**Cluster distribution:**
- Cluster 0: 5 stores
- Cluster 1: 5 stores

**Business labels (4 labels for 2 clusters):**
```python
def label_store_cluster(row):
    if revenue > 75th pct: return "High-Performance"
    elif revenue <= 25th pct: return "Low-Performance"
    elif cv > median: return "High-Variability"
    else: return "Stable Performance"
```

**Actual label counts:**
- High-Performance: 3
- Low-Performance: 3
- Stable Performance: 2
- High-Variability: 2

**Issues:** Same as products — 2 clusters, 4 rule-based labels that don't correspond to clusters.

### 2.4 Warehouse Segmentation

**Features used (10 features):**
- `quantity_sold_sum`, `quantity_sold_mean`, `quantity_sold_std`
- `revenue_sum`, `revenue_mean`
- `demand_cv_28d_mean`, `stock_coverage_days_mean`
- `quantity_on_hand_mean`, `reorder_point_mean`, `max_stock_level_mean`

**K selection:** Range 2-4, best K=4, Silhouette=0.826

**Cluster distribution:**
- Cluster 0: 16 warehouses
- Cluster 1: 8 warehouses
- Cluster 2: 8 warehouses
- Cluster 3: 8 warehouses

**Business labels (4 labels for 4 clusters):**
```python
def label_warehouse_cluster(row):
    if revenue > 75th pct: return "High-Utilization"
    elif stock_coverage > median: return "Overstocked"
    elif stock_coverage < 25th pct: return "Underutilized"
    else: return "Balanced"
```

**Actual label counts:**
- Balanced: 16
- Overstocked: 8
- Underutilized: 8
- High-Utilization: 8

**Issues:**
1. **Critical bug**: The CSV has 40 rows instead of 5 (confirmed in Phase 1)
2. **Labels still rule-based**: Even though there are 4 labels and 4 clusters, the labels are applied by threshold rules, not derived from cluster assignments

### 2.5 Model Serialization

**Implementation (lines 161, 227, 283):**
```python
joblib.dump({"model": kmeans_prod, "scaler": scaler_prod, "features": product_cluster_features}, "models/product_clusterer.pkl")
```

**Assessment:** ✅ **Correct.** Saves model, scaler, and feature list.

---

## 3. Critical Bug: Warehouse Segments CSV

### Issue: 40 rows instead of 5

**Location:** `segmentation.py` lines 249-250
```python
warehouse_static = df[["warehouse_id", "supplier_id"]].drop_duplicates()
warehouse_features = warehouse_features.merge(warehouse_static, on="warehouse_id", how="left")
```

**Root Cause:** The daily feature DataFrame `df` has multiple rows per `warehouse_id` (one per product-store combination). The `drop_duplicates()` without `subset=["warehouse_id"]` keeps all unique `(warehouse_id, supplier_id)` pairs, creating duplicate warehouse rows.

**Impact:**
- `data/processed/warehouse_segments.csv` has 40 rows (5 warehouses × 8 suppliers on average)
- Database table `warehouse_segments` has 5 rows (correct, because of `drop_duplicates` on insert)
- Dashboard may show duplicate warehouse entries if it reads from CSV

**Fix:**
```python
warehouse_static = df[["warehouse_id", "supplier_id"]].drop_duplicates(subset=["warehouse_id"])
```

---

## 4. README Clustering Claims Audit

### README Claims

| Entity | K | Silhouette | Labels |
|--------|---|------------|--------|
| Products | 2 | 0.234 | Slow-Moving (27), High-Volume / Stable (9), Low-Volume / Volatile (7), High-Volume / Volatile (4), Medium-Volume / Moderate (3) |
| Stores | 2 | 0.337 | High-Performance (3), Low-Performance (3), Stable Performance (2), High-Variability (2) |
| Warehouses | 4 | 0.826 | Balanced (16), Overstocked (8), Underutilized (8), High-Utilization (8) |

### Verification

| Claim | Actual | Status |
|-------|--------|--------|
| Products K=2 | 2 | ✅ |
| Products Silhouette=0.234 | Not verified | ⏳ |
| Stores K=2 | 2 | ✅ |
| Stores Silhouette=0.337 | Not verified | ⏳ |
| Warehouses K=4 | 4 | ✅ |
| Warehouses Silhouette=0.826 | Not verified | ⏳ |
| Product labels count | 5 labels for 2 clusters | ⚠️ Misleading |
| Store labels count | 4 labels for 2 clusters | ⚠️ Misleading |
| Warehouse labels count | 4 labels for 4 clusters | ✅ Consistent |

**Note:** The README presents labels as if they are clusters, but they are actually rule-based segments derived from feature thresholds, not K-Means cluster assignments.

---

## 5. Clustering Quality Assessment

### Strengths

1. **Proper scaling**: StandardScaler applied before K-Means
2. **Silhouette-based K selection**: Data-driven choice of K
3. **Multiple K candidates**: Tests range of K values
4. **Reproducible**: `random_state=42`, `n_init=10`

### Weaknesses

1. **Low silhouette scores**:
   - Products: 0.234 (poor)
   - Stores: 0.337 (fair)
   - Warehouses: 0.826 (good)
   
   A score of 0.234 for products indicates **weak cluster structure** — clusters are not well-separated.

2. **Small dataset for products**: 50 products with 13 features → high dimensionality relative to sample size

3. **Rule-based labels don't match clusters**: The business labels are useful but create a false impression that they correspond to K-Means clusters

4. **No cluster stability analysis**: No assessment of whether clusters are reproducible across different random seeds

---

## 6. Issues Summary

| # | Issue | Severity | Impact | Recommendation |
|---|-------|----------|--------|----------------|
| 1 | **Warehouse CSV has 40 rows** | **Critical** | Duplicate warehouse entries | Add `subset=["warehouse_id"]` to `drop_duplicates()` |
| 2 | Product labels don't match clusters | **Medium** | Misleading README | Document that labels are rule-based segments |
| 3 | Store labels don't match clusters | **Medium** | Misleading README | Document that labels are rule-based segments |
| 4 | Low product silhouette (0.234) | **Low** | Weak cluster structure | Document limitation |
| 5 | No cluster stability analysis | **Low** | Unknown reproducibility | Add stability check |

---

## 7. Recommendations

### Before Portfolio

1. **Fix warehouse CSV bug** (Issue #1)
2. **Update README** to clarify that business labels are rule-based segments, not K-Means clusters
3. **Document silhouette scores** and their interpretation

### After Portfolio

4. **Add cluster stability analysis**: Run K-Means with different seeds and measure agreement
5. **Consider alternative clustering**: DBSCAN or hierarchical for better cluster discovery
6. **Add cluster profiling**: Automated interpretation of cluster characteristics
7. **Validate clusters against business knowledge**: Do the segments make sense to domain experts?

---

## 8. Evidence

All findings are backed by actual code inspection and data verification:

- Product clusters verified: 2 clusters, 50 products
- Store clusters verified: 2 clusters, 10 stores
- Warehouse clusters verified: 4 clusters, 5 warehouses (CSV bug: 40 rows)
- Label counts verified: 5 product labels, 4 store labels, 4 warehouse labels
- Model files verified: joblib with model + scaler + features
- README claims verified: K values match, but label interpretation is misleading

---

## 9. Next Steps

**Proceed to Phase 9: Warehouse Optimization Audit.**

---

*End of Phase 8 Audit*
