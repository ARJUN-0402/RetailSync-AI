# Inventory Intelligence Methodology

## Overview

This document describes the inventory risk detection and intelligence pipeline for RetailSync AI.

**Output:** `data/processed/inventory_intelligence.csv`
**Alerts Table:** `inventory_alerts` in SQLite database
**Snapshot Date:** 2025-08-08 (latest available)

---

## Risk Categories

### 1. Stockout Risk

**Purpose:** Identify products at risk of running out of stock.

**Risk Levels:**
- **HIGH:** `quantity_on_hand <= 0` OR `quantity_on_hand <= reorder_point * 0.5`
- **MEDIUM:** `quantity_on_hand <= reorder_point` (but > 0)
- **LOW:** `quantity_on_hand > reorder_point`

**Adjustment:** Risk is escalated if forecasted 7-day demand exceeds current stock.

**Business Logic:** Stockouts lead to lost sales and customer dissatisfaction. The reorder point is the supplier's recommended trigger for restocking.

---

### 2. Overstock Risk

**Purpose:** Identify products with excess inventory that ties up capital and storage space.

**Risk Levels:**
- **HIGH:** `quantity_on_hand > max_stock_level * 1.5`
- **MEDIUM:** `quantity_on_hand > max_stock_level`
- **LOW:** `quantity_on_hand <= max_stock_level`

**Adjustment:** Risk is escalated for products with high demand variability (CV > 3.0).

**Business Logic:** Overstock increases holding costs and risk of obsolescence. The max_stock_level represents the optimal inventory ceiling.

---

### 3. Dead Stock Detection

**Purpose:** Identify inventory that is unlikely to sell.

**Criteria:**
- `quantity_on_hand > max_stock_level * 0.8` (high inventory)
- AND (`forecast_demand_14d == 0` OR `demand_cv_28d == 0`) (no recent demand)
- AND `stock_coverage_days > 90` (excess coverage)

**Additional Check:** Products with `quantity_on_hand > 0` but zero sales in the last 28 days.

**Business Logic:** Dead stock represents tied-up capital with no revenue potential. Early identification allows for clearance sales or returns.

---

### 4. Reorder Urgency

**Purpose:** Prioritize which products need immediate restocking.

**Urgency Levels:**
- **URGENT:** `quantity_on_hand <= reorder_point`
- **SOON:** `quantity_on_hand > reorder_point` AND `stock_coverage_days <= 7`
- **MONITOR:** `quantity_on_hand > reorder_point` AND `stock_coverage_days <= 14`
- **NONE:** `stock_coverage_days > 14`

**Business Logic:** Urgency is based on how quickly current stock will be depleted given forecasted demand.

---

### 5. Composite Risk Score

**Purpose:** Combine multiple risk dimensions into a single actionable score.

**Formula:**
```
composite_risk_score = (
    stockout_score * 0.35 +
    overstock_score * 0.25 +
    dead_stock_score * 0.20 +
    reorder_score * 0.20
)
```

**Risk Levels:**
- **LOW:** 0-25
- **MEDIUM:** 26-50
- **HIGH:** 51-75
- **CRITICAL:** 76-100

**Weights:** Stockout risk is weighted highest (35%) because stockouts directly impact revenue and customer satisfaction.

---

## Results Summary

| Metric | Count | Percentage |
|---|---|---|
| Total Product-Store Combinations | 500 | 100% |
| Stockout Risk (HIGH) | 16 | 3.2% |
| Stockout Risk (MEDIUM) | 169 | 33.8% |
| Overstock Risk (HIGH) | 110 | 22.0% |
| Overstock Risk (MEDIUM) | 305 | 61.0% |
| Dead Stock | 0 | 0.0% |
| Urgent Reorder | 185 | 37.0% |
| Soon Reorder | 0 | 0.0% |
| Critical Composite Risk | 3 | 0.6% |
| High Composite Risk | 158 | 31.6% |

**Key Findings:**
- 37% of product-store combinations require urgent restocking
- 83% of combinations face some level of overstock risk
- 36.6% face stockout risk (HIGH or MEDIUM)
- Only 17% have LOW composite risk

---

## Recommendations

| Action | Count | Description |
|---|---|---|
| Place emergency order | 185 | Below reorder point, immediate action needed |
| Review upcoming demand | 185 | Overstock risk, consider reducing orders |
| Run promotion or reduce orders | 70 | High overstock risk, clearance recommended |
| Monitor | 60 | Low risk, routine monitoring sufficient |

---

## Limitations

1. **Static reorder points:** The current reorder points are fixed. A dynamic reorder point based on lead time and demand variability would be more robust.

2. **Weekly inventory snapshots:** Daily inventory interpolation introduces uncertainty.

3. **No supplier reliability integration:** Lead times and reliability scores are available but not yet integrated into reorder calculations.

4. **Binary dead stock criteria:** The current criteria are conservative. A probabilistic approach would better identify slow-moving items.

5. **No markdown optimization:** Clearance pricing strategies are not modeled.

---

## Reproducibility

```bash
# Run inventory intelligence
python src/inventory/inventory_intelligence.py

# Load alerts into database
python src/inventory/load_alerts.py
```

**Outputs:**
- `data/processed/inventory_intelligence.csv` — Full analysis for all 500 product-store combinations
- `data/processed/inventory_intelligence_summary.csv` — Summary statistics
- `database/retailsync.db` — `inventory_alerts` table with 785 alerts

---

*Generated by `src/inventory/inventory_intelligence.py` and `src/inventory/load_alerts.py`*
