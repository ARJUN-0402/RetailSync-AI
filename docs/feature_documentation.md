# Feature Engineering Documentation

## Overview

This document describes the feature engineering pipeline for RetailSync AI demand forecasting and inventory intelligence.

**Output:** `data/processed/features_daily.csv`
**Rows:** 365,000 (730 days × 50 products × 10 stores)
**Features:** 74 columns

---

## Feature Categories

### 1. Identifiers (3 features)
- `product_id` — Unique product identifier
- `store_id` — Unique store identifier
- `date` — Date of observation

### 2. Static Attributes (7 features)
- `category` — Product category (Electronics, Clothing, Groceries, Home Goods, Beauty, Toys)
- `subcategory` — Product subcategory (A, B, C)
- `store_type` — Store type (Urban, Suburban, Rural)
- `city` — Store city
- `state` — Store state
- `supplier_id` — Supplier identifier
- `warehouse_id` — Warehouse identifier

### 3. Product Attributes (4 features)
- `unit_price` — Retail selling price (USD)
- `cost_price` — Supplier cost price (USD)
- `weight_kg` — Product weight in kg
- `volume_m3` — Product volume in cubic meters

### 4. Supplier Attributes (2 features)
- `lead_time_days` — Average supplier lead time
- `reliability_score` — Supplier reliability (0.7-0.99)

### 5. Lag Features (8 features)
Historical demand and revenue values from previous days:
- `demand_lag_1d`, `demand_lag_7d`, `demand_lag_14d`, `demand_lag_28d`
- `revenue_lag_1d`, `revenue_lag_7d`, `revenue_lag_14d`, `revenue_lag_28d`

**Purpose:** Capture autocorrelation and recent demand patterns.

### 6. Rolling Features (12 features)
Time-window statistics computed over past windows (excluding current day):
- `demand_rolling_mean_7d`, `demand_rolling_mean_14d`, `demand_rolling_mean_28d`
- `demand_rolling_std_7d`, `demand_rolling_std_14d`, `demand_rolling_std_28d`
- `demand_rolling_max_7d`, `demand_rolling_max_14d`, `demand_rolling_max_28d`
- `demand_rolling_min_7d`, `demand_rolling_min_14d`, `demand_rolling_min_28d`

**Purpose:** Capture recent demand trends, volatility, and range.

### 7. Expanding Features (2 features)
Cumulative statistics from the beginning of the time series:
- `demand_expanding_mean` — Cumulative average demand
- `demand_expanding_std` — Cumulative standard deviation of demand

**Purpose:** Capture long-term demand level and variability.

### 8. Time Features (12 features)
Calendar and cyclical features:
- `day_of_week` — 0=Monday, 6=Sunday
- `day_of_month` — Day of month (1-31)
- `month` — Month (1-12)
- `quarter` — Quarter (1-4)
- `year` — Year
- `is_weekend` — Binary indicator for weekend
- `is_month_start` — Binary indicator for first 5 days of month
- `is_month_end` — Binary indicator for last 5 days of month
- `month_sin`, `month_cos` — Cyclical encoding of month
- `dow_sin`, `dow_cos` — Cyclical encoding of day of week

**Purpose:** Capture seasonal, weekly, and monthly patterns.

### 9. Price Features (2 features)
- `price_vs_cost` — Absolute margin (unit_price - cost_price)
- `price_margin_pct` — Percentage margin

**Purpose:** Capture pricing context and profitability.

### 10. Promotion Features (3 features)
- `promotion` — Current day promotion flag (0/1)
- `promotion_last_7d` — Number of promotion days in last 7 days
- `promotion_last_14d` — Number of promotion days in last 14 days

**Purpose:** Capture promotional impact and promotion history.

### 11. Inventory Features (7 features)
- `quantity_on_hand` — Current inventory level
- `reorder_point` — Reorder threshold
- `max_stock_level` — Maximum stock level
- `stock_coverage_days` — Estimated days of stock remaining (based on 7d rolling mean demand)
- `stock_vs_reorder` — Difference between current stock and reorder point
- `stock_vs_max` — Difference between current stock and max stock level
- `stock_to_max_ratio` — Ratio of current stock to max stock level

**Purpose:** Capture inventory position and risk indicators.

### 12. Demand Variability Features (2 features)
- `demand_cv_28d` — Coefficient of variation (std/mean) over last 28 days
- `zero_demand_pct_28d` — Proportion of zero-demand days in last 28 days

**Purpose:** Identify products with stable vs volatile demand patterns.

### 13. Aggregate Features (2 features)
- `category_avg_demand` — Average daily demand for the product's category on that date
- `store_type_avg_demand` — Average daily demand for the store's type on that date

**Purpose:** Capture category and store-type level demand patterns.

### 14. Target Variables (6 features)
Future values to predict:
- `target_demand_1d`, `target_revenue_1d` — Next day demand and revenue
- `target_demand_7d`, `target_revenue_7d` — Next 7 days demand and revenue
- `target_demand_14d`, `target_revenue_14d` — Next 14 days demand and revenue

**Purpose:** Multi-horizon forecasting targets.

---

## Data Leakage Prevention

All features are constructed using **only past or current information**:

1. **Lag features** use `.shift(lag)` to access values from `lag` days ago
2. **Rolling features** use `.shift(1).rolling(window)` to exclude the current day
3. **Expanding features** use `.shift(1).expanding()` to exclude the current day
4. **Target variables** use `.shift(-horizon)` to access future values
5. **Aggregate features** are computed using only current and past data

**Validation:** `tests/validate_features.py` confirms:
- Lag feature accuracy: 99.8%
- Rolling feature accuracy: 99.8%
- Target variable accuracy: 99.9%
- No suspicious future correlations detected

---

## Feature Selection Guidance

For demand forecasting models, recommended feature subsets:

### Baseline Model
- Lag features (1d, 7d)
- Time features (day_of_week, month)
- Static attributes (category, store_type)

### Intermediate Model
- All lag features
- Rolling features (7d, 14d)
- Time features with cyclical encoding
- Price and promotion features

### Advanced Model
- All features except expanding features (which may cause overfitting)
- Demand variability features
- Inventory features
- Aggregate features

---

## Known Limitations

1. **Zero-inflation:** ~81% of target values are zero, reflecting that many product-store combinations have no sales on many days. This is realistic for retail but requires appropriate model choices (e.g., zero-inflated models, hurdle models).

2. **Missing inventory data:** Inventory is weekly, so daily inventory features are forward-filled. This introduces some uncertainty in daily inventory estimates.

3. **Static attributes:** Product and store characteristics are assumed constant over time. In reality, products may be discontinued or stores may be renovated.

4. **No external features:** Weather, holidays, local events, and competitor activity are not included. These could significantly improve forecast accuracy.

---

*Generated by `src/features/feature_engineering.py`*
*Validated by `tests/validate_features.py`*
