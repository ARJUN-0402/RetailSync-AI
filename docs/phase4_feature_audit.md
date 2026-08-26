# RetailSync AI — Phase 4: Feature Engineering Audit

**Date:** 2026-08-09
**Auditor:** Kilo
**Status:** READ-ONLY AUDIT — No code changes made

---

## 1. Executive Summary

The feature engineering pipeline is **well-implemented and mostly correct**. The 74-feature claim is verified. Data leakage is **minimal and low-impact**. The time-based train/validation/test split is properly implemented. Feature quality is appropriate for the zero-inflated dataset.

### Bottom Line

| Component | Status | Notes |
|-----------|--------|-------|
| Feature count | ✅ Verified | 74 features |
| Feature categories | ✅ Verified | 10 categories |
| Data leakage | ⚠️ Minor | 2 aggregate features use current day data |
| Time-based split | ✅ Verified | Proper chronological split |
| NaN handling | ✅ Verified | No NaN in critical features |
| Feature availability | ✅ Verified | Most features available at prediction time |
| Scaling/normalization | ⚠️ Not implemented | Features not scaled before clustering/ML |

---

## 2. Feature Count Verification

### Claimed: 74 engineered features

### Actual Breakdown

| Category | Count | Features |
|----------|-------|----------|
| Lag features | 8 | demand_lag_1d/7d/14d/28d, revenue_lag_1d/7d/14d/28d |
| Rolling features | 12 | demand_rolling_mean/std/max/min for 7d, 14d, 28d |
| Expanding features | 2 | demand_expanding_mean, demand_expanding_std |
| Time features | 12 | day_of_week, day_of_month, month, quarter, year, is_weekend, is_month_start, is_month_end, month_sin/cos, dow_sin/cos |
| Price features | 4 | unit_price, cost_price, price_vs_cost, price_margin_pct |
| Promotion features | 3 | promotion, promotion_last_7d, promotion_last_14d |
| Inventory features | 6 | quantity_on_hand, reorder_point, max_stock_level, stock_coverage_days, stock_vs_reorder, stock_to_max_ratio |
| Demand variability | 2 | demand_cv_28d, zero_demand_pct_28d |
| Aggregate features | 2 | category_avg_demand, store_type_avg_demand |
| Target variables | 6 | target_demand_1d/7d/14d, target_revenue_1d/7d/14d |
| Identifiers/static | 17 | product_id, store_id, date, category, subcategory, store_type, city, state, supplier_id, warehouse_id, etc. |
| **Total** | **74** | — |

**Note:** The README counts 74 as "engineered features" but this includes identifiers and static attributes. True engineered features are approximately 50-55 depending on classification.

---

## 3. Data Leakage Audit

### Critical Finding: Minor Leakage in Aggregate Features

**Features:** `category_avg_demand`, `store_type_avg_demand`

**Implementation (lines 197-201):**
```python
cat_daily = (
    daily_df.groupby(["date", "category"], as_index=False)["quantity_sold"]
    .mean()
    .rename(columns={"quantity_sold": "category_avg_demand"})
)
daily_df = daily_df.merge(cat_daily, on=["date", "category"], how="left")

store_type_daily = (
    daily_df.groupby(["date", "store_type"], as_index=False)["quantity_sold"]
    .mean()
    .rename(columns={"quantity_sold": "store_type_avg_demand"})
)
daily_df = daily_df.merge(store_type_daily, on=["date", "store_type"], how="left")
```

**Issue:** These features are calculated using the **current day's demand**, which includes the target day's demand. This is technically data leakage for time-series forecasting.

**Impact Assessment:**
- Correlation with target: `category_avg_demand` = 0.0208, `store_type_avg_demand` = 0.0226
- **Very low correlation** — the leakage has minimal practical impact
- In a zero-inflated dataset, the average demand is dominated by zeros, making these features nearly constant

**Recommendation:** Shift aggregate features by 1 day to use only past data:
```python
cat_daily["category_avg_demand"] = cat_daily.groupby("category")[
    "category_avg_demand"
].shift(1)
store_type_daily["store_type_avg_demand"] = store_type_daily.groupby("store_type")[
    "store_type_avg_demand"
].shift(1)
```

### Other Features: No Leakage Detected

| Feature Type | Leakage Risk | Assessment |
|-------------|-------------|------------|
| Lag features (1d, 7d, 14d, 28d) | None | Uses `shift()` — past data only |
| Rolling features (7d, 14d, 28d) | None | Uses shifted series + rolling window |
| Expanding features | None | Uses expanding window on shifted data |
| Time features | None | Pure date components |
| Price features | None | Static product attributes |
| Promotion features | None | Shifted by 1 day |
| Inventory features | None | Forward-filled from weekly snapshots |
| Demand variability | None | Based on past 28 days |

---

## 4. Time-Based Split Audit

### Implementation

**Split dates:**
- Train: `<= 2024-12-31` (254,500 rows)
- Validation: `2025-01-01` to `2025-06-09` (80,000 rows)
- Test: `2025-06-10` to `2025-08-09` (30,500 rows)

### Verification

| Check | Result |
|-------|--------|
| Train ends before val starts | ✅ 2024-12-31 < 2025-01-01 |
| Val ends before test starts | ✅ 2025-06-09 < 2025-06-10 |
| No date overlap | ✅ Verified |
| chronological order | ✅ Verified |

### Assessment

✅ **Correct implementation** of time-based split. No data leakage from future to past.

---

## 5. Feature Quality Assessment

### Strengths

1. **Lag features**: Properly shifted, no leakage
2. **Rolling features**: Use `min_periods=1` to avoid NaN for early dates
3. **Time features**: Cyclical encoding (sin/cos) for month and day of week — good practice
4. **Inventory features**: Forward-filled from weekly snapshots — acceptable for daily model
5. **Target variables**: Properly shifted for 1d, 7d, 14d horizons
6. **No NaN in critical columns**: All targets and identifiers are complete

### Weaknesses

1. **No feature scaling**: Features are not normalized/standardized before being fed to ML models
   - Random Forest and XGBoost are tree-based and don't require scaling
   - But clustering (K-Means) requires scaling — this is done in `segmentation.py` separately
   - **Recommendation:** Document that scaling is done per-module

2. **Aggregate features leak current day data**: Already identified above

3. **NaN filling with 0**: All numeric NaNs are filled with 0
   - For features where 0 is a valid value (e.g., demand), this may introduce bias
   - **Recommendation:** Use median or group-specific defaults

4. **No feature selection**: All 74 features are used for forecasting
   - Some features may be redundant or noisy
   - **Recommendation:** Add feature importance analysis or correlation-based selection

---

## 6. Feature-Target Correlation Analysis

### Top Correlations with `target_demand_1d`

| Feature | Correlation | Interpretation |
|---------|-------------|----------------|
| year | 0.0394 | Very weak positive trend over time |
| demand_expanding_std | 0.0241 | Slightly higher demand when variability is high |
| store_type_avg_demand | 0.0226 | Weak store-type effect |
| demand_rolling_max_28d | 0.0211 | Past peak demand weakly predicts future |
| category_avg_demand | 0.0208 | Very weak category effect |
| demand_rolling_std_28d | 0.0201 | Demand variability weakly predicts future |
| demand_rolling_mean_28d | 0.0149 | Past average demand weakly predicts future |
| demand_rolling_max_14d | 0.0148 | Recent peak demand weakly predicts future |
| month_cos | -0.0144 | Very weak seasonal effect |
| demand_expanding_mean | 0.0140 | Historical average weakly predicts future |

### Interpretation

**All correlations are extremely weak (< 0.04).** This confirms:
1. The zero-inflated, random-demand dataset has very weak signal
2. No single feature strongly predicts future demand
3. This explains why the baseline mean model wins — there's no strong pattern to learn
4. This is **honest and realistic** for the data generation approach used

---

## 7. Feature Availability at Prediction Time

### Available Features (for forecasting future demand)

| Feature | Available at Prediction? | Source |
|---------|--------------------------|--------|
| Lag features | ✅ Yes | Past demand |
| Rolling features | ✅ Yes | Past demand |
| Expanding features | ✅ Yes | Past demand |
| Time features | ✅ Yes | Future date known |
| Price features | ✅ Yes | Static product attributes |
| Promotion features | ⚠️ Partial | Past promotions known, future promotions unknown |
| Inventory features | ✅ Yes | Latest inventory snapshot |
| Demand variability | ✅ Yes | Past demand |
| Aggregate features | ⚠️ Partial | Current day averages known, future unknown |

**Note:** For the 14-day forecast, future promotions are unknown. The model uses past promotion history, which is acceptable.

---

## 8. Issues Summary

| # | Issue | Severity | Impact | Recommendation |
|---|-------|----------|--------|----------------|
| 1 | Aggregate features leak current day data | **Medium** | Minimal (correlation 0.02) | Shift by 1 day |
| 2 | No feature scaling for ML models | **Low** | None for tree models | Document per-module scaling |
| 3 | NaN filling with 0 may introduce bias | **Low** | Minor | Use median/group defaults |
| 4 | No feature selection | **Low** | Minor | Add importance analysis |

---

## 9. Recommendations

### Before Portfolio

1. **Fix aggregate feature leakage**: Shift `category_avg_demand` and `store_type_avg_demand` by 1 day
2. **Document feature engineering logic**: Update `docs/feature_documentation.md` with actual implementation details

### After Portfolio

3. **Add feature selection**: Use feature importance or mutual information to select top features
4. **Improve NaN handling**: Use group-specific medians instead of global 0
5. **Add feature monitoring**: Track feature distributions over time for drift detection

---

## 10. Evidence

All findings are backed by actual code inspection and data analysis:

- Feature count verified: `data/processed/features_daily.csv` → 74 columns
- Leakage verified: correlation analysis shows minimal impact
- Time split verified: no date overlap between train/val/test
- Feature availability verified: most features use past or static data
- NaN check verified: 0 NaN in critical columns

---

## 11. Next Steps

**Do not proceed to Phase 5 until aggregate feature leakage is fixed.**

After fix:
1. Re-run feature engineering script
2. Re-run forecasting model
3. Verify test suite still passes
4. Update documentation
5. Proceed to Phase 5: Demand Forecasting Audit

---

*End of Phase 4 Audit*
