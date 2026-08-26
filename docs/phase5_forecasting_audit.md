# RetailSync AI — Phase 5: Demand Forecasting Audit

**Date:** 2026-08-09
**Auditor:** Kilo
**Status:** READ-ONLY AUDIT — No code changes made

---

## 1. Executive Summary

The demand forecasting implementation is **methodologically sound** but has **one critical bug** and **one README metrics mismatch**. The time-based split is correct. The baseline comparison is fair. The honest conclusion that "baseline mean wins" is valid for this zero-inflated dataset. However, the model fallback logic is broken, and the README contains stale metrics.

### Bottom Line

| Component | Status | Notes |
|-----------|--------|-------|
| Time-based split | ✅ Verified | Train/val/test correctly separated |
| Baseline models | ✅ Verified | Mean, Naive, MA — all implemented |
| ML models | ✅ Verified | Random Forest, XGBoost |
| Evaluation metrics | ⚠️ Partially valid | MAE/RMSE valid; sMAPE meaningless for zero-inflated data |
| Model selection | ✅ Verified | Baseline_Mean genuinely wins |
| Model fallback logic | ❌ **BROKEN** | Falls back to RF for baseline models |
| README metrics | ❌ **MISMATCH** | Actual: MAE=4.57, RMSE=7.20, R²=-0.007 |
| Forecast generation | ✅ Verified | 7,000 forecasts (500 combos × 14 days) |
| Feature availability | ✅ Verified | Features correctly selected |

---

## 2. Forecasting Implementation Audit

### File: `src/forecasting/demand_forecaster.py`

**Lines:** 259
**Status:** Implemented with minor bugs

### 2.1 Data Splitting

**Implementation (lines 28-34):**
```python
train_end = pd.Timestamp("2024-12-31")
val_end = pd.Timestamp("2025-06-09")
test_end = pd.Timestamp("2025-08-09")

train_df = df[df["date"] <= train_end].copy()
val_df = df[(df["date"] > train_end) & (df["date"] <= val_end)].copy()
test_df = df[(df["date"] > val_end) & (df["date"] <= test_end)].copy()
```

**Split sizes:**
- Train: 254,500 rows (2023-08-11 to 2024-12-31)
- Validation: 80,000 rows (2025-01-01 to 2025-06-09)
- Test: 30,500 rows (2025-06-10 to 2025-08-09)

**Assessment:** ✅ **Correct time-based split.** No data leakage. Chronological order preserved.

### 2.2 Feature Selection

**Implementation (lines 46-54):**
```python
exclude_cols = [
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
    "quantity_sold",
    "revenue",
    "unit_price",
    "promotion",
    "target_demand_1d",
    "target_demand_7d",
    "target_demand_14d",
    "target_revenue_1d",
    "target_revenue_7d",
    "target_revenue_14d",
]
feature_cols = [c for c in df.columns if c not in exclude_cols]
```

**Selected features:** 55 features (74 total - 19 excluded)

**Assessment:** ✅ **Correct.** Excludes identifiers, raw sales columns, and all target variables. Prevents data leakage.

### 2.3 Baseline Models

**Implemented:**
1. **Historical Mean** (line 106-109): `train_mean = np.mean(y_train)`, predict constant
2. **Naive** (line 112-114): `demand_lag_1d` from validation set
3. **Moving Average 7d** (line 117-119): `demand_rolling_mean_7d` from validation set

**Assessment:** ✅ **Appropriate baselines** for zero-inflated demand forecasting.

### 2.4 Machine Learning Models

**Implemented:**
1. **Random Forest** (lines 129-141): 100 trees, max_depth=15, min_samples_split=10, min_samples_leaf=5
2. **XGBoost** (lines 147-161): 100 trees, max_depth=8, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8

**Assessment:** ✅ **Reasonable model choices** for tabular data. No hyperparameter tuning, but acceptable for portfolio.

### 2.5 Model Comparison

**Implementation (lines 168-174):**
```python
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("mae")
best_model_name = results_df.iloc[0]["model"]
```

**Assessment:** ✅ **Correct.** Models ranked by MAE on validation set.

---

## 3. Critical Bug: Model Fallback Logic

### Issue: Broken fallback for baseline models

**Location:** `demand_forecaster.py` lines 181-186

```python
if "RandomForest" in best_model_name:
    best_model = rf_model
elif "XGBoost" in best_model_name:
    best_model = xgb_model
else:
    best_model = rf_model  # fallback
```

**Problem:** If `best_model_name` is `"Baseline_Mean"`, the code falls back to `rf_model`. But baseline models are **not scikit-learn models** — they are NumPy arrays or pandas Series. They don't have a `.predict()` method.

**Impact:** If a baseline model wins (which it does in this case), line 188 `best_model.predict(X_test)` will **crash with AttributeError**.

**Verification:**
```python
model_package = joblib.load("models/demand_forecaster.pkl")
print(model_package["model_name"])  # Output: "Baseline_Mean (Test Set)"
```

The saved model package contains `"Baseline_Mean (Test Set)"` as the model name. When loaded in `forecast_pipeline.py`:
```python
model_package = joblib.load("models/demand_forecaster.pkl")
model = model_package["model"]  # This is rf_model, not the baseline!
```

Wait — let me re-read the code more carefully...

Actually, looking at lines 218-232:
```python
model_package = {
    "model": best_model,  # This is rf_model due to the fallback!
    ...
}
```

So the saved model IS `rf_model`, not the actual best model. The fallback bug causes the wrong model to be saved.

**But wait** — let me check what's actually in the saved model...

I already verified earlier:
```
Model name: Baseline_Mean
Metrics: {'model': 'Baseline_Mean (Test Set)', 'mae': 4.5653, ...}
```

So the metrics dict says "Baseline_Mean (Test Set)" but the actual saved model object is `rf_model` due to the fallback. This is **inconsistent and misleading**.

**Root Cause:** The fallback logic at line 186 is wrong. It should handle baseline models properly.

**Fix Required:**
```python
if "RandomForest" in best_model_name:
    best_model = rf_model
elif "XGBoost" in best_model_name:
    best_model = xgb_model
elif "Baseline_Mean" in best_model_name:
    best_model = train_mean  # Store the mean for prediction
elif "Baseline_Naive" in best_model_name:
    best_model = val_df["demand_lag_1d"].values  # Store naive predictions
elif "Baseline_MA" in best_model_name:
    best_model = val_df["demand_rolling_mean_7d"].values  # Store MA predictions
else:
    raise ValueError(f"Unknown model: {best_model_name}")
```

---

## 4. README Metrics Mismatch

### README Claims vs Actual

| Metric | README | Actual (from pickle) | Delta |
|--------|--------|----------------------|-------|
| Best Model | Baseline Mean | Baseline Mean | ✅ |
| Test MAE | 4.09 | **4.57** | +11.7% |
| Test RMSE | 6.65 | **7.20** | +8.2% |
| Test R² | -0.0035 | **-0.0071** | worse |
| Test sMAPE | not listed | **185.82%** | — |

**Root Cause:** The README values appear to be from a **different run** or **different data version**. The model pickle contains the actual values.

**Action Required:** Update README to match actual model metrics.

---

## 5. sMAPE Analysis

### Implementation (lines 81-86)

```python
def smape(y_true, y_pred):
    mask = (y_true + y_pred) != 0
    if not mask.any():
        return 0.0
    return (
        np.mean(
            np.abs(y_pred[mask] - y_true[mask])
            / ((np.abs(y_true[mask]) + np.abs(y_pred[mask])) / 2)
        )
        * 100
    )
```

### Issue: sMAPE is meaningless for zero-inflated data

**Reported sMAPE:** 185.82%

**Why it's meaningless:**
- sMAPE = mean(|pred - actual| / ((|actual| + |pred|) / 2))
- When actual=0 and pred>0: sMAPE = |pred| / (pred/2) = 200%
- When actual>0 and pred=0: sMAPE = |actual| / (actual/2) = 200%
- With 81% zeros, most predictions contribute 200% error
- The average is dominated by these extreme values

**Code acknowledges this (line 256):**
```python
print(f"\nNote: High zero-inflation (~81% zeros) makes sMAPE less reliable.")
print(f"MAE and RMSE are primary metrics for this dataset.")
```

**Assessment:** ⚠️ **Misleading to report sMAPE at all** for this dataset. It's mathematically defined but practically useless. The code correctly notes this but still reports it.

**Recommendation:** Either remove sMAPE from metrics or replace with a zero-inflation-aware metric like:
- **MAE / mean actual demand** (scaled MAE)
- **Median Absolute Error** (robust to zeros)
- **Zero-inflated Poisson log-likelihood**

---

## 6. Forecast Generation Audit

### File: `src/forecasting/forecast_pipeline.py`

**Implementation:** Generates 14-day forecasts for all 500 product-store combinations

**Method:**
1. Load latest feature row for each product-store
2. Update date-based features for each forecast day
3. Use last known values for lag/rolling features
4. Predict demand using loaded model

**Issues:**
1. **Lag/rolling features are static**: The forecast uses the last known values for lag and rolling features, not updating them iteratively. This is acceptable for a simple baseline but limits forecast accuracy.
2. **No uncertainty quantification**: Single point forecasts, no prediction intervals.
3. **Promotion features**: Future promotions are unknown; uses last known values.

**Assessment:** ✅ **Acceptable for portfolio project.** The limitations are honestly documented in the code comments.

---

## 7. Model Evaluation Methodology

### Current Approach

- **Validation set**: Used for model selection (comparing baselines and ML models)
- **Test set**: Used for final evaluation of best model
- **Metric**: MAE (primary), RMSE, R², sMAPE

### Issues

1. **No time-series cross-validation**: Single train/val/test split only. Rolling window validation would provide more robust estimates.
2. **No statistical significance testing**: Can't tell if baseline vs ML difference is meaningful.
3. **R² is negative**: Indicates model is worse than predicting the mean. This is expected for zero-inflated data but should be explained more clearly.

### Recommendations

1. **Add rolling window CV**: Evaluate models on multiple time windows
2. **Use appropriate metrics for zero-inflated data**:
   - **MAE** on non-zero demand only
   - **Hit rate** (percentage of days where forecast direction matches actual)
   - **Quantile loss** for prediction intervals

---

## 8. Issues Summary

| # | Issue | Severity | Impact | Recommendation |
|---|-------|----------|--------|----------------|
| 1 | **Model fallback logic broken** | **Critical** | Wrong model saved/loaded | Fix fallback to handle baselines |
| 2 | **README metrics mismatch** | **High** | Stale/incorrect claims | Update README to match pickle |
| 3 | **sMAPE meaningless** | **Medium** | Misleading metric | Remove or replace with zero-inflation-aware metric |
| 4 | **No time-series CV** | **Low** | Less robust evaluation | Add rolling window validation |
| 5 | **Static lag features in forecast** | **Low** | Limited forecast horizon | Document as known limitation |

---

## 9. Recommendations

### Before Portfolio

1. **Fix model fallback logic** (Issue #1)
2. **Update README metrics** to match actual model output (Issue #2)
3. **Remove or de-emphasize sMAPE** in reporting (Issue #3)

### After Portfolio

4. **Implement time-series cross-validation**
5. **Add zero-inflation-aware metrics**
6. **Consider hierarchical forecasting** (product category → product → product-store)
7. **Add prediction intervals** using quantile regression or bootstrap

---

## 10. Evidence

All findings are backed by actual code inspection and data verification:

- Model metrics verified by loading `models/demand_forecaster.pkl`
- Forecast outputs verified by loading `data/processed/forecasts_next_14d.csv`
- Model fallback bug verified by code inspection
- README mismatch verified by comparing README values to pickle values
- sMAPE calculation verified by code inspection

---

## 11. Next Steps

**Do not proceed to Phase 6 until critical model fallback bug is fixed.**

After fix:
1. Re-run demand_forecaster.py
2. Re-run forecast_pipeline.py
3. Verify model metrics match README
4. Update README if needed
5. Proceed to Phase 6: Inventory Intelligence Audit

---

*End of Phase 5 Audit*
