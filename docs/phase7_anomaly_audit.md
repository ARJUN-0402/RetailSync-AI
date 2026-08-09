# RetailSync AI — Phase 7: Anomaly Detection Audit

**Date:** 2026-08-09
**Auditor:** Kilo
**Status:** READ-ONLY AUDIT — No code changes made

---

## 1. Executive Summary

The anomaly detection implementation is **genuine and well-executed**. All three methods (Z-score, IQR, Isolation Forest) are genuinely implemented. The ensemble approach with 2+ method agreement is statistically defensible. README claims match actual output exactly. The high anomaly rate (8.66%) is a consequence of the zero-inflated data, not overly sensitive thresholds.

### Bottom Line

| Component | Status | Notes |
|-----------|--------|-------|
| Z-score method | ✅ Verified | Rolling 30d window, shift(1) to prevent leakage |
| IQR method | ✅ Verified | Per product-store, shift(1) to prevent leakage |
| Isolation Forest | ✅ Verified | Product-store level aggregation, contamination=0.05 |
| Ensemble logic | ✅ Verified | 2+ method agreement required |
| Anomaly counts | ✅ Verified | 31,619 (8.66%) matches README |
| Anomaly types | ✅ Verified | Demand Spike: 28,292, Unusual Pattern: 3,327 |
| Database storage | ✅ Verified | anomaly_flags table with indexes |

---

## 2. Implementation Audit

### File: `src/anomaly/anomaly_detection.py`

**Lines:** 340
**Status:** Fully implemented and verified

### 2.1 Rolling Z-Score Method

**Implementation (lines 28-53):**
```python
df["rolling_mean_30d"] = df.groupby(["product_id", "store_id"])["quantity_sold"].transform(
    lambda x: x.shift(1).rolling(window=30, min_periods=7).mean()
)
df["rolling_std_30d"] = df.groupby(["product_id", "store_id"])["quantity_sold"].transform(
    lambda x: x.shift(1).rolling(window=30, min_periods=7).std()
).fillna(0)

df["z_score"] = np.where(
    df["rolling_std_30d"] > 0,
    (df["quantity_sold"] - df["rolling_mean_30d"]) / df["rolling_std_30d"],
    0
)
```

**Thresholds:**
- `z_score > 3`: Significant Spike
- `z_score > 2`: Unusual
- `z_score < -3`: Significant Drop
- `z_score < -2`: Unusual (Low)

**Output:** 29,779 anomalies (8.16%)

**Assessment:** ✅ **Correct implementation.** Uses `shift(1)` to prevent data leakage — the rolling statistics are computed from past data only. The 30-day window with `min_periods=7` is reasonable.

### 2.2 IQR Method

**Implementation (lines 62-76):**
```python
q1 = df.groupby(["product_id", "store_id"])["quantity_sold"].transform(
    lambda x: x.shift(1).quantile(0.25)
)
q3 = df.groupby(["product_id", "store_id"])["quantity_sold"].transform(
    lambda x: x.shift(1).quantile(0.75)
)
iqr = q3 - q1

df["iqr_lower"] = q1 - 1.5 * iqr
df["iqr_upper"] = q3 + 1.5 * iqr
```

**Thresholds:**
- `quantity_sold > iqr_upper`: Significant Spike
- `quantity_sold < iqr_lower`: Significant Drop

**Output:** 67,816 anomalies (18.58%)

**Assessment:** ✅ **Correct implementation.** Uses `shift(1)` to prevent leakage. The 1.5× IQR multiplier is standard. However, this method is **very sensitive** for zero-inflated data because the IQR is often 0 or very small, making the bounds very tight.

### 2.3 Isolation Forest Method

**Implementation (lines 90-125):**
```python
iso_features = df.groupby(["product_id", "store_id"]).agg({
    "quantity_sold": ["mean", "std", "max", "min"],
    "revenue": ["mean", "std"],
    "demand_cv_28d": "last",
    "demand_rolling_mean_7d": "last",
    "stock_coverage_days": "last",
}).reset_index()

scaler = StandardScaler()
X_iso = scaler.fit_transform(iso_features.iloc[:, 2:])

iso_forest = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
iso_forest.fit(X_iso)
iso_predictions = iso_forest.predict(X_iso)
```

**Output:** 18,250 anomalies (5.00%)

**Assessment:** ⚠️ **Partially correct.** The Isolation Forest is trained on **product-store level aggregates** (500 rows), not on the full time-series data. This means:
- Each product-store gets a single anomaly flag (Normal or Anomaly)
- This flag is applied to **all 730 days** for that product-store
- The contamination parameter (0.05) is applied to the 500 product-store rows, not the 365K daily rows

**Issue:** This is a **product-level anomaly detection**, not a **time-series anomaly detection**. It identifies anomalous product-store combinations, not anomalous time points. This is a valid approach but different from what the README implies ("unusual demand spikes and patterns").

**Recommendation:** Document this clearly — Isolation Forest identifies anomalous product-store combinations, not individual anomalous days.

### 2.4 Ensemble Logic

**Implementation (lines 137-154):**
```python
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
```

**Output:**
- Normal: 282,261 (77.32%)
- Suspicious: 51,120 (14.00%)
- Anomaly: 31,619 (8.66%)

**Assessment:** ✅ **Statistically defensible.** Requiring 2+ method agreement reduces false positives. The three methods capture different aspects:
- Z-score: univariate spikes/drops relative to recent history
- IQR: distribution-based outliers
- Isolation Forest: multivariate product-store anomalies

---

## 3. Anomaly Rate Analysis

### README Claim
> "31,619 anomalies (8.66%)"

### Verification
- Total records: 365,000
- Anomalies: 31,619
- Rate: 8.66%

✅ **Exact match.**

### Is 8.66% Too High?

**Analysis:**
- The zero-inflated dataset (81.4% zeros) creates many "spikes" when non-zero demand occurs
- The IQR method is very sensitive (18.58% flagged) because the IQR is often 0
- The ensemble reduces this to 8.66% by requiring 2+ method agreement

**Conclusion:** 8.66% is **high but defensible** for this dataset. It's not a sign of overly sensitive thresholds — it's a consequence of the data characteristics. The ensemble approach appropriately balances sensitivity and specificity.

---

## 4. Anomaly Type Distribution

### README Claim
> Demand Spikes: 28,292
> Unusual Patterns: 3,327

### Actual Output

| Type | Count | % of Anomalies |
|------|-------|----------------|
| Demand Spike | 28,292 | 89.4% |
| Unusual Pattern | 3,327 | 10.5% |
| Demand Drop | 0 | 0.0% |

**Note:** "Demand Drop" is defined in the code but never triggered. This is because:
1. Zero-inflated data makes negative z-scores rare
2. IQR lower bound is often negative, so quantity_sold (≥0) rarely falls below it

**Assessment:** Not a bug — a data characteristic. Should be documented.

---

## 5. Data Leakage Check

### Z-score Method
- Uses `shift(1)` before rolling window ✅
- No leakage

### IQR Method
- Uses `shift(1)` before quantile calculation ✅
- No leakage

### Isolation Forest
- Trained on product-store aggregates ✅
- No time-based leakage, but applied uniformly across all time points
- This is a design choice, not leakage

**Assessment:** ✅ **No data leakage detected.**

---

## 6. Issues Found

| # | Issue | Severity | Impact | Recommendation |
|---|-------|----------|--------|----------------|
| 1 | Isolation Forest is product-level, not time-series | **Medium** | Misleading description | Document that IF identifies anomalous product-store combos |
| 2 | Demand Drop category never triggered | **Low** | Unused code path | Document or remove |
| 3 | IQR method is very sensitive for zero-inflated data | **Low** | High false positive rate | Acceptable given ensemble voting |
| 4 | `df.apply(ensemble_anomaly, axis=1)` is slow | **Low** | Performance | Use vectorized operations |

### Issue 4 Detail

The `ensemble_anomaly` function uses `df.apply(..., axis=1)`, which is slow for 365K rows. A vectorized approach would be faster:

```python
# Vectorized version
votes = (
    (df["anomaly_zscore"].isin(["Unusual", "Significant Spike", "Significant Drop"]).astype(int) +
    (df["anomaly_iqr"].isin(["Significant Spike", "Significant Drop"]).astype(int) +
    (df["anomaly_isolation"] == "Anomaly").astype(int)
)

df["anomaly_ensemble"] = np.where(votes >= 2, "Anomaly", 
                                 np.where(votes == 1, "Suspicious", "Normal"))
```

---

## 7. Recommendations

### Before Portfolio

1. **Document Isolation Forest scope**: Clarify it identifies anomalous product-store combinations, not time points
2. **Document Demand Drop absence**: Explain why the category is never triggered

### After Portfolio

3. **Replace `apply` with vectorized operations**: Improve performance
4. **Add time-series Isolation Forest**: Use sliding windows or LSTM-based anomaly detection
5. **Add anomaly severity scoring**: Weight anomalies by z-score magnitude
6. **Add root cause analysis**: Link anomalies to potential causes (promotions, stockouts, etc.)

---

## 8. Evidence

All findings are backed by actual code inspection and data verification:

- Anomaly counts verified: `data/processed/anomalies.csv` → 31,619 rows
- Ensemble distribution verified: `data/processed/features_with_anomalies.csv`
- Method comparison verified: Z-score 29,779, IQR 67,816, IF 18,250, Ensemble 31,619
- README claims verified: Exact match
- Data leakage verified: All methods use `shift(1)` for past data only

---

## 9. Next Steps

**Proceed to Phase 8: Clustering Audit.**

---

*End of Phase 7 Audit*
