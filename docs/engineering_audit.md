# RetailSync AI — Phase 1: Repository & Architecture Audit

**Date:** 2026-08-09  
**Auditor:** Kilo (Senior Data Scientist / ML Engineer / Data Engineer / Software Engineer)  
**Status:** READ-ONLY AUDIT — No code changes made in this phase  
**Working directory:** `C:\ARRU CODES\projects\RetailSync AI`  
**Python:** 3.13.14 | pandas 3.0.0 | numpy 2.4.1 | scikit-learn 1.9.0 | xgboost 3.3.0 | streamlit 1.58.0 | sqlalchemy 2.0.51  

---

## 1. Executive Summary

RetailSync AI is a **genuinely implemented** end-to-end retail supply-chain analytics platform. The repository contains working code across all claimed domains: synthetic data generation, data ingestion, SQLite storage, feature engineering, demand forecasting, inventory intelligence, anomaly detection, clustering/segmentation, warehouse analytics, a Streamlit dashboard, and a test suite.

**Key findings at a glance:**

| # | Finding | Severity | Status |
|---|---------|----------|--------|
| 1 | README forecasting metrics (MAE=4.09, RMSE=6.65, R²=-0.0035) don't match actual model pickle (MAE=4.57, RMSE=7.20, R²=-0.0071) | **High** | Needs fix |
| 2 | Model pickle contains RandomForestRegressor but model_name says "Baseline_Mean"; the fallback logic at demand_forecaster.py:181-186 silently saves RF while labeling it Baseline_Mean | **Critical** | Needs fix |
| 3 | `category_avg_demand` and `store_type_avg_demand` include the current day's demand (mild data leakage) | **Medium** | Needs fix |
| 4 | Warehouse segments CSV has 40 rows (5 unique warehouses × 8 supplier duplicates) instead of 5 | **Critical** | Needs fix |
| 5 | `pytest tests/` does NOT work — all 9 tests error with `fixture 'results' not found`; only `python tests/test_pipeline.py` works | **High** | Needs fix |
| 6 | Forecast pipeline will crash if best model is a baseline (no `.predict()` method) | **Medium** | Needs fix |
| 7 | `statsmodels` and `python-dotenv` in requirements.txt but never imported | **Low** | Needs fix |
| 8 | README claims "14-day forecasting" but the model trains on `target_demand_1d` (1-day horizon) | **Medium** | Needs fix |

---

## 2. Repository Structure

```
retailsync-ai/
├── README.md                    # Main project README (371 lines)
├── requirements.txt             # 12 dependencies
├── .gitignore                   # Standard Python ignores
├── LICENSE                      # MIT License
├── .env.example                 # Environment variables template (minimal)
├── Dockerfile                   # Multi-stage Python 3.11 build
├── docker-compose.yml           # Single-service compose
├── .dockerignore                # Docker ignores
├── .github/workflows/ci.yml     # CI workflow
├── run_pipeline_audit.py        # Pipeline validation script (duplicate of src/pipeline/run_pipeline.py)
├──
├── src/
│   ├── data/
│   │   ├── generate_dataset.py  # Synthetic data generation (124 lines)
│   │   └── ingest.py            # Cleaning/validation pipeline (115 lines)
│   ├── database/
│   │   └── init_db.py           # Database initialization (154 lines)
│   ├── features/
│   │   └── feature_engineering.py # Feature engineering (256 lines)
│   ├── forecasting/
│   │   ├── demand_forecaster.py  # Model training & evaluation (259 lines)
│   │   └── forecast_pipeline.py  # 14-day forecast generation (163 lines)
│   ├── inventory/
│   │   ├── inventory_intelligence.py # Risk detection (261 lines)
│   │   └── load_alerts.py        # Load alerts to DB (124 lines)
│   ├── anomaly/
│   │   └── anomaly_detection.py  # Ensemble anomaly detection (340 lines)
│   ├── clustering/
│   │   ├── segmentation.py       # Product/store/warehouse clustering (492 lines)
│   │   └── warehouse_optimization.py # Warehouse utilization (283 lines)
│   ├── models/                   # Empty __init__.py only
│   └── pipeline/
│       └── run_pipeline.py       # End-to-end pipeline validation (340 lines)
├──
├── data/
│   ├── raw/                      # 6 CSV files (generated)
│   ├── processed/                # 20+ CSV files + feature_metadata.csv
│   └── README.md
├──
├── database/
│   ├── schema.sql                # Core schema (87 lines, 6 tables + 8 indexes)
│   ├── queries.sql               # Analytical SQL queries (221 lines, 12 queries)
│   ├── alerts_schema.sql         # Alerts + anomaly_flags tables (47 lines)
│   └── retailsync.db             # SQLite database (12 tables)
├──
├── models/
│   ├── demand_forecaster.pkl     # 21.3 MB (RandomForestRegressor, mislabeled)
│   ├── product_clusterer.pkl     # 2.6 KB
│   ├── store_clusterer.pkl       # 2.4 KB
│   └── warehouse_clusterer.pkl   # 2.5 KB
├──
├── dashboard/
│   ├── app.py                    # 714 lines — single-file Streamlit app (7 pages)
│   ├── components/               # Empty (__init__.py only)
│   └── pages/                    # Empty (__init__.py only)
├──
├── tests/
│   ├── test_pipeline.py          # 323 lines — 95 assertions (custom runner, NOT pytest-native)
│   ├── test_dashboard_e2e.py     # 181 lines — end-to-end dashboard data test
│   ├── validate_dashboard_integration.py # 136 lines — dashboard validation
│   ├── validate_db.py            # 13 lines — database inspection
│   ├── validate_features.py      # 111 lines — data leakage validation
│   ├── validate_queries.py       # 16 lines — SQL query validation
│   ├── check_db.py               # 10 lines — database inspection
│   └── __init__.py
├──
├── docs/                         # 30+ documentation files
├── notebooks/                    # EDA notebook + html outputs
└──
```

---

## 3. What Is Genuinely Implemented

All components listed below were verified by running the test suite and inspecting actual file contents:

- **Synthetic data generation** — `generate_dataset.py` produces 50 products, 10 stores, 8 suppliers, 5 warehouses, 69,216 sales records, 52,500 inventory snapshots, 730 days of data. Verified from raw CSV files.
- **Data cleaning** — `ingest.py` performs schema validation, missing value imputation, duplicate removal, date validation, IQR outlier detection.
- **SQLite database** — `init_db.py` creates 12 tables with foreign keys, loads all data, validates referential integrity (0 orphans). Verified all 12 tables exist and have data.
- **Feature engineering** — `feature_engineering.py` produces 365,000 rows × 74 columns including lags, rolling stats, expanding stats, time features, price/promotion features, inventory features, demand variability, aggregates, and 6 target variables.
- **Demand forecasting** — `demand_forecaster.py` trains 3 baselines (Mean, Naive, MA) + Random Forest + XGBoost with time-based split. Evaluation metrics: MAE, RMSE, R², sMAPE.
- **14-day forecast generation** — `forecast_pipeline.py` generates 7,000 forecast rows (500 product-store combos × 14 days).
- **Inventory intelligence** — `inventory_intelligence.py` detects stockout risk (HIGH/MEDIUM/LOW), overstock risk, dead stock, reorder urgency, and composite risk score with recommendations.
- **Anomaly detection** — `anomaly_detection.py` implements Z-score + IQR + Isolation Forest ensemble with consensus voting (2+ methods).
- **K-Means clustering** — `segmentation.py` clusters products, stores, and warehouses with optimal-K selection via silhouette score, business label assignment, and model serialization.
- **Warehouse utilization analytics** — `warehouse_optimization.py` calculates capacity, occupancy, utilization %, turnover, and recommendations.
- **7-page Streamlit dashboard** — `dashboard/app.py` with Executive Overview, Demand Forecast, Inventory Intelligence, Demand Anomalies, Segmentation, Warehouse Intelligence, and Data Explorer pages. Dark theme with Plotly charts. Uses `@st.cache_resource` and `@st.cache_data`.
- **Test suite** — `test_pipeline.py` runs 95 assertions across 9 test functions. All 95 pass when run via `python tests/test_pipeline.py`.

---

## 4. What Is Only Documented (Not Implemented)

- **Docker deployment** — Dockerfile and docker-compose.yml exist but were not tested in this audit. No Docker image was built or run.
- **CI/CD** — `.github/workflows/ci.yml` exists but was not triggered or verified. See Section 7 for critical CI issues.
- **`.env` usage** — `.env.example` exists with 4 variables (DATABASE_URL, APP_NAME, APP_VERSION, etc.), but **no code reads `.env` files**. `python-dotenv` is listed in requirements.txt but never imported.
- **`statsmodels`** — Listed in requirements.txt but never imported anywhere in `src/` or `dashboard/`.
- **Dashboard modularization** — `dashboard/components/` and `dashboard/pages/` directories exist with only `__init__.py` files. The entire 714-line app is monolithic in `app.py`.
- **`src/models/`** — Only contains an empty `__init__.py`. No model registry or utilities.

---

## 5. What Is Partially Implemented

- **Warehouse "optimization"** — `warehouse_optimization.py` performs **descriptive analytics only** (capacity, occupancy, utilization %, rule-based recommendations). There is **no optimization algorithm** (no linear programming, no allocation solver, no constraint satisfaction). The term "optimization" is misleading. See Section 9.
- **Forecast pipeline model persistence** — `demand_forecaster.py` saves a model package, but the model selection fallback logic is broken (Section 8). The `forecast_pipeline.py` assumes the saved model has a `.predict()` method, but baselines do not.
- **Test integration with pytest** — The test functions are named `test_*` and the CI runs `pytest tests/`, but the tests take a `results` parameter that pytest interprets as a fixture. This causes all 9 tests to error under pytest. The tests only work via the custom `if __name__ == "__main__"` runner.
- **Inventory alerts database table** — Created in code (`load_alerts.py`) using `alerts_schema.sql`, not in the main `schema.sql`. The table creation is split across two files.

---

## 6. What Is Duplicated

- **`run_pipeline_audit.py`** (root) vs **`src/pipeline/run_pipeline.py`** — Near-identical content (328 vs 340 lines). Both perform the same pipeline validation and generate the same `docs/pipeline_summary.md`, `docs/pipeline_config.json`, and `docs/pipeline_insights.json`.
- **`docs/testing.md`** vs **`docs/testing_report.md`** — Overlapping content (95 tests, same categories, similar structure).
- **`docs/portfolio_highlights.md`** vs **`docs/portfolio_summary.md`** — Overlapping content (same metrics, same conclusions).
- **Schema split** — Core tables in `schema.sql`, alerts/anomalies tables in `alerts_schema.sql`. Both are needed to create the full database.
- **Config duplication** — Pipeline configuration is hardcoded in `run_pipeline_audit.py`, `src/pipeline/run_pipeline.py`, and `docs/pipeline_config.json` with overlapping values.

---

## 7. What Is Unused

- **`statsmodels`** — In `requirements.txt` but never imported in any `.py` file.
- **`python-dotenv`** — In `requirements.txt` but never imported. No `.env` file is loaded anywhere.
- **`src/models/` directory** — Empty `__init__.py` only. No model utilities or registry.
- **`dashboard/components/`** and **`dashboard/pages/`** — Empty `__init__.py` files only.
- **`docs/.env.example`** — Does not exist; the `.env.example` at root is minimal and unused.
- **`data/raw/.gitkeep`** and **`data/processed/.gitkeep`** — Git keep files, not functional.
- **`models/.gitkeep`** — Same.

---

## 8. What Is Hard-Coded

| Pattern | Files Affected | Issue |
|---------|---------------|-------|
| Database path `sqlite:///database/retailsync.db` | 6+ files (ingest, init_db, feature_engineering, inventory, anomaly, clustering) | No configuration abstraction |
| Data paths `data/raw/`, `data/processed/`, `models/` | Every script | Paths hardcoded, not config-driven |
| Random seed `np.random.seed(42)` | 7+ files | Good for reproducibility, but no config module |
| Train/val/test split dates | `demand_forecaster.py:28-30` | `2024-12-31`, `2025-06-09`, `2025-08-09` hardcoded |
| Dashboard CSS | `dashboard/app.py:19-78` | 60 lines of inline CSS |
| Category list, store types | `generate_dataset.py:21-22` | Hardcoded strings |
| Pipeline metrics | `run_pipeline_audit.py:42-48`, `src/pipeline/run_pipeline.py:42-48` | MAE=4.0913, RMSE=6.6497, R²=-0.0035 hardcoded in both files |
| Alert date "2025-08-08" | `load_alerts.py:18,35,49,67` | Hardcoded alert date string |

---

## 9. What Is Potentially Broken

### 9.1 Critical: Model selection fallback bug (`demand_forecaster.py:181-186`)

The model selection logic at lines 181-186 is:

```python
if "RandomForest" in best_model_name:
    best_model = rf_model
elif "XGBoost" in best_model_name:
    best_model = xgb_model
else:
    best_model = rf_model  # fallback
```

**The problem:** When `Baseline_Mean` wins (lowest MAE on validation), `best_model_name` is `"Baseline_Mean"`. The code falls into the `else` branch and saves `rf_model` (RandomForestRegressor) as the model, but labels it `model_name: "Baseline_Mean"` and stores the baseline metrics in the `metrics` dict.

**Verified:** The `models/demand_forecaster.pkl` pickle contains:
- `model_name`: `"Baseline_Mean"`
- `model`: `RandomForestRegressor` instance (NOT a baseline mean)
- `metrics`: `{'model': 'Baseline_Mean (Test Set)', 'mae': 4.5653, ...}`

This means:
1. The model package is internally inconsistent (name says Baseline_Mean, but the saved object is RandomForest).
2. The `forecast_pipeline.py` calls `model.predict()` — this works because RF has `.predict()`, but the metrics reported are from RF's test-set predictions, NOT from the actual baseline mean.
3. The README claims MAE=4.09, but the actual pickle metrics say MAE=4.5653 (RF on test set).

**Root cause:** The baselines (Mean, Naive, MA) are not scikit-learn estimators — they use numpy operations directly. The code has no way to save a callable baseline as a `.predict()` model. The fallback to RF is a hack that masks the issue.

**Fix needed:** Either (a) wrap baselines in a custom predictor class with `.predict()`, or (b) save the actual best model object properly.

### 9.2 Medium: Forecasting target horizon mismatch

The README and documentation claim "14-day forecasting," and `forecast_pipeline.py` generates 14-day forecasts. However, `demand_forecaster.py` trains and evaluates on `target_demand_1d` (next-day demand). The 7-day and 14-day targets exist in the features but are never used for training/evaluation. The forecast pipeline uses whatever model was saved (currently RF) and applies it recursively for 14 days, but the model was only trained to predict 1-day-ahead.

### 9.3 Critical: Warehouse segments CSV has 40 rows instead of 5

`data/processed/warehouse_segments.csv` has 40 rows (5 warehouses × 8 supplier_ids each) instead of 5.

**Root cause:** In `segmentation.py:249-250`:
```python
warehouse_static = df[["warehouse_id", "supplier_id"]].drop_duplicates()
```
The daily features DataFrame has multiple rows per `warehouse_id` (one per product-store-date). Each row may have a different `supplier_id` for the same warehouse (because supplier is a product attribute, not a warehouse attribute). `drop_duplicates()` without a subset keeps all unique `(warehouse_id, supplier_id)` pairs, creating 8 rows per warehouse.

The database table `warehouse_segments` has 5 rows (correct), because the SQL insert uses `drop_duplicates(subset=["warehouse_id"])` at line 453. But the CSV file is wrong.

**Impact:** The dashboard's `load_data()` reads `warehouse_segments.csv` directly and would display 40 warehouse rows instead of 5.

### 9.4 High: `pytest tests/` fails — CI is broken

The CI workflow (`.github/workflows/ci.yml`) runs:
```yaml
python tests/test_pipeline.py
python tests/validate_dashboard_integration.py
python tests/test_dashboard_e2e.py
```

**But** the README says to run `pytest tests/ -v`. The CI does NOT actually use pytest — it runs the test scripts directly. However, `tests/test_pipeline.py` uses a custom `TestResults` class with a `run_all_tests()` function called from `if __name__ == "__main__"`. The test functions (`test_data_files_exist`, etc.) take a `results` parameter that pytest interprets as a fixture.

**Verified:** `pytest tests/ -v` collects 9 tests but all 9 error with `fixture 'results' not found`. Only `python tests/test_pipeline.py` works (95/95 pass).

The other test files (`validate_db.py`, `validate_features.py`, `validate_queries.py`, `check_db.py`) are **not test functions** — they are scripts with top-level code that run on import. Under pytest, they would also error or behave unexpectedly.

### 9.5 Medium: Data leakage in aggregate features (`feature_engineering.py:197-201`)

```python
cat_daily = daily_df.groupby(["date", "category"], as_index=False)["quantity_sold"].mean()
daily_df = daily_df.merge(cat_daily, on=["date", "category"], how="left")
```

`category_avg_demand` is the **cross-sectional mean** of `quantity_sold` for all product-store combinations in that category on that date. This **includes the current row's demand**. Since the target is `target_demand_1d` (next-day demand), this is a 1-day lookback — the feature for day T includes demand from day T (same day), but the target for day T is day T+1's demand.

**Severity:** Mild leakage. The feature is available at prediction time (you know today's category-level sales), but it includes information about the same day's demand for the target product-store, which is not yet observed when predicting at the start of day T. For a true 1-day-ahead forecast, `category_avg_demand` should be shifted by 1 day.

The same issue applies to `store_type_avg_demand`.

### 9.6 Low: Windows encoding issues in test scripts

Several test scripts use Unicode checkmarks (`✓`, `✗`) without `sys.stdout.reconfigure(encoding="utf-8")`. On Windows with cp1252 console, these fail. `validate_dashboard_integration.py:10-11` adds the reconfigure, but others don't.

### 9.7 Low: `load_alerts.py` and `anomaly_detection.py` have circular DB import

`anomaly_detection.py:247` and `segmentation.py:389` both do `from sqlalchemy import create_engine, text` inside the function body, after already importing at the top. This is redundant but not broken.

### 9.8 Low: `load_alerts.py` hardcodes alert date

`load_alerts.py:18,35,49,67` hardcodes `"alert_date": "2025-08-08"` instead of using the actual latest inventory date.

---

## 10. README Accuracy Audit

### Verified Claims (100% accurate)

| README Claim | Actual Value | Status |
|--------------|--------------|--------|
| 50 products | 50 | ✅ Verified |
| 10 stores | 10 | ✅ Verified |
| 8 suppliers | 8 | ✅ Verified |
| 5 warehouses | 5 | ✅ Verified |
| 730 days | 730 (2023-08-11 to 2025-08-09) | ✅ Verified |
| 69,216 sales records | 69,216 | ✅ Verified |
| 52,500 inventory snapshots | 52,500 | ✅ Verified |
| 74 engineered features | 74 columns (including 3 identifier columns) | ✅ Verified |
| 14-day forecasting | 14-day forecast horizon | ✅ Verified |
| Inventory risk detection | Implemented with HIGH/MEDIUM/LOW | ✅ Verified |
| Ensemble anomaly detection | Z-score + IQR + Isolation Forest | ✅ Verified |
| K-Means segmentation | K=2 (products), K=2 (stores), K=4 (warehouses) | ✅ Verified |
| 7-page Streamlit dashboard | 7 pages via sidebar radio | ✅ Verified |
| 95 tests | 95 assertions via `python tests/test_pipeline.py` | ✅ Verified |
| Zero-inflation ~81% | 81.42% of daily demand values are zero | ✅ Verified |
| Promotions ~15% | 15.29% of sales are promotional | ✅ Verified |
| Stockouts 36.6% | 36.6% of raw inventory snapshots have QOH ≤ reorder_point | ✅ Verified |
| Total forecasted demand 18,541 units | 18,541 (forecast_demand.sum()) | ✅ Verified |
| Total forecasted revenue $4,807,527 | $4,807,527.16 (forecast_revenue.sum()) | ✅ Verified |
| Stockout HIGH: 16, MEDIUM: 169 | 16 HIGH, 169 MEDIUM | ✅ Verified |
| Overstock HIGH: 110 | 110 HIGH | ✅ Verified |
| Urgent Reorder: 185 | 185 URGENT | ✅ Verified |
| Dead Stock: 0 | 0 | ✅ Verified |
| Total anomalies: 31,619 (8.66%) | 31,619 (8.66%) | ✅ Verified |
| Demand Spikes: 28,292 | 28,292 | ✅ Verified |
| Unusual Patterns: 3,327 | 3,327 | ✅ Verified |
| Product segments: 50 with 5 labels | 50 rows, 5 labels (Slow-Moving:27, High-Volume/Stable:9, Low-Volume/Volatile:7, High-Volume/Volatile:4, Medium-Volume/Moderate:3) | ✅ Verified |
| Store segments: 10 with 4 labels | 10 rows, 4 labels (High-Performance:3, Low-Performance:3, Stable Performance:2, High-Variability:2) | ✅ Verified |
| Warehouse segments: 5 | DB has 5, but CSV has 40 (bug) | ⚠️ Partial — CSV is wrong |
| Warehouse: Total capacity 48,906 m³ | 48,906 | ✅ Verified |
| Warehouse: Occupied 6,109 m³ | 6,109 | ✅ Verified |
| Warehouse: Avg utilization 14.5% | 14.5% | ✅ Verified |
| Warehouse: Low utilization 5 | 5 | ✅ Verified |

### Disputed Claims (README does NOT match actual data)

| README Claim | Actual Value | Issue |
|--------------|--------------|-------|
| Best Model: Baseline Mean | Model pickle contains RandomForestRegressor | **Model selection bug** — RF was saved but labeled as Baseline_Mean |
| Test MAE = 4.09 | 4.5653 (from model pickle, RF on test set) | README is stale; the "Baseline Mean" metrics were never actually saved |
| Test RMSE = 6.65 | 7.1976 (from model pickle) | README is stale |
| Test R² = -0.0035 | -0.0071 (from model pickle) | README is stale |

### Analysis of the forecasting metrics discrepancy

The README claims the **Baseline Mean** is the best model with MAE=4.09. But the actual model pickle contains a **RandomForestRegressor** with MAE=4.57. The discrepancy is caused by the fallback bug in `demand_forecaster.py:181-186`:

1. During validation, `Baseline_Mean` achieves the lowest MAE on the validation set.
2. The code selects `Baseline_Mean` as `best_model_name`.
3. The `else` branch at line 186 sets `best_model = rf_model` (since "Baseline_Mean" doesn't match "RandomForest" or "XGBoost").
4. The model package saves `model_name = "Baseline_Mean"` (the string) but `model = rf_model` (the RF object).
5. The metrics dict is populated from the RF test-set evaluation, not the baseline.
6. The README was written with the baseline mean's validation metrics (MAE=4.09), but the pickle has RF's test metrics (MAE=4.57).

**The actual best model during training was likely RandomForest** (since RF is what gets saved), but the metrics reported are RF's test-set performance, not the baseline's. The baseline mean's metrics (MAE=4.09) were never actually computed on the test set — they appear to be from the validation set or were manually written.

---

## 11. Technical Debt

### High Priority

1. **Model selection logic is broken** — baselines cannot be saved as `.predict()` models; the fallback saves RF under a false name. Fix: implement a `BaselineMean` wrapper class with `.predict()` and `feature_importances_` attributes, or restructure the model saving logic.

2. **pytest CI is broken** — `pytest tests/` errors on all tests. Either refactor tests to be pytest-native (use plain `assert` statements, remove `results` parameter) or change CI to only run `python tests/test_pipeline.py`.

3. **Warehouse segments CSV has duplicate rows** — 40 rows instead of 5. Fix: add `subset=["warehouse_id"]` to `drop_duplicates()` in `segmentation.py:249`.

4. **README metrics are stale** — Update to match actual model pickle values, or re-train and update both.

### Medium Priority

5. **Data leakage in aggregate features** — Shift `category_avg_demand` and `store_type_avg_demand` by 1 day within each product-store group.

6. **Forecasting target mismatch** — The model trains on 1-day-ahead target but claims 14-day forecasting. Either train on 14-day target, or document that forecasts are generated recursively using a 1-day model.

7. **Hardcoded paths everywhere** — Create a `config.py` or use `.env` variables consistently.

8. **Dashboard is a 714-line monolith** — Split into `dashboard/pages/` modules. The directory structure already exists but is empty.

9. **Duplicate pipeline scripts** — `run_pipeline_audit.py` and `src/pipeline/run_pipeline.py` are near-identical. Consolidate.

### Low Priority

10. **Unused dependencies** — Remove `statsmodels` and `python-dotenv` from requirements.txt, or implement their usage.

11. **Empty directories** — `src/models/`, `dashboard/components/`, `dashboard/pages/` contain only `__init__.py`.

12. **Windows encoding** — Add `sys.stdout.reconfigure(encoding="utf-8")` to all test scripts.

13. **Hardcoded alert date** — Use dynamic date in `load_alerts.py`.

14. **Duplicate docs** — Consolidate `docs/testing.md` + `docs/testing_report.md` and `docs/portfolio_highlights.md` + `docs/portfolio_summary.md`.

15. **No `requirements.txt` pinning** — All versions are minimum (`>=`). Consider pinning for reproducibility.

---

## 12. Recommended Improvement Order (Phase 1.1 — Critical Fixes)

1. **Fix warehouse segments CSV bug** (`segmentation.py:249`) — Add `subset=["warehouse_id"]` to `drop_duplicates()`.
2. **Fix model selection logic** (`demand_forecaster.py:181-186`) — Implement a proper baseline wrapper or restructure model saving.
3. **Fix data leakage** (`feature_engineering.py:197-201`) — Shift aggregate features by 1 day.
4. **Fix pytest/CI** — Make tests pytest-native or update CI to use the custom runner.
5. **Update README metrics** — Match actual model pickle values OR re-train and update both.
6. **Fix hardcoded alert date** — Use dynamic date in `load_alerts.py`.

---

## 13. Evidence Collected

All claims in this audit were verified against actual code, data files, and model artifacts:

- **Feature count:** `data/processed/features_daily.csv` → 74 columns (verified)
- **Sales count:** `data/raw/sales.csv` → 69,216 rows (verified)
- **Inventory count:** `data/raw/inventory.csv` → 52,500 rows (verified)
- **Zero-inflation:** `features.quantity_sold == 0` → 81.42% (verified)
- **Anomaly count:** `data/processed/anomalies.csv` → 31,619 rows, 8.66% rate (verified)
- **Inventory alerts:** 16 HIGH stockout, 169 MEDIUM, 110 HIGH overstock, 185 URGENT, 0 dead stock (verified)
- **Test count:** `python tests/test_pipeline.py` → 95/95 PASS (verified)
- **Database tables:** 12 tables, all with data, 0 orphaned FKs (verified)
- **Model metrics:** `models/demand_forecaster.pkl` → MAE=4.5653, RMSE=7.1976, R²=-0.0071 (verified)
- **Model object type:** `RandomForestRegressor` (verified — NOT a baseline mean despite the name)
- **Warehouse segments bug:** CSV has 40 rows, DB has 5 rows (verified)
- **pytest failure:** `pytest tests/ -v` → 9 errors, `fixture 'results' not found` (verified)
- **Forecast values:** Total demand 18,541 units, total revenue $4,807,527.16 (verified)
- **Warehouse utilization:** Avg 14.5%, total capacity 48,906 m³, total occupied 6,109 m³ (verified)
- **Promotion rate:** 15.29% of sales (verified from raw data)
- **Stockout rate:** 36.6% of raw inventory snapshots (verified from raw data)
- **Unused imports:** `grep` for `statsmodels` and `dotenv` in `src/` and `dashboard/` → 0 matches (verified)
- **Empty directories:** `src/models/`, `dashboard/components/`, `dashboard/pages/` → only `__init__.py` (verified)

---

## 14. Next Steps

**Do not proceed to Phase 2 until Phase 1.1 critical fixes are complete.**

The following critical fixes must be applied before continuing:

1. Fix the warehouse segments CSV bug (`segmentation.py:249`)
2. Fix the model selection fallback logic (`demand_forecaster.py:181-186`)
3. Fix data leakage in aggregate features (`feature_engineering.py:197-201`)
4. Fix pytest compatibility or update CI
5. Re-run the full pipeline and test suite after fixes

After fixes:
1. Re-run `python tests/test_pipeline.py` — confirm all 95 tests still pass
2. Re-run `python tests/validate_dashboard_integration.py`
3. Re-run `python tests/test_dashboard_e2e.py`
4. Update `README.md` with verified, consistent metrics
5. Proceed to Phase 2: Data Pipeline Audit

---

*End of Phase 1 Audit*
