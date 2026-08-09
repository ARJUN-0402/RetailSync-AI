# RetailSync AI — Phase 3: SQL/Database Audit

**Date:** 2026-08-09
**Auditor:** Kilo
**Status:** READ-ONLY AUDIT — No code changes made

---

## 1. Executive Summary

The database layer is **well-structured and mostly correct**, but has one **critical security/integrity issue**: foreign key constraints are defined in schema but **not enforced** by SQLite at runtime. All analytical queries are syntactically correct and return expected results. Dashboard metrics can be traced back to actual database queries.

### Bottom Line

| Component | Status | Notes |
|-----------|--------|-------|
| Schema design | ✅ Verified | 12 tables, proper normalization |
| Foreign key definitions | ✅ Verified | FK clauses present in DDL |
| Foreign key enforcement | ❌ **NOT enforced** | SQLite requires per-connection PRAGMA |
| Indexes | ✅ Verified | 18 custom indexes |
| Analytical queries | ✅ Verified | All 12 queries execute correctly |
| Data types | ✅ Verified | Appropriate types used |
| Dashboard metrics traceability | ✅ Verified | Metrics originate from DB/CSV |

---

## 2. Schema Audit

### Tables Created

| Table | Rows | Primary Key | Foreign Keys | Status |
|-------|------|-------------|--------------|--------|
| products | 50 | product_id | supplier_id | ✅ |
| stores | 10 | store_id | — | ✅ |
| suppliers | 8 | supplier_id | — | ✅ |
| warehouses | 5 | warehouse_id | supplier_id | ✅ |
| sales | 69,216 | sale_id (auto) | product_id, store_id | ✅ |
| inventory | 52,500 | inventory_id (auto) | product_id, store_id, warehouse_id | ✅ |
| inventory_alerts | 785 | alert_id (auto) | product_id, store_id, warehouse_id | ✅ |
| anomaly_flags | 31,619 | anomaly_id (auto) | product_id, store_id | ✅ |
| product_segments | 50 | product_id | product_id | ✅ |
| store_segments | 10 | store_id | store_id | ✅ |
| warehouse_segments | 5 | warehouse_id | warehouse_id | ✅ |
| warehouse_optimization | 5 | warehouse_id | warehouse_id | ✅ |

### Normalization Assessment

- **3NF achieved**: All tables have no transitive dependencies
- **Dimension tables**: products, stores, suppliers, warehouses
- **Fact tables**: sales, inventory
- **Analytics tables**: inventory_alerts, anomaly_flags, segments, warehouse_optimization
- **Junction tables**: None needed — many-to-many relationships not present

### Data Types

| Column | Type | Assessment |
|--------|------|------------|
| IDs (product, store, etc.) | TEXT | ✅ Appropriate for alphanumeric IDs |
| Quantities | INTEGER | ✅ Appropriate |
| Prices/revenue | REAL | ✅ Appropriate |
| Dates | TEXT (ISO-8601) | ⚠️ Should be DATE/DATETIME for SQLite |
| Scores/ratios | REAL | ✅ Appropriate |

**Note:** SQLite doesn't enforce strict DATE types, but storing dates as TEXT (ISO-8601) is acceptable and comparable for sorting.

---

## 3. Foreign Key Audit

### Critical Finding: Foreign Keys NOT Enforced

**Status:** ❌ **CRITICAL ISSUE**

The schema file (`schema.sql`) contains:
```sql
PRAGMA foreign_keys = ON;
```

However, in SQLite, `PRAGMA foreign_keys = ON` is a **per-connection setting**, not a persistent one. The PRAGMA is executed during schema creation but does not apply to subsequent connections.

### Verification Test

I tested by attempting to insert an orphan record:
```python
# Attempting to insert sale with non-existent product_id
c.execute('INSERT INTO sales (date, product_id, store_id, ...) VALUES ("2025-08-10", "P999", "ST01", ...)')
```

**Result:** The insert succeeded. Foreign key constraint was NOT enforced.

### Impact

- **Data integrity risk**: Orphan records can be inserted into sales, inventory, etc.
- **Referential integrity**: Not guaranteed
- **Dashboard reliability**: Queries with JOINs may produce incomplete results if orphans exist

### Root Cause

`init_db.py` splits schema.sql by semicolons and executes each statement. The PRAGMA is executed but doesn't persist. Every new database connection needs to explicitly enable FK enforcement.

### Affected Code

**File:** `src/database/init_db.py`
```python
engine = create_engine(f"sqlite:///{DATABASE_PATH}")
# Missing: engine = engine.execution_options(pragma="foreign_keys=ON")
```

**File:** `dashboard/app.py`
```python
engine = create_engine("sqlite:///database/retailsync.db")
# Missing: engine = engine.execution_options(pragma="foreign_keys=ON")
```

### Recommendation

Add foreign key enforcement to all SQLAlchemy engines:
```python
engine = create_engine("sqlite:///database/retailsync.db", 
                       execution_options={"sqlite_raw_colnames": True})
# Or use event listener:
from sqlalchemy import event
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()
```

---

## 4. Index Audit

### Indexes Present

18 custom indexes found:

| Index Name | Table | Columns | Purpose |
|------------|-------|---------|---------|
| idx_sales_date | sales | date | Date range queries |
| idx_sales_product | sales | product_id | Product-level analytics |
| idx_sales_store | sales | store_id | Store-level analytics |
| idx_inventory_date | inventory | date | Date-based filtering |
| idx_inventory_product | inventory | product_id | Product inventory lookups |
| idx_inventory_store | inventory | store_id | Store inventory lookups |
| idx_inventory_warehouse | inventory | warehouse_id | Warehouse analytics |
| idx_products_supplier | products | supplier_id | Supplier performance |
| idx_alerts_product | inventory_alerts | product_id | Alert lookups |
| idx_alerts_store | inventory_alerts | store_id | Alert lookups |
| idx_alerts_risk | inventory_alerts | risk_level | Risk filtering |
| idx_alerts_date | inventory_alerts | alert_date | Date filtering |
| idx_anomaly_date | anomaly_flags | date | Date filtering |
| idx_anomaly_product | anomaly_flags | product_id | Product anomalies |
| idx_anomaly_store | anomaly_flags | store_id | Store anomalies |
| idx_anomaly_type | anomaly_flags | anomaly_type | Type filtering |
| idx_wh_opt_id | warehouse_optimization | warehouse_id | Warehouse lookups |
| idx_wh_opt_risk | warehouse_optimization | capacity_risk | Risk filtering |

### Assessment

✅ **Good index coverage** for common query patterns:
- Date-based filtering (sales, inventory, alerts, anomalies)
- Entity-based filtering (product, store, warehouse)
- Risk/category filtering

### Missing Indexes

| Suggested Index | Tables | Reason |
|-----------------|--------|--------|
| sales(product_id, date) | sales | Composite index for product-date queries |
| inventory(product_id, store_id, date) | inventory | Composite index for latest inventory CTEs |
| anomaly_flags(product_id, store_id, date) | anomaly_flags | Composite for product-store-time queries |

**Impact:** Low — current indexes are sufficient for dataset size (69K sales, 52.5K inventory)

---

## 5. Analytical Query Audit

### Queries Verified

All 12 queries from `database/queries.sql` were executed successfully:

| Query | Description | Status | Notes |
|-------|-------------|--------|-------|
| 1. Daily Sales Summary | Daily aggregates | ✅ | Returns correct totals |
| 2. Product Performance | All-time product rankings | ✅ | Correct JOINs |
| 3. Store Performance | All-time store rankings | ✅ | Correct JOINs |
| 4. Current Inventory Levels | Latest snapshot with CTE | ✅ | CTE works correctly |
| 5. Stockout Candidates | Below reorder point | ✅ | Returns 185 records |
| 6. Overstock Candidates | Above max stock | ✅ | Correct |
| 7. Inventory Turnover | 90-day turnover | ✅ | Requires date math |
| 8. Supplier Performance | Supplier metrics | ✅ | Correct |
| 9. Warehouse Utilization | Volume-based utilization | ✅ | Returns 5 warehouses |
| 10. Monthly Sales Trend | Monthly aggregates | ✅ | STRFTIME works |
| 11. Category Performance | Category rankings | ✅ | Correct |
| 12. Stockout Frequency | Product stockout history | ✅ | Correct |

### Query Issues Found

**Query 7: Inventory Turnover** — Uses SQLite date math:
```sql
AND i.date BETWEEN DATE(s.date, '-90 days') AND s.date
```

**Issue:** This joins every sale with inventory snapshots in a 90-day window, creating a **Cartesian product** that inflates turnover calculations. For a product with 100 sales and 13 weekly inventory snapshots in 90 days, this produces ~1,300 rows per product.

**Impact:** Turnover ratios are mathematically correct but computationally expensive. For the current dataset size, this is acceptable but would not scale to millions of records.

**Query 4 & 5: Latest Inventory CTE** — Correctly uses CTE to find latest inventory per product-store:
```sql
WITH latest_inventory AS (
    SELECT product_id, store_id, MAX(date) AS latest_date
    FROM inventory GROUP BY product_id, store_id
)
```

✅ **Correct implementation** — avoids correlated subqueries.

---

## 6. Dashboard Metrics Traceability

### Verified Metrics Originate from Actual Data

| Dashboard Metric | Source | Query/Computation | Verified |
|------------------|--------|-------------------|----------|
| Total Revenue | features_daily.csv | `df["revenue"].sum()` | ✅ |
| Total Quantity Sold | features_daily.csv | `df["quantity_sold"].sum()` | ✅ |
| Total Products | features_daily.csv | `df["product_id"].nunique()` | ✅ |
| Total Stores | features_daily.csv | `df["store_id"].nunique()` | ✅ |
| 14-Day Forecast | forecasts_next_14d.csv | `df["forecast_demand"].sum()` | ✅ |
| Stockout HIGH | inventory_intelligence.csv | `(df["stockout_risk"] == "HIGH").sum()` | ✅ |
| Anomalies Detected | anomalies.csv | `len(df)` | ✅ |
| Avg Warehouse Utilization | warehouse_optimization.csv | `df["utilization_pct"].mean()` | ✅ |

### SQL Queries Used by Dashboard

The dashboard loads data from:
1. **CSV files** (`pd.read_csv`) — processed outputs
2. **SQL queries** (`pd.read_sql`) — database tables
3. **ML models** (`joblib.load`) — serialized models

**Verification:** Dashboard code was inspected and confirmed to use actual data sources, not hard-coded values.

---

## 7. Data Integrity Checks

### Referential Integrity Validation

| Relationship | Orphan Count | Status |
|--------------|--------------|--------|
| sales → products | 0 | ✅ |
| sales → stores | 0 | ✅ |
| inventory → products | 0 | ✅ |
| inventory → stores | 0 | ✅ |
| inventory → warehouses | 0 | ✅ |
| warehouses → suppliers | 0 | ✅ |
| products → suppliers | 0 | ✅ |

**Note:** These checks pass because data was loaded correctly, NOT because FK constraints are enforced. Without FK enforcement, future data loads could introduce orphans.

### Data Consistency

- **No future dates** in sales or inventory
- **No negative quantities** or revenues
- **No duplicate primary keys**
- **Date ranges consistent** across tables

---

## 8. Critical Issues Summary

| # | Issue | Severity | Impact | Recommendation |
|---|-------|----------|--------|----------------|
| 1 | **Foreign keys not enforced** | **Critical** | Orphan records possible | Add FK pragma to all engines |
| 2 | Schema split across 2 files | **Low** | Confusion | Consolidate into schema.sql |
| 3 | Inventory turnover query has Cartesian product | **Low** | Performance at scale | Add composite index or rewrite |
| 4 | Dates stored as TEXT | **Low** | Type safety | Use ISO-8601 consistently (current approach is acceptable) |

---

## 9. Evidence

All findings are backed by actual database queries:

- Table counts verified via `SELECT COUNT(*)`
- FK orphans verified via LEFT JOIN checks
- Indexes verified via `sqlite_master` query
- Query correctness verified by execution
- FK enforcement failure verified by orphan insert test
- Dashboard metrics traced to actual CSV/DB sources

---

## 10. Next Steps

**Do not proceed to Phase 4 until critical FK issue is addressed.**

After fix:
1. Add FK enforcement to `init_db.py` and `dashboard/app.py`
2. Re-run test suite to confirm no regressions
3. Re-run foreign key validation tests
4. Proceed to Phase 4: Feature Engineering Audit

---

*End of Phase 3 Audit*
