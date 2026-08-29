-- RetailSync AI Database Schema
-- SQLite with foreign key enforcement

PRAGMA foreign_keys = ON;

-- Dimension: Products
CREATE TABLE IF NOT EXISTS products (
    product_id TEXT PRIMARY KEY,
    product_name TEXT NOT NULL,
    category TEXT NOT NULL,
    subcategory TEXT NOT NULL,
    unit_price REAL NOT NULL,
    cost_price REAL NOT NULL,
    supplier_id TEXT NOT NULL,
    weight_kg REAL NOT NULL,
    volume_m3 REAL NOT NULL,
    launch_date TEXT NOT NULL
);

-- Dimension: Stores
CREATE TABLE IF NOT EXISTS stores (
    store_id TEXT PRIMARY KEY,
    store_name TEXT NOT NULL,
    city TEXT NOT NULL,
    state TEXT NOT NULL,
    store_type TEXT NOT NULL,
    opening_date TEXT NOT NULL
);

-- Dimension: Suppliers
CREATE TABLE IF NOT EXISTS suppliers (
    supplier_id TEXT PRIMARY KEY,
    supplier_name TEXT NOT NULL,
    country TEXT NOT NULL,
    lead_time_days INTEGER NOT NULL,
    reliability_score REAL NOT NULL
);

-- Dimension: Warehouses
CREATE TABLE IF NOT EXISTS warehouses (
    warehouse_id TEXT PRIMARY KEY,
    warehouse_name TEXT NOT NULL,
    city TEXT NOT NULL,
    state TEXT NOT NULL,
    capacity_m3 INTEGER NOT NULL,
    supplier_id TEXT NOT NULL,
    FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_id)
);

-- Fact: Sales
CREATE TABLE IF NOT EXISTS sales (
    sale_id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    product_id TEXT NOT NULL,
    store_id TEXT NOT NULL,
    quantity_sold INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    discount_pct REAL NOT NULL,
    promotion INTEGER NOT NULL,
    revenue REAL NOT NULL,
    FOREIGN KEY (product_id) REFERENCES products(product_id),
    FOREIGN KEY (store_id) REFERENCES stores(store_id)
);

-- Fact: Inventory
CREATE TABLE IF NOT EXISTS inventory (
    inventory_id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    product_id TEXT NOT NULL,
    store_id TEXT NOT NULL,
    quantity_on_hand INTEGER NOT NULL,
    reorder_point INTEGER NOT NULL,
    max_stock_level INTEGER NOT NULL,
    warehouse_id TEXT NOT NULL,
    FOREIGN KEY (product_id) REFERENCES products(product_id),
    FOREIGN KEY (store_id) REFERENCES stores(store_id),
    FOREIGN KEY (warehouse_id) REFERENCES warehouses(warehouse_id)
);

-- Indexes for query performance
CREATE INDEX IF NOT EXISTS idx_sales_date ON sales(date);
CREATE INDEX IF NOT EXISTS idx_sales_product ON sales(product_id);
CREATE INDEX IF NOT EXISTS idx_sales_store ON sales(store_id);
CREATE INDEX IF NOT EXISTS idx_inventory_date ON inventory(date);
CREATE INDEX IF NOT EXISTS idx_inventory_product ON inventory(product_id);
CREATE INDEX IF NOT EXISTS idx_inventory_store ON inventory(store_id);
CREATE INDEX IF NOT EXISTS idx_inventory_warehouse ON inventory(warehouse_id);
CREATE INDEX IF NOT EXISTS idx_products_supplier ON products(supplier_id);

-- ============================================================
-- ANALYTICS TABLES
-- Written by the pipeline stages after database initialization.
-- Column names must match the DataFrames produced by the modules
-- listed against each table.
-- ============================================================

-- Analytics: Inventory alerts (written by src/inventory/load_alerts.py)
CREATE TABLE IF NOT EXISTS inventory_alerts (
    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id TEXT NOT NULL,
    store_id TEXT NOT NULL,
    warehouse_id TEXT,
    alert_date TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    reason TEXT,
    quantity_on_hand REAL,
    reorder_point REAL,
    max_stock_level REAL,
    stock_coverage_days REAL,
    forecast_demand_7d REAL,
    recommended_action TEXT,
    FOREIGN KEY (product_id) REFERENCES products(product_id),
    FOREIGN KEY (store_id) REFERENCES stores(store_id),
    FOREIGN KEY (warehouse_id) REFERENCES warehouses(warehouse_id)
);

CREATE INDEX IF NOT EXISTS idx_alerts_product ON inventory_alerts(product_id);
CREATE INDEX IF NOT EXISTS idx_alerts_store ON inventory_alerts(store_id);
CREATE INDEX IF NOT EXISTS idx_alerts_risk ON inventory_alerts(risk_level);
CREATE INDEX IF NOT EXISTS idx_alerts_date ON inventory_alerts(alert_date);
CREATE INDEX IF NOT EXISTS idx_alerts_type ON inventory_alerts(alert_type);

-- Analytics: Demand anomaly flags (written by src/anomaly/anomaly_detection.py)
CREATE TABLE IF NOT EXISTS anomaly_flags (
    anomaly_id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    product_id TEXT NOT NULL,
    store_id TEXT NOT NULL,
    z_score REAL,
    anomaly_type TEXT NOT NULL,
    detection_methods TEXT,
    quantity_sold REAL,
    FOREIGN KEY (product_id) REFERENCES products(product_id),
    FOREIGN KEY (store_id) REFERENCES stores(store_id)
);

CREATE INDEX IF NOT EXISTS idx_anomaly_date ON anomaly_flags(date);
CREATE INDEX IF NOT EXISTS idx_anomaly_product ON anomaly_flags(product_id);
CREATE INDEX IF NOT EXISTS idx_anomaly_store ON anomaly_flags(store_id);
CREATE INDEX IF NOT EXISTS idx_anomaly_type ON anomaly_flags(anomaly_type);

-- Analytics: Product segments (written by src/clustering/segmentation.py)
CREATE TABLE IF NOT EXISTS product_segments (
    product_id TEXT PRIMARY KEY,
    total_revenue REAL,
    avg_demand REAL,
    demand_cv REAL,
    zero_demand_pct REAL,
    cluster INTEGER,
    cluster_label TEXT,
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

-- Analytics: Store segments (written by src/clustering/segmentation.py)
CREATE TABLE IF NOT EXISTS store_segments (
    store_id TEXT PRIMARY KEY,
    total_revenue REAL,
    avg_demand REAL,
    demand_cv REAL,
    zero_demand_pct REAL,
    cluster INTEGER,
    cluster_label TEXT,
    FOREIGN KEY (store_id) REFERENCES stores(store_id)
);

-- Analytics: Warehouse segments (written by src/clustering/segmentation.py)
CREATE TABLE IF NOT EXISTS warehouse_segments (
    warehouse_id TEXT PRIMARY KEY,
    total_quantity REAL,
    avg_stock_coverage REAL,
    capacity_m3 REAL,
    utilization_pct REAL,
    turnover REAL,
    cluster INTEGER,
    cluster_label TEXT,
    FOREIGN KEY (warehouse_id) REFERENCES warehouses(warehouse_id)
);

-- Analytics: Warehouse optimization (written by src/clustering/warehouse_optimization.py)
CREATE TABLE IF NOT EXISTS warehouse_optimization (
    warehouse_id TEXT PRIMARY KEY,
    total_quantity REAL,
    distinct_products INTEGER,
    occupied_volume_m3 REAL,
    avg_quantity_per_product REAL,
    max_quantity REAL,
    min_quantity REAL,
    warehouse_name TEXT,
    city TEXT,
    state TEXT,
    capacity_m3 REAL,
    utilization_pct REAL,
    available_capacity_m3 REAL,
    capacity_risk TEXT,
    capacity_risk_reason TEXT,
    inventory_turnover REAL,
    recommendation TEXT,
    FOREIGN KEY (warehouse_id) REFERENCES warehouses(warehouse_id)
);
