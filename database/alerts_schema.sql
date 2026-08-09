-- Inventory Alerts Table
CREATE TABLE IF NOT EXISTS inventory_alerts (
    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
    product_id TEXT NOT NULL,
    store_id TEXT NOT NULL,
    warehouse_id TEXT,
    alert_date TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    reason TEXT NOT NULL,
    quantity_on_hand REAL,
    reorder_point REAL,
    max_stock_level REAL,
    stock_coverage_days REAL,
    forecast_demand_7d REAL,
    recommended_action TEXT NOT NULL,
    FOREIGN KEY (product_id) REFERENCES products(product_id),
    FOREIGN KEY (store_id) REFERENCES stores(store_id),
    FOREIGN KEY (warehouse_id) REFERENCES warehouses(warehouse_id)
);

CREATE INDEX IF NOT EXISTS idx_alerts_product ON inventory_alerts(product_id);
CREATE INDEX IF NOT EXISTS idx_alerts_store ON inventory_alerts(store_id);
CREATE INDEX IF NOT EXISTS idx_alerts_risk ON inventory_alerts(risk_level);
CREATE INDEX IF NOT EXISTS idx_alerts_date ON inventory_alerts(alert_date);

-- Anomaly Flags Table
CREATE TABLE IF NOT EXISTS anomaly_flags (
    anomaly_id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    product_id TEXT NOT NULL,
    store_id TEXT NOT NULL,
    quantity_sold REAL,
    anomaly_type TEXT NOT NULL,
    z_score REAL,
    method_agreement INTEGER,
    category TEXT,
    store_type TEXT,
    city TEXT,
    FOREIGN KEY (product_id) REFERENCES products(product_id),
    FOREIGN KEY (store_id) REFERENCES stores(store_id)
);

CREATE INDEX IF NOT EXISTS idx_anomaly_date ON anomaly_flags(date);
CREATE INDEX IF NOT EXISTS idx_anomaly_product ON anomaly_flags(product_id);
CREATE INDEX IF NOT EXISTS idx_anomaly_store ON anomaly_flags(store_id);
CREATE INDEX IF NOT EXISTS idx_anomaly_type ON anomaly_flags(anomaly_type);
