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
