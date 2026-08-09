-- RetailSync AI Analytical SQL Queries
-- These queries power the dashboard and analytics modules.

-- 1. Daily Sales Summary
SELECT
    date,
    SUM(quantity_sold) AS total_quantity_sold,
    SUM(revenue) AS total_revenue,
    AVG(unit_price) AS avg_unit_price,
    COUNT(DISTINCT product_id) AS products_sold,
    SUM(CASE WHEN promotion = 1 THEN quantity_sold ELSE 0 END) AS promotional_quantity
FROM sales
GROUP BY date
ORDER BY date DESC;

-- 2. Product Performance (all-time)
SELECT
    p.product_id,
    p.product_name,
    p.category,
    p.unit_price,
    SUM(s.quantity_sold) AS total_quantity_sold,
    SUM(s.revenue) AS total_revenue,
    COUNT(DISTINCT s.date) AS days_sold,
    AVG(s.quantity_sold) AS avg_daily_quantity,
    AVG(s.unit_price) AS avg_selling_price
FROM sales s
JOIN products p ON s.product_id = p.product_id
GROUP BY p.product_id, p.product_name, p.category, p.unit_price
ORDER BY total_revenue DESC;

-- 3. Store Performance (all-time)
SELECT
    st.store_id,
    st.store_name,
    st.city,
    st.state,
    st.store_type,
    SUM(s.revenue) AS total_revenue,
    SUM(s.quantity_sold) AS total_quantity_sold,
    COUNT(DISTINCT s.date) AS active_days,
    AVG(s.revenue) AS avg_daily_revenue
FROM sales s
JOIN stores st ON s.store_id = st.store_id
GROUP BY st.store_id, st.store_name, st.city, st.state, st.store_type
ORDER BY total_revenue DESC;

-- 4. Current Inventory Levels (latest snapshot)
WITH latest_inventory AS (
    SELECT
        product_id,
        store_id,
        MAX(date) AS latest_date
    FROM inventory
    GROUP BY product_id, store_id
)
SELECT
    i.product_id,
    p.product_name,
    p.category,
    i.store_id,
    st.store_name,
    st.city,
    i.quantity_on_hand,
    i.reorder_point,
    i.max_stock_level,
    i.warehouse_id,
    w.warehouse_name,
    w.capacity_m3,
    ROUND(i.quantity_on_hand * 1.0 / NULLIF(i.reorder_point, 0), 2) AS stock_coverage_ratio
FROM inventory i
JOIN latest_inventory li ON i.product_id = li.product_id AND i.store_id = li.store_id AND i.date = li.latest_date
JOIN products p ON i.product_id = p.product_id
JOIN stores st ON i.store_id = st.store_id
JOIN warehouses w ON i.warehouse_id = w.warehouse_id
ORDER BY i.quantity_on_hand ASC;

-- 5. Stockout Candidates
WITH latest_inventory AS (
    SELECT
        product_id,
        store_id,
        MAX(date) AS latest_date
    FROM inventory
    GROUP BY product_id, store_id
)
SELECT
    i.product_id,
    p.product_name,
    p.category,
    i.store_id,
    st.store_name,
    i.quantity_on_hand,
    i.reorder_point,
    i.quantity_on_hand - i.reorder_point AS shortage,
    ROUND(i.quantity_on_hand * 1.0 / NULLIF(i.reorder_point, 0), 2) AS stock_coverage_ratio,
    i.warehouse_id
FROM inventory i
JOIN latest_inventory li ON i.product_id = li.product_id AND i.store_id = li.store_id AND i.date = li.latest_date
JOIN products p ON i.product_id = p.product_id
JOIN stores st ON i.store_id = st.store_id
WHERE i.quantity_on_hand <= i.reorder_point
ORDER BY shortage ASC;

-- 6. Overstock / Excess Inventory Candidates
WITH latest_inventory AS (
    SELECT
        product_id,
        store_id,
        MAX(date) AS latest_date
    FROM inventory
    GROUP BY product_id, store_id
)
SELECT
    i.product_id,
    p.product_name,
    p.category,
    i.store_id,
    st.store_name,
    i.quantity_on_hand,
    i.max_stock_level,
    i.quantity_on_hand - i.max_stock_level AS excess,
    ROUND(i.quantity_on_hand * 1.0 / NULLIF(i.max_stock_level, 0), 2) AS stock_ratio
FROM inventory i
JOIN latest_inventory li ON i.product_id = li.product_id AND i.store_id = li.store_id AND i.date = li.latest_date
JOIN products p ON i.product_id = p.product_id
JOIN stores st ON i.store_id = st.store_id
WHERE i.quantity_on_hand > i.max_stock_level
ORDER BY excess DESC;

-- 7. Inventory Turnover (product-level, last 90 days)
SELECT
    p.product_id,
    p.product_name,
    p.category,
    SUM(s.quantity_sold) AS total_sold_90d,
    AVG(i.quantity_on_hand) AS avg_inventory_90d,
    ROUND(SUM(s.quantity_sold) * 1.0 / NULLIF(AVG(i.quantity_on_hand), 0), 2) AS turnover_ratio
FROM sales s
JOIN products p ON s.product_id = p.product_id
JOIN inventory i ON s.product_id = i.product_id AND s.store_id = i.store_id
    AND i.date BETWEEN DATE(s.date, '-90 days') AND s.date
WHERE s.date >= DATE('now', '-90 days')
GROUP BY p.product_id, p.product_name, p.category
ORDER BY turnover_ratio DESC;

-- 8. Supplier Performance
SELECT
    sup.supplier_id,
    sup.supplier_name,
    sup.country,
    sup.lead_time_days,
    sup.reliability_score,
    COUNT(DISTINCT p.product_id) AS products_supplied,
    SUM(s.quantity_sold) AS total_units_sold,
    SUM(s.revenue) AS total_revenue
FROM suppliers sup
JOIN products p ON sup.supplier_id = p.supplier_id
JOIN sales s ON p.product_id = s.product_id
GROUP BY sup.supplier_id, sup.supplier_name, sup.country, sup.lead_time_days, sup.reliability_score
ORDER BY total_revenue DESC;

-- 9. Warehouse Utilization
SELECT
    w.warehouse_id,
    w.warehouse_name,
    w.city,
    w.state,
    w.capacity_m3,
    COUNT(DISTINCT i.product_id) AS distinct_products,
    SUM(i.quantity_on_hand * p.volume_m3) AS occupied_volume_m3,
    ROUND(SUM(i.quantity_on_hand * p.volume_m3) * 1.0 / NULLIF(w.capacity_m3, 0), 4) AS utilization_ratio,
    ROUND(SUM(i.quantity_on_hand * p.volume_m3) * 1.0 / NULLIF(w.capacity_m3, 0) * 100, 2) AS utilization_pct
FROM warehouses w
JOIN inventory i ON w.warehouse_id = i.warehouse_id
JOIN products p ON i.product_id = p.product_id
WHERE i.date = (SELECT MAX(date) FROM inventory)
GROUP BY w.warehouse_id, w.warehouse_name, w.city, w.state, w.capacity_m3
ORDER BY utilization_pct DESC;

-- 10. Monthly Sales Trend
SELECT
    STRFTIME('%Y-%m', date) AS month,
    SUM(quantity_sold) AS total_quantity_sold,
    SUM(revenue) AS total_revenue,
    COUNT(DISTINCT product_id) AS products_sold,
    COUNT(DISTINCT store_id) AS active_stores,
    SUM(CASE WHEN promotion = 1 THEN revenue ELSE 0 END) AS promotional_revenue
FROM sales
GROUP BY STRFTIME('%Y-%m', date)
ORDER BY month;

-- 11. Category Performance
SELECT
    p.category,
    SUM(s.quantity_sold) AS total_quantity_sold,
    SUM(s.revenue) AS total_revenue,
    COUNT(DISTINCT p.product_id) AS product_count,
    AVG(s.unit_price) AS avg_unit_price,
    SUM(CASE WHEN s.promotion = 1 THEN s.quantity_sold ELSE 0 END) AS promotional_quantity
FROM sales s
JOIN products p ON s.product_id = p.product_id
GROUP BY p.category
ORDER BY total_revenue DESC;

-- 12. Stockout Frequency by Product (last 90 days)
SELECT
    p.product_id,
    p.product_name,
    p.category,
    COUNT(*) AS stockout_snapshots,
    MIN(i.quantity_on_hand) AS min_stock,
    MAX(i.quantity_on_hand) AS max_stock,
    AVG(i.quantity_on_hand) AS avg_stock
FROM inventory i
JOIN products p ON i.product_id = p.product_id
WHERE i.quantity_on_hand <= i.reorder_point
    AND i.date >= DATE('now', '-90 days')
GROUP BY p.product_id, p.product_name, p.category
ORDER BY stockout_snapshots DESC
LIMIT 20;
