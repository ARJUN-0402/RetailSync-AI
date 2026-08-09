from sqlalchemy import create_engine, text

engine = create_engine("sqlite:///database/retailsync.db")
with engine.connect() as conn:
    queries = {
        "daily_sales": "SELECT date, SUM(quantity_sold), SUM(revenue) FROM sales GROUP BY date ORDER BY date DESC LIMIT 3",
        "product_performance": "SELECT p.product_id, p.category, SUM(s.quantity_sold), SUM(s.revenue) FROM sales s JOIN products p ON s.product_id = p.product_id GROUP BY p.product_id ORDER BY SUM(s.revenue) DESC LIMIT 3",
        "store_performance": "SELECT st.store_id, st.city, SUM(s.revenue) FROM sales s JOIN stores st ON s.store_id = st.store_id GROUP BY st.store_id ORDER BY SUM(s.revenue) DESC LIMIT 3",
        "stockout_candidates": "WITH latest_inventory AS (SELECT product_id, store_id, MAX(date) AS latest_date FROM inventory GROUP BY product_id, store_id) SELECT i.product_id, i.store_id, i.quantity_on_hand, i.reorder_point FROM inventory i JOIN latest_inventory li ON i.product_id = li.product_id AND i.store_id = li.store_id AND i.date = li.latest_date WHERE i.quantity_on_hand <= i.reorder_point LIMIT 5",
        "warehouse_utilization": "SELECT w.warehouse_id, w.capacity_m3, SUM(i.quantity_on_hand * p.volume_m3) AS occupied FROM warehouses w JOIN inventory i ON w.warehouse_id = i.warehouse_id JOIN products p ON i.product_id = p.product_id WHERE i.date = (SELECT MAX(date) FROM inventory) GROUP BY w.warehouse_id LIMIT 3",
    }
    for name, sql in queries.items():
        result = conn.execute(text(sql)).fetchall()
        print(f"\n{name}:")
        for row in result:
            print(f"  {row}")
