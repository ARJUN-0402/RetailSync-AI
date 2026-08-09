import pandas as pd
import os
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

DATABASE_PATH = "database/retailsync.db"
PROCESSED_DIR = "data/processed"

TABLES = {
    "products": "products.csv",
    "stores": "stores.csv",
    "suppliers": "suppliers.csv",
    "warehouses": "warehouses.csv",
    "sales": "sales.csv",
    "inventory": "inventory.csv",
}

def create_database():
    if os.path.exists(DATABASE_PATH):
        os.remove(DATABASE_PATH)
        print(f"Removed existing database: {DATABASE_PATH}")

    engine = create_engine(f"sqlite:///{DATABASE_PATH}")
    with engine.connect() as conn:
        with open("database/schema.sql", "r", encoding="utf-8") as f:
            schema_sql = f.read()
        statements = [s.strip() for s in schema_sql.split(";") if s.strip()]
        for stmt in statements:
            conn.execute(text(stmt))
        conn.commit()
        print("Schema created successfully.")
    return engine

def load_data(engine):
    for table_name, filename in TABLES.items():
        filepath = os.path.join(PROCESSED_DIR, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing processed file: {filepath}")
        df = pd.read_csv(filepath)
        df.to_sql(table_name, con=engine, if_exists="append", index=False)
        print(f"Loaded {len(df)} rows into {table_name}")

def validate_relationships(engine):
    queries = {
        "sales_product_fk": """
            SELECT COUNT(*) as orphan_sales
            FROM sales s
            LEFT JOIN products p ON s.product_id = p.product_id
            WHERE p.product_id IS NULL
        """,
        "sales_store_fk": """
            SELECT COUNT(*) as orphan_sales
            FROM sales s
            LEFT JOIN stores st ON s.store_id = st.store_id
            WHERE st.store_id IS NULL
        """,
        "inventory_product_fk": """
            SELECT COUNT(*) as orphan_inventory
            FROM inventory i
            LEFT JOIN products p ON i.product_id = p.product_id
            WHERE p.product_id IS NULL
        """,
        "inventory_store_fk": """
            SELECT COUNT(*) as orphan_inventory
            FROM inventory i
            LEFT JOIN stores st ON i.store_id = st.store_id
            WHERE st.store_id IS NULL
        """,
        "inventory_warehouse_fk": """
            SELECT COUNT(*) as orphan_inventory
            FROM inventory i
            LEFT JOIN warehouses w ON i.warehouse_id = w.warehouse_id
            WHERE w.warehouse_id IS NULL
        """,
        "warehouses_supplier_fk": """
            SELECT COUNT(*) as orphan_warehouses
            FROM warehouses w
            LEFT JOIN suppliers s ON w.supplier_id = s.supplier_id
            WHERE s.supplier_id IS NULL
        """,
        "products_supplier_fk": """
            SELECT COUNT(*) as orphan_products
            FROM products p
            LEFT JOIN suppliers s ON p.supplier_id = s.supplier_id
            WHERE s.supplier_id IS NULL
        """,
    }
    with engine.connect() as conn:
        for name, sql in queries.items():
            result = conn.execute(text(sql)).fetchone()
            orphan_count = result[0]
            status = "OK" if orphan_count == 0 else f"FAIL ({orphan_count} orphans)"
            print(f"  {name}: {status}")

def run_sample_analytics(engine):
    print("\n--- Sample Analytics ---")
    queries = {
        "daily_sales": """
            SELECT date, SUM(quantity_sold) as total_qty, SUM(revenue) as total_revenue
            FROM sales
            GROUP BY date
            ORDER BY date DESC
            LIMIT 5
        """,
        "product_performance": """
            SELECT p.product_id, p.category, SUM(s.quantity_sold) as total_qty, SUM(s.revenue) as total_revenue
            FROM sales s
            JOIN products p ON s.product_id = p.product_id
            GROUP BY p.product_id, p.category
            ORDER BY total_revenue DESC
            LIMIT 5
        """,
        "store_performance": """
            SELECT st.store_id, st.city, st.store_type, SUM(s.revenue) as total_revenue
            FROM sales s
            JOIN stores st ON s.store_id = st.store_id
            GROUP BY st.store_id, st.city, st.store_type
            ORDER BY total_revenue DESC
            LIMIT 5
        """,
        "inventory_levels": """
            SELECT i.product_id, i.store_id, i.quantity_on_hand, i.reorder_point
            FROM inventory i
            WHERE i.date = (SELECT MAX(date) FROM inventory)
            ORDER BY i.quantity_on_hand ASC
            LIMIT 5
        """,
        "stockout_candidates": """
            SELECT i.product_id, i.store_id, i.quantity_on_hand, i.reorder_point
            FROM inventory i
            WHERE i.date = (SELECT MAX(date) FROM inventory)
              AND i.quantity_on_hand <= i.reorder_point
            LIMIT 5
        """,
    }
    with engine.connect() as conn:
        for name, sql in queries.items():
            result = conn.execute(text(sql)).fetchall()
            print(f"\n{name}:")
            for row in result:
                print(f"  {row}")

def main():
    print("=== RetailSync AI Database Initialization ===")
    engine = create_database()
    print("\nLoading processed data into database...")
    load_data(engine)
    print("\nValidating referential integrity...")
    validate_relationships(engine)
    run_sample_analytics(engine)
    print("\nDatabase initialization complete.")

if __name__ == "__main__":
    main()
