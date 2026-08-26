"""Database initialization and management for RetailSync AI."""

from __future__ import annotations

import logging
import os

import pandas as pd
from sqlalchemy import create_engine, text

from src.config import settings
from src.exceptions import DatabaseError

logger = logging.getLogger(__name__)

TABLES = {
    "products": "products.csv",
    "stores": "stores.csv",
    "suppliers": "suppliers.csv",
    "warehouses": "warehouses.csv",
    "sales": "sales.csv",
    "inventory": "inventory.csv",
}


def create_database(db_path: str | None = None) -> str:
    """Create the SQLite database schema.

    Args:
        db_path: Optional database path override.

    Returns:
        Database URL string.
    """
    path = db_path or settings.database.path
    logger.info("Creating database at %s", path)

    if os.path.exists(path):
        os.remove(path)
        logger.info("Removed existing database: %s", path)

    engine = create_engine(f"sqlite:///{path}")
    schema_path = os.path.join(str(settings.paths.database), "schema.sql")
    with engine.connect() as conn:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_sql = f.read()
        statements = [s.strip() for s in schema_sql.split(";") if s.strip()]
        for stmt in statements:
            conn.execute(text(stmt))
        conn.commit()
    logger.info("Schema created successfully")
    return f"sqlite:///{path}"


def load_data(engine, processed_dir: str | None = None) -> None:
    """Load processed CSV data into the database.

    Args:
        engine: SQLAlchemy engine.
        processed_dir: Optional processed data directory override.
    """
    directory = processed_dir or str(settings.paths.processed_data)
    for table_name, filename in TABLES.items():
        filepath = os.path.join(directory, filename)
        if not os.path.exists(filepath):
            raise DatabaseError(f"Missing processed file: {filepath}")
        df = pd.read_csv(filepath)
        df.to_sql(table_name, con=engine, if_exists="append", index=False)
        logger.info("Loaded %d rows into %s", len(df), table_name)


def validate_relationships(engine) -> None:
    """Validate referential integrity of the database."""
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
            logger.info("  %s: %s", name, status)


def run_sample_analytics(engine) -> None:
    """Run sample analytics queries for validation."""
    logger.info("Running sample analytics...")
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
            logger.info("Sample query %s returned %d rows", name, len(result))


def main() -> None:
    """Main entry point for database initialization."""
    from src.utils.logging import setup_logging
    setup_logging(__name__)

    logger.info("=== RetailSync AI Database Initialization ===")
    engine = create_engine(create_database())
    logger.info("Loading processed data into database...")
    load_data(engine)
    logger.info("Validating referential integrity...")
    validate_relationships(engine)
    run_sample_analytics(engine)
    logger.info("Database initialization complete.")


if __name__ == "__main__":
    main()
