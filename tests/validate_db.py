from sqlalchemy import create_engine, text

engine = create_engine("sqlite:///database/retailsync.db")
with engine.connect() as conn:
    tables = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()
    print("Tables:", [r[0] for r in tables])
    for table in ["products", "stores", "suppliers", "warehouses", "sales", "inventory"]:
        count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).fetchone()[0]
        print(f"{table} rows: {count}")
    sales_idx = conn.execute(text("PRAGMA index_list(sales)")).fetchall()
    print("Sales indexes:", [r[1] for r in sales_idx])
    inv_idx = conn.execute(text("PRAGMA index_list(inventory)")).fetchall()
    print("Inventory indexes:", [r[1] for r in inv_idx])
