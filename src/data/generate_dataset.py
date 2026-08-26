import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

np.random.seed(42)

OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_PRODUCTS = 50
NUM_STORES = 10
NUM_SUPPLIERS = 8
NUM_WAREHOUSES = 5
DAYS = 730  # ~2 years of daily data

end_date = date(2025, 8, 9)
start_date = pd.Timestamp(end_date) - timedelta(days=DAYS - 1)
dates = pd.date_range(start=start_date, end=end_date, freq="D")

CATEGORIES = ["Electronics", "Clothing", "Groceries", "Home Goods", "Beauty", "Toys"]
SEASONS = ["Winter", "Spring", "Summer", "Fall"]

products = pd.DataFrame(
    {
        "product_id": [f"P{str(i).zfill(3)}" for i in range(1, NUM_PRODUCTS + 1)],
        "product_name": [f"Product {i}" for i in range(1, NUM_PRODUCTS + 1)],
        "category": np.random.choice(CATEGORIES, NUM_PRODUCTS),
        "subcategory": np.random.choice(["A", "B", "C"], NUM_PRODUCTS),
        "unit_price": np.round(np.random.uniform(5, 500, NUM_PRODUCTS), 2),
        "cost_price": np.round(np.random.uniform(3, 300, NUM_PRODUCTS), 2),
        "supplier_id": np.random.choice(
            [f"S{str(i).zfill(2)}" for i in range(1, NUM_SUPPLIERS + 1)], NUM_PRODUCTS
        ),
        "weight_kg": np.round(np.random.uniform(0.1, 20, NUM_PRODUCTS), 2),
        "volume_m3": np.round(np.random.uniform(0.01, 0.5, NUM_PRODUCTS), 3),
        "launch_date": np.random.choice(dates, NUM_PRODUCTS),
    }
)

stores = pd.DataFrame(
    {
        "store_id": [f"ST{str(i).zfill(2)}" for i in range(1, NUM_STORES + 1)],
        "store_name": [f"Store {i}" for i in range(1, NUM_STORES + 1)],
        "city": np.random.choice(
            [
                "New York",
                "Los Angeles",
                "Chicago",
                "Houston",
                "Phoenix",
                "Philadelphia",
                "San Antonio",
                "San Diego",
                "Dallas",
                "San Jose",
            ],
            NUM_STORES,
        ),
        "state": np.random.choice(
            ["NY", "CA", "IL", "TX", "AZ", "PA", "TX", "CA", "TX", "CA"], NUM_STORES
        ),
        "store_type": np.random.choice(["Urban", "Suburban", "Rural"], NUM_STORES),
        "opening_date": np.random.choice(dates[: DAYS // 2], NUM_STORES),
    }
)

suppliers = pd.DataFrame(
    {
        "supplier_id": [f"S{str(i).zfill(2)}" for i in range(1, NUM_SUPPLIERS + 1)],
        "supplier_name": [f"Supplier {i}" for i in range(1, NUM_SUPPLIERS + 1)],
        "country": np.random.choice(
            ["USA", "China", "Germany", "Japan", "Mexico", "Vietnam", "India", "UK"],
            NUM_SUPPLIERS,
        ),
        "lead_time_days": np.random.randint(1, 30, NUM_SUPPLIERS),
        "reliability_score": np.round(np.random.uniform(0.7, 0.99, NUM_SUPPLIERS), 2),
    }
)

warehouses = pd.DataFrame(
    {
        "warehouse_id": [f"WH{str(i).zfill(2)}" for i in range(1, NUM_WAREHOUSES + 1)],
        "warehouse_name": [f"Warehouse {i}" for i in range(1, NUM_WAREHOUSES + 1)],
        "city": np.random.choice(
            ["Dallas", "Chicago", "Los Angeles", "Atlanta", "Seattle"], NUM_WAREHOUSES
        ),
        "state": np.random.choice(["TX", "IL", "CA", "GA", "WA"], NUM_WAREHOUSES),
        "capacity_m3": np.random.randint(5000, 20000, NUM_WAREHOUSES),
        "supplier_id": np.random.choice(
            [f"S{str(i).zfill(2)}" for i in range(1, NUM_SUPPLIERS + 1)], NUM_WAREHOUSES
        ),
    }
)

# Generate sales with realistic patterns
sales_rows = []
for dt in dates:
    month = dt.month
    season = (
        "Winter"
        if month in [12, 1, 2]
        else "Spring"
        if month in [3, 4, 5]
        else "Summer"
        if month in [6, 7, 8]
        else "Fall"
    )
    for store_id in stores["store_id"]:
        n_products = np.random.randint(5, 15)
        sampled_products = np.random.choice(
            products["product_id"], n_products, replace=False
        )
        for product_id in sampled_products:
            product = products[products["product_id"] == product_id].iloc[0]
            base_demand = np.random.randint(1, 20)
            seasonal_factor = (
                1.3
                if product["category"] in ["Electronics"] and month in [11, 12]
                else 1.2
                if product["category"] == "Clothing" and month in [6, 7, 8]
                else 1.0
            )
            trend_factor = 1 + (dt - start_date).days / (DAYS * 2)
            quantity = max(
                0, int(np.random.poisson(base_demand * seasonal_factor * trend_factor))
            )
            revenue = round(quantity * product["unit_price"], 2)
            sales_rows.append(
                {
                    "date": dt,
                    "product_id": product_id,
                    "store_id": store_id,
                    "quantity_sold": quantity,
                    "unit_price": product["unit_price"],
                    "revenue": revenue,
                    "promotion": np.random.choice([0, 1], p=[0.85, 0.15]),
                }
            )

sales = pd.DataFrame(sales_rows)

# Generate inventory snapshots (weekly)
inventory_dates = dates[::7]
inventory_rows = []
for dt in inventory_dates:
    for store_id in stores["store_id"]:
        for product_id in products["product_id"]:
            product = products[products["product_id"] == product_id].iloc[0]
            base_stock = np.random.randint(20, 200)
            inventory_rows.append(
                {
                    "date": dt,
                    "product_id": product_id,
                    "store_id": store_id,
                    "quantity_on_hand": np.random.randint(0, base_stock),
                    "reorder_point": np.random.randint(10, 50),
                    "max_stock_level": base_stock,
                    "warehouse_id": np.random.choice(warehouses["warehouse_id"]),
                }
            )

inventory = pd.DataFrame(inventory_rows)

# Save raw data
products.to_csv(os.path.join(OUTPUT_DIR, "products.csv"), index=False)
stores.to_csv(os.path.join(OUTPUT_DIR, "stores.csv"), index=False)
suppliers.to_csv(os.path.join(OUTPUT_DIR, "suppliers.csv"), index=False)
warehouses.to_csv(os.path.join(OUTPUT_DIR, "warehouses.csv"), index=False)
sales.to_csv(os.path.join(OUTPUT_DIR, "sales.csv"), index=False)
inventory.to_csv(os.path.join(OUTPUT_DIR, "inventory.csv"), index=False)

print(f"Generated raw data in {OUTPUT_DIR}/")
print(f"Products: {len(products)}")
print(f"Stores: {len(stores)}")
print(f"Suppliers: {len(suppliers)}")
print(f"Warehouses: {len(warehouses)}")
print(f"Sales records: {len(sales)}")
print(f"Inventory records: {len(inventory)}")
