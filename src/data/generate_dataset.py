"""Synthetic retail dataset generation for RetailSync AI."""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

from src.config import settings
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

np.random.seed(42)

NUM_PRODUCTS = 50
NUM_STORES = 10
NUM_SUPPLIERS = 8
NUM_WAREHOUSES = 5
DAYS = 730

end_date = date(2025, 8, 9)
start_date = pd.Timestamp(end_date) - timedelta(days=DAYS - 1)
dates = pd.date_range(start=start_date, end=end_date, freq="D")

CATEGORIES = ["Electronics", "Clothing", "Groceries", "Home Goods", "Beauty", "Toys"]
STORE_TYPES = ["Urban", "Suburban", "Rural"]

CATEGORY_PROPERTIES = {
    "Electronics": {"base_demand": 7.0, "price_elasticity": 1.6, "promo_lift": 1.35, "holiday_lift": 1.4},
    "Clothing": {"base_demand": 5.5, "price_elasticity": 2.0, "promo_lift": 1.45, "holiday_lift": 1.25},
    "Groceries": {"base_demand": 10.0, "price_elasticity": 1.1, "promo_lift": 1.15, "holiday_lift": 1.05},
    "Home Goods": {"base_demand": 3.5, "price_elasticity": 1.4, "promo_lift": 1.25, "holiday_lift": 1.15},
    "Beauty": {"base_demand": 2.8, "price_elasticity": 2.1, "promo_lift": 1.55, "holiday_lift": 1.35},
    "Toys": {"base_demand": 4.0, "price_elasticity": 1.7, "promo_lift": 1.4, "holiday_lift": 1.7},
}

STORE_PROPERTIES = {
    "Urban": {"multiplier": 1.25},
    "Suburban": {"multiplier": 1.0},
    "Rural": {"multiplier": 0.7},
}

HOLIDAYS = {
    (1, 1): 1.25,
    (1, 15): 1.08,
    (2, 14): 1.35,
    (5, 27): 1.15,
    (7, 4): 1.25,
    (9, 2): 1.12,
    (11, 11): 1.15,
    (11, 28): 1.55,
    (11, 29): 1.75,
    (12, 24): 1.35,
    (12, 25): 1.45,
}


def generate() -> None:
    """Generate synthetic retail data and save to raw data directory."""
    settings.paths.ensure_dirs()
    output_dir = str(settings.paths.raw_data)
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Generating synthetic retail dataset...")

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

    products = pd.DataFrame(
        {
            "product_id": [f"P{str(i).zfill(3)}" for i in range(1, NUM_PRODUCTS + 1)],
            "product_name": [f"Product {i}" for i in range(1, NUM_PRODUCTS + 1)],
            "category": [CATEGORIES[i % len(CATEGORIES)] for i in range(NUM_PRODUCTS)],
            "subcategory": [["A", "B", "C"][i % 3] for i in range(NUM_PRODUCTS)],
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
            "city": [
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
            "state": ["NY", "CA", "IL", "TX", "AZ", "PA", "TX", "CA", "TX", "CA"],
            "store_type": [STORE_TYPES[i % len(STORE_TYPES)] for i in range(NUM_STORES)],
            "opening_date": np.random.choice(dates[: DAYS // 2], NUM_STORES),
        }
    )

    idx = pd.MultiIndex.from_product(
        [dates, stores["store_id"].values, products["product_id"].values],
        names=["date", "store_id", "product_id"],
    )
    sales = pd.DataFrame(index=idx).reset_index()
    sales = sales.merge(stores[["store_id", "store_type"]], on="store_id", how="left")
    sales = sales.merge(
        products[["product_id", "category", "unit_price", "cost_price"]], on="product_id", how="left"
    )

    dt = sales["date"]
    month = dt.dt.month
    day_of_year = dt.dt.dayofyear
    day_of_week = dt.dt.dayofweek

    cat_props = sales["category"].map(CATEGORY_PROPERTIES)
    base_demand = cat_props.map(lambda x: x["base_demand"])
    elasticity = cat_props.map(lambda x: x["price_elasticity"])
    promo_lift = cat_props.map(lambda x: x["promo_lift"])
    store_mult = sales["store_type"].map(STORE_PROPERTIES).map(lambda x: x["multiplier"])

    yearly_factor = 1 + 0.12 * np.sin(2 * np.pi * day_of_year / 365)
    weekly_factor = 1.0 + 0.08 * np.sin(2 * np.pi * day_of_week / 7)
    monthly_factor = 1 + 0.08 * np.sin(2 * np.pi * month / 12)
    days_since_start = (dt - pd.Timestamp(start_date)).dt.days
    trend_factor = 1 + days_since_start / (DAYS * 2.5)

    holiday_factor = sales.apply(
        lambda row: HOLIDAYS.get((row["date"].month, row["date"].day), 1.0), axis=1
    )

    category_monthly = monthly_factor.copy()
    electronics_mask = (sales["category"] == "Electronics") & (month.isin([11, 12]))
    clothing_mask = (sales["category"] == "Clothing") & (month.isin([6, 7, 8]))
    toys_mask = (sales["category"] == "Toys") & (month == 12)
    groceries_mask = (sales["category"] == "Groceries") & (month == 11)
    category_monthly[electronics_mask] *= 1.3
    category_monthly[clothing_mask] *= 1.2
    category_monthly[toys_mask] *= 1.35
    category_monthly[groceries_mask] *= 1.08

    promotion = np.random.choice([0, 1], size=len(sales), p=[0.82, 0.18])
    promo_factor = np.where(promotion == 1, promo_lift, 1.0)

    discount_pct = np.random.choice(
        [0, 0.10, 0.15, 0.20, 0.25], size=len(sales), p=[0.55, 0.18, 0.12, 0.10, 0.05]
    )
    discount_factor = 1 + elasticity * discount_pct

    product_noise = np.random.uniform(0.75, 1.25, size=len(sales))
    spike_mask = np.random.random(len(sales)) < 0.04
    spike_factor = np.where(spike_mask, np.random.uniform(3.0, 5.5, size=len(sales)), 1.0)

    lambda_demand = (
        base_demand
        * store_mult
        * weekly_factor
        * category_monthly
        * yearly_factor
        * holiday_factor
        * promo_factor
        * discount_factor
        * trend_factor
        * product_noise
        * spike_factor
    )

    lambda_demand = np.maximum(0.1, lambda_demand)
    quantity_sold = np.random.poisson(lambda_demand)

    sales["quantity_sold"] = quantity_sold
    sales["promotion"] = promotion
    sales["discount_pct"] = discount_pct
    sales["revenue"] = np.round(sales["quantity_sold"] * sales["unit_price"] * (1 - sales["discount_pct"]), 2)

    sales = sales[
        [
            "date",
            "product_id",
            "store_id",
            "quantity_sold",
            "unit_price",
            "discount_pct",
            "promotion",
            "revenue",
        ]
    ]

    logger.info("Generated %s sales records", f"{len(sales):,}")
    logger.info("Zero demand pct: %.2f%%", (sales["quantity_sold"] == 0).mean() * 100)

    inventory_dates = dates[::7]
    inventory_rows = []
    for dt in inventory_dates:
        for _, store in stores.iterrows():
            for _, product in products.iterrows():
                hist = sales[
                    (sales["product_id"] == product["product_id"])
                    & (sales["store_id"] == store["store_id"])
                    & (sales["date"] < dt)
                ]
                avg_daily_demand = float(hist["quantity_sold"].mean()) if len(hist) > 0 else 1.0
                base_stock = max(20, int(avg_daily_demand * 14 + np.random.uniform(10, 60)))
                inventory_rows.append(
                    {
                        "date": dt,
                        "product_id": product["product_id"],
                        "store_id": store["store_id"],
                        "quantity_on_hand": max(0, int(np.random.normal(base_stock, base_stock * 0.4))),
                        "reorder_point": max(5, int(avg_daily_demand * 5 + np.random.uniform(5, 20))),
                        "max_stock_level": base_stock,
                        "warehouse_id": np.random.choice(warehouses["warehouse_id"]),
                    }
                )

    inventory = pd.DataFrame(inventory_rows)
    logger.info("Generated %s inventory records", f"{len(inventory):,}")

    products.to_csv(os.path.join(output_dir, "products.csv"), index=False)
    stores.to_csv(os.path.join(output_dir, "stores.csv"), index=False)
    suppliers.to_csv(os.path.join(output_dir, "suppliers.csv"), index=False)
    warehouses.to_csv(os.path.join(output_dir, "warehouses.csv"), index=False)
    sales.to_csv(os.path.join(output_dir, "sales.csv"), index=False)
    inventory.to_csv(os.path.join(output_dir, "inventory.csv"), index=False)

    logger.info("Generated raw data in %s/", output_dir)
    logger.info("Products: %d", len(products))
    logger.info("Stores: %d", len(stores))
    logger.info("Suppliers: %d", len(suppliers))
    logger.info("Warehouses: %d", len(warehouses))
    logger.info("Sales records: %d", len(sales))
    logger.info("Inventory records: %d", len(inventory))
    logger.info("Zero demand pct: %.2f%%", (sales["quantity_sold"] == 0).mean() * 100)
    logger.info("Avg daily demand per product-store: %.2f", sales["quantity_sold"].mean())


def main() -> None:
    """Main entry point."""
    setup_logging(__name__)
    logger.info("=== RetailSync AI - Dataset Generation ===")
    generate()
    logger.info("Dataset generation complete.")


if __name__ == "__main__":
    main()
