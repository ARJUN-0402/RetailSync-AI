import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

REQUIRED_SCHEMA = {
    "products": ["product_id", "product_name", "category", "unit_price", "cost_price", "supplier_id"],
    "stores": ["store_id", "store_name", "city", "state", "store_type"],
    "suppliers": ["supplier_id", "supplier_name", "country", "lead_time_days"],
    "warehouses": ["warehouse_id", "warehouse_name", "city", "state", "capacity_m3", "supplier_id"],
    "sales": ["date", "product_id", "store_id", "quantity_sold", "revenue", "promotion"],
    "inventory": ["date", "product_id", "store_id", "quantity_on_hand", "reorder_point", "max_stock_level", "warehouse_id"],
}

def validate_schema(df, name):
    required = REQUIRED_SCHEMA.get(name, [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")
    logger.info(f"Schema valid for {name}")

def handle_missing(df, name):
    missing_counts = df.isnull().sum()
    if missing_counts.any():
        logger.warning(f"Missing values in {name}:\n{missing_counts[missing_counts > 0]}")
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in [np.float64, np.int64]:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown")
        logger.info(f"Filled missing values in {name}")
    else:
        logger.info(f"No missing values in {name}")
    return df

def handle_duplicates(df, name, subset=None):
    dup_count = df.duplicated(subset=subset).sum()
    if dup_count:
        logger.warning(f"Found {dup_count} duplicates in {name}")
        df = df.drop_duplicates(subset=subset)
        logger.info(f"Removed duplicates from {name}")
    else:
        logger.info(f"No duplicates in {name}")
    return df

def validate_dates(df, name, date_col="date"):
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        invalid = df[date_col].isnull().sum()
        if invalid:
            logger.warning(f"Invalid dates in {name}: {invalid} rows")
            df = df.dropna(subset=[date_col])
        logger.info(f"Dates validated in {name}")
    return df

def check_outliers(df, name, method="iqr"):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_counts = {}
    for col in numeric_cols:
        if col in ["product_id", "store_id", "supplier_id", "warehouse_id"]:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        if not outliers.empty:
            outlier_counts[col] = len(outliers)
            logger.info(f"Outliers detected in {name}.{col}: {len(outliers)} records")
    if not outlier_counts:
        logger.info(f"No significant outliers in {name}")
    return df

def clean_table(name):
    logger.info(f"Cleaning {name}...")
    df = pd.read_csv(os.path.join(RAW_DIR, f"{name}.csv"))
    validate_schema(df, name)
    df = handle_missing(df, name)
    subset = None
    if name == "products":
        subset = ["product_id"]
    elif name == "stores":
        subset = ["store_id"]
    elif name == "suppliers":
        subset = ["supplier_id"]
    elif name == "warehouses":
        subset = ["warehouse_id"]
    elif name == "sales":
        subset = ["date", "product_id", "store_id"]
    elif name == "inventory":
        subset = ["date", "product_id", "store_id"]
    df = handle_duplicates(df, name, subset=subset)
    df = validate_dates(df, name)
    df = check_outliers(df, name)
    return df

def main():
    tables = ["products", "stores", "suppliers", "warehouses", "sales", "inventory"]
    for table in tables:
        cleaned = clean_table(table)
        output_path = os.path.join(PROCESSED_DIR, f"{table}.csv")
        cleaned.to_csv(output_path, index=False)
        logger.info(f"Saved cleaned {table} to {output_path} (rows: {len(cleaned)})")
    logger.info("Data ingestion and cleaning complete.")

if __name__ == "__main__":
    main()
