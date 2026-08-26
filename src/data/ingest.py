"""Data ingestion and cleaning for RetailSync AI."""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

from src.config import settings
from src.exceptions import DataError
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

REQUIRED_SCHEMA = {
    "products": [
        "product_id",
        "product_name",
        "category",
        "unit_price",
        "cost_price",
        "supplier_id",
    ],
    "stores": ["store_id", "store_name", "city", "state", "store_type"],
    "suppliers": ["supplier_id", "supplier_name", "country", "lead_time_days"],
    "warehouses": [
        "warehouse_id",
        "warehouse_name",
        "city",
        "state",
        "capacity_m3",
        "supplier_id",
    ],
    "sales": [
        "date",
        "product_id",
        "store_id",
        "quantity_sold",
        "unit_price",
        "discount_pct",
        "revenue",
        "promotion",
    ],
    "inventory": [
        "date",
        "product_id",
        "store_id",
        "quantity_on_hand",
        "reorder_point",
        "max_stock_level",
        "warehouse_id",
    ],
}


def validate_schema(df: pd.DataFrame, name: str) -> None:
    """Validate that a DataFrame has the required schema columns."""
    required = REQUIRED_SCHEMA.get(name, [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataError(f"{name} missing required columns: {missing}")
    logger.info("Schema valid for %s", name)


def handle_missing(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Detect and fill missing values."""
    missing_counts = df.isnull().sum()
    if missing_counts.any():
        logger.warning(
            "Missing values in %s:\n%s", name, missing_counts[missing_counts > 0]
        )
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype in [np.float64, np.int64]:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(
                        df[col].mode().iloc[0]
                        if not df[col].mode().empty
                        else "Unknown"
                    )
        logger.info("Filled missing values in %s", name)
    else:
        logger.info("No missing values in %s", name)
    return df


def handle_duplicates(df: pd.DataFrame, name: str, subset: list | None = None) -> pd.DataFrame:
    """Detect and remove duplicate rows."""
    dup_count = df.duplicated(subset=subset).sum()
    if dup_count:
        logger.warning("Found %d duplicates in %s", dup_count, name)
        df = df.drop_duplicates(subset=subset)
        logger.info("Removed duplicates from %s", name)
    else:
        logger.info("No duplicates in %s", name)
    return df


def validate_dates(df: pd.DataFrame, name: str, date_col: str = "date") -> pd.DataFrame:
    """Validate and coerce date columns."""
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        invalid = df[date_col].isnull().sum()
        if invalid:
            logger.warning("Invalid dates in %s: %d rows", name, invalid)
            df = df.dropna(subset=[date_col])
        logger.info("Dates validated in %s", name)
    return df


def check_outliers(df: pd.DataFrame, name: str, method: str = "iqr") -> pd.DataFrame:
    """Log outlier counts for numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_counts: dict[str, int] = {}
    for col in numeric_cols:
        if col in ["product_id", "store_id", "supplier_id", "warehouse_id"]:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        if not outliers.empty:
            outlier_counts[col] = len(outliers)
            logger.info("Outliers detected in %s.%s: %d records", name, col, len(outliers))
    if not outlier_counts:
        logger.info("No significant outliers in %s", name)
    return df


def clean_table(name: str) -> pd.DataFrame:
    """Clean a single raw data table."""
    logger.info("Cleaning %s...", name)
    raw_path = os.path.join(str(settings.paths.raw_data), f"{name}.csv")
    if not os.path.exists(raw_path):
        raise DataError(f"Raw file not found: {raw_path}")
    df = pd.read_csv(raw_path)
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
    elif name in ("sales", "inventory"):
        subset = ["date", "product_id", "store_id"]
    df = handle_duplicates(df, name, subset=subset)
    df = validate_dates(df, name)
    df = check_outliers(df, name)
    return df


def main() -> None:
    """Main entry point for data ingestion."""
    setup_logging(__name__)
    logger.info("=== RetailSync AI - Data Ingestion ===")
    tables = ["products", "stores", "suppliers", "warehouses", "sales", "inventory"]
    for table in tables:
        cleaned = clean_table(table)
        output_path = os.path.join(str(settings.paths.processed_data), f"{table}.csv")
        cleaned.to_csv(output_path, index=False)
        logger.info("Saved cleaned %s to %s (rows: %d)", table, output_path, len(cleaned))
    logger.info("Data ingestion and cleaning complete.")


if __name__ == "__main__":
    main()
