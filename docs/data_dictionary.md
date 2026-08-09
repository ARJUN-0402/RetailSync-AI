# Data Dictionary

## Overview

RetailSync AI uses a synthetic retail dataset generated for portfolio and demonstration purposes.

**Source:** Synthetic data generated programmatically (`src/data/generate_dataset.py`).
**Rationale:** Real retail sales data is proprietary. Synthetic data allows full demonstration of the pipeline while maintaining realistic patterns.

**Date range:** 2023-08-11 to 2025-08-09 (730 days)
**Granularity:** Daily sales; weekly inventory snapshots

## Tables

### products
| Column | Type | Description |
|---|---|---|
| product_id | string | Unique product identifier (P001-P050) |
| product_name | string | Product display name |
| category | string | Product category (Electronics, Clothing, Groceries, Home Goods, Beauty, Toys) |
| subcategory | string | Subcategory (A, B, C) |
| unit_price | float | Retail selling price (USD) |
| cost_price | float | Supplier cost price (USD) |
| supplier_id | string | Foreign key to suppliers |
| weight_kg | float | Product weight in kilograms |
| volume_m3 | float | Product volume in cubic meters |
| launch_date | date | Product launch date |

### stores
| Column | Type | Description |
|---|---|---|
| store_id | string | Unique store identifier (ST01-ST10) |
| store_name | string | Store display name |
| city | string | Store city |
| state | string | Store state code |
| store_type | string | Urban, Suburban, or Rural |
| opening_date | date | Store opening date |

### suppliers
| Column | Type | Description |
|---|---|---|
| supplier_id | string | Unique supplier identifier (S01-S08) |
| supplier_name | string | Supplier display name |
| country | string | Supplier country |
| lead_time_days | int | Average lead time in days |
| reliability_score | float | Supplier reliability (0.7-0.99) |

### warehouses
| Column | Type | Description |
|---|---|---|
| warehouse_id | string | Unique warehouse identifier (WH01-WH05) |
| warehouse_name | string | Warehouse display name |
| city | string | Warehouse city |
| state | string | Warehouse state code |
| capacity_m3 | int | Storage capacity in cubic meters |
| supplier_id | string | Foreign key to suppliers |

### sales
| Column | Type | Description |
|---|---|---|
| date | date | Sale date |
| product_id | string | Foreign key to products |
| store_id | string | Foreign key to stores |
| quantity_sold | int | Units sold |
| unit_price | float | Unit selling price at time of sale |
| revenue | float | Total revenue (quantity_sold * unit_price) |
| promotion | int | 1 if promotional sale, 0 otherwise |

### inventory
| Column | Type | Description |
|---|---|---|
| date | date | Inventory snapshot date (weekly) |
| product_id | string | Foreign key to products |
| store_id | string | Foreign key to stores |
| quantity_on_hand | int | Current stock level |
| reorder_point | int | Reorder trigger threshold |
| max_stock_level | int | Maximum stock level |
| warehouse_id | string | Foreign key to warehouses |

## Data Quality Summary

- **Missing values:** None detected in raw data
- **Duplicates:** None detected after deduplication by natural keys
- **Date validity:** All dates valid
- **Outliers:** Present in quantity_sold, revenue, and inventory levels (expected in retail data; retained for analysis)

## Assumptions

1. Sales are generated at daily granularity for a subset of products per store.
2. Inventory snapshots are taken weekly.
3. Promotions are randomly assigned with ~15% probability per sale.
4. Seasonal effects are embedded for Electronics (holiday season) and Clothing (summer).
5. Warehouse capacity is randomly assigned; no real-world warehouse data is used.
