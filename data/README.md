# Data Directory

## Structure

- `raw/` — Original, immutable data files
- `processed/` — Cleaned and transformed data ready for analysis

## Guidelines

1. Never modify files in `raw/`. Treat them as immutable.
2. All data processing pipelines should output to `processed/`.
3. Document data sources and transformations in notebooks.

## Current Dataset

- **Type:** Synthetic retail data
- **Generator:** `src/data/generate_dataset.py`
- **Ingestion:** `src/data/ingest.py`
- **Tables:** products, stores, suppliers, warehouses, sales, inventory
- **Schema:** See `docs/data_dictionary.md`

## Data Pipeline

```
generate_dataset.py → data/raw/*.csv → ingest.py → data/processed/*.csv
```
