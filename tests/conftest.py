"""Shared pytest configuration and fixtures for RetailSync AI tests.

Database-backed tests must never read a developer's local
``database/retailsync.db``: ``src/database/init_db.py`` recreates that file from
scratch, so its analytics tables (``inventory_alerts``, ``anomaly_flags``,
``*_segments``, ``warehouse_optimization``) are empty until the later pipeline
stages are re-run. Any test asserting on those rows is therefore asserting on
whichever pipeline command a developer happened to run last.

:func:`build_pipeline_database` removes that dependency. It creates an isolated
temporary database from the canonical ``database/schema.sql`` and populates it
by running the real (cheap) pipeline stages over a small deterministic dataset.
Model training and the 14-day forecast stage are deliberately skipped: no
analytics table depends on them, so the fixture stays fast.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

# The canonical schema. It is copied (never re-written) into the temporary
# database directory because init_db reads "<settings.paths.database>/schema.sql".
SCHEMA_FILE = os.path.join(PROJECT_ROOT, "database", "schema.sql")

# Deterministic fixture dataset. Small enough to stay fast, big enough for the
# pipeline's windows: 30-day rolling anomaly stats, 28-day variability windows,
# 14-day target horizons, and clustering with k up to 5 (needs >= 6 entities).
FIXTURE_START_DATE = "2024-01-01"
FIXTURE_DAYS = 120
FIXTURE_PRODUCTS = 8
FIXTURE_STORES = 6
FIXTURE_WAREHOUSES = 3
FIXTURE_SUPPLIERS = 3

# Days (offsets from FIXTURE_START_DATE) that carry a deliberate demand spike so
# anomaly detection has something real to flag. Kept inside the window that
# survives feature engineering's 14-day target shift.
SPIKE_DAYS = (60, 85)


@dataclass(frozen=True)
class PipelineArtifacts:
    """Locations of one isolated, fully populated pipeline run."""

    db_path: str
    processed_dir: str


def _product_id(idx: int) -> str:
    return f"P{idx + 1:03d}"


def _store_id(idx: int) -> str:
    return f"ST{idx + 1:02d}"


def _warehouse_id(idx: int) -> str:
    return f"WH{idx + 1:02d}"


def _supplier_id(idx: int) -> str:
    return f"SUP{idx + 1:03d}"


def _demand(product_idx: int, store_idx: int, day: int) -> int:
    """Deterministic daily demand (no RNG, so results are reproducible)."""
    base = 20 + 3 * product_idx + 5 * store_idx
    wobble = ((day % 7) - 3) + (((product_idx * 5 + store_idx * 3 + day * 11) % 7) - 3)
    quantity = max(base + wobble, 1)
    if day in SPIKE_DAYS and (product_idx + store_idx) % 2 == 0:
        quantity *= 9
    return quantity


def _stock_levels(product_idx: int, store_idx: int) -> tuple[int, int]:
    reorder_point = 20 + 5 * ((product_idx + store_idx) % 3)
    return reorder_point, reorder_point * 6


def _source_frames() -> dict[str, pd.DataFrame]:
    """Build the six source tables the database is loaded from.

    The frames intentionally mirror ``database/schema.sql`` column-for-column so
    ``init_db.load_data`` can append them unchanged.
    """
    dates = pd.date_range(FIXTURE_START_DATE, periods=FIXTURE_DAYS, freq="D")

    suppliers = pd.DataFrame(
        [
            {
                "supplier_id": _supplier_id(k),
                "supplier_name": f"Fixture Supplier {k + 1}",
                "country": ["USA", "India", "Germany"][k % 3],
                "lead_time_days": 5 + 2 * k,
                "reliability_score": round(0.80 + 0.05 * k, 2),
            }
            for k in range(FIXTURE_SUPPLIERS)
        ]
    )

    products = pd.DataFrame(
        [
            {
                "product_id": _product_id(i),
                "product_name": f"Fixture Product {i + 1}",
                "category": ["Electronics", "Grocery", "Apparel", "Home"][i % 4],
                "subcategory": f"Fixture Sub {i % 4 + 1}",
                "unit_price": round(10.0 + 5.0 * i, 2),
                "cost_price": round((10.0 + 5.0 * i) * 0.6, 2),
                "supplier_id": _supplier_id(i % FIXTURE_SUPPLIERS),
                "weight_kg": round(0.5 + 0.25 * i, 2),
                "volume_m3": round(0.01 + 0.005 * i, 3),
                "launch_date": "2023-01-01",
            }
            for i in range(FIXTURE_PRODUCTS)
        ]
    )

    stores = pd.DataFrame(
        [
            {
                "store_id": _store_id(j),
                "store_name": f"Fixture Store {j + 1}",
                "city": f"City {j + 1}",
                "state": ["CA", "NY", "TX"][j % 3],
                "store_type": ["Flagship", "Standard", "Express"][j % 3],
                "opening_date": "2022-06-01",
            }
            for j in range(FIXTURE_STORES)
        ]
    )

    warehouses = pd.DataFrame(
        [
            {
                "warehouse_id": _warehouse_id(w),
                "warehouse_name": f"Fixture Warehouse {w + 1}",
                "city": f"Hub {w + 1}",
                "state": ["CA", "NY", "TX"][w % 3],
                "capacity_m3": 400 + 150 * w,
                "supplier_id": _supplier_id(w % FIXTURE_SUPPLIERS),
            }
            for w in range(FIXTURE_WAREHOUSES)
        ]
    )

    unit_prices = dict(zip(products["product_id"], products["unit_price"]))

    sales_rows = []
    for day, date in enumerate(dates):
        promotion = 1 if day % 14 == 0 else 0
        discount = 0.10 if promotion else 0.0
        for i in range(FIXTURE_PRODUCTS):
            price = unit_prices[_product_id(i)]
            for j in range(FIXTURE_STORES):
                quantity = _demand(i, j, day)
                sales_rows.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "product_id": _product_id(i),
                        "store_id": _store_id(j),
                        "quantity_sold": quantity,
                        "unit_price": price,
                        "discount_pct": discount,
                        "promotion": promotion,
                        "revenue": round(quantity * price * (1 - discount), 2),
                    }
                )

    # Weekly inventory snapshots. The final snapshot deliberately covers every
    # risk bucket that load_alerts.py turns into an alert row (stockout,
    # overstock, urgent reorder) so the alert stage has real work to do.
    snapshot_days = list(range(0, FIXTURE_DAYS, 7))
    if snapshot_days[-1] != FIXTURE_DAYS - 1:
        snapshot_days.append(FIXTURE_DAYS - 1)
    last_day = snapshot_days[-1]

    inventory_rows = []
    for day in snapshot_days:
        for i in range(FIXTURE_PRODUCTS):
            for j in range(FIXTURE_STORES):
                reorder_point, max_stock_level = _stock_levels(i, j)
                if day == last_day:
                    bucket = (i * FIXTURE_STORES + j) % 4
                    if bucket == 0:
                        quantity_on_hand = 0
                    elif bucket == 1:
                        quantity_on_hand = reorder_point // 2
                    elif bucket == 2:
                        quantity_on_hand = int(max_stock_level * 1.6)
                    else:
                        quantity_on_hand = int(max_stock_level * 0.5)
                else:
                    quantity_on_hand = int(max_stock_level * 0.7) - (day % 3) * 2
                inventory_rows.append(
                    {
                        "date": dates[day].strftime("%Y-%m-%d"),
                        "product_id": _product_id(i),
                        "store_id": _store_id(j),
                        "quantity_on_hand": quantity_on_hand,
                        "reorder_point": reorder_point,
                        "max_stock_level": max_stock_level,
                        "warehouse_id": _warehouse_id((i + j) % FIXTURE_WAREHOUSES),
                    }
                )

    return {
        "products": products,
        "stores": stores,
        "suppliers": suppliers,
        "warehouses": warehouses,
        "sales": pd.DataFrame(sales_rows),
        "inventory": pd.DataFrame(inventory_rows),
    }


def _write_source_csvs(directory: Path) -> None:
    """Write the deterministic source tables as CSVs init_db can load."""
    for name, frame in _source_frames().items():
        frame.to_csv(directory / f"{name}.csv", index=False)


def build_pipeline_database(root: Path) -> PipelineArtifacts:
    """Create and populate an isolated analytics database under ``root``.

    Every table required by the pipeline tests is filled by the production code
    paths themselves (no hand-written analytics rows), running against
    temporary database/processed/models directories so neither the real
    database nor the real model artifacts are touched.
    """
    from src.anomaly import anomaly_detection
    from src.clustering import segmentation, warehouse_optimization
    from src.config import settings
    from src.database import init_db
    from src.features import feature_engineering
    from src.inventory import inventory_intelligence, load_alerts

    db_dir = root / "database"
    processed_dir = root / "processed"
    # Clusterer artifacts must land here instead of the real models/ directory.
    models_dir = root / "models"
    for directory in (db_dir, processed_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(SCHEMA_FILE, db_dir / "schema.sql")
    db_file = db_dir / "retailsync.db"

    _write_source_csvs(processed_dir)

    patch = pytest.MonkeyPatch()
    try:
        # Pipeline modules resolve the database either through
        # settings.database.path or through settings.paths.database; redirect
        # both, plus the processed-data and models directories.
        patch.setenv("DATABASE_PATH", str(db_file))
        patch.setattr(settings.database, "path", str(db_file))
        patch.setattr(settings.database, "url", f"sqlite:///{db_file}")
        patch.setattr(settings.paths, "database", db_dir)
        patch.setattr(settings.paths, "processed_data", processed_dir)
        patch.setattr(settings.paths, "models", models_dir)

        engine = create_engine(init_db.create_database())
        init_db.load_data(engine, str(processed_dir))

        features = feature_engineering.engineer()
        inventory_intelligence.detect_risks()
        load_alerts.load()
        anomaly_detection.detect()

        inventory = pd.read_sql("SELECT * FROM inventory", engine)
        inventory["date"] = pd.to_datetime(inventory["date"])
        segmentation.segment_products(features)
        segmentation.segment_stores(features)
        segmentation.segment_warehouses(features, inventory)
        warehouse_optimization.analyze()
    finally:
        patch.undo()

    return PipelineArtifacts(
        db_path=str(db_file),
        processed_dir=str(processed_dir),
    )


@pytest.fixture(scope="session")
def pipeline_db(tmp_path_factory) -> PipelineArtifacts:
    """Deterministic analytics database shared by the database-backed tests.

    Treat it as read-only: it is built once per session and shared, so a test
    that inserts, updates or deletes rows would make the other consumers
    order-dependent. A test that needs to mutate a database must call
    :func:`build_pipeline_database` with its own ``tmp_path`` instead.
    """
    return build_pipeline_database(tmp_path_factory.mktemp("pipeline"))


@pytest.fixture(scope="session")
def source_data_dir(tmp_path_factory) -> str:
    """Directory of deterministic source CSVs (products ... inventory).

    Tests that only need a freshly initialized, data-loaded database can load
    these instead of the developer's ``data/processed`` exports.
    """
    directory = tmp_path_factory.mktemp("source")
    _write_source_csvs(directory)
    return str(directory)
