"""Regression tests for logging formatting and database schema initialization.

These tests guard against the two CI blockers:

1. The custom logging layer must not mutate ``record.args`` into strings,
   otherwise ``%d``/``%f`` interpolation raises ``TypeError``.
2. The canonical schema (``database/schema.sql``) must create every analytics
   table (``inventory_alerts`` and friends) so a clean CI database is usable by
   every pipeline stage.
"""

from __future__ import annotations

import logging
import os
import sqlite3

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from src.database import init_db
from src.utils import logging as logging_module
from src.utils.logging import redact_text, sanitize_log_args


class _ListHandler(logging.Handler):
    """Handler that records formatted output instead of emitting it."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        # Run the formatter (which applies secret redaction) on each record.
        self.records.append(self.format(record))


@pytest.fixture
def capture_handler():
    """Attach a capturing handler using the secret-redacting formatter."""
    logger = logging.getLogger("retailsync_regression")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False
    handler = _ListHandler()
    handler.setFormatter(logging_module.SecretRedactingFormatter("%(message)s"))
    logger.addHandler(handler)
    return logger, handler


def _emitted(handler) -> str:
    return "".join(handler.records)


# ---------------------------------------------------------------------------
# Logging formatting regression tests
# ---------------------------------------------------------------------------


def test_logging_int_percent_d(capture_handler):
    logger, handler = capture_handler
    logger.info("count=%d", 68)
    assert "count=68" in _emitted(handler)


def test_logging_float_percent_f(capture_handler):
    logger, handler = capture_handler
    logger.info("duration=%.1fs", 19.6)
    assert "duration=19.6s" in _emitted(handler)


def test_logging_string_percent_s(capture_handler):
    logger, handler = capture_handler
    logger.info("name=%s", "RandomForest")
    assert "name=RandomForest" in _emitted(handler)


def test_logging_multiple_args(capture_handler):
    logger, handler = capture_handler
    logger.info("multi %d %s %.2f", 1, "two", 3.14159)
    assert "multi 1 two 3.14" in _emitted(handler)


def test_logging_none_arg(capture_handler):
    logger, handler = capture_handler
    logger.info("none=%s", None)
    assert "none=None" in _emitted(handler)


def test_logging_no_args(capture_handler):
    logger, handler = capture_handler
    logger.info("no args at all")
    assert "no args at all" in _emitted(handler)


def test_logging_literal_percent_is_preserved(capture_handler):
    logger, handler = capture_handler
    logger.info("High utilization (>80%%): %d", 3)
    assert "High utilization (>80%): 3" in _emitted(handler)


def test_logging_levels(capture_handler):
    logger, handler = capture_handler
    logger.warning("warn %d", 7)
    logger.error("err %.2f", 1.5)
    logger.debug("dbg %s", "x")
    out = _emitted(handler)
    assert "warn 7" in out and "err 1.50" in out and "dbg x" in out


def test_logging_exception(capture_handler):
    logger, handler = capture_handler
    try:
        raise ValueError("boom")
    except ValueError:
        logger.exception("exception with arg %d", 42)
    out = _emitted(handler)
    assert "exception with arg 42" in out
    assert "ValueError" in out


def test_logging_preserves_numeric_types():
    """Numeric args must stay numeric until % interpolation occurs."""
    sanitized = sanitize_log_args((68, 19.6, "RandomForest", None, True))
    assert sanitized[0] == 68
    assert isinstance(sanitized[0], int)
    assert sanitized[1] == 19.6
    assert isinstance(sanitized[1], float)
    assert sanitized[3] is None
    assert sanitized[4] is True


def test_logging_numpy_types(capture_handler):
    logger, handler = capture_handler
    logger.info("numpy int %d", np.int64(12))
    logger.info("numpy float %.1f", np.float64(3.456))
    out = _emitted(handler)
    assert "numpy int 12" in out
    assert "numpy float 3.5" in out


def test_logging_redacts_key_value_secret(capture_handler):
    logger, handler = capture_handler
    logger.info("api_key=%s", "sk-super-secret-value")
    assert "sk-super-secret-value" not in _emitted(handler)
    assert "api_key=[REDACTED]" in _emitted(handler)


def test_logging_redacts_bearer_token(capture_handler):
    logger, handler = capture_handler
    logger.info("Authorization: Bearer %s", "abc123token")
    out = _emitted(handler)
    assert "abc123token" not in out
    assert "Authorization: [REDACTED]" in out


def test_logging_does_not_redact_benign_text(capture_handler):
    logger, handler = capture_handler
    logger.info("Primary key product_id is fine")
    assert "Primary key product_id is fine" in _emitted(handler)


def test_redact_text_helper():
    assert redact_text("api_key=sk-abc and count=5") == "api_key=[REDACTED] and count=5"
    assert redact_text("Primary key product_id is fine") == "Primary key product_id is fine"


# ---------------------------------------------------------------------------
# Database schema regression tests (run against a clean temporary database)
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_db(tmp_path) -> str:
    """Create a fresh, data-loaded database from the canonical schema."""
    db_file = str(tmp_path / "retailsync_test.db")
    _use_temp_db(db_file)
    init_db.main()
    return db_file


EXPECTED_TABLES = [
    "products",
    "stores",
    "suppliers",
    "warehouses",
    "sales",
    "inventory",
    "inventory_alerts",
    "anomaly_flags",
    "product_segments",
    "store_segments",
    "warehouse_segments",
    "warehouse_optimization",
]


def test_clean_db_has_all_expected_tables(clean_db):
    conn = sqlite3.connect(clean_db)
    tables = {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    conn.close()
    missing = [t for t in EXPECTED_TABLES if t not in tables]
    assert not missing, f"Missing tables: {missing}"


def test_inventory_alerts_columns_match_usage(clean_db):
    conn = sqlite3.connect(clean_db)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(inventory_alerts)")}
    conn.close()
    required = {
        "alert_id",
        "product_id",
        "store_id",
        "warehouse_id",
        "alert_date",
        "alert_type",
        "risk_level",
        "reason",
        "quantity_on_hand",
        "reorder_point",
        "max_stock_level",
        "stock_coverage_days",
        "forecast_demand_7d",
        "recommended_action",
    }
    assert required.issubset(cols), f"Missing columns: {required - cols}"


def test_inventory_alerts_crud(clean_db):
    engine = create_engine(f"sqlite:///{clean_db}")
    with engine.connect() as conn:
        conn.execute(
            text(
                "DELETE FROM inventory_alerts WHERE alert_date = :date"
            ),
            {"date": "2025-01-01"},
        )
        conn.commit()

        df = pd.DataFrame(
            [
                {
                    "product_id": "P001",
                    "store_id": "S001",
                    "warehouse_id": "W001",
                    "alert_date": "2025-01-01",
                    "alert_type": "Stockout Risk",
                    "risk_level": "HIGH",
                    "reason": "Out of stock",
                    "quantity_on_hand": 0.0,
                    "reorder_point": 10.0,
                    "max_stock_level": 50.0,
                    "stock_coverage_days": 12.5,
                    "forecast_demand_7d": 30.0,
                    "recommended_action": "Immediate reorder required",
                }
            ]
        )
        df.to_sql("inventory_alerts", con=engine, if_exists="append", index=False)

        count = conn.execute(
            text("SELECT COUNT(*) FROM inventory_alerts")
        ).fetchone()[0]
        assert count == 1
        row = conn.execute(
            text("SELECT product_id, risk_level FROM inventory_alerts")
        ).fetchone()
        assert row[0] == "P001" and row[1] == "HIGH"


def test_anomaly_flags_columns_match_usage(clean_db):
    engine = create_engine(f"sqlite:///{clean_db}")
    df = pd.DataFrame(
        [
            {
                "date": "2025-01-01",
                "product_id": "P001",
                "store_id": "S001",
                "z_score": 3.2,
                "anomaly_type": "Demand Spike",
                "detection_methods": "Z-score, IQR",
                "quantity_sold": 120.0,
            }
        ]
    )
    df.to_sql("anomaly_flags", con=engine, if_exists="append", index=False)
    with engine.connect() as conn:
        cols = {r[1] for r in conn.execute(text("PRAGMA table_info(anomaly_flags)"))}
    assert {"detection_methods", "z_score", "anomaly_type"}.issubset(cols)


def _use_temp_db(db_file: str) -> None:
    """Point the shared ``settings.database`` at a temp file in-place.

    We mutate the already-imported ``settings`` object rather than reloading the
    module, so other tests keep seeing a consistent config. ``DATABASE_PATH`` is
    also set so freshly imported copies derive the same path.
    """
    os.environ["DATABASE_PATH"] = db_file
    from src.config import settings

    settings.database.path = db_file


def test_segments_persist_to_db(tmp_path):
    """segmentation.py must write its segment tables to the database."""
    from src.clustering import segmentation

    db_file = str(tmp_path / "retailsync_seg.db")
    _use_temp_db(db_file)
    init_db.main()

    features = pd.read_csv("data/processed/features_daily.csv", parse_dates=["date"])
    engine = create_engine(f"sqlite:///{db_file}")
    inventory_df = pd.read_sql("SELECT * FROM inventory", engine)
    inventory_df["date"] = pd.to_datetime(inventory_df["date"])

    segmentation.segment_products(features)
    segmentation.segment_stores(features)
    segmentation.segment_warehouses(features, inventory_df)

    with engine.connect() as conn:
        for table in ["product_segments", "store_segments", "warehouse_segments"]:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).fetchone()[0]
            assert count > 0, f"{table} was not persisted"


def test_load_alerts_against_clean_db(tmp_path):
    """load_alerts.load() should insert rows into a freshly initialized DB."""
    from src.inventory import load_alerts

    db_file = str(tmp_path / "retailsync_alerts.db")
    _use_temp_db(db_file)
    init_db.main()

    load_alerts.load()

    engine = create_engine(f"sqlite:///{db_file}")
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM inventory_alerts")).fetchone()[0]
    assert count > 0, "load_alerts did not insert any rows"
