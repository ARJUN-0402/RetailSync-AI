"""
Test suite for RetailSync AI pipeline.

Database-backed tests use the ``pipeline_db`` fixture (see tests/conftest.py),
which builds a temporary database from the canonical ``database/schema.sql`` and
populates it by running the real pipeline stages over a small deterministic
dataset. They must not read the local ``database/retailsync.db``, whose
analytics tables are empty whenever database initialization ran more recently
than the alert/anomaly/segmentation stages.

Run with:
    python -m pytest tests/test_pipeline.py -v
    # or
    python tests/test_pipeline.py
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import joblib
import pandas as pd
from sqlalchemy import create_engine, text

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

REQUIRED_FILES = [
    "data/processed/features_daily.csv",
    "data/processed/forecasts_next_14d.csv",
    "data/processed/inventory_intelligence.csv",
    "data/processed/anomalies.csv",
    "data/processed/product_segments.csv",
    "data/processed/store_segments.csv",
    "data/processed/warehouse_segments.csv",
    "data/processed/warehouse_optimization.csv",
    "models/demand_forecaster.pkl",
    "models/product_clusterer.pkl",
    "models/store_clusterer.pkl",
    "models/warehouse_clusterer.pkl",
]

REQUIRED_TABLES = [
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

CRITICAL_FEATURE_COLS = ["product_id", "store_id", "date", "target_demand_1d"]


class TestResults:
    """Track test results for the standalone runner (backward compatibility)."""

    __test__ = False  # Prevent pytest from collecting this class

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def record(self, test_name, passed, message=""):
        if passed:
            self.passed += 1
            print(f"  [PASS] {test_name}")
        else:
            self.failed += 1
            self.errors.append(f"{test_name}: {message}")
            print(f"  [FAIL] {test_name}: {message}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'=' * 60}")
        print(f"TEST SUMMARY: {self.passed}/{total} passed")
        if self.errors:
            print("\nFAILED TESTS:")
            for error in self.errors:
                print(f"  - {error}")
        print(f"{'=' * 60}")
        return self.failed == 0


# ============================================================
# TEST SUITE
# ============================================================


def test_data_files_exist():
    """Test that all required data files exist and are non-empty."""
    print("\n=== Testing Data Files ===")
    for file_path in REQUIRED_FILES:
        full_path = os.path.join(PROJECT_ROOT, file_path)
        assert os.path.exists(full_path), f"File does not exist: {file_path}"
        size = os.path.getsize(full_path)
        assert size > 0, f"File is empty: {file_path} (size: {size} bytes)"
        print(f"  [PASS] {file_path} ({size:,} bytes)")


def test_database_tables(pipeline_db):
    """Test that database tables exist and have data.

    Runs against the deterministic fixture database so the row counts reflect an
    actual pipeline run instead of whatever state a local database happens to be
    left in.
    """
    print("\n=== Testing Database Tables ===")
    db_path = pipeline_db.db_path
    assert os.path.exists(db_path), f"Database not found at {db_path}"

    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        tables = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table'")
        ).fetchall()
        table_names = [r[0] for r in tables]

        for table in REQUIRED_TABLES:
            assert table in table_names, f"Table does not exist: {table}"
            count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).fetchone()[0]
            assert count > 0, f"Table has no data: {table} (count: {count})"
            print(f"  [PASS] Table {table}: {count} rows")


def test_feature_engineering():
    """Test feature engineering output."""
    print("\n=== Testing Feature Engineering ===")
    features_path = os.path.join(PROCESSED_DIR, "features_daily.csv")
    assert os.path.exists(features_path), f"Features file missing: {features_path}"

    df = pd.read_csv(features_path, parse_dates=["date"])

    assert len(df) > 0, "Features DataFrame is empty"
    print(f"  [PASS] Features not empty ({len(df):,} rows)")

    for col in ["date", "product_id", "store_id", "target_demand_1d"]:
        assert col in df.columns, f"Missing column: {col}"
        print(f"  [PASS] Has {col} column")

    assert any("lag" in c for c in df.columns), "Missing lag features"
    print("  [PASS] Has lag features")

    assert any("rolling" in c for c in df.columns), "Missing rolling features"
    print("  [PASS] Has rolling features")

    assert any(c in df.columns for c in ["day_of_week", "month", "year"]), (
        "Missing time features"
    )
    print("  [PASS] Has time features")

    for col in CRITICAL_FEATURE_COLS:
        nan_count = df[col].isna().sum()
        assert nan_count == 0, f"NaN values in {col}: {nan_count}"
        print(f"  [PASS] No NaN in {col}")


def test_forecasting():
    """Test forecasting model and outputs."""
    print("\n=== Testing Forecasting ===")
    model_path = os.path.join(MODELS_DIR, "demand_forecaster.pkl")
    assert os.path.exists(model_path), f"Model file missing: {model_path}"

    model_package = joblib.load(model_path)
    assert "model" in model_package, "Model package missing 'model' key"
    assert "feature_cols" in model_package, "Model package missing 'feature_cols' key"
    assert "metrics" in model_package, "Model package missing 'metrics' key"
    assert "model_name" in model_package, "Model package missing 'model_name' key"
    print("  [PASS] Model package has all required keys")
    print(f"  [PASS] Model name: {model_package['model_name']}")
    print(f"  [PASS] Model type: {type(model_package['model']).__name__}")

    # Verify the model has a predict method (critical for forecast_pipeline.py)
    assert hasattr(model_package["model"], "predict"), (
        "Model does not have predict() method"
    )
    print("  [PASS] Model has predict() method")

    # Verify model_name matches the actual model type (no mismatch)
    model_name = model_package["model_name"]
    model_type = type(model_package["model"]).__name__
    if "Baseline" in model_name:
        assert (
            "Predictor" in model_type
            or "Regressor" not in model_type
            or model_type == "BaselineMeanPredictor"
        ), f"Model name '{model_name}' does not match model type '{model_type}'"
        print(
            f"  [PASS] Model name '{model_name}' is consistent with model type '{model_type}'"
        )

    forecasts_path = os.path.join(PROCESSED_DIR, "forecasts_next_14d.csv")
    assert os.path.exists(forecasts_path), f"Forecasts file missing: {forecasts_path}"
    forecasts = pd.read_csv(forecasts_path, parse_dates=["date"])
    assert len(forecasts) > 0, "Forecasts DataFrame is empty"
    assert "forecast_demand" in forecasts.columns, "Missing forecast_demand column"
    assert (forecasts["forecast_demand"] >= 0).all(), "Forecasts must be non-negative"
    print(f"  [PASS] Forecasts: {len(forecasts)} rows, all non-negative")


def test_inventory_intelligence():
    """Test inventory intelligence outputs."""
    print("\n=== Testing Inventory Intelligence ===")
    inv_path = os.path.join(PROCESSED_DIR, "inventory_intelligence.csv")
    assert os.path.exists(inv_path), f"Inventory file missing: {inv_path}"

    df = pd.read_csv(inv_path)
    assert len(df) > 0, "Inventory intelligence DataFrame is empty"
    print(f"  [PASS] Inventory data: {len(df)} rows")

    for col in [
        "stockout_risk",
        "overstock_risk",
        "reorder_urgency",
        "composite_risk_score",
    ]:
        assert col in df.columns, f"Missing column: {col}"
        print(f"  [PASS] Has {col} column")

    assert df["stockout_risk"].isin(["HIGH", "MEDIUM", "LOW"]).all(), (
        "Invalid stockout_risk values"
    )
    print("  [PASS] Valid stockout risk values")

    assert df["overstock_risk"].isin(["HIGH", "MEDIUM", "LOW"]).all(), (
        "Invalid overstock_risk values"
    )
    print("  [PASS] Valid overstock risk values")


def test_anomaly_detection(pipeline_db):
    """Test anomaly detection outputs.

    Uses the fixture pipeline run, so both the CSV output and the
    ``anomaly_flags`` rows come from the same deterministic execution of
    ``src/anomaly/anomaly_detection.py``.
    """
    print("\n=== Testing Anomaly Detection ===")
    anomalies_path = os.path.join(pipeline_db.processed_dir, "anomalies.csv")
    assert os.path.exists(anomalies_path), f"Anomalies file missing: {anomalies_path}"

    df = pd.read_csv(anomalies_path, parse_dates=["date"])
    assert len(df) > 0, "Anomalies DataFrame is empty"
    print(f"  [PASS] Anomalies: {len(df)} rows")

    for col in ["anomaly_type", "z_score"]:
        assert col in df.columns, f"Missing column: {col}"
        print(f"  [PASS] Has {col} column")

    valid_types = ["Demand Spike", "Demand Drop", "Unusual Pattern"]
    assert df["anomaly_type"].isin(valid_types).all(), "Invalid anomaly types"
    print("  [PASS] Valid anomaly types")

    db_path = pipeline_db.db_path
    assert os.path.exists(db_path), f"Database not found at {db_path}"
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM anomaly_flags")).fetchone()[0]
        assert count > 0, f"No anomaly flags in database (count: {count})"
        assert count == len(df), (
            f"anomaly_flags rows ({count}) do not match anomalies.csv ({len(df)})"
        )
        print(f"  [PASS] Anomaly flags in DB: {count}")


def test_clustering():
    """Test clustering/segmentation outputs."""
    print("\n=== Testing Clustering ===")

    # Product segments
    prod_path = os.path.join(PROCESSED_DIR, "product_segments.csv")
    assert os.path.exists(prod_path), f"Product segments missing: {prod_path}"
    df = pd.read_csv(prod_path)
    assert len(df) > 0, "Product segments DataFrame is empty"
    assert "cluster" in df.columns or "product_cluster" in df.columns, (
        "Missing cluster column in product segments"
    )
    # Products should be exactly 50
    assert len(df) == 50, f"Expected 50 products, got {len(df)}"
    print(f"  [PASS] Product segments: {len(df)} products")

    # Store segments
    store_path = os.path.join(PROCESSED_DIR, "store_segments.csv")
    assert os.path.exists(store_path), f"Store segments missing: {store_path}"
    df = pd.read_csv(store_path)
    assert len(df) > 0, "Store segments DataFrame is empty"
    assert len(df) == 10, f"Expected 10 stores, got {len(df)}"
    print(f"  [PASS] Store segments: {len(df)} stores")

    # Warehouse segments
    wh_path = os.path.join(PROCESSED_DIR, "warehouse_segments.csv")
    assert os.path.exists(wh_path), f"Warehouse segments missing: {wh_path}"
    df = pd.read_csv(wh_path)
    assert len(df) > 0, "Warehouse segments DataFrame is empty"
    assert len(df) == 5, f"Expected 5 warehouses, got {len(df)}"
    print(f"  [PASS] Warehouse segments: {len(df)} warehouses")

    # Model files
    for model_name in ["product_clusterer", "store_clusterer", "warehouse_clusterer"]:
        model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
        assert os.path.exists(model_path), f"Clusterer model missing: {model_name}"
        model = joblib.load(model_path)
        assert "model" in model, f"Missing 'model' key in {model_name}"
        assert "scaler" in model, f"Missing 'scaler' key in {model_name}"
        assert "features" in model, f"Missing 'features' key in {model_name}"
        print(f"  [PASS] {model_name}: {type(model['model']).__name__}")


def test_pipeline_integration():
    """Test end-to-end pipeline integration."""
    print("\n=== Testing Pipeline Integration ===")

    base = os.path.join(PROCESSED_DIR, "features_daily.csv")
    assert os.path.exists(base), f"Features file missing: {base}"

    forecasts_path = os.path.join(PROCESSED_DIR, "forecasts_next_14d.csv")
    assert os.path.exists(forecasts_path), f"Forecasts file missing: {forecasts_path}"

    inv_path = os.path.join(PROCESSED_DIR, "inventory_intelligence.csv")
    assert os.path.exists(inv_path), f"Inventory file missing: {inv_path}"

    anomalies_path = os.path.join(PROCESSED_DIR, "anomalies.csv")
    assert os.path.exists(anomalies_path), f"Anomalies file missing: {anomalies_path}"

    features = pd.read_csv(base, parse_dates=["date"])
    forecasts = pd.read_csv(forecasts_path, parse_dates=["date"])
    pd.read_csv(inv_path)
    pd.read_csv(anomalies_path, parse_dates=["date"])

    print("  [PASS] All main outputs loadable")

    features_date_range = (features["date"].min(), features["date"].max())
    forecasts_date_range = (forecasts["date"].min(), forecasts["date"].max())

    assert features_date_range[0] < features_date_range[1], (
        "Invalid features date range"
    )
    print("  [PASS] Features date range valid")

    assert forecasts_date_range[0] >= features_date_range[1], (
        "Forecasts should be future dates"
    )
    print("  [PASS] Forecasts are future dates")


def test_dashboard_artifacts():
    """Test dashboard can be built."""
    print("\n=== Testing Dashboard ===")
    dashboard_path = os.path.join(PROJECT_ROOT, "dashboard", "app.py")
    assert os.path.exists(dashboard_path), f"Dashboard app missing: {dashboard_path}"

    with open(dashboard_path, "r", encoding="utf-8") as f:
        code = f.read()

    assert "import streamlit" in code, "Dashboard missing streamlit import"
    print("  [PASS] Dashboard has imports")

    assert "load_data" in code, "Dashboard missing load_data function"
    print("  [PASS] Dashboard has load_data")

    assert "page =" in code or "radio" in code, "Dashboard missing page navigation"
    print("  [PASS] Dashboard has pages")


# ============================================================
# STANDALONE RUNNER (backward compatibility)
# ============================================================


def run_all_tests():
    """Run all tests using the custom runner (for backward compatibility).

    Wraps each test function in a try/except to report pass/fail
    the same way the original TestResults-based runner did. The database-backed
    tests need the same deterministic fixture pytest builds, so it is created
    here in a throwaway temporary directory.
    """
    print("=" * 60)
    print("RETAILSYNC AI - TEST SUITE")
    print("=" * 60)

    sys.path.insert(0, PROJECT_ROOT)
    from tests.conftest import build_pipeline_database

    # SQLite connection pools keep the fixture database file open on Windows, so
    # the temporary directory is removed best-effort instead of strictly.
    tmp_dir = tempfile.mkdtemp(prefix="retailsync_tests_")
    try:
        pipeline_db = build_pipeline_database(Path(tmp_dir))

        test_funcs = [
            ("test_data_files_exist", test_data_files_exist),
            ("test_database_tables", lambda: test_database_tables(pipeline_db)),
            ("test_feature_engineering", test_feature_engineering),
            ("test_forecasting", test_forecasting),
            ("test_inventory_intelligence", test_inventory_intelligence),
            ("test_anomaly_detection", lambda: test_anomaly_detection(pipeline_db)),
            ("test_clustering", test_clustering),
            ("test_pipeline_integration", test_pipeline_integration),
            ("test_dashboard_artifacts", test_dashboard_artifacts),
        ]

        passed = 0
        failed = 0
        errors = []

        for func_name, func in test_funcs:
            try:
                func()
                passed += 1
            except (
                AssertionError,
                RuntimeError,
                ValueError,
                KeyError,
                AttributeError,
            ) as e:
                failed += 1
                errors.append(f"{func_name}: ERROR - {e}")
                print(f"  [ERROR] {func_name}: {e}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"TEST SUMMARY: {passed}/{total} test functions passed")
    if errors:
        print("\nFAILED TESTS:")
        for error in errors:
            print(f"  - {error}")
    print(f"{'=' * 60}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
