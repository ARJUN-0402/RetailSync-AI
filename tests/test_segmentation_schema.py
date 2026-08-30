"""Regression tests for segmentation schema contract."""

import os

import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
DB_PATH = os.path.join(PROJECT_ROOT, "database", "retailsync.db")

SEGMENT_FILES = {
    "product": "product_segments.csv",
    "store": "store_segments.csv",
    "warehouse": "warehouse_segments.csv",
}

SEGMENT_ID_COLS = {
    "product": "product_id",
    "store": "store_id",
    "warehouse": "warehouse_id",
}

CANONICAL_CLUSTER_COL = "cluster_label"
CANONICAL_CLUSTER_ID_COL = "cluster"


def _load_segment(name: str) -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, SEGMENT_FILES[name])
    assert os.path.exists(path), f"Missing segment file: {path}"
    return pd.read_csv(path)


@pytest.mark.parametrize("name", ["product", "store", "warehouse"])
def test_segment_dataframe_has_canonical_columns(name):
    df = _load_segment(name)
    assert CANONICAL_CLUSTER_COL in df.columns, f"{name}: missing {CANONICAL_CLUSTER_COL}"
    assert CANONICAL_CLUSTER_ID_COL in df.columns, f"{name}: missing {CANONICAL_CLUSTER_ID_COL}"
    assert SEGMENT_ID_COLS[name] in df.columns, f"{name}: missing {SEGMENT_ID_COLS[name]}"


@pytest.mark.parametrize("name", ["product", "store", "warehouse"])
def test_segment_label_counts_no_keyerror(name):
    df = _load_segment(name)
    counts = df[CANONICAL_CLUSTER_COL].value_counts().reset_index()
    assert len(counts) > 0, f"{name}: label counts are empty"
    assert "cluster" not in counts.columns or counts.columns[0] != "cluster" or len(counts) > 0


@pytest.mark.parametrize("name", ["product", "store", "warehouse"])
def test_segment_dashboard_render_operations(name):
    df = _load_segment(name)
    label_counts = df[CANONICAL_CLUSTER_COL].value_counts().reset_index()
    label_counts.columns = ["cluster", "count"]
    assert len(label_counts) > 0

    if name == "product":
        assert "total_revenue" in df.columns
        assert "demand_cv" in df.columns
    else:
        assert "total_revenue" in df.columns or "total_quantity" in df.columns


def test_product_segmentation_uses_canonical_cluster_label():
    df = _load_segment("product")
    assert CANONICAL_CLUSTER_COL in df.columns
    counts = df[CANONICAL_CLUSTER_COL].value_counts().reset_index()
    counts.columns = ["cluster", "count"]
    assert len(counts) > 0


def test_store_segmentation_uses_canonical_cluster_label():
    df = _load_segment("store")
    assert CANONICAL_CLUSTER_COL in df.columns
    counts = df[CANONICAL_CLUSTER_COL].value_counts().reset_index()
    counts.columns = ["cluster", "count"]
    assert len(counts) > 0


def test_warehouse_segmentation_uses_canonical_cluster_label():
    df = _load_segment("warehouse")
    assert CANONICAL_CLUSTER_COL in df.columns
    counts = df[CANONICAL_CLUSTER_COL].value_counts().reset_index()
    counts.columns = ["cluster", "count"]
    assert len(counts) > 0


def test_database_segment_schema_matches_csv():
    import sqlalchemy

    engine = sqlalchemy.create_engine(f"sqlite:///{DB_PATH}")
    for name, filename in SEGMENT_FILES.items():
        table = f"{name}_segments"
        df = pd.read_sql(f"SELECT * FROM {table} LIMIT 1", engine)
        csv_df = _load_segment(name)
        csv_cols = set(csv_df.columns)
        db_cols = set(df.columns)
        assert db_cols == csv_cols, f"{table}: DB columns {db_cols} != CSV columns {csv_cols}"
