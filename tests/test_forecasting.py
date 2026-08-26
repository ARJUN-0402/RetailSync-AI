"""
Tests for the RetailSync AI demand forecasting pipeline.

Run with:
    pytest tests/test_forecasting.py -v
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def _load_features():
    path = os.path.join(PROCESSED_DIR, "features_daily.csv")
    assert os.path.exists(path), f"Missing {path}"
    return pd.read_csv(path, parse_dates=["date"])


def test_feature_count_and_types():
    df = _load_features()
    assert len(df) > 0, "Features DataFrame is empty"

    required = [
        "date", "product_id", "store_id", "quantity_sold", "revenue",
        "target_demand_1d", "target_demand_7d", "target_demand_14d",
        "demand_lag_1d", "demand_lag_7d", "demand_lag_14d", "demand_lag_28d",
        "demand_rolling_mean_7d", "demand_rolling_std_7d",
        "day_of_week", "month", "quarter", "is_weekend",
        "promotion", "discount_pct", "unit_price", "cost_price",
        "stock_coverage_days", "is_holiday", "week_of_year",
    ]
    for col in required:
        assert col in df.columns, f"Missing required feature: {col}"

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    assert len(numeric_cols) >= 30, f"Expected >= 30 numeric features, got {len(numeric_cols)}"


def test_time_based_split():
    df = _load_features()
    train_end = pd.Timestamp("2024-12-31")
    val_end = pd.Timestamp("2025-06-09")
    test_end = pd.Timestamp("2025-08-09")

    train = df[df["date"] <= train_end]
    val = df[(df["date"] > train_end) & (df["date"] <= val_end)]
    test = df[(df["date"] > val_end) & (df["date"] <= test_end)]

    assert len(train) > 0, "Train set is empty"
    assert len(val) > 0, "Validation set is empty"
    assert len(test) > 0, "Test set is empty"

    assert train["date"].max() <= train_end
    assert val["date"].max() <= val_end
    assert test["date"].max() <= test_end

    assert train["date"].max() < val["date"].min(), "Train/val overlap"
    assert val["date"].max() < test["date"].min(), "Val/test overlap"


def test_no_future_leakage_in_lags():
    df = _load_features()
    df = df.sort_values(["product_id", "store_id", "date"]).reset_index(drop=True)

    sample = df.head(20000).copy()
    for idx, row in sample.head(500).iterrows():
        if pd.isna(row["demand_lag_1d"]):
            continue
        past = sample[
            (sample["date"] == row["date"] - pd.Timedelta(days=1))
            & (sample["product_id"] == row["product_id"])
            & (sample["store_id"] == row["store_id"])
        ]
        if not past.empty:
            assert abs(past.iloc[0]["quantity_sold"] - row["demand_lag_1d"]) < 1e-9


def test_no_future_leakage_in_rolling():
    df = _load_features()
    df = df.sort_values(["product_id", "store_id", "date"]).reset_index(drop=True)

    sample = df.head(20000).copy()
    for idx, row in sample.head(500).iterrows():
        if pd.isna(row["demand_rolling_mean_7d"]):
            continue
        hist = sample[
            (sample["date"] < row["date"])
            & (sample["product_id"] == row["product_id"])
            & (sample["store_id"] == row["store_id"])
            & (sample["date"] >= row["date"] - pd.Timedelta(days=7))
        ]
        if len(hist) > 0:
            expected = hist["quantity_sold"].mean()
            assert abs(expected - row["demand_rolling_mean_7d"]) < 1e-6


def test_targets_are_future():
    df = _load_features()
    df = df.sort_values(["product_id", "store_id", "date"]).reset_index(drop=True)

    sample = df.head(20000).copy()
    for idx, row in sample.head(500).iterrows():
        if pd.isna(row["target_demand_1d"]):
            continue
        future = sample[
            (sample["date"] == row["date"] + pd.Timedelta(days=1))
            & (sample["product_id"] == row["product_id"])
            & (sample["store_id"] == row["store_id"])
        ]
        if not future.empty:
            assert abs(future.iloc[0]["quantity_sold"] - row["target_demand_1d"]) < 1e-9


def test_aggregate_features_shifted():
    df = _load_features()
    df = df.sort_values(["product_id", "store_id", "date"]).reset_index(drop=True)

    for idx, row in df.head(2000).iterrows():
        if pd.isna(row["category_avg_demand"]) or row["category_avg_demand"] == 0:
            continue
        prev_day = row["date"] - pd.Timedelta(days=1)
        prev = df[(df["date"] == prev_day) & (df["category"] == row["category"])]
        if not prev.empty:
            prev_mean = prev["quantity_sold"].mean()
            assert abs(prev_mean - row["category_avg_demand"]) < 1e-6, (
                "category_avg_demand does not match previous day's mean"
            )


def test_model_package_structure():
    path = os.path.join(MODELS_DIR, "demand_forecaster.pkl")
    assert os.path.exists(path), f"Missing model: {path}"

    pkg = joblib.load(path)
    for key in ["model", "feature_cols", "metrics", "model_name", "trained_at", "data_splits"]:
        assert key in pkg, f"Model package missing key: {key}"

    assert hasattr(pkg["model"], "predict"), "Model lacks predict method"
    assert len(pkg["feature_cols"]) > 0, "Empty feature_cols"


def test_model_prediction_output():
    path = os.path.join(MODELS_DIR, "demand_forecaster.pkl")
    pkg = joblib.load(path)
    model = pkg["model"]
    feature_cols = pkg["feature_cols"]

    df = _load_features()
    test = df[df["date"] > pd.Timestamp("2025-06-09")].copy()
    X_test = test[feature_cols].fillna(0).values
    y_test = test["target_demand_1d"].values

    preds = model.predict(X_test)
    assert len(preds) == len(y_test), "Prediction length mismatch"
    assert np.all(preds >= 0), "Predictions must be non-negative"
    assert np.any(preds > 0), "All predictions are zero"

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    assert np.isfinite(mae), "MAE is not finite"
    assert np.isfinite(rmse), "RMSE is not finite"
    assert np.isfinite(r2), "R² is not finite"
    assert mae >= 0, "MAE must be non-negative"
    assert rmse >= 0, "RMSE must be non-negative"


def test_evaluation_metrics():
    y_true = np.array([1.0, 2.0, 0.0, 4.0, 5.0])
    y_pred = np.array([1.5, 2.5, 0.5, 3.5, 4.5])

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    assert abs(mae - 0.5) < 1e-6, f"MAE mismatch: {mae}"
    assert abs(rmse - 0.5) < 1e-6, f"RMSE mismatch: {rmse}"

    mask = (y_true + y_pred) != 0
    smape = (
        np.mean(np.abs(y_pred[mask] - y_true[mask]) / ((np.abs(y_true[mask]) + np.abs(y_pred[mask])) / 2))
        * 100
    )
    assert smape > 0, "sMAPE should be positive"

    mape_mask = y_true != 0
    mape = np.mean(np.abs((y_true[mape_mask] - y_pred[mape_mask]) / y_true[mape_mask])) * 100
    assert mape > 0, "MAPE should be positive"


def test_model_persistence_roundtrip():
    path = os.path.join(MODELS_DIR, "demand_forecaster.pkl")
    pkg1 = joblib.load(path)
    model1 = pkg1["model"]
    feature_cols1 = pkg1["feature_cols"]

    test_path = os.path.join(MODELS_DIR, "test_roundtrip.pkl")
    joblib.dump(pkg1, test_path)
    pkg2 = joblib.load(test_path)
    model2 = pkg2["model"]
    feature_cols2 = pkg2["feature_cols"]

    df = _load_features()
    test = df[df["date"] > pd.Timestamp("2025-06-09")].copy()
    X_test = test[feature_cols1].fillna(0).values[:100]

    pred1 = model1.predict(X_test)
    pred2 = model2.predict(X_test)

    np.testing.assert_allclose(pred1, pred2, rtol=1e-10)
    assert feature_cols1 == feature_cols2

    os.remove(test_path)


def test_forecasts_shape_and_range():
    forecasts_path = os.path.join(PROCESSED_DIR, "forecasts_next_14d.csv")
    assert os.path.exists(forecasts_path), f"Missing {forecasts_path}"

    forecasts = pd.read_csv(forecasts_path, parse_dates=["date"])
    assert len(forecasts) > 0, "Forecasts are empty"

    expected = 50 * 10 * 14
    assert len(forecasts) == expected, f"Expected {expected} forecasts, got {len(forecasts)}"

    assert (forecasts["forecast_demand"] >= 0).all(), "Negative forecast demand"
    assert (forecasts["forecast_revenue"] >= 0).all(), "Negative forecast revenue"

    features = _load_features()
    forecast_start = forecasts["date"].min()
    feature_end = features["date"].max()
    assert forecast_start > feature_end, "Forecasts should be in the future"


def test_baseline_models_exist():
    from src.models.baselines import BaselineMeanPredictor, NaivePredictor, MovingAveragePredictor

    mean_model = BaselineMeanPredictor()
    naive_model = NaivePredictor()
    ma_model = MovingAveragePredictor()

    X = np.random.randn(100, 5)
    y = np.random.randn(100)

    mean_model.fit(X, y)
    p1 = mean_model.predict(X[:10])
    assert len(p1) == 10
    assert np.all(p1 == p1[0])

    df = pd.DataFrame({"demand_lag_1d": np.random.randn(100)})
    p2 = naive_model.predict(df)
    assert len(p2) == 100

    df2 = pd.DataFrame({"demand_rolling_mean_7d": np.random.randn(100)})
    p3 = ma_model.predict(df2)
    assert len(p3) == 100
