"""Demand forecasting pipeline for RetailSync AI."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.config import settings
from src.exceptions import ModelError, DataError, PipelineError
from src.models.baselines import (
    BaselineMeanPredictor,
    MovingAveragePredictor,
    NaivePredictor,
)
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

np.random.seed(42)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    mask = (y_true + y_pred) != 0
    if not mask.any():
        return 0.0
    return (
        np.mean(
            np.abs(y_pred[mask] - y_true[mask])
            / ((np.abs(y_true[mask]) + np.abs(y_pred[mask])) / 2)
        )
        * 100
    )


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
    """Evaluate model with multiple metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    s = smape(y_true, y_pred)

    logger.info(
        "%s -> MAE: %.4f, RMSE: %.4f, R2: %.4f, sMAPE: %.2f%%",
        model_name, mae, rmse, r2, s,
    )
    return {"model": model_name, "mae": mae, "rmse": rmse, "r2": r2, "smape": s}


def train() -> dict:
    """Train forecasting models and return the best model package."""
    setup_logging(__name__)
    logger.info("=== RetailSync AI - Demand Forecasting Pipeline ===")

    features_path = os.path.join(str(settings.paths.processed_data), "features_daily.csv")
    if not os.path.exists(features_path):
        raise DataError(f"Features file not found: {features_path}")

    df = pd.read_csv(features_path, parse_dates=["date"])
    logger.info("Dataset shape: %s", df.shape)
    logger.info("Date range: %s to %s", df["date"].min().date(), df["date"].max().date())

    train_end = pd.Timestamp("2024-12-31")
    val_end = pd.Timestamp("2025-06-09")
    test_end = pd.Timestamp("2025-08-09")

    train_df = df[df["date"] <= train_end].copy()
    val_df = df[(df["date"] > train_end) & (df["date"] <= val_end)].copy()
    test_df = df[(df["date"] > val_end) & (df["date"] <= test_end)].copy()

    logger.info(
        "Train: %s to %s (%d rows)",
        train_df["date"].min().date(), train_df["date"].max().date(), len(train_df),
    )
    logger.info(
        "Validation: %s to %s (%d rows)",
        val_df["date"].min().date(), val_df["date"].max().date(), len(val_df),
    )
    logger.info(
        "Test: %s to %s (%d rows)",
        test_df["date"].min().date(), test_df["date"].max().date(), len(test_df),
    )

    exclude_cols = [
        "date", "product_id", "store_id", "category", "subcategory", "store_type",
        "city", "state", "supplier_id", "warehouse_id", "quantity_sold", "revenue",
        "unit_price", "promotion", "target_demand_1d", "target_demand_7d",
        "target_demand_14d", "target_revenue_1d", "target_revenue_7d",
        "target_revenue_14d",
    ]

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    logger.info("Selected %d features", len(feature_cols))

    non_numeric = [c for c in feature_cols if df[c].dtype == "object"]
    if non_numeric:
        logger.warning("Non-numeric features found: %s", non_numeric)
        feature_cols = [c for c in feature_cols if c not in non_numeric]

    X_train = train_df[feature_cols].values
    y_train = train_df["target_demand_1d"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["target_demand_1d"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["target_demand_1d"].values

    logger.info(
        "Shapes - train: %s, val: %s, test: %s", X_train.shape, X_val.shape, X_test.shape
    )

    results = []

    logger.info("Training Baseline 1: Historical Mean...")
    baseline_mean_model = BaselineMeanPredictor(n_features=len(feature_cols))
    baseline_mean_model.fit(X_train, y_train)
    y_pred_mean = baseline_mean_model.predict(val_df[feature_cols].values)
    results.append(evaluate_model(y_val, y_pred_mean, "Baseline_Mean"))

    logger.info("Training Baseline 2: Naive (lag_1d)...")
    naive_model = NaivePredictor(lag_col="demand_lag_1d", n_features=len(feature_cols))
    y_pred_naive = naive_model.predict(val_df)
    results.append(evaluate_model(y_val, y_pred_naive, "Baseline_Naive"))

    logger.info("Training Baseline 3: Moving Average (7d)...")
    ma_model = MovingAveragePredictor(
        rolling_col="demand_rolling_mean_7d", n_features=len(feature_cols)
    )
    y_pred_ma = ma_model.predict(val_df)
    results.append(evaluate_model(y_val, y_pred_ma, "Baseline_MA_7d"))

    logger.info("Training Random Forest...")
    start_time = time.time()
    rf_model = RandomForestRegressor(
        n_estimators=100, max_depth=15, min_samples_split=10, min_samples_leaf=5,
        random_state=42, n_jobs=-1, verbose=0,
    )
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_val)
    rf_time = time.time() - start_time
    results.append(evaluate_model(y_val, y_pred_rf, f"RandomForest ({rf_time:.1f}s)"))
    logger.info("RF training time: %.1fs", rf_time)

    logger.info("Training XGBoost...")
    start_time = time.time()
    xgb_model = XGBRegressor(
        n_estimators=100, max_depth=8, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0,
    )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_val)
    xgb_time = time.time() - start_time
    results.append(evaluate_model(y_val, y_pred_xgb, f"XGBoost ({xgb_time:.1f}s)"))
    logger.info("XGB training time: %.1fs", xgb_time)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("mae")
    logger.info("Model Ranking by MAE:\n%s", results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["model"]
    logger.info("Best model: %s", best_model_name)

    if best_model_name == "Baseline_Mean":
        best_model = baseline_mean_model
        y_pred_test = best_model.predict(test_df[feature_cols].values)
    elif best_model_name == "Baseline_Naive":
        best_model = naive_model
        y_pred_test = best_model.predict(test_df)
    elif best_model_name == "Baseline_MA_7d":
        best_model = ma_model
        y_pred_test = best_model.predict(test_df)
    elif "RandomForest" in best_model_name:
        best_model = rf_model
        y_pred_test = best_model.predict(X_test)
    elif "XGBoost" in best_model_name:
        best_model = xgb_model
        y_pred_test = best_model.predict(X_test)
    else:
        best_model = rf_model
        y_pred_test = best_model.predict(X_test)

    test_metrics = evaluate_model(y_test, y_pred_test, f"{best_model_name} (Test Set)")

    if hasattr(best_model, "feature_importances_") and not isinstance(
        best_model, BaselineMeanPredictor
    ):
        importance_df = pd.DataFrame(
            {"feature": feature_cols, "importance": best_model.feature_importances_}
        ).sort_values("importance", ascending=False)

        logger.info("Top 15 Most Important Features:\n%s", importance_df.head(15).to_string(index=False))

        docs_dir = str(settings.paths.docs)
        os.makedirs(docs_dir, exist_ok=True)
        importance_df.to_csv(os.path.join(docs_dir, "feature_importance.csv"), index=False)
        logger.info("Saved feature importance to docs/feature_importance.csv")
    else:
        logger.info("Model does not support feature importance.")

    os.makedirs(str(settings.paths.models), exist_ok=True)

    model_package = {
        "model": best_model,
        "feature_cols": feature_cols,
        "train_mean": getattr(baseline_mean_model, "train_mean_", 0.0),
        "metrics": test_metrics,
        "model_name": best_model_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_splits": {
            "train_end": train_end.isoformat(),
            "val_end": val_end.isoformat(),
            "test_end": test_end.isoformat(),
        },
    }

    model_path = os.path.join(str(settings.paths.models), "demand_forecaster.pkl")
    joblib.dump(model_package, model_path)
    logger.info("Saved model to %s", model_path)

    sample_indices = np.random.choice(len(test_df), min(10, len(test_df)), replace=False)
    for idx in sample_indices[:5]:
        row = test_df.iloc[idx]
        actual = row["target_demand_1d"]
        pred = y_pred_test[idx]
        logger.info(
            "Sample: %s | %s | %s | Actual: %.1f | Predicted: %.1f",
            row["date"].date(), row["product_id"], row["store_id"], actual, pred,
        )

    logger.info("Best model: %s", best_model_name)
    logger.info("Test MAE: %.4f", test_metrics["mae"])
    logger.info("Test RMSE: %.4f", test_metrics["rmse"])
    logger.info("Test R2: %.4f", test_metrics["r2"])
    logger.info("Test sMAPE: %.2f%%", test_metrics["smape"])
    logger.info("Note: High zero-inflation makes sMAPE less reliable. MAE and RMSE are primary metrics.")
    logger.info("Forecasting pipeline complete.")

    return model_package


def main() -> None:
    """Main entry point."""
    try:
        train()
    except (DataError, ModelError, PipelineError):
        raise
    except Exception as exc:
        raise PipelineError(f"Forecasting pipeline failed: {exc}") from exc


if __name__ == "__main__":
    main()
