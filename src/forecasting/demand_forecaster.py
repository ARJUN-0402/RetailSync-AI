import pandas as pd
import numpy as np
import os
import time
import joblib
from datetime import datetime, timedelta

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

np.random.seed(42)

print("=== RETAILSYNC AI - DEMAND FORECASTING PIPELINE ===\n")

# Load data
print("Loading feature data...")
df = pd.read_csv("data/processed/features_daily.csv", parse_dates=["date"])
print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

# ============================================================
# 1. TIME-BASED TRAIN/VALIDATION/TEST SPLIT
# ============================================================
print("\n=== 1. DATA SPLITTING ===")

# Use time-based split to avoid data leakage
train_end = pd.Timestamp("2024-12-31")
val_end = pd.Timestamp("2025-06-09")
test_end = pd.Timestamp("2025-08-09")

train_df = df[df["date"] <= train_end].copy()
val_df = df[(df["date"] > train_end) & (df["date"] <= val_end)].copy()
test_df = df[(df["date"] > val_end) & (df["date"] <= test_end)].copy()

print(f"Train: {train_df['date'].min().date()} to {train_df['date'].max().date()} ({len(train_df):,} rows)")
print(f"Validation: {val_df['date'].min().date()} to {val_df['date'].max().date()} ({len(val_df):,} rows)")
print(f"Test: {test_df['date'].min().date()} to {test_df['date'].max().date()} ({len(test_df):,} rows)")

# ============================================================
# 2. FEATURE SELECTION
# ============================================================
print("\n=== 2. FEATURE SELECTION ===")

# Define feature columns (exclude identifiers, targets, and raw sales columns)
exclude_cols = [
    "date", "product_id", "store_id", "category", "subcategory", "store_type",
    "city", "state", "supplier_id", "warehouse_id",
    "quantity_sold", "revenue", "unit_price", "promotion",
    "target_demand_1d", "target_demand_7d", "target_demand_14d",
    "target_revenue_1d", "target_revenue_7d", "target_revenue_14d"
]

feature_cols = [c for c in df.columns if c not in exclude_cols]
print(f"Selected {len(feature_cols)} features")

# Check for any remaining non-numeric columns
non_numeric = [c for c in feature_cols if df[c].dtype == 'object']
if non_numeric:
    print(f"WARNING: Non-numeric features found: {non_numeric}")
    feature_cols = [c for c in feature_cols if c not in non_numeric]

X_train = train_df[feature_cols].values
y_train = train_df["target_demand_1d"].values

X_val = val_df[feature_cols].values
y_val = val_df["target_demand_1d"].values

X_test = test_df[feature_cols].values
y_test = test_df["target_demand_1d"].values

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# ============================================================
# 3. BASELINE MODELS
# ============================================================
print("\n=== 3. BASELINE MODELS ===")

def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    mask = (y_true + y_pred) != 0
    if not mask.any():
        return 0.0
    return np.mean(np.abs(y_pred[mask] - y_true[mask]) / ((np.abs(y_true[mask]) + np.abs(y_pred[mask])) / 2)) * 100

def evaluate_model(y_true, y_pred, model_name):
    """Evaluate model with multiple metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    s = smape(y_true, y_pred)
    
    print(f"\n{model_name}:")
    print(f"  MAE:   {mae:.4f}")
    print(f"  RMSE:  {rmse:.4f}")
    print(f"  R²:    {r2:.4f}")
    print(f"  sMAPE: {s:.2f}%")
    
    return {"model": model_name, "mae": mae, "rmse": rmse, "r2": r2, "smape": s}

results = []

# Baseline 1: Historical Mean (train set mean)
print("\nTraining Baseline 1: Historical Mean...")
train_mean = np.mean(y_train)
y_pred_mean = np.full(len(y_val), train_mean)
results.append(evaluate_model(y_val, y_pred_mean, "Baseline_Mean"))

# Baseline 2: Naive (last known value = lag_1d)
print("\nTraining Baseline 2: Naive (lag_1d)...")
y_pred_naive = val_df["demand_lag_1d"].values
results.append(evaluate_model(y_val, y_pred_naive, "Baseline_Naive"))

# Baseline 3: Moving Average (7-day rolling mean)
print("\nTraining Baseline 3: Moving Average (7d)...")
y_pred_ma = val_df["demand_rolling_mean_7d"].values
results.append(evaluate_model(y_val, y_pred_ma, "Baseline_MA_7d"))

# ============================================================
# 4. MACHINE LEARNING MODELS
# ============================================================
print("\n=== 4. MACHINE LEARNING MODELS ===")

# Model 1: Random Forest
print("\nTraining Random Forest...")
start_time = time.time()
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    verbose=0
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_val)
rf_time = time.time() - start_time
results.append(evaluate_model(y_val, y_pred_rf, f"RandomForest ({rf_time:.1f}s)"))
print(f"  Training time: {rf_time:.1f}s")

# Model 2: XGBoost
print("\nTraining XGBoost...")
start_time = time.time()
xgb_model = XGBRegressor(
    n_estimators=100,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_val)
xgb_time = time.time() - start_time
results.append(evaluate_model(y_val, y_pred_xgb, f"XGBoost ({xgb_time:.1f}s)"))
print(f"  Training time: {xgb_time:.1f}s")

# ============================================================
# 5. MODEL COMPARISON
# ============================================================
print("\n=== 5. MODEL COMPARISON ===")

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("mae")
print("\nModel Ranking by MAE:")
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]["model"]
print(f"\nBest model: {best_model_name}")

# ============================================================
# 6. BEST MODEL EVALUATION ON TEST SET
# ============================================================
print("\n=== 6. TEST SET EVALUATION ===")

if "RandomForest" in best_model_name:
    best_model = rf_model
elif "XGBoost" in best_model_name:
    best_model = xgb_model
else:
    best_model = rf_model  # fallback

y_pred_test = best_model.predict(X_test)
test_metrics = evaluate_model(y_test, y_pred_test, f"{best_model_name} (Test Set)")

# ============================================================
# 7. FEATURE IMPORTANCE
# ============================================================
print("\n=== 7. FEATURE IMPORTANCE ===")

if hasattr(best_model, "feature_importances_"):
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": best_model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    print("\nTop 15 Most Important Features:")
    print(importance_df.head(15).to_string(index=False))
    
    # Save feature importance
    importance_df.to_csv("docs/feature_importance.csv", index=False)
    print(f"\nSaved feature importance to docs/feature_importance.csv")
else:
    print("Model does not support feature importance.")

# ============================================================
# 8. SAVE MODEL
# ============================================================
print("\n=== 8. SAVING MODEL ===")

os.makedirs("models", exist_ok=True)

model_package = {
    "model": best_model,
    "feature_cols": feature_cols,
    "train_mean": train_mean,
    "metrics": test_metrics,
    "model_name": best_model_name,
    "trained_at": datetime.now().isoformat(),
    "data_splits": {
        "train_end": train_end.isoformat(),
        "val_end": val_end.isoformat(),
        "test_end": test_end.isoformat()
    }
}

joblib.dump(model_package, "models/demand_forecaster.pkl")
print(f"Saved model to models/demand_forecaster.pkl")

# ============================================================
# 9. SAMPLE PREDICTIONS
# ============================================================
print("\n=== 9. SAMPLE PREDICTIONS ===")

sample_indices = np.random.choice(len(test_df), min(10, len(test_df)), replace=False)
for idx in sample_indices[:5]:
    row = test_df.iloc[idx]
    actual = row["target_demand_1d"]
    pred = y_pred_test[idx]
    print(f"  {row['date'].date()} | {row['product_id']} | {row['store_id']} | Actual: {actual:.1f} | Predicted: {pred:.1f}")

# ============================================================
# 10. SUMMARY
# ============================================================
print("\n=== 10. FORECASTING SUMMARY ===")
print(f"Best model: {best_model_name}")
print(f"Test MAE: {test_metrics['mae']:.4f}")
print(f"Test RMSE: {test_metrics['rmse']:.4f}")
print(f"Test R²: {test_metrics['r2']:.4f}")
print(f"Test sMAPE: {test_metrics['smape']:.2f}%")
print(f"\nNote: High zero-inflation (~81% zeros) makes sMAPE less reliable.")
print(f"MAE and RMSE are primary metrics for this dataset.")

print("\nForecasting pipeline complete.")
