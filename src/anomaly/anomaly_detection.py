"""Demand anomaly detection for RetailSync AI."""

from __future__ import annotations

import logging
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text

from src.config import settings
from src.exceptions import DataError, PipelineError
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

np.random.seed(42)


def detect() -> pd.DataFrame:
    """Run anomaly detection and return anomalies dataframe."""
    setup_logging(__name__)
    logger.info("=== RetailSync AI - Demand Anomaly Detection ===")

    features_path = os.path.join(str(settings.paths.processed_data), "features_daily.csv")
    if not os.path.exists(features_path):
        raise DataError(f"Features file not found: {features_path}")

    df = pd.read_csv(features_path, parse_dates=["date"])
    logger.info("Dataset shape: %s", df.shape)
    logger.info("Date range: %s to %s", df["date"].min().date(), df["date"].max().date())

    df = df.sort_values(["product_id", "store_id", "date"]).reset_index(drop=True)

    start_time = time.time()
    df["rolling_mean_30d"] = df.groupby(["product_id", "store_id"])[
        "quantity_sold"
    ].transform(lambda x: x.shift(1).rolling(window=30, min_periods=7).mean())
    df["rolling_std_30d"] = (
        df.groupby(["product_id", "store_id"])["quantity_sold"]
        .transform(lambda x: x.shift(1).rolling(window=30, min_periods=7).std())
        .fillna(0)
    )

    df["z_score"] = np.where(
        df["rolling_std_30d"] > 0,
        (df["quantity_sold"] - df["rolling_mean_30d"]) / df["rolling_std_30d"],
        0,
    )

    df["anomaly_zscore"] = "Normal"
    df.loc[df["z_score"] > 2, "anomaly_zscore"] = "Unusual"
    df.loc[df["z_score"] > 3, "anomaly_zscore"] = "Significant Spike"
    df.loc[df["z_score"] < -2, "anomaly_zscore"] = "Unusual (Low)"
    df.loc[df["z_score"] < -3, "anomaly_zscore"] = "Significant Drop"

    zscore_time = time.time() - start_time
    logger.info("Rolling Z-score completed in %.1fs", zscore_time)
    logger.info("Z-score distribution:\n%s", df["anomaly_zscore"].value_counts().to_string())

    start_time = time.time()
    q1 = df.groupby(["product_id", "store_id"])["quantity_sold"].transform(
        lambda x: x.shift(1).quantile(0.25)
    )
    q3 = df.groupby(["product_id", "store_id"])["quantity_sold"].transform(
        lambda x: x.shift(1).quantile(0.75)
    )
    iqr = q3 - q1

    df["iqr_lower"] = q1 - 1.5 * iqr
    df["iqr_upper"] = q3 + 1.5 * iqr

    df["anomaly_iqr"] = "Normal"
    df.loc[df["quantity_sold"] > df["iqr_upper"], "anomaly_iqr"] = "Significant Spike"
    df.loc[df["quantity_sold"] < df["iqr_lower"], "anomaly_iqr"] = "Significant Drop"
    logger.info("IQR method completed in %.1fs", time.time() - start_time)
    logger.info("IQR distribution:\n%s", df["anomaly_iqr"].value_counts().to_string())

    start_time = time.time()
    agg = df.groupby(["product_id", "store_id"]).agg(
        avg_demand=("quantity_sold", "mean"),
        std_demand=("quantity_sold", "std"),
        max_demand=("quantity_sold", "max"),
        zero_pct=("quantity_sold", lambda x: (x == 0).mean()),
    ).reset_index()
    agg = agg.fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(agg[["avg_demand", "std_demand", "max_demand", "zero_pct"]])

    iso = IsolationForest(contamination=0.05, random_state=42)
    agg["anomaly_iso"] = iso.fit_predict(X_scaled)
    agg["anomaly_iso"] = agg["anomaly_iso"].map({1: "Normal", -1: "Anomaly"})

    df = df.merge(
        agg[["product_id", "store_id", "anomaly_iso"]], on=["product_id", "store_id"], how="left"
    )
    logger.info("Isolation Forest completed in %.1fs", time.time() - start_time)
    logger.info("Isolation Forest distribution:\n%s", df["anomaly_iso"].value_counts().to_string())

    df["anomaly_votes"] = 0
    df.loc[df["anomaly_zscore"].isin(["Significant Spike", "Significant Drop", "Unusual", "Unusual (Low)"]), "anomaly_votes"] += 1
    df.loc[df["anomaly_iqr"].isin(["Significant Spike", "Significant Drop"]), "anomaly_votes"] += 1
    df.loc[df["anomaly_iso"] == "Anomaly", "anomaly_votes"] += 1

    df["is_anomaly"] = df["anomaly_votes"] >= 2
    anomalies = df[df["is_anomaly"]].copy()
    logger.info("Total anomalies (2+ votes): %d (%.2f%%)", len(anomalies), len(anomalies) / len(df) * 100)

    def classify_anomaly(row):
        if row["z_score"] > 3:
            return "Demand Spike"
        if row["z_score"] < -3:
            return "Demand Drop"
        return "Unusual Pattern"

    anomalies["anomaly_type"] = anomalies.apply(classify_anomaly, axis=1)
    anomalies["detection_methods"] = anomalies.apply(
        lambda row: ", ".join(
            filter(None, [
                "Z-score" if row["anomaly_votes"] >= 1 and row["anomaly_zscore"] != "Normal" else "",
                "IQR" if row["anomaly_iqr"] != "Normal" else "",
                "Isolation Forest" if row["anomaly_iso"] == "Anomaly" else "",
            ])
        ),
        axis=1,
    )

    output_path = os.path.join(str(settings.paths.processed_data), "anomalies.csv")
    anomalies.to_csv(output_path, index=False)
    logger.info("Saved anomalies to %s", output_path)

    db_path = os.path.join(str(settings.paths.database), "retailsync.db")
    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM anomaly_flags"))
        conn.commit()

    anomalies[["date", "product_id", "store_id", "z_score", "anomaly_type", "detection_methods", "quantity_sold"]].to_sql(
        "anomaly_flags", con=engine, if_exists="append", index=False
    )
    logger.info("Loaded %d anomaly flags into database", len(anomalies))

    logger.info("Demand spikes: %d", (anomalies["anomaly_type"] == "Demand Spike").sum())
    logger.info("Unusual patterns: %d", (anomalies["anomaly_type"] == "Unusual Pattern").sum())

    return anomalies


def main() -> None:
    """Main entry point."""
    try:
        detect()
    except (DataError, PipelineError):
        raise
    except Exception as exc:
        raise PipelineError(f"Anomaly detection failed: {exc}") from exc


if __name__ == "__main__":
    main()
