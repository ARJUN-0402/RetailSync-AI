"""Health check utilities for RetailSync AI.

Provides functions to verify application health including database
connectivity, model availability, and configuration validity.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)


def check_database(db_path: str | None = None) -> dict[str, Any]:
    """Check database connectivity and basic health.

    Args:
        db_path: Path to SQLite database. Defaults to configured path.

    Returns:
        Dict with status, details, and any errors.
    """
    from src.config import settings

    path = db_path or settings.database.path
    result: dict[str, Any] = {"status": "unhealthy", "path": path, "error": None}

    if not os.path.exists(path):
        result["error"] = f"Database file not found: {path}"
        return result

    try:
        conn = sqlite3.connect(path, timeout=5)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()

        result["status"] = "healthy"
        result["table_count"] = len(tables)
        result["tables"] = tables[:10]
    except Exception as exc:
        result["error"] = str(exc)
        logger.error("Database health check failed: %s", exc)

    return result


def check_models(models_dir: str | None = None) -> dict[str, Any]:
    """Check model artifact availability.

    Args:
        models_dir: Directory containing model files.

    Returns:
        Dict with status, available models, and missing models.
    """
    from src.config import settings

    directory = models_dir or str(settings.paths.models)
    expected = [
        "demand_forecaster.pkl",
        "product_clusterer.pkl",
        "store_clusterer.pkl",
        "warehouse_clusterer.pkl",
    ]

    result: dict[str, Any] = {
        "status": "unhealthy",
        "directory": directory,
        "available": [],
        "missing": [],
    }

    if not os.path.isdir(directory):
        result["error"] = f"Models directory not found: {directory}"
        return result

    for model_file in expected:
        path = os.path.join(directory, model_file)
        if os.path.exists(path):
            result["available"].append(model_file)
        else:
            result["missing"].append(model_file)

    if not result["missing"]:
        result["status"] = "healthy"
    else:
        result["error"] = f"Missing models: {result['missing']}"

    return result


def check_data_files(processed_dir: str | None = None) -> dict[str, Any]:
    """Check required data file availability.

    Args:
        processed_dir: Directory containing processed data files.

    Returns:
        Dict with status, available files, and missing files.
    """
    from src.config import settings

    directory = processed_dir or str(settings.paths.processed_data)
    expected = [
        "features_daily.csv",
        "forecasts_next_14d.csv",
        "inventory_intelligence.csv",
        "anomalies.csv",
        "product_segments.csv",
        "store_segments.csv",
        "warehouse_segments.csv",
        "warehouse_optimization.csv",
        "products.csv",
        "stores.csv",
        "suppliers.csv",
        "warehouses.csv",
    ]

    result: dict[str, Any] = {
        "status": "unhealthy",
        "directory": directory,
        "available": [],
        "missing": [],
    }

    if not os.path.isdir(directory):
        result["error"] = f"Processed data directory not found: {directory}"
        return result

    for data_file in expected:
        path = os.path.join(directory, data_file)
        if os.path.exists(path):
            result["available"].append(data_file)
        else:
            result["missing"].append(data_file)

    if not result["missing"]:
        result["status"] = "healthy"
    else:
        result["error"] = f"Missing data files: {result['missing']}"

    return result


def check_configuration() -> dict[str, Any]:
    """Verify that required configuration exists.

    Returns:
        Dict with status and any configuration issues.
    """
    from src.config import settings

    result: dict[str, Any] = {"status": "healthy", "issues": []}

    if not settings.paths.database.exists():
        result["issues"].append("Database directory does not exist")
    if not settings.paths.processed_data.exists():
        result["issues"].append("Processed data directory does not exist")
    if not settings.paths.models.exists():
        result["issues"].append("Models directory does not exist")

    if result["issues"]:
        result["status"] = "degraded"
        result["error"] = "; ".join(result["issues"])

    return result


def validate_runtime_artifacts() -> dict[str, Any]:
    """Validate that all required deployment artifacts are present.

    Returns:
        Dict with status, present artifacts, and missing artifacts with
        specific file paths for deployment diagnostics.
    """
    from src.config import settings

    processed_dir = settings.paths.processed_data
    models_dir = settings.paths.models
    database_path = settings.database.path

    required_data = [
        "features_daily.csv",
        "forecasts_next_14d.csv",
        "inventory_intelligence.csv",
        "anomalies.csv",
        "product_segments.csv",
        "store_segments.csv",
        "warehouse_segments.csv",
        "warehouse_optimization.csv",
        "products.csv",
        "stores.csv",
        "suppliers.csv",
        "warehouses.csv",
    ]

    required_models = [
        "demand_forecaster.pkl",
        "product_clusterer.pkl",
        "store_clusterer.pkl",
        "warehouse_clusterer.pkl",
    ]

    result: dict[str, Any] = {
        "status": "healthy",
        "processed_dir": str(processed_dir),
        "models_dir": str(models_dir),
        "database_path": str(database_path),
        "missing": [],
    }

    if not processed_dir.exists():
        result["status"] = "unhealthy"
        result["missing"].append(f"Processed data directory not found: {processed_dir}")
    else:
        for fname in required_data:
            if not (processed_dir / fname).exists():
                result["status"] = "unhealthy"
                result["missing"].append(f"Missing deployment artifact: {processed_dir / fname}")

    if not models_dir.exists():
        result["status"] = "unhealthy"
        result["missing"].append(f"Models directory not found: {models_dir}")
    else:
        for fname in required_models:
            if not (models_dir / fname).exists():
                result["status"] = "unhealthy"
                result["missing"].append(f"Missing deployment artifact: {models_dir / fname}")

    if not Path(database_path).exists():
        result["status"] = "unhealthy"
        result["missing"].append(f"Missing deployment artifact: {database_path}")

    return result


def get_health_status() -> dict[str, Any]:
    """Return comprehensive application health status.

    Returns:
        Dict with overall status and component checks.
    """
    db_check = check_database()
    model_check = check_models()
    data_check = check_data_files()
    config_check = check_configuration()

    components = {
        "database": db_check,
        "models": model_check,
        "data_files": data_check,
        "configuration": config_check,
    }

    all_healthy = all(c.get("status") == "healthy" for c in components.values())
    any_degraded = any(c.get("status") == "degraded" for c in components.values())

    if all_healthy:
        overall = "healthy"
    elif any_degraded:
        overall = "degraded"
    else:
        overall = "unhealthy"

    return {
        "status": overall,
        "app": settings.app.name,
        "version": settings.app.version,
        "environment": settings.app.environment,
        "components": components,
    }
