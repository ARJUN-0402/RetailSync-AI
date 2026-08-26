"""Input validation utilities for RetailSync AI.

Provides reusable validation functions for model inputs, uploaded data,
filters, API inputs, database values, and configuration.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when input validation fails."""


def validate_product_id(value: Any) -> str:
    if not value or not isinstance(value, str):
        raise ValidationError("product_id must be a non-empty string")
    if not re.match(r"^P\d{3}$", value):
        raise ValidationError(f"Invalid product_id format: {value}")
    return value


def validate_store_id(value: Any) -> str:
    if not value or not isinstance(value, str):
        raise ValidationError("store_id must be a non-empty string")
    if not re.match(r"^ST\d{2}$", value):
        raise ValidationError(f"Invalid store_id format: {value}")
    return value


def validate_warehouse_id(value: Any) -> str:
    if not value or not isinstance(value, str):
        raise ValidationError("warehouse_id must be a non-empty string")
    if not re.match(r"^WH\d{2}$", value):
        raise ValidationError(f"Invalid warehouse_id format: {value}")
    return value


def validate_positive_int(value: Any, name: str = "value") -> int:
    try:
        ivalue = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be an integer") from exc
    if ivalue <= 0:
        raise ValidationError(f"{name} must be positive, got {ivalue}")
    return ivalue


def validate_non_negative_int(value: Any, name: str = "value") -> int:
    try:
        ivalue = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be an integer") from exc
    if ivalue < 0:
        raise ValidationError(f"{name} must be non-negative, got {ivalue}")
    return ivalue


def validate_float_range(value: Any, min_val: float, max_val: float, name: str = "value") -> float:
    try:
        fvalue = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be a number") from exc
    if not (min_val <= fvalue <= max_val):
        raise ValidationError(f"{name} must be between {min_val} and {max_val}, got {fvalue}")
    return fvalue


def validate_csv_upload(file_obj: Any, allowed_extensions: tuple[str, ...] = (".csv",)) -> str:
    if file_obj is None:
        raise ValidationError("No file uploaded")
    filename = getattr(file_obj, "name", "")
    if not filename:
        raise ValidationError("Uploaded file has no name")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed_extensions:
        raise ValidationError(f"Invalid file type: {ext}. Allowed: {allowed_extensions}")
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)
    return filename


def validate_date(value: Any) -> str:
    if isinstance(value, str):
        return value
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    raise ValidationError(f"Invalid date value: {value}")


def validate_filter_string(value: Any, max_length: int = 100) -> str:
    if not isinstance(value, str):
        raise ValidationError("Filter must be a string")
    if len(value) > max_length:
        raise ValidationError(f"Filter string exceeds max length {max_length}")
    if not re.match(r"^[a-zA-Z0-9_\- ]+$", value):
        raise ValidationError("Filter contains invalid characters")
    return value.strip()


def validate_config_value(key: str, value: Any) -> Any:
    """Validate a configuration value."""
    if not key or not isinstance(key, str):
        raise ValidationError("Config key must be a non-empty string")
    if not re.match(r"^[A-Z][A-Z0-9_]*$", key):
        raise ValidationError(f"Invalid config key format: {key}")
    return value
