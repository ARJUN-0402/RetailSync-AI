"""Utility functions for business metrics calculations."""

from __future__ import annotations

import math


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that returns a default value when denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator


def smape(y_true: float, y_pred: float) -> float:
    """Symmetric Mean Absolute Percentage Error for a single pair."""
    if y_true == 0 and y_pred == 0:
        return 0.0
    denom = abs(y_true) + abs(y_pred)
    if denom == 0:
        return 0.0
    return abs(y_pred - y_true) / (denom / 2.0) * 100.0


def compute_smape_batch(y_true, y_pred) -> float:
    """Compute sMAPE over arrays."""
    mask = (y_true + y_pred) != 0
    if not mask.any():
        return 0.0
    return (
        abs(y_pred[mask] - y_true[mask]) / ((abs(y_true[mask]) + abs(y_pred[mask])) / 2)
    ).mean() * 100.0


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))


def clean_series(series):
    """Replace inf and -inf with NaN, then fill NaN with 0."""
    return series.replace([float("inf"), float("-inf")], float("nan")).fillna(0)


def safe_num(value, default: float = 0.0) -> float:
    """Convert value to float, returning default if invalid."""
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default
