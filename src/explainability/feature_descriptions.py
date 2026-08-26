"""Human-readable descriptions for model features.

This mapping translates raw model feature names (and, where it matters, their
actual values) into business-friendly phrases used by the natural-language
explanation. It is *not* a list of feature-importance values - the phrases are
built dynamically from the feature's SHAP contribution and its real value, so
the generated explanation always reflects the live model output.
"""

from __future__ import annotations

import pandas as pd

# Curated labels for the most decision-relevant features. Anything not listed
# falls back to a title-cased version of the column name.
FEATURE_LABELS: dict[str, str] = {
    "discount_pct": "discount depth",
    "promotion_last_7d": "recent promotion activity",
    "promotion_last_14d": "promotion over the last two weeks",
    "is_holiday": "holiday timing",
    "is_weekend": "weekend timing",
    "is_month_start": "month-start timing",
    "is_month_end": "month-end timing",
    "demand_lag_1d": "yesterday's demand",
    "demand_lag_7d": "demand 7 days ago",
    "demand_lag_14d": "demand 14 days ago",
    "demand_lag_28d": "demand 28 days ago",
    "revenue_lag_1d": "yesterday's revenue",
    "demand_rolling_mean_7d": "recent 7-day average demand",
    "demand_rolling_mean_14d": "recent 14-day average demand",
    "demand_rolling_mean_28d": "recent 28-day average demand",
    "demand_rolling_std_7d": "weekly demand volatility",
    "demand_rolling_std_28d": "monthly demand volatility",
    "demand_rolling_max_7d": "recent peak demand",
    "demand_rolling_min_7d": "recent low demand",
    "demand_expanding_mean": "long-run average demand",
    "demand_cv_28d": "demand variability",
    "zero_demand_pct_28d": "recent zero-demand share",
    "store_avg_demand": "store's usual demand",
    "product_avg_demand": "product's usual demand",
    "category_avg_demand": "category's usual demand",
    "store_type_avg_demand": "store-type usual demand",
    "stock_coverage_days": "stock coverage",
    "stock_vs_reorder": "stock versus reorder point",
    "stock_vs_max": "stock versus maximum level",
    "stock_to_max_ratio": "stock-to-capacity ratio",
    "quantity_on_hand": "inventory on hand",
    "reorder_point": "reorder point",
    "max_stock_level": "maximum stock level",
    "effective_price": "effective price",
    "price_margin_pct": "price margin",
    "price_vs_cost": "price versus cost",
    "holiday_multiplier": "holiday multiplier",
    "day_of_week": "day of week",
    "day_of_month": "day of month",
    "week_of_year": "week of year",
    "month": "month of year",
    "quarter": "fiscal quarter",
    "trend_linear": "overall demand trend",
}


def feature_label(feature: str) -> str:
    """Return a readable label for a feature name."""
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


def _is_active_flag(value: float) -> bool:
    return pd.notna(value) and float(value) > 0.5


def describe_contribution(
    feature: str,
    value: float,
    median: float | None,
) -> str:
    """Build a business-friendly phrase for a single feature's contribution.

    The phrase is generated from the feature name and its *actual* value
    (compared against the background median), so it adapts to whatever the
    model is currently looking at - it is never hard-coded.
    """
    label = feature_label(feature)
    active = _is_active_flag(value)

    # Known on/off signals.
    if feature in {"promotion_last_7d", "promotion_last_14d"}:
        return "the product is under promotion" if active else "the product is not promoted"
    if feature == "is_holiday":
        return "the forecast date is a holiday" if active else "the forecast date is not a holiday"
    if feature == "is_weekend":
        return "the forecast date is a weekend" if active else "the forecast date is a weekday"
    if feature in {"is_month_start", "is_month_end"}:
        return f"the date is near a month {'start' if feature == 'is_month_start' else 'end'}"

    # Demand-magnitude features: compare to the typical level.
    if median is not None and pd.notna(value):
        above = value > median
        rel = "above normal" if above else "below normal"
        # Holiday multiplier and margins read a little differently.
        if feature in {"holiday_multiplier", "price_margin_pct", "discount_pct", "price_vs_cost"}:
            return f"{label} is {rel} (value {value:.2f})"
        return f"{label} is {rel}"

    return f"{label} (value {value})"
