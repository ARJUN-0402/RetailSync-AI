"""Business assumptions and configuration for RetailSync AI KPI calculations.

All financial assumptions are explicitly defined here so they can be
audited, adjusted, and traced back to source data.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BusinessConfig:
    """Container for business assumptions used in KPI calculations.

    Attributes:
        carrying_cost_pct: Annual carrying cost as a fraction of unit cost.
            Industry reference: 20-30% for general merchandise.
        stockout_cost_rate: Stockout cost as a fraction of lost revenue.
            1.0 means 100% of lost revenue is treated as cost.
            Can be increased to include customer churn/opportunity cost.
        stockout_high_risk_multiplier: Estimated stockout units for HIGH risk
            items as a fraction of current shortfall (quantity_on_hand <= 0).
            Used when actual shortfall cannot be measured directly.
        stockout_medium_risk_units: Flat estimate of units at risk for MEDIUM
            stockout risk items.
        overstock_excess_threshold: Threshold for excess inventory as a
            multiplier of max_stock_level. Default 1.0 means anything above
            max_stock_level is excess.
        lead_time_default_days: Default lead time in days when supplier data
            is unavailable.
        safety_stock_z_score: Z-score used for safety stock calculations
            under normal demand distribution.
        forecast_accuracy_period_days: Period over which forecast accuracy
            is evaluated.
        revenue_protected_confidence: Confidence level (0-1) for the
            potential revenue protected estimate. Lower = more conservative.
    """

    carrying_cost_pct: float = 0.25
    stockout_cost_rate: float = 1.0
    stockout_high_risk_multiplier: float = 1.0
    stockout_medium_risk_units: float = 5.0
    overstock_excess_threshold: float = 1.0
    lead_time_default_days: int = 14
    safety_stock_z_score: float = 1.65
    forecast_accuracy_period_days: int = 14
    revenue_protected_confidence: float = 0.5

    def __post_init__(self) -> None:
        if not 0 <= self.carrying_cost_pct <= 1:
            raise ValueError("carrying_cost_pct must be between 0 and 1")
        if not 0 <= self.stockout_cost_rate <= 2:
            raise ValueError("stockout_cost_rate must be between 0 and 2")
        if self.overstock_excess_threshold < 0:
            raise ValueError("overstock_excess_threshold must be non-negative")
        if self.lead_time_default_days <= 0:
            raise ValueError("lead_time_default_days must be positive")
        if self.safety_stock_z_score <= 0:
            raise ValueError("safety_stock_z_score must be positive")
        if self.forecast_accuracy_period_days <= 0:
            raise ValueError("forecast_accuracy_period_days must be positive")
        if not 0 <= self.revenue_protected_confidence <= 1:
            raise ValueError("revenue_protected_confidence must be between 0 and 1")

    @classmethod
    def conservative(cls) -> "BusinessConfig":
        """Return a conservative assumption set (higher costs, lower confidence)."""
        return cls(
            carrying_cost_pct=0.30,
            stockout_cost_rate=1.5,
            stockout_high_risk_multiplier=1.5,
            stockout_medium_risk_units=8.0,
            overstock_excess_threshold=1.0,
            lead_time_default_days=21,
            safety_stock_z_score=2.0,
            forecast_accuracy_period_days=14,
            revenue_protected_confidence=0.3,
        )

    @classmethod
    def aggressive(cls) -> "BusinessConfig":
        """Return an aggressive assumption set (lower costs, higher confidence)."""
        return cls(
            carrying_cost_pct=0.20,
            stockout_cost_rate=0.8,
            stockout_high_risk_multiplier=0.8,
            stockout_medium_risk_units=3.0,
            overstock_excess_threshold=0.8,
            lead_time_default_days=7,
            safety_stock_z_score=1.28,
            forecast_accuracy_period_days=7,
            revenue_protected_confidence=0.7,
        )
