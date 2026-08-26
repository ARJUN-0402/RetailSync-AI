"""Tests for RetailSync AI business metrics.

Run with:
    pytest tests/test_business_metrics.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.business_metrics.config import BusinessConfig
from src.business_metrics.kpi import (
    compute_executive_kpis,
    compute_forecast_accuracy,
    compute_inventory_carrying_cost,
    compute_overstock_value,
    compute_potential_revenue_protected,
    compute_stockout_cost,
)
from src.business_metrics.reorder import generate_reorder_recommendations
from src.business_metrics.utils import (
    clean_series,
    compute_smape_batch,
    safe_div,
    smape,
)

# ============================================================
# Helper fixtures
# ============================================================


@pytest.fixture
def base_config():
    return BusinessConfig(
        carrying_cost_pct=0.25,
        stockout_cost_rate=1.0,
        stockout_high_risk_multiplier=1.0,
        stockout_medium_risk_units=5.0,
        overstock_excess_threshold=1.0,
        lead_time_default_days=14,
        safety_stock_z_score=1.65,
        forecast_accuracy_period_days=14,
        revenue_protected_confidence=0.5,
    )


@pytest.fixture
def sample_features():
    dates = pd.date_range("2025-01-01", periods=220, freq="D")
    products = [f"P{i:03d}" for i in range(1, 4)]
    stores = [f"ST{i:02d}" for i in range(1, 3)]

    rows = []
    rng = np.random.default_rng(42)
    for date in dates:
        for product in products:
            for store in stores:
                rows.append(
                    {
                        "date": date,
                        "product_id": product,
                        "store_id": store,
                        "category": "Electronics",
                        "quantity_sold": float(rng.integers(0, 20)),
                        "target_demand_1d": float(rng.integers(0, 20)),
                        "unit_price": 100.0,
                        "cost_price": 50.0,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def sample_products():
    return pd.DataFrame(
        {
            "product_id": [f"P{i:03d}" for i in range(1, 4)],
            "product_name": [f"Product {i}" for i in range(1, 4)],
            "category": ["Electronics", "Clothing", "Groceries"],
            "unit_price": [100.0, 50.0, 10.0],
            "cost_price": [50.0, 25.0, 5.0],
            "supplier_id": ["S01", "S02", "S03"],
        }
    )


@pytest.fixture
def sample_inv_intel():
    return pd.DataFrame(
        {
            "inventory_id": [1, 2, 3, 4, 5],
            "date": ["2025-08-08"] * 5,
            "product_id": ["P001", "P002", "P003", "P004", "P005"],
            "store_id": ["ST01", "ST01", "ST01", "ST01", "ST01"],
            "quantity_on_hand": [0, 10, 200, 50, -5],
            "reorder_point": [20, 20, 100, 30, 10],
            "max_stock_level": [100, 100, 150, 60, 50],
            "warehouse_id": ["WH01", "WH01", "WH01", "WH01", "WH01"],
            "stockout_risk": ["HIGH", "MEDIUM", "LOW", "LOW", "HIGH"],
            "overstock_risk": ["LOW", "HIGH", "HIGH", "MEDIUM", "LOW"],
            "reorder_urgency": ["URGENT", "SOON", "NONE", "MONITOR", "URGENT"],
            "forecast_demand_7d": [5.0, 10.0, 30.0, 15.0, 2.0],
            "forecast_demand_14d": [10.0, 20.0, 60.0, 30.0, 4.0],
            "demand_cv_28d": [0.5, 1.2, 0.8, 0.3, 2.5],
            "stock_coverage_days": [0.0, 1.0, 6.67, 3.33, -2.5],
        }
    )


@pytest.fixture
def sample_forecasts():
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-08-10", periods=14, freq="D").repeat(2),
            "product_id": ["P001"] * 14 + ["P002"] * 14,
            "store_id": ["ST01"] * 14 + ["ST01"] * 14,
            "forecast_demand": np.linspace(10, 24, 14).tolist() + np.linspace(5, 19, 14).tolist(),
            "forecast_revenue": np.linspace(1000, 2400, 14).tolist() + np.linspace(250, 950, 14).tolist(),
            "unit_price": [100.0] * 14 + [50.0] * 14,
            "category": ["Electronics"] * 14 + ["Clothing"] * 14,
            "store_type": ["Urban"] * 14 + ["Urban"] * 14,
            "model": ["TestModel"] * 28,
        }
    )


@pytest.fixture
def sample_suppliers():
    return pd.DataFrame(
        {
            "supplier_id": ["S01", "S02", "S03"],
            "supplier_name": ["Supplier 1", "Supplier 2", "Supplier 3"],
            "country": ["USA", "China", "Germany"],
            "lead_time_days": [14, 7, 21],
            "reliability_score": [0.9, 0.85, 0.95],
        }
    )


# ============================================================
# UTILS TESTS
# ============================================================


class TestUtils:
    def test_safe_div_normal(self):
        assert safe_div(10, 2) == 5.0

    def test_safe_div_zero_denominator(self):
        assert safe_div(10, 0) == 0.0

    def test_safe_div_custom_default(self):
        assert safe_div(10, 0, default=-1.0) == -1.0

    def test_smape_identical(self):
        assert smape(10, 10) == 0.0

    def test_smape_both_zero(self):
        assert smape(0, 0) == 0.0

    def test_smape_nonzero(self):
        val = smape(0, 10)
        assert val == 200.0

    def test_smape_half(self):
        val = smape(10, 5)
        assert abs(val - 66.666) < 0.01

    def test_compute_smape_batch(self):
        y_true = np.array([1.0, 2.0, 0.0, 4.0])
        y_pred = np.array([1.5, 2.5, 0.5, 3.5])
        val = compute_smape_batch(y_true, y_pred)
        assert val > 0

    def test_clean_series(self):
        s = pd.Series([1.0, np.inf, -np.inf, np.nan, 5.0])
        cleaned = clean_series(s)
        assert cleaned.tolist() == [1.0, 0.0, 0.0, 0.0, 5.0]

    def test_clamp(self):
        from src.business_metrics.utils import clamp
        assert clamp(5, 0, 10) == 5
        assert clamp(-5, 0, 10) == 0
        assert clamp(15, 0, 10) == 10

    def test_safe_num_valid(self):
        from src.business_metrics.utils import safe_num
        assert safe_num(5.5) == 5.5

    def test_safe_num_nan(self):
        from src.business_metrics.utils import safe_num
        assert safe_num(float("nan")) == 0.0

    def test_safe_num_inf(self):
        from src.business_metrics.utils import safe_num
        assert safe_num(float("inf")) == 0.0

    def test_safe_num_none(self):
        from src.business_metrics.utils import safe_num
        assert safe_num(None) == 0.0


# ============================================================
# CONFIG TESTS
# ============================================================


class TestConfig:
    def test_default_config(self):
        cfg = BusinessConfig()
        assert cfg.carrying_cost_pct == 0.25
        assert cfg.stockout_cost_rate == 1.0

    def test_conservive_factory(self):
        cfg = BusinessConfig.conservative()
        assert cfg.carrying_cost_pct == 0.30
        assert cfg.lead_time_default_days == 21

    def test_aggressive_factory(self):
        cfg = BusinessConfig.aggressive()
        assert cfg.carrying_cost_pct == 0.20
        assert cfg.lead_time_default_days == 7

    def test_invalid_carrying_cost(self):
        with pytest.raises(ValueError):
            BusinessConfig(carrying_cost_pct=-0.1)

    def test_invalid_stockout_cost_rate(self):
        with pytest.raises(ValueError):
            BusinessConfig(stockout_cost_rate=2.5)

    def test_invalid_lead_time(self):
        with pytest.raises(ValueError):
            BusinessConfig(lead_time_default_days=0)

    def test_invalid_confidence(self):
        with pytest.raises(ValueError):
            BusinessConfig(revenue_protected_confidence=1.5)


# ============================================================
# FORECAST ACCURACY TESTS
# ============================================================


class TestForecastAccuracy:
    def test_no_data(self, base_config):
        result = compute_forecast_accuracy(None, config=base_config)
        assert "note" in result
        assert result["overall"] == {}

    def test_missing_columns(self, base_config):
        df = pd.DataFrame({"date": [pd.Timestamp("2025-07-01")]})
        result = compute_forecast_accuracy(df, config=base_config)
        assert "Missing required columns" in result["note"]

    def test_no_test_data(self, base_config):
        df = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-01")],
                "product_id": ["P001"],
                "store_id": ["ST01"],
                "category": ["A"],
                "target_demand_1d": [5.0],
            }
        )
        result = compute_forecast_accuracy(df, config=base_config)
        assert "No test data" in result["note"]

    def test_with_model_package(self, base_config, sample_features):
        class FakeModel:
            def predict(self, X):
                return np.ones(len(X), dtype=float)

        # Use a real numeric column from sample_features as a fake feature
        fake_model = FakeModel()
        pkg = {
            "model": fake_model,
            "feature_cols": ["unit_price", "cost_price"],
            "model_name": "FakeModel",
        }
        result = compute_forecast_accuracy(sample_features, model_package=pkg, config=base_config)
        assert "overall" in result
        assert result["overall"]["model"] == "FakeModel"
        assert result["overall"]["test_rows"] > 0
        assert "by_product" in result
        assert "by_store" in result
        assert "by_category" in result

    def test_metrics_non_negative(self, base_config, sample_features):
        class FakeModel:
            def predict(self, X):
                return np.zeros(len(X), dtype=float)

        pkg = {
            "model": FakeModel(),
            "feature_cols": ["unit_price", "cost_price"],
            "model_name": "ZeroModel",
        }
        result = compute_forecast_accuracy(sample_features, model_package=pkg, config=base_config)
        mae = result["overall"]["mae"]
        rmse = result["overall"]["rmse"]
        assert mae >= 0
        assert rmse >= 0


# ============================================================
# INVENTORY CARRYING COST TESTS
# ============================================================


class TestInventoryCarryingCost:
    def test_empty_inputs(self, base_config):
        result = compute_inventory_carrying_cost(None, None, base_config)
        assert result["estimated_carrying_cost"] == 0.0

    def test_basic_calculation(self, base_config, sample_inv_intel, sample_products):
        result = compute_inventory_carrying_cost(
            sample_inv_intel, sample_products, base_config, period_days=365
        )
        # P001: 0 * 50 = 0
        # P002: 10 * 25 = 250
        # P003: 200 * 5 = 1000
        # P004: 50 * 0 (missing in products) = 0
        # P005: 0 (negative clamped) * 0 (missing in products) = 0
        expected_value = 250 + 1000
        assert result["total_inventory_value"] == expected_value
        expected_cost = expected_value * (0.25 / 365) * 365
        assert result["estimated_carrying_cost"] == expected_cost

    def test_negative_quantity_clamped(self, base_config, sample_inv_intel, sample_products):
        result = compute_inventory_carrying_cost(
            sample_inv_intel, sample_products, base_config
        )
        # P005 has quantity_on_hand = -5, should be clamped to 0
        p005_value = result["by_product"][result["by_product"]["product_id"] == "P005"]
        if not p005_value.empty:
            assert p005_value.iloc[0]["inventory_value"] == 0.0

    def test_zero_products(self, base_config, sample_inv_intel):
        result = compute_inventory_carrying_cost(
            sample_inv_intel, pd.DataFrame(), base_config
        )
        assert result["total_inventory_value"] == 0.0


# ============================================================
# STOCKOUT COST TESTS
# ============================================================


class TestStockoutCost:
    def test_empty_inputs(self, base_config):
        result = compute_stockout_cost(None, None, base_config)
        assert result["estimated_stockout_cost"] == 0.0

    def test_high_risk_estimation(self, base_config, sample_inv_intel, sample_products):
        result = compute_stockout_cost(sample_inv_intel, sample_products, base_config)
        # P001: HIGH risk, qty=0, unit_price missing (merged from products = 50)
        # Actually unit_price comes from products merged on product_id
        # P001 has unit_price 100 in sample_products
        # stockout_units = 0 * 1.0 = 0
        # P005: HIGH risk, qty=-5, stockout_units = abs(-5) * 1.0 = 5, unit_price=10
        # lost_revenue = 5 * 10 = 50
        # stockout_cost = 50 * 1.0 = 50
        assert result["high_risk_count"] == 2  # P001 and P005
        assert result["estimated_stockout_units"] > 0

    def test_medium_risk_flat_units(self, base_config, sample_inv_intel, sample_products):
        result = compute_stockout_cost(sample_inv_intel, sample_products, base_config)
        # P002: MEDIUM risk -> 5.0 flat units
        assert result["medium_risk_count"] == 1

    def test_zero_inventory_no_stockout(self, base_config):
        inv = pd.DataFrame(
            {
                "product_id": ["P001"],
                "store_id": ["ST01"],
                "quantity_on_hand": [0],
                "stockout_risk": ["HIGH"],
                "unit_price": [100.0],
            }
        )
        products = pd.DataFrame(
            {
                "product_id": ["P001"],
                "unit_price": [100.0],
                "cost_price": [50.0],
            }
        )
        result = compute_stockout_cost(inv, products, base_config)
        # quantity_on_hand = 0, abs(0) * 1.0 = 0
        assert result["estimated_stockout_units"] == 0.0
        assert result["estimated_stockout_cost"] == 0.0

    def test_missing_unit_price(self, base_config):
        inv = pd.DataFrame(
            {
                "product_id": ["P001"],
                "store_id": ["ST01"],
                "quantity_on_hand": [10],
                "stockout_risk": ["HIGH"],
            }
        )
        products = pd.DataFrame(
            {
                "product_id": ["P001"],
                "unit_price": [0.0],
                "cost_price": [50.0],
            }
        )
        result = compute_stockout_cost(inv, products, base_config)
        assert result["estimated_lost_revenue"] == 0.0


# ============================================================
# OVERSTOCK VALUE TESTS
# ============================================================


class TestOverstockValue:
    def test_empty_inputs(self, base_config):
        result = compute_overstock_value(None, None, base_config)
        assert result["overstock_inventory_value"] == 0.0

    def test_basic_calculation(self, base_config, sample_inv_intel, sample_products):
        result = compute_overstock_value(
            sample_inv_intel, sample_products, base_config
        )
        # P002: qty=10, max=100, excess=0
        # P003: qty=200, max=150, excess=50, cost=5 => value=250
        # P004: qty=50, max=60, excess=0
        assert result["overstock_items_count"] >= 1
        assert result["excess_units"] >= 50

    def test_no_excess_when_below_max(self, base_config):
        inv = pd.DataFrame(
            {
                "product_id": ["P001"],
                "store_id": ["ST01"],
                "quantity_on_hand": [50],
                "max_stock_level": [100],
                "overstock_risk": ["LOW"],
            }
        )
        products = pd.DataFrame(
            {
                "product_id": ["P001"],
                "cost_price": [10.0],
            }
        )
        result = compute_overstock_value(inv, products, base_config)
        assert result["excess_units"] == 0.0
        assert result["overstock_inventory_value"] == 0.0

    def test_negative_quantity_clamped(self, base_config):
        inv = pd.DataFrame(
            {
                "product_id": ["P001"],
                "store_id": ["ST01"],
                "quantity_on_hand": [-10],
                "max_stock_level": [100],
                "overstock_risk": ["LOW"],
            }
        )
        products = pd.DataFrame(
            {
                "product_id": ["P001"],
                "cost_price": [10.0],
            }
        )
        result = compute_overstock_value(inv, products, base_config)
        assert result["excess_units"] == 0.0


# ============================================================
# POTENTIAL REVENUE PROTECTED TESTS
# ============================================================


class TestPotentialRevenueProtected:
    def test_empty_inputs(self, base_config):
        result = compute_potential_revenue_protected({}, {}, base_config)
        assert result["estimated_revenue_protected"] == 0.0

    def test_basic_estimate(self, base_config):
        stockout = {"estimated_lost_revenue": 1000.0}
        overstock = {"overstock_inventory_value": 500.0}
        result = compute_potential_revenue_protected(
            stockout, overstock, base_config
        )
        # 1000 + 500*0.3 = 1150; * 0.5 confidence = 575
        assert abs(result["estimated_revenue_protected"] - 575.0) < 0.01

    def test_high_confidence(self):
        cfg = BusinessConfig(revenue_protected_confidence=1.0)
        stockout = {"estimated_lost_revenue": 200.0}
        overstock = {"overstock_inventory_value": 300.0}
        result = compute_potential_revenue_protected(
            stockout, overstock, cfg
        )
        assert abs(result["estimated_revenue_protected"] - 290.0) < 0.01

    def test_zero_confidence(self):
        cfg = BusinessConfig(revenue_protected_confidence=0.0)
        stockout = {"estimated_lost_revenue": 200.0}
        overstock = {"overstock_inventory_value": 300.0}
        result = compute_potential_revenue_protected(
            stockout, overstock, cfg
        )
        assert result["estimated_revenue_protected"] == 0.0


# ============================================================
# EXECUTIVE KPIs TESTS
# ============================================================


class TestExecutiveKpis:
    def test_empty_inputs(self, base_config):
        result = compute_executive_kpis(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}, {}, {}, {}, {}, base_config
        )
        assert result["total_inventory_value"] == 0.0
        assert result["estimated_carrying_cost"] == 0.0

    def test_populated_kpis(self, base_config, sample_inv_intel, sample_products, sample_forecasts):
        stockout = compute_stockout_cost(sample_inv_intel, sample_products, base_config)
        overstock = compute_overstock_value(sample_inv_intel, sample_products, base_config)
        carrying = compute_inventory_carrying_cost(
            sample_inv_intel, sample_products, base_config
        )
        revenue = compute_potential_revenue_protected(stockout, overstock, base_config)
        forecast_acc = compute_forecast_accuracy(
            pd.DataFrame(), model_package=None, config=base_config
        )

        result = compute_executive_kpis(
            sample_inv_intel,
            sample_products,
            sample_forecasts,
            stockout,
            overstock,
            carrying,
            revenue,
            forecast_acc,
            base_config,
        )

        assert "total_inventory_value" in result
        assert "estimated_carrying_cost" in result
        assert "stockout_exposure" in result
        assert "overstock_value" in result
        assert "products_requiring_reorder" in result
        assert "potential_revenue_protected" in result
        assert "forecast_accuracy_smape" in result
        assert result["total_inventory_value"] >= 0


# ============================================================
# REORDER RECOMMENDATION TESTS
# ============================================================


class TestReorderRecommendations:
    def test_empty_input(self, base_config):
        result = generate_reorder_recommendations(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), base_config
        )
        assert result.empty

    def test_basic_recommendations(self, base_config, sample_inv_intel, sample_products, sample_forecasts, sample_suppliers):
        result = generate_reorder_recommendations(
            sample_inv_intel, sample_products, sample_forecasts, sample_suppliers, base_config
        )
        assert not result.empty
        assert "recommended_quantity" in result.columns
        assert "reorder_urgency_computed" in result.columns
        assert "reorder_reasoning" in result.columns
        assert "expected_coverage_days" in result.columns

    def test_zero_inventory_gets_quantity(self, base_config):
        inv = pd.DataFrame(
            {
                "product_id": ["P001"],
                "store_id": ["ST01"],
                "quantity_on_hand": [0],
                "reorder_point": [20],
                "max_stock_level": [100],
                "forecast_demand_7d": [10.0],
                "forecast_demand_14d": [20.0],
                "demand_cv_28d": [0.5],
            }
        )
        products = pd.DataFrame(
            {
                "product_id": ["P001"],
                "cost_price": [50.0],
                "supplier_id": ["S01"],
            }
        )
        forecasts = pd.DataFrame(
            {
                "date": pd.date_range("2025-08-10", periods=14, freq="D"),
                "product_id": ["P001"] * 14,
                "store_id": ["ST01"] * 14,
                "forecast_demand": [5.0] * 14,
                "forecast_revenue": [250.0] * 14,
                "unit_price": [50.0] * 14,
                "category": ["Electronics"] * 14,
                "store_type": ["Urban"] * 14,
                "model": ["Test"] * 14,
            }
        )
        suppliers = pd.DataFrame(
            {
                "supplier_id": ["S01"],
                "lead_time_days": [14],
            }
        )
        result = generate_reorder_recommendations(
            inv, products, forecasts, suppliers, base_config
        )
        assert not result.empty
        assert result.iloc[0]["recommended_quantity"] > 0

    def test_adequate_stock_low_urgency(self, base_config):
        inv = pd.DataFrame(
            {
                "product_id": ["P001"],
                "store_id": ["ST01"],
                "quantity_on_hand": [500],
                "reorder_point": [20],
                "max_stock_level": [100],
                "forecast_demand_7d": [5.0],
                "forecast_demand_14d": [10.0],
                "demand_cv_28d": [0.1],
            }
        )
        products = pd.DataFrame(
            {
                "product_id": ["P001"],
                "cost_price": [50.0],
                "supplier_id": ["S01"],
            }
        )
        forecasts = pd.DataFrame(
            {
                "date": pd.date_range("2025-08-10", periods=14, freq="D"),
                "product_id": ["P001"] * 14,
                "store_id": ["ST01"] * 14,
                "forecast_demand": [1.0] * 14,
                "forecast_revenue": [50.0] * 14,
                "unit_price": [50.0] * 14,
                "category": ["Electronics"] * 14,
                "store_type": ["Urban"] * 14,
                "model": ["Test"] * 14,
            }
        )
        suppliers = pd.DataFrame(
            {
                "supplier_id": ["S01"],
                "lead_time_days": [14],
            }
        )
        result = generate_reorder_recommendations(
            inv, products, forecasts, suppliers, base_config
        )
        assert not result.empty
        urgency = result.iloc[0]["reorder_urgency_computed"]
        assert urgency in ["NONE", "MONITOR"]


# ============================================================
# EDGE CASE TESTS
# ============================================================


class TestEdgeCases:
    def test_zero_sales(self, base_config):
        inv = pd.DataFrame(
            {
                "product_id": ["P001"],
                "store_id": ["ST01"],
                "quantity_on_hand": [0],
                "reorder_point": [20],
                "max_stock_level": [100],
                "stockout_risk": ["HIGH"],
                "overstock_risk": ["LOW"],
            }
        )
        products = pd.DataFrame(
            {
                "product_id": ["P001"],
                "unit_price": [100.0],
                "cost_price": [50.0],
            }
        )
        stockout = compute_stockout_cost(inv, products, base_config)
        assert stockout["estimated_stockout_units"] == 0.0

    def test_missing_values_in_inventory(self, base_config, sample_products):
        inv = pd.DataFrame(
            {
                "product_id": ["P001"],
                "store_id": ["ST01"],
                "quantity_on_hand": [np.nan],
                "reorder_point": [np.nan],
                "max_stock_level": [np.nan],
                "stockout_risk": ["HIGH"],
                "overstock_risk": ["LOW"],
            }
        )
        result = compute_overstock_value(inv, sample_products, base_config)
        assert result["excess_units"] == 0.0

    def test_negative_cost_price(self, base_config):
        inv = pd.DataFrame(
            {
                "product_id": ["P001"],
                "store_id": ["ST01"],
                "quantity_on_hand": [10],
                "max_stock_level": [100],
                "overstock_risk": ["LOW"],
            }
        )
        products = pd.DataFrame(
            {
                "product_id": ["P001"],
                "cost_price": [-50.0],
            }
        )
        result = compute_overstock_value(inv, products, base_config)
        assert result["overstock_inventory_value"] == 0.0

    def test_extreme_demand_forecast(self, base_config):
        inv = pd.DataFrame(
            {
                "product_id": ["P001"],
                "store_id": ["ST01"],
                "quantity_on_hand": [10],
                "reorder_point": [20],
                "max_stock_level": [100],
                "forecast_demand_7d": [1e9],
                "forecast_demand_14d": [2e9],
                "demand_cv_28d": [10.0],
            }
        )
        products = pd.DataFrame(
            {
                "product_id": ["P001"],
                "cost_price": [50.0],
                "supplier_id": ["S01"],
            }
        )
        forecasts = pd.DataFrame(
            {
                "date": pd.date_range("2025-08-10", periods=14, freq="D"),
                "product_id": ["P001"] * 14,
                "store_id": ["ST01"] * 14,
                "forecast_demand": [1e8] * 14,
                "forecast_revenue": [1e10] * 14,
                "unit_price": [100.0] * 14,
                "category": ["Electronics"] * 14,
                "store_type": ["Urban"] * 14,
                "model": ["Test"] * 14,
            }
        )
        suppliers = pd.DataFrame(
            {
                "supplier_id": ["S01"],
                "lead_time_days": [14],
            }
        )
        result = generate_reorder_recommendations(
            inv, products, forecasts, suppliers, base_config
        )
        assert not result.empty
        assert np.isfinite(result["recommended_quantity"].iloc[0])
        assert result["recommended_quantity"].iloc[0] >= 0
