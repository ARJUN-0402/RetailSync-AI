"""Tests for RetailSync AI input validation utilities."""

from __future__ import annotations

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.validation import (
    ValidationError,
    validate_csv_upload,
    validate_date,
    validate_filter_string,
    validate_float_range,
    validate_non_negative_int,
    validate_positive_int,
    validate_product_id,
    validate_store_id,
    validate_warehouse_id,
)


class TestValidation:
    def test_validate_product_id_valid(self):
        assert validate_product_id("P001") == "P001"

    def test_validate_product_id_invalid_format(self):
        with pytest.raises(ValidationError):
            validate_product_id("INVALID")

    def test_validate_product_id_empty(self):
        with pytest.raises(ValidationError):
            validate_product_id("")

    def test_validate_store_id_valid(self):
        assert validate_store_id("ST01") == "ST01"

    def test_validate_store_id_invalid_format(self):
        with pytest.raises(ValidationError):
            validate_store_id("INVALID")

    def test_validate_warehouse_id_valid(self):
        assert validate_warehouse_id("WH01") == "WH01"

    def test_validate_warehouse_id_invalid_format(self):
        with pytest.raises(ValidationError):
            validate_warehouse_id("INVALID")

    def test_validate_positive_int_valid(self):
        assert validate_positive_int(5) == 5

    def test_validate_positive_int_zero(self):
        with pytest.raises(ValidationError):
            validate_positive_int(0)

    def test_validate_positive_int_negative(self):
        with pytest.raises(ValidationError):
            validate_positive_int(-1)

    def test_validate_positive_int_string(self):
        with pytest.raises(ValidationError):
            validate_positive_int("abc")

    def test_validate_non_negative_int_valid(self):
        assert validate_non_negative_int(0) == 0

    def test_validate_non_negative_int_negative(self):
        with pytest.raises(ValidationError):
            validate_non_negative_int(-1)

    def test_validate_float_range_valid(self):
        assert validate_float_range(0.5, 0.0, 1.0) == 0.5

    def test_validate_float_range_below_min(self):
        with pytest.raises(ValidationError):
            validate_float_range(-0.1, 0.0, 1.0)

    def test_validate_float_range_above_max(self):
        with pytest.raises(ValidationError):
            validate_float_range(1.5, 0.0, 1.0)

    def test_validate_date_string(self):
        assert validate_date("2025-01-01") == "2025-01-01"

    def test_validate_date_datetime(self):
        import datetime
        dt = datetime.date(2025, 1, 1)
        assert validate_date(dt) == "2025-01-01"

    def test_validate_date_invalid(self):
        with pytest.raises(ValidationError):
            validate_date(12345)

    def test_validate_filter_string_valid(self):
        assert validate_filter_string("Electronics") == "Electronics"

    def test_validate_filter_string_too_long(self):
        with pytest.raises(ValidationError):
            validate_filter_string("a" * 101)

    def test_validate_filter_string_invalid_chars(self):
        with pytest.raises(ValidationError):
            validate_filter_string("test@script")

    def test_validate_csv_upload_valid(self):
        class FakeFile:
            name = "data.csv"
        result = validate_csv_upload(FakeFile())
        assert result == "data.csv"

    def test_validate_csv_upload_none(self):
        with pytest.raises(ValidationError):
            validate_csv_upload(None)

    def test_validate_csv_upload_no_name(self):
        class FakeFile:
            pass
        with pytest.raises(ValidationError):
            validate_csv_upload(FakeFile())

    def test_validate_csv_upload_wrong_extension(self):
        class FakeFile:
            name = "data.txt"
        with pytest.raises(ValidationError):
            validate_csv_upload(FakeFile())
