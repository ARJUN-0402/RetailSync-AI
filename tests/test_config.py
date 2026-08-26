"""Tests for RetailSync AI configuration and health modules."""

from __future__ import annotations

import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.config import Settings, settings
from src.health import check_configuration, check_data_files, check_database, check_models, get_health_status


class TestConfig:
    def test_settings_singleton(self):
        assert settings is not None
        assert isinstance(settings, Settings)

    def test_app_name(self):
        assert settings.app.name is not None
        assert len(settings.app.name) > 0

    def test_database_path_configured(self):
        assert settings.database.path is not None
        assert len(settings.database.path) > 0

    def test_paths_exist_or_creatable(self):
        settings.paths.ensure_dirs()
        assert settings.paths.processed_data.exists()
        assert settings.paths.models.exists()
        assert settings.paths.database.exists()


class TestHealth:
    def test_check_database_returns_dict(self):
        result = check_database()
        assert isinstance(result, dict)
        assert "status" in result
        assert "path" in result

    def test_check_models_returns_dict(self):
        result = check_models()
        assert isinstance(result, dict)
        assert "status" in result
        assert "available" in result
        assert "missing" in result

    def test_check_data_files_returns_dict(self):
        result = check_data_files()
        assert isinstance(result, dict)
        assert "status" in result
        assert "available" in result
        assert "missing" in result

    def test_check_configuration_returns_dict(self):
        result = check_configuration()
        assert isinstance(result, dict)
        assert "status" in result

    def test_get_health_status_returns_dict(self):
        result = get_health_status()
        assert isinstance(result, dict)
        assert "status" in result
        assert "app" in result
        assert "version" in result
        assert "components" in result
        assert "database" in result["components"]
        assert "models" in result["components"]
        assert "data_files" in result["components"]
        assert "configuration" in result["components"]
