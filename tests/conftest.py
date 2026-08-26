"""Shared pytest configuration and fixtures for RetailSync AI tests."""

from __future__ import annotations

import os

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


@pytest.fixture
def project_root() -> str:
    return PROJECT_ROOT


@pytest.fixture
def processed_dir() -> str:
    return os.path.join(PROJECT_ROOT, "data", "processed")


@pytest.fixture
def models_dir() -> str:
    return os.path.join(PROJECT_ROOT, "models")


@pytest.fixture
def db_path() -> str:
    return os.path.join(PROJECT_ROOT, "database", "retailsync.db")
