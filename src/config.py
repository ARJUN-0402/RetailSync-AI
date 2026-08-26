"""Centralized configuration for RetailSync AI.

All configuration values are loaded from environment variables with sensible
defaults. Never hardcode secrets or environment-specific paths.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DatabaseConfig:
    url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///database/retailsync.db"))
    path: str = field(default_factory=lambda: os.getenv("DATABASE_PATH", "database/retailsync.db"))

    def __post_init__(self) -> None:
        if not self.url:
            self.url = "sqlite:///database/retailsync.db"
        if not self.path:
            self.path = str(Path(self.url.replace("sqlite:///", "").replace("sqlite://", "")).resolve())


@dataclass
class PathsConfig:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    raw_data: Path = field(default_factory=lambda: Path(os.getenv("RAW_DATA_PATH", "data/raw")))
    processed_data: Path = field(default_factory=lambda: Path(os.getenv("PROCESSED_DATA_PATH", "data/processed")))
    models: Path = field(default_factory=lambda: Path(os.getenv("MODELS_PATH", "models")))
    docs: Path = field(default_factory=lambda: Path(os.getenv("DOCS_PATH", "docs")))
    database: Path = field(default_factory=lambda: Path(os.getenv("DATABASE_DIR", "database")))
    logs: Path = field(default_factory=lambda: Path(os.getenv("LOG_DIR", "logs")))

    def ensure_dirs(self) -> None:
        for path in [self.raw_data, self.processed_data, self.models, self.docs, self.database, self.logs]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class AppConfig:
    name: str = field(default_factory=lambda: os.getenv("APP_NAME", "RetailSync AI"))
    version: str = field(default_factory=lambda: os.getenv("APP_VERSION", "2.0.0"))
    environment: str = field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    debug: bool = field(default_factory=lambda: os.getenv("APP_DEBUG", "false").lower() in ("1", "true", "yes"))


@dataclass
class LoggingConfig:
    level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    file: str = field(default_factory=lambda: os.getenv("LOG_FILE", "logs/retailsync.log"))


@dataclass
class DashboardConfig:
    port: int = field(default_factory=lambda: int(os.getenv("STREAMLIT_PORT", "8501")))
    host: str = field(default_factory=lambda: os.getenv("STREAMLIT_HOST", "localhost"))
    server_port: int = field(default_factory=lambda: int(os.getenv("STREAMLIT_SERVER_PORT", "8501")))
    server_address: str = field(default_factory=lambda: os.getenv("STREAMLIT_SERVER_ADDRESS", "0.0.0.0"))


@dataclass
class Settings:
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    app: AppConfig = field(default_factory=AppConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)

    @classmethod
    def load(cls) -> "Settings":
        settings = cls()
        settings.paths.ensure_dirs()
        return settings


settings = Settings.load()
