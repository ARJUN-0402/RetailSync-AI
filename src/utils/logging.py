"""Structured logging setup for RetailSync AI.

Provides a configured logger that writes to both console and file.
Never logs secrets or sensitive credentials.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

from src.config import settings


_sensitive_patterns = [
    "api_key", "apikey", "secret", "password", "token", "credential",
    "auth", "private_key", "access_key",
]


class _SecretFilter(logging.Filter):
    """Filter that redacts potential secrets from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self._redact(str(record.msg))
        if record.args:
            record.args = tuple(self._redact(str(a)) for a in record.args)
        return True

    def _redact(self, text: str) -> str:
        lower = text.lower()
        for pattern in _sensitive_patterns:
            if pattern in lower:
                return "[REDACTED]"
        return text


def setup_logging(name: Optional[str] = None) -> logging.Logger:
    """Set up and return a logger with structured output.

    Args:
        name: Logger name. Defaults to the module name.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, settings.logging.level.upper(), logging.INFO))

    formatter = logging.Formatter(
        fmt=settings.logging.format,
        datefmt=settings.logging.date_format,
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(_SecretFilter())
    logger.addHandler(console_handler)

    try:
        settings.paths.logs.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(settings.logging.file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(_SecretFilter())
        logger.addHandler(file_handler)
    except OSError:
        logger.warning("Could not create log file handler: %s", settings.logging.file)

    logger.propagate = False
    return logger
