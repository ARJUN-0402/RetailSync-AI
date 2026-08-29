"""Structured logging setup for RetailSync AI.

Provides a configured logger that writes to both console and file.

Secret redaction is implemented in a :class:`logging.Formatter` subclass so that
standard ``%``-style logging semantics are fully preserved. The formatter never
mutates the shared :class:`logging.LogRecord`, and it never coerces formatting
arguments to strings, so calls such as::

    logger.info("Features: %d", len(feature_cols))
    logger.info("RF training time: %.1fs", fit_time)

interpolate exactly as the standard library does.
"""

from __future__ import annotations

import copy
import logging
import re
import sys
from collections.abc import Mapping
from typing import Any, Optional

from src.config import settings

REDACTION_PLACEHOLDER = "[REDACTED]"

# Key names that indicate the *following value* is a credential.
SENSITIVE_KEY_PATTERNS: tuple[str, ...] = (
    "api[_-]?key",
    "apikey",
    "secret[_-]?key",
    "secret",
    "password",
    "passwd",
    "pwd",
    "access[_-]?token",
    "refresh[_-]?token",
    "token",
    "credentials",
    "credential",
    "private[_-]?key",
    "access[_-]?key",
    "session[_-]?key",
    "authorization",
    "auth",
    "key",
)

# Keys unambiguous enough that a whitespace separator is also treated as an
# assignment (e.g. ``Bearer abc123``). Deliberately excludes bare "key"/"auth"
# to avoid redacting benign text such as "Primary key product_id".
_UNAMBIGUOUS_KEY_PATTERNS: tuple[str, ...] = (
    "bearer",
    "api[_-]?key",
    "apikey",
    "secret[_-]?key",
    "secret",
    "password",
    "passwd",
    "access[_-]?token",
    "refresh[_-]?token",
    "token",
    "credentials",
    "credential",
    "private[_-]?key",
    "access[_-]?key",
)

# A quoted string, an auth-scheme prefixed credential (``Bearer abc123``), or a
# run of characters up to the next delimiter.
_VALUE = r"""(?:"[^"]*"|'[^']*'|(?:Bearer|Token|Basic|JWT)\s+\S+|[^\s,;)\]}]+)"""

_KEY_VALUE_RE = re.compile(
    r"(?i)\b(" + "|".join(SENSITIVE_KEY_PATTERNS) + r")\b(\s*[:=]\s*)" + _VALUE
)

_KEY_SPACE_VALUE_RE = re.compile(
    r"(?i)\b(" + "|".join(_UNAMBIGUOUS_KEY_PATTERNS) + r")\b(\s+)" + _VALUE
)

# Matches a bare sensitive key name used as an entire string, e.g. an argument
# that is literally "api_key".
_BARE_SENSITIVE_RE = re.compile(
    r"(?i)^\s*(?:" + "|".join(SENSITIVE_KEY_PATTERNS) + r")\s*$"
)


def redact_text(text: str) -> str:
    """Redact credential values from an already-rendered log line.

    Only the *value* portion of a ``key=value`` (or ``key: value``) pair is
    replaced, so timestamps, logger names and surrounding context survive.

    Args:
        text: Rendered log text.

    Returns:
        Text with sensitive values replaced by ``[REDACTED]``.
    """
    if not text:
        return text
    result = _KEY_VALUE_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTION_PLACEHOLDER}", text)
    result = _KEY_SPACE_VALUE_RE.sub(
        lambda m: f"{m.group(1)}{m.group(2)}{REDACTION_PLACEHOLDER}", result
    )
    return result


def _redact_value(value: Any) -> Any:
    """Redact a single logging argument, preserving non-string types.

    Numbers, booleans, ``None`` and arbitrary objects are returned untouched so
    that ``%d``, ``%f`` and ``%.1f`` interpolation keeps working. Only ``str``
    values are inspected, and only sensitive substrings are replaced.
    """
    if isinstance(value, str):
        if _BARE_SENSITIVE_RE.match(value):
            return REDACTION_PLACEHOLDER
        return redact_text(value)
    return value


def sanitize_log_args(args: Any) -> Any:
    """Return a redacted copy of ``record.args`` that preserves value types.

    Args:
        args: The ``record.args`` value: ``None``, a tuple of positional
            arguments, or a single mapping for ``%(name)s`` style formatting.

    Returns:
        A sanitized copy in the same shape as the input. Numeric arguments are
        never converted to strings.
    """
    if not args:
        return args

    if isinstance(args, Mapping):
        sanitized: dict[Any, Any] = {}
        for key, value in args.items():
            if isinstance(key, str) and _BARE_SENSITIVE_RE.match(key):
                sanitized[key] = REDACTION_PLACEHOLDER
            else:
                sanitized[key] = _redact_value(value)
        return sanitized

    if isinstance(args, tuple):
        return tuple(_redact_value(value) for value in args)

    return _redact_value(args)


class SecretRedactingFormatter(logging.Formatter):
    """Formatter that redacts secrets without breaking ``%`` interpolation.

    The record is shallow-copied before sanitization so the original record is
    left intact for other handlers, and ``record.args`` types are preserved so
    the standard formatter can apply ``%d``/``%f``/``%s`` normally. Redaction of
    the rendered text happens *after* interpolation.
    """

    def format(self, record: logging.LogRecord) -> str:
        safe_record = copy.copy(record)
        safe_record.args = sanitize_log_args(record.args)
        return redact_text(super().format(safe_record))


def setup_logging(name: Optional[str] = None) -> logging.Logger:
    """Set up and return a logger with structured output.

    Args:
        name: Logger name. Defaults to the root logger.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, settings.logging.level.upper(), logging.INFO))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        SecretRedactingFormatter(
            fmt=settings.logging.format,
            datefmt=settings.logging.date_format,
        )
    )
    logger.addHandler(console_handler)

    try:
        settings.paths.logs.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(settings.logging.file, encoding="utf-8")
        file_handler.setFormatter(
            SecretRedactingFormatter(
                fmt=settings.logging.format,
                datefmt=settings.logging.date_format,
            )
        )
        logger.addHandler(file_handler)
    except OSError:
        logger.warning("Could not create log file handler: %s", settings.logging.file)

    logger.propagate = False
    return logger
