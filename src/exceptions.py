"""Custom exceptions for RetailSync AI pipeline components.

Provides a hierarchy of exceptions for different failure modes,
enabling callers to catch specific errors and degrade gracefully.
"""

from __future__ import annotations


class RetailSyncError(Exception):
    """Base exception for all RetailSync AI errors."""


class ConfigurationError(RetailSyncError):
    """Raised when required configuration is missing or invalid."""


class DataError(RetailSyncError):
    """Raised when data loading, validation, or processing fails."""


class ModelError(RetailSyncError):
    """Raised when model training, loading, or prediction fails."""


class DatabaseError(RetailSyncError):
    """Raised when database operations fail."""


class PipelineError(RetailSyncError):
    """Raised when pipeline execution fails."""


class ValidationError(RetailSyncError):
    """Raised when input validation fails."""


class APIError(RetailSyncError):
    """Raised when external API calls fail."""
