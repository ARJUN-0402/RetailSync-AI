"""Custom exceptions for the AI analyst layer."""

from __future__ import annotations


class AIAnalystError(Exception):
    """Base exception for AI analyst errors."""


class MissingConfigurationError(AIAnalystError):
    """Raised when required configuration is missing."""


class LLMProviderError(AIAnalystError):
    """Raised when the LLM provider returns an error."""


class RetrievalError(AIAnalystError):
    """Raised when documentation retrieval fails."""


class ToolExecutionError(AIAnalystError):
    """Raised when a data access tool fails."""


class GroundingError(AIAnalystError):
    """Raised when response grounding fails."""
