"""Configuration for the AI analyst layer.

All secrets are loaded from environment variables.
Never hardcode credentials or API keys.
"""

from __future__ import annotations

import os
from dataclasses import dataclass



@dataclass
class AIAnalystConfig:
    """Container for AI analyst configuration."""

    llm_provider: str = "openai"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str = ""
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    retrieval_top_k: int = 3
    retrieval_chunk_size: int = 800
    enable_rag: bool = True
    enable_tools: bool = True
    max_tool_rounds: int = 4
    offline_mode: bool = False

    def __post_init__(self) -> None:
        if not self.llm_api_key and not self.offline_mode:
            self.llm_api_key = os.getenv("RETAILSYNC_AI_API_KEY", "")
        if not self.llm_base_url:
            self.llm_base_url = os.getenv("RETAILSYNC_AI_BASE_URL", "")
        if not self.llm_model:
            self.llm_model = os.getenv("RETAILSYNC_AI_MODEL", "gpt-4o-mini")
        if not self.llm_provider:
            self.llm_provider = os.getenv("RETAILSYNC_AI_PROVIDER", "openai")
        if os.getenv("RETAILSYNC_AI_OFFLINE_MODE", "").lower() in ("1", "true", "yes"):
            self.offline_mode = True
        if os.getenv("RETAILSYNC_AI_DISABLE_TOOLS", "").lower() in ("1", "true", "yes"):
            self.enable_tools = False
        if os.getenv("RETAILSYNC_AI_DISABLE_RAG", "").lower() in ("1", "true", "yes"):
            self.enable_rag = False

    @property
    def is_configured(self) -> bool:
        """Return True if the LLM is ready to be called."""
        if self.offline_mode:
            return False
        return bool(self.llm_api_key or self.llm_provider in ("ollama",))

    @classmethod
    def from_env(cls) -> "AIAnalystConfig":
        """Build config from environment variables."""
        return cls()
