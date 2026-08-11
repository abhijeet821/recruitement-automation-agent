"""Provider selection — the single place that knows which backend is in use."""

from __future__ import annotations

import logging
import threading

from matching.config import MatchingConfig, get_config
from matching.llm.base import LLMError, LLMProvider, LLMUnavailable
from matching.llm.fake import FakeProvider
from matching.llm.gemini import GeminiProvider
from matching.llm.ollama import OllamaProvider

logger = logging.getLogger(__name__)

_PROVIDERS = {
    "ollama": OllamaProvider,
    "gemini": GeminiProvider,
    "fake": FakeProvider,
}

_cache: dict[str, LLMProvider] = {}
_lock = threading.Lock()


def get_provider(config: MatchingConfig | None = None) -> LLMProvider:
    """Return the configured provider, memoised per (provider, model) pair.

    Memoising matters because each provider owns a connection pool and an
    embedding cache; rebuilding one per request would throw both away.
    """
    config = config or get_config()
    key = f"{config.provider}:{getattr(config, f'{config.provider}_model', '')}"

    with _lock:
        if key in _cache:
            return _cache[key]

        provider_cls = _PROVIDERS.get(config.provider)
        if provider_cls is None:
            raise LLMError(
                f"Unknown LLM_PROVIDER '{config.provider}'. "
                f"Valid options: {', '.join(sorted(_PROVIDERS))}"
            )
        provider = provider_cls(config)
        _cache[key] = provider
        logger.info("LLM provider initialised: %s", key)
        return provider


def reset_provider_cache() -> None:
    """Drop memoised providers (used by tests that swap configuration)."""
    with _lock:
        _cache.clear()


__all__ = ["get_provider", "reset_provider_cache", "LLMError", "LLMUnavailable"]
