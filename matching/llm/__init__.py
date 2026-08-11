"""LLM provider abstraction (Ollama / Gemini / deterministic fake)."""

from matching.llm.base import LLMError, LLMProvider, LLMUnavailable
from matching.llm.factory import get_provider, reset_provider_cache
from matching.llm.json_utils import (
    JSONExtractionError,
    coerce_dict,
    coerce_list,
    extract_json,
    strip_reasoning,
)

__all__ = [
    "LLMError",
    "LLMProvider",
    "LLMUnavailable",
    "get_provider",
    "reset_provider_cache",
    "JSONExtractionError",
    "coerce_dict",
    "coerce_list",
    "extract_json",
    "strip_reasoning",
]
