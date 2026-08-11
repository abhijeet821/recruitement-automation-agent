"""
Gemini provider — the hosted option for production.

The SDK is imported lazily so the default (Ollama) install never needs it, and
so a missing optional dependency produces an actionable message instead of an
ImportError at startup.
"""

from __future__ import annotations

import logging

import numpy as np

from matching.config import MatchingConfig
from matching.llm.base import LLMError, LLMProvider, LLMUnavailable
from matching.llm.cache import EmbeddingCache
from matching.llm.json_utils import coerce_dict, extract_json

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, config: MatchingConfig):
        if not config.gemini_api_key:
            raise LLMUnavailable(
                "LLM_PROVIDER=gemini but GEMINI_API_KEY is empty. "
                "Set the key, or use LLM_PROVIDER=ollama for local development."
            )
        self.config = config
        self.model = config.gemini_model
        self.embed_model = config.gemini_embed_model
        self._client = None
        self._cache = EmbeddingCache(
            config.cache_dir, config.gemini_embed_model, config.cache_enabled
        )

    @property
    def client(self):
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise LLMUnavailable(
                    "The Gemini provider needs the google-genai SDK: "
                    "pip install -r requirements-gemini.txt"
                ) from exc
            self._client = genai.Client(api_key=self.config.gemini_api_key)
        return self._client

    def _config_obj(self, **kwargs):
        from google.genai import types

        return types.GenerateContentConfig(**kwargs)

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> str:
        kwargs: dict = {"temperature": temperature}
        if system:
            kwargs["system_instruction"] = system
        if max_tokens:
            kwargs["max_output_tokens"] = max_tokens
        try:
            response = self.client.models.generate_content(
                model=self.model, contents=prompt, config=self._config_obj(**kwargs)
            )
        except Exception as exc:  # SDK raises a wide range of transport errors
            raise LLMError(f"Gemini generation failed: {exc}") from exc
        return (response.text or "").strip()

    def generate_json(
        self,
        prompt: str,
        *,
        schema: dict,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> dict:
        kwargs: dict = {
            "temperature": temperature,
            "response_mime_type": "application/json",
            "response_schema": schema,
        }
        if system:
            kwargs["system_instruction"] = system
        try:
            response = self.client.models.generate_content(
                model=self.model, contents=prompt, config=self._config_obj(**kwargs)
            )
        except Exception as exc:
            raise LLMError(f"Gemini JSON generation failed: {exc}") from exc
        return coerce_dict(extract_json(response.text or ""))

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)

        vectors: list[np.ndarray | None] = [self._cache.get(t) for t in texts]
        missing = [i for i, v in enumerate(vectors) if v is None]

        if missing:
            try:
                response = self.client.models.embed_content(
                    model=self.embed_model, contents=[texts[i] for i in missing]
                )
            except Exception as exc:
                raise LLMError(f"Gemini embedding failed: {exc}") from exc
            fresh = [e.values for e in response.embeddings]
            for slot, vector in zip(missing, fresh, strict=True):
                arr = np.asarray(vector, dtype=np.float32)
                vectors[slot] = arr
                self._cache.put(texts[slot], arr)

        return self._normalise_rows(np.vstack([v for v in vectors]))
