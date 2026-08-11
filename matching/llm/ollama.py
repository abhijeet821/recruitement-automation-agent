"""
Ollama provider — the local default for development and testing.

Talks to the Ollama REST API directly with ``requests``; no extra SDK. Two
capabilities of modern Ollama do most of the heavy lifting:

* ``format: <json-schema>`` constrains decoding to a schema, which is far more
  reliable than asking a 7-14B model to "reply with JSON" and hoping.
* ``think: false`` suppresses qwen3's reasoning preamble. We still strip
  ``<think>`` tags downstream, because the flag is ignored by some builds.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import requests

from matching.config import MatchingConfig
from matching.llm.base import LLMError, LLMProvider, LLMUnavailable
from matching.llm.cache import EmbeddingCache
from matching.llm.json_utils import coerce_dict, extract_json

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, config: MatchingConfig):
        self.config = config
        self.host = config.ollama_host.rstrip("/")
        self.model = config.ollama_model
        self.embed_model = config.ollama_embed_model
        self._session = requests.Session()
        self._cache = EmbeddingCache(
            config.cache_dir, config.ollama_embed_model, config.cache_enabled
        )

    # ── transport ────────────────────────────────────────────

    def _post(self, path: str, payload: dict) -> dict:
        """POST with bounded retries and exponential backoff.

        Only transport-level errors are retried. A 4xx means the request itself
        is wrong (missing model, bad schema) and retrying it just wastes time,
        so it fails immediately with the server's message.
        """
        url = f"{self.host}{path}"
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries):
            try:
                response = self._session.post(
                    url, json=payload, timeout=self.config.request_timeout
                )
            except requests.exceptions.ConnectionError as exc:
                raise LLMUnavailable(
                    f"Cannot reach Ollama at {self.host}. Is `ollama serve` running?"
                ) from exc
            except requests.exceptions.Timeout as exc:
                last_error = exc
                logger.warning(
                    "Ollama timeout on %s (attempt %d/%d)",
                    path, attempt + 1, self.config.max_retries,
                )
            else:
                if response.status_code == 404:
                    raise LLMError(
                        f"Ollama model not found. Run: ollama pull {payload.get('model')}"
                    )
                if 400 <= response.status_code < 500:
                    raise LLMError(
                        f"Ollama rejected the request ({response.status_code}): "
                        f"{response.text[:300]}"
                    )
                if response.status_code >= 500:
                    last_error = LLMError(f"Ollama server error {response.status_code}")
                    logger.warning("Ollama 5xx on %s, retrying", path)
                else:
                    return response.json()

            if attempt < self.config.max_retries - 1:
                time.sleep(self.config.retry_backoff ** attempt)

        raise LLMError(f"Ollama request to {path} failed: {last_error}")

    # ── generation ───────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> str:
        options: dict = {"temperature": temperature}
        if max_tokens:
            options["num_predict"] = max_tokens

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": options,
        }
        if system:
            payload["system"] = system

        data = self._post("/api/generate", payload)
        from matching.llm.json_utils import strip_reasoning

        return strip_reasoning(data.get("response", "")).strip()

    def generate_json(
        self,
        prompt: str,
        *,
        schema: dict,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> dict:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            # Schema-constrained decoding: the sampler can only emit tokens that
            # keep the output a valid instance of `schema`.
            "format": schema,
            "options": {
                "temperature": temperature,
                # Resumes are long; the default 2k context silently truncates
                # them, which looks like a bad extraction rather than a config
                # problem. 8k covers essentially every real CV.
                "num_ctx": 8192,
            },
        }
        if system:
            payload["system"] = system

        data = self._post("/api/generate", payload)
        raw = data.get("response", "")
        return coerce_dict(extract_json(raw))

    # ── embeddings ───────────────────────────────────────────

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)

        vectors: list[np.ndarray | None] = [self._cache.get(t) for t in texts]
        missing = [i for i, v in enumerate(vectors) if v is None]

        if missing:
            payload = {"model": self.embed_model, "input": [texts[i] for i in missing]}
            data = self._post("/api/embed", payload)
            fresh = data.get("embeddings")
            if not fresh:
                raise LLMError(
                    f"Ollama returned no embeddings for model '{self.embed_model}'. "
                    f"Run: ollama pull {self.embed_model}"
                )
            if len(fresh) != len(missing):
                raise LLMError(
                    f"Embedding count mismatch: asked {len(missing)}, got {len(fresh)}"
                )
            for slot, vector in zip(missing, fresh, strict=True):
                arr = np.asarray(vector, dtype=np.float32)
                vectors[slot] = arr
                self._cache.put(texts[slot], arr)

        return self._normalise_rows(np.vstack([v for v in vectors]))

    # ── diagnostics ──────────────────────────────────────────

    def list_models(self) -> list[str]:
        try:
            response = self._session.get(f"{self.host}/api/tags", timeout=10)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise LLMUnavailable(f"Cannot reach Ollama at {self.host}: {exc}") from exc
        return [m["name"] for m in response.json().get("models", [])]
