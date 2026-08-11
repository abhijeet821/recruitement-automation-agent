"""
The provider interface every LLM backend implements.

Three operations are enough for this system:

* ``generate``       — free text (job descriptions)
* ``generate_json``  — a dict conforming to a JSON schema (extraction, rubric)
* ``embed``          — dense vectors (semantic skill and JD/resume matching)

Keeping the surface this small is what makes swapping local Ollama for hosted
Gemini a one-line configuration change instead of a refactor.
"""

from __future__ import annotations

import abc
import logging

import numpy as np

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Any provider-level failure: transport, timeout, malformed output."""


class LLMUnavailable(LLMError):
    """The backend could not be reached at all (server down, bad host)."""


class LLMProvider(abc.ABC):
    """Abstract LLM backend."""

    name: str = "base"

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> str:
        """Return free-form text."""

    @abc.abstractmethod
    def generate_json(
        self,
        prompt: str,
        *,
        schema: dict,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> dict:
        """Return a dict, constrained to ``schema`` where the backend supports it."""

    @abc.abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an ``(n, d)`` float32 array of L2-normalised embeddings.

        Normalising here means every caller can use a plain dot product as
        cosine similarity, and no call site has to remember to normalise.
        """

    def health(self) -> tuple[bool, str]:
        """Cheap reachability probe for the UI and the doctor command."""
        try:
            self.embed(["health check"])
            return True, "ok"
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            return False, str(exc)

    # ── shared helpers ───────────────────────────────────────

    @staticmethod
    def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        # Guard the zero vector: an all-zero embedding would divide by zero and
        # poison every downstream similarity with NaN.
        norms[norms == 0] = 1.0
        return matrix / norms
