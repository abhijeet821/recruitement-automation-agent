"""
Deterministic stub provider.

Unit tests must exercise the full pipeline — parsing, features, scoring,
ensembling — without a model server, and must give the same answer every run.
This provider fabricates output that is *structurally* valid (it walks the
requested JSON schema and fills in type-correct defaults) and produces
embeddings from a seeded hash of the text.

The embeddings are not semantically meaningful, but they are stable and
self-consistent: identical strings embed identically and different strings embed
differently, which is exactly the property the pipeline tests rely on.
"""

from __future__ import annotations

import hashlib

import numpy as np

from matching.config import MatchingConfig
from matching.llm.base import LLMProvider

_DIM = 256


class FakeProvider(LLMProvider):
    name = "fake"

    def __init__(self, config: MatchingConfig | None = None, canned: dict | None = None):
        self.config = config
        # Tests can inject exact payloads keyed by a substring of the prompt.
        self.canned = canned or {}
        self.calls: list[tuple[str, str]] = []

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append(("generate", prompt))
        for needle, payload in self.canned.items():
            if needle in prompt and isinstance(payload, str):
                return payload
        return "Generated text for testing purposes."

    def generate_json(
        self,
        prompt: str,
        *,
        schema: dict,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> dict:
        self.calls.append(("generate_json", prompt))
        for needle, payload in self.canned.items():
            if needle in prompt and isinstance(payload, dict):
                return payload
        return _skeleton(schema)

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls.append(("embed", "|".join(texts)[:120]))
        if not texts:
            return np.zeros((0, _DIM), dtype=np.float32)
        rows = []
        for text in texts:
            seed = int.from_bytes(
                hashlib.sha256(text.encode()).digest()[:8], "big", signed=False
            )
            rng = np.random.default_rng(seed % (2**32))
            rows.append(rng.normal(size=_DIM).astype(np.float32))
        return self._normalise_rows(np.vstack(rows))


def _skeleton(schema: dict) -> dict:
    """Build a minimal type-correct instance of a JSON schema."""
    node_type = schema.get("type", "object")

    if node_type == "object":
        out = {}
        for key, sub in (schema.get("properties") or {}).items():
            out[key] = _skeleton_value(sub)
        return out
    return {}


def _skeleton_value(schema: dict):
    node_type = schema.get("type", "string")
    if isinstance(node_type, list):
        node_type = next((t for t in node_type if t != "null"), "string")

    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]
    if node_type == "object":
        return _skeleton(schema)
    if node_type == "array":
        return []
    if node_type in ("number", "integer"):
        return 0
    if node_type == "boolean":
        return False
    return ""
