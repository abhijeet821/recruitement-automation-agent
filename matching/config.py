"""
Runtime configuration for the matching engine.

Read from environment variables so the package stays usable outside Django
(tests, the evaluation harness, notebooks). Django's settings module calls
``set_config()`` at startup to override with values it has already parsed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass(frozen=True)
class MatchingConfig:
    """Everything the engine needs to know about its environment."""

    # ── LLM provider ─────────────────────────────────────────
    # "ollama" (local, the dev/test default), "gemini" (hosted), or "fake"
    # (deterministic stub used by the unit tests so CI needs no model server).
    provider: str = "ollama"

    ollama_host: str = "http://localhost:11434"
    # qwen3:14b follows JSON schemas reliably and is strong at the extraction
    # and rubric tasks. llama3.1:8b is the lighter fallback.
    ollama_model: str = "qwen3:14b"
    # bge-m3 produces 1024-d embeddings and handles the short, jargon-dense
    # strings (skill names, bullet points) this system compares.
    ollama_embed_model: str = "bge-m3"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_embed_model: str = "text-embedding-004"

    # ── Networking ───────────────────────────────────────────
    request_timeout: float = 180.0
    max_retries: int = 3
    retry_backoff: float = 1.5

    # ── Caching ──────────────────────────────────────────────
    # Embeddings are pure functions of (model, text), so they cache perfectly.
    # This turns a re-scoring run from minutes into milliseconds.
    cache_enabled: bool = True
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "hireai")

    # ── GitHub enrichment ────────────────────────────────────
    # Unauthenticated GitHub allows 60 req/hr; a token raises it to 5000.
    github_token: str = ""
    github_api: str = "https://api.github.com"
    github_max_repos: int = 30

    # ── Scoring behaviour ────────────────────────────────────
    # Blind screening: redact name/school/location/etc. before the resume text
    # ever reaches the rubric model.
    blind_screening: bool = True
    # Skip the (slow) LLM rubric pass and score from features alone.
    rubric_enabled: bool = True

    def with_overrides(self, **kwargs) -> MatchingConfig:
        clean = {k: v for k, v in kwargs.items() if v is not None}
        return replace(self, **clean)


def load_config_from_env() -> MatchingConfig:
    return MatchingConfig(
        provider=os.environ.get("LLM_PROVIDER", "ollama").strip().lower(),
        ollama_host=os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/"),
        ollama_model=os.environ.get("OLLAMA_MODEL", "qwen3:14b"),
        ollama_embed_model=os.environ.get("OLLAMA_EMBED_MODEL", "bge-m3"),
        gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
        gemini_model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        gemini_embed_model=os.environ.get("GEMINI_EMBED_MODEL", "text-embedding-004"),
        request_timeout=_env_float("LLM_TIMEOUT", 180.0),
        max_retries=_env_int("LLM_MAX_RETRIES", 3),
        cache_enabled=_env_bool("MATCHING_CACHE", True),
        cache_dir=Path(
            os.environ.get("MATCHING_CACHE_DIR", str(Path.home() / ".cache" / "hireai"))
        ),
        github_token=os.environ.get("GITHUB_TOKEN", ""),
        github_max_repos=_env_int("GITHUB_MAX_REPOS", 30),
        blind_screening=_env_bool("BLIND_SCREENING", True),
        rubric_enabled=_env_bool("RUBRIC_ENABLED", True),
    )


_config: MatchingConfig | None = None


def get_config() -> MatchingConfig:
    global _config
    if _config is None:
        _config = load_config_from_env()
    return _config


def set_config(config: MatchingConfig) -> None:
    """Install an explicit config (used by Django settings and by tests)."""
    global _config
    _config = config
