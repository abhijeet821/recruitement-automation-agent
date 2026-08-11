"""
Content-addressed disk cache for embeddings.

An embedding is a pure function of (model, text), so it caches perfectly and
never needs invalidation — the model name is part of the key, so switching
models simply misses rather than serving stale vectors.

This is not a micro-optimisation. Re-scoring a campaign after a weight change
re-embeds every skill string and resume; with a cold cache that is minutes of
GPU time, with a warm one it is milliseconds. It also makes the evaluation
harness cheap to re-run, which is what makes iterating on the scorer practical.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Thread-safe, process-safe-enough cache of text -> vector."""

    def __init__(self, cache_dir: Path, model: str, enabled: bool = True):
        self.enabled = enabled
        self.model = model
        self.dir = Path(cache_dir) / "embeddings" / _slug(model)
        self._memory: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        if self.enabled:
            try:
                self.dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.warning("Embedding cache disabled (%s): %s", self.dir, exc)
                self.enabled = False

    def _key(self, text: str) -> str:
        digest = hashlib.sha256(f"{self.model}\x00{text}".encode()).hexdigest()
        return digest

    def _path(self, key: str) -> Path:
        # Two-level fan-out keeps directory listings small once a few thousand
        # resumes have been processed.
        return self.dir / key[:2] / f"{key}.npy"

    def get(self, text: str) -> np.ndarray | None:
        if not self.enabled:
            return None
        key = self._key(text)
        with self._lock:
            hit = self._memory.get(key)
        if hit is not None:
            return hit
        path = self._path(key)
        if not path.exists():
            return None
        try:
            vector = np.load(path)
        except (OSError, ValueError) as exc:
            # A truncated file from an interrupted write: drop it and recompute.
            logger.debug("Discarding unreadable cache entry %s: %s", path, exc)
            path.unlink(missing_ok=True)
            return None
        with self._lock:
            self._memory[key] = vector
        return vector

    def put(self, text: str, vector: np.ndarray) -> None:
        if not self.enabled:
            return
        key = self._key(text)
        with self._lock:
            self._memory[key] = vector
        path = self._path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-rename so a crash mid-write cannot leave a partial file
            # that a later read would mistake for a valid vector.
            tmp = path.with_suffix(".npy.tmp")
            np.save(tmp, vector)
            tmp.replace(path)
        except OSError as exc:
            logger.debug("Could not persist embedding cache entry: %s", exc)

    def clear(self) -> int:
        removed = 0
        with self._lock:
            self._memory.clear()
        if self.dir.exists():
            for path in self.dir.rglob("*.npy"):
                path.unlink(missing_ok=True)
                removed += 1
        return removed


class JSONCache:
    """Small persistent key -> JSON-value cache.

    Used for skill-verification verdicts. Those repeat heavily: every candidate
    in a campaign is checked against the *same* requirement list, and applicants
    for one role share most of their skills. Caching the verdicts turns an
    O(candidates x requirements) model workload into roughly O(distinct pairs),
    which is what makes the verification stage affordable at all.
    """

    def __init__(self, cache_dir: Path, namespace: str, enabled: bool = True):
        self.enabled = enabled
        self.path = Path(cache_dir) / f"{_slug(namespace)}.json"
        self._data: dict[str, object] = {}
        self._lock = threading.Lock()
        self._dirty = False
        if self.enabled:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                if self.path.exists():
                    self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                logger.warning("Could not load %s cache: %s", namespace, exc)
                self._data = {}

    @staticmethod
    def key(*parts: str) -> str:
        return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:32]

    def get(self, key: str):
        if not self.enabled:
            return None
        with self._lock:
            return self._data.get(key)

    def put(self, key: str, value) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._data[key] = value
            self._dirty = True

    def flush(self) -> None:
        """Persist pending writes. Cheap to call; a no-op when nothing changed."""
        if not self.enabled or not self._dirty:
            return
        with self._lock:
            try:
                tmp = self.path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(self._data), encoding="utf-8")
                tmp.replace(self.path)
                self._dirty = False
            except OSError as exc:
                logger.debug("Could not persist %s: %s", self.path, exc)


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in text)
