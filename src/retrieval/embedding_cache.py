"""Disk-backed embedding cache for repeated queries (Task 4.09).

The agent re-encodes the same or near-identical query strings on
loopbacks, in eval reruns, and during dev iteration. Encoding a single
query through `BAAI/bge-base-en-v1.5` on CPU is cheap (~50ms) but adds
up across a 30-query eval pass that loops twice on average. This module
sidecar-caches each (model, prefix, query) → vector pair on disk so the
second call is an `np.load` instead of a forward pass.

Storage layout
--------------
Cache dir defaults to `data/cache/embeddings/` (relative to CWD; the
repo's `.gitignore` already covers `data/cache/`). Each entry is a
single `.npy` file named after a sha256 hex digest of
`f"{model}|{prefix}|{query}"`. We don't put the raw text in the
filename because queries can be hundreds of characters and contain
characters that aren't filename-safe across platforms.

Concurrency
-----------
A second writer racing with the first will just overwrite the file with
identical bytes — fine. We don't bother with a lock; the cost of two
parallel encodes once in a while is far less than the cost of holding
a file lock across a `model.encode()` call.

Typical usage
-------------
    cache = EmbeddingCache()
    vec = cache.embed(
        model,                # any object with an `.encode(text, ...)` method
        query="cash rate today",
        prefix="Represent this sentence for searching relevant passages: ",
        model_name="BAAI/bge-base-en-v1.5",
    )

Pure-functional `key_for(...)` and `path_for(...)` helpers are exposed
so tests can poke at the wire without driving a real model.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data/cache/embeddings")
ENV_CACHE_DIR = "CADASTREAI_EMBED_CACHE_DIR"


def _resolve_cache_dir(cache_dir: Path | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir)
    env = os.environ.get(ENV_CACHE_DIR)
    if env:
        return Path(env)
    return DEFAULT_CACHE_DIR


def key_for(model_name: str, prefix: str, query: str) -> str:
    """Stable sha256 hex digest for the cache key.

    Splitting on the literal `|` separator is fine because we never
    parse the digest back out — collisions of the form
    `("a|b", "c") vs ("a", "b|c")` would still hash to the same key,
    which is acceptable: both deterministically map to the same vector.
    """
    raw = f"{model_name}|{prefix}|{query}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def path_for(cache_dir: Path, model_name: str, prefix: str, query: str) -> Path:
    return cache_dir / f"{key_for(model_name, prefix, query)}.npy"


class EmbeddingCache:
    """Tiny disk cache: one .npy file per (model, prefix, query) triple."""

    def __init__(self, cache_dir: Path | None = None, *, enabled: bool = True) -> None:
        self.cache_dir = _resolve_cache_dir(cache_dir)
        self.enabled = enabled
        self.hits = 0
        self.misses = 0

    def get(self, model_name: str, prefix: str, query: str):
        """Return the cached vector or None on miss / when disabled."""
        if not self.enabled:
            return None
        import numpy as np

        path = path_for(self.cache_dir, model_name, prefix, query)
        if not path.exists():
            return None
        try:
            arr = np.load(path)
        except Exception as exc:  # noqa: BLE001 — corrupted file shouldn't crash
            log.warning("embedding cache: failed to load %s (%s); ignoring", path, exc)
            return None
        self.hits += 1
        return arr

    def set(self, model_name: str, prefix: str, query: str, vector) -> None:
        """Persist a vector to disk. No-op when disabled."""
        if not self.enabled:
            return
        import numpy as np

        path = path_for(self.cache_dir, model_name, prefix, query)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.asarray(vector))

    def embed(
        self,
        model,
        query: str,
        *,
        prefix: str = "",
        model_name: str | None = None,
        normalize_embeddings: bool = True,
    ):
        """Get-or-compute helper. Returns a numpy vector.

        `model_name` defaults to `getattr(model, "name", "")`. We fall
        back to the empty string rather than e.g. `repr(model)` so the
        cache key doesn't churn across runs (object reprs include
        memory addresses).
        """
        import numpy as np

        name = (
            model_name
            or getattr(model, "name", "")
            or getattr(model, "model_name", "")
            or ""
        )
        cached = self.get(name, prefix, query)
        if cached is not None:
            return cached
        self.misses += 1
        text = prefix + query
        vec = np.asarray(
            model.encode(text, normalize_embeddings=normalize_embeddings)
        )
        self.set(name, prefix, query, vec)
        return vec


__all__ = [
    "DEFAULT_CACHE_DIR",
    "ENV_CACHE_DIR",
    "EmbeddingCache",
    "key_for",
    "path_for",
]
