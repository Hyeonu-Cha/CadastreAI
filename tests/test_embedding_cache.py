"""Unit tests for the disk-backed embedding cache (Task 4.09).

Coverage:
  - key_for is deterministic and changes when any of (model, prefix,
    query) change.
  - get/set round-trips a numpy vector through .npy on disk.
  - embed() is a cache-miss → encode-and-persist on first call,
    cache-hit (no encode) on second call with the same inputs.
  - changing the model name busts the cache.
  - changing the query prefix busts the cache.
  - hits / misses counters track behaviour for telemetry.
  - corrupted file on disk → graceful miss + log, not a crash.
  - enabled=False short-circuits both reads and writes.
  - embed() honours the model's `name` / `model_name` attribute when
    no explicit model_name is passed.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.retrieval.embedding_cache import (
    EmbeddingCache,
    key_for,
    path_for,
)


class FakeModel:
    """Stand-in SentenceTransformer that records call counts."""

    def __init__(self, *, name: str = "fake-model", dim: int = 4):
        self.name = name
        self.dim = dim
        self.calls = 0

    def encode(self, text: str, *, normalize_embeddings: bool = True):
        self.calls += 1
        # Deterministic vector that depends on the input text so we can
        # tell whether the cache served the same value back.
        seed = sum(ord(c) for c in text) % 2**31
        rng = np.random.default_rng(seed)
        v = rng.normal(size=self.dim).astype(np.float32)
        if normalize_embeddings:
            v /= max(np.linalg.norm(v), 1e-8)
        return v


# ---------- key_for / path_for ---------------------------------


def test_key_for_is_deterministic():
    a = key_for("M", "P", "Q")
    b = key_for("M", "P", "Q")
    assert a == b
    assert len(a) == 64  # sha256 hex


def test_key_for_changes_with_each_field():
    base = key_for("M", "P", "Q")
    assert key_for("M2", "P", "Q") != base
    assert key_for("M", "P2", "Q") != base
    assert key_for("M", "P", "Q2") != base


def test_path_for_uses_key_as_filename(tmp_path):
    p = path_for(tmp_path, "M", "P", "Q")
    assert p.parent == tmp_path
    assert p.suffix == ".npy"
    assert p.stem == key_for("M", "P", "Q")


# ---------- get / set round trip --------------------------------


def test_set_then_get_round_trips_vector(tmp_path):
    cache = EmbeddingCache(cache_dir=tmp_path)
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    cache.set("M", "P", "Q", vec)
    out = cache.get("M", "P", "Q")
    assert out is not None
    assert np.array_equal(out, vec)


def test_get_returns_none_on_miss(tmp_path):
    cache = EmbeddingCache(cache_dir=tmp_path)
    assert cache.get("M", "P", "missing") is None


# ---------- embed() integration --------------------------------


def test_embed_is_miss_then_hit(tmp_path):
    cache = EmbeddingCache(cache_dir=tmp_path)
    model = FakeModel(name="fake-model")

    v1 = cache.embed(model, "cash rate today", prefix="QP: ")
    assert model.calls == 1
    assert cache.misses == 1
    assert cache.hits == 0

    v2 = cache.embed(model, "cash rate today", prefix="QP: ")
    assert model.calls == 1, "second call should be a cache hit, not re-encode"
    assert cache.hits == 1
    assert np.array_equal(v1, v2)


def test_embed_different_query_misses_again(tmp_path):
    cache = EmbeddingCache(cache_dir=tmp_path)
    model = FakeModel()
    cache.embed(model, "query A", prefix="P: ")
    cache.embed(model, "query B", prefix="P: ")
    assert model.calls == 2
    assert cache.misses == 2


def test_embed_different_prefix_misses(tmp_path):
    """Prefix is part of the key — query embedding prefix vs passage
    prefix must not collide."""
    cache = EmbeddingCache(cache_dir=tmp_path)
    model = FakeModel()
    cache.embed(model, "Q", prefix="QUERY: ")
    cache.embed(model, "Q", prefix="PASSAGE: ")
    assert model.calls == 2


def test_embed_different_model_name_misses(tmp_path):
    cache = EmbeddingCache(cache_dir=tmp_path)
    cache.embed(FakeModel(name="m1"), "Q", prefix="P: ", model_name="m1")
    cache.embed(FakeModel(name="m2"), "Q", prefix="P: ", model_name="m2")
    assert cache.misses == 2


def test_embed_uses_model_name_attr_when_unspecified(tmp_path):
    cache = EmbeddingCache(cache_dir=tmp_path)
    model = FakeModel(name="auto-name")
    cache.embed(model, "Q", prefix="P: ")
    cache.embed(model, "Q", prefix="P: ")
    assert cache.hits == 1, "second call should hit despite no explicit model_name"


# ---------- robustness ----------------------------------------


def test_corrupted_cache_file_falls_through_to_miss(tmp_path):
    """A garbage .npy on disk shouldn't crash the cache."""
    cache = EmbeddingCache(cache_dir=tmp_path)
    bad_path = path_for(tmp_path, "M", "P", "Q")
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_bytes(b"not a numpy file")
    assert cache.get("M", "P", "Q") is None


def test_disabled_cache_short_circuits_reads_and_writes(tmp_path):
    cache = EmbeddingCache(cache_dir=tmp_path, enabled=False)
    model = FakeModel()
    cache.embed(model, "Q", prefix="P: ")
    cache.embed(model, "Q", prefix="P: ")
    assert model.calls == 2, "disabled cache must always re-encode"
    # Nothing should have been persisted to disk.
    assert not any(tmp_path.iterdir())


def test_set_creates_parent_dirs(tmp_path):
    nested = tmp_path / "deep" / "nested"
    cache = EmbeddingCache(cache_dir=nested)
    cache.set("M", "P", "Q", np.array([1.0]))
    assert nested.exists()


# ---------- env var override ----------------------------------


def test_env_var_overrides_default(tmp_path, monkeypatch):
    monkeypatch.setenv("CADASTREAI_EMBED_CACHE_DIR", str(tmp_path))
    cache = EmbeddingCache()
    assert cache.cache_dir == tmp_path


def test_explicit_arg_beats_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CADASTREAI_EMBED_CACHE_DIR", "/tmp/should-not-be-used")
    cache = EmbeddingCache(cache_dir=tmp_path)
    assert cache.cache_dir == tmp_path


# ---------- normalize_embeddings flag ------------------------


def test_normalize_embeddings_flag_is_passed_through(tmp_path):
    """If the caller asks for non-normalized vectors, the cache shouldn't
    silently rewrite them at retrieval time — the on-disk vector is
    whatever the model returned, full stop."""
    cache = EmbeddingCache(cache_dir=tmp_path)

    class CapturingModel:
        name = "cap"

        def __init__(self):
            self.last_kwargs = None

        def encode(self, text, *, normalize_embeddings: bool = True):
            self.last_kwargs = {"normalize_embeddings": normalize_embeddings}
            return np.array([1.0, 2.0, 3.0])

    m = CapturingModel()
    cache.embed(m, "Q", prefix="P: ", normalize_embeddings=False)
    assert m.last_kwargs == {"normalize_embeddings": False}


# ---------- pytest goldens ----------------------------------


@pytest.mark.parametrize(
    "model,prefix,query",
    [
        ("BAAI/bge-base-en-v1.5", "Represent this sentence: ", "cash rate"),
        ("BAAI/bge-base-en-v1.5", "", "cash rate"),
    ],
)
def test_key_for_stable_across_runs(model: str, prefix: str, query: str):
    """Sanity: hashing the same triple twice on the same Python build
    gives the same key. (We don't pin to a hex value because hash
    backends can change across major Python versions; the contract is
    just determinism within a process.)"""
    assert key_for(model, prefix, query) == key_for(model, prefix, query)
