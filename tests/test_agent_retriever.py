"""Tests for `_agent_retriever` env-driven selection (Task 3.25).

The agent shipped on dense-only retrieval in v1/v2; v3 makes it
configurable and defaults to hybrid (BGE dense + BM25 RRF) per the
honest-split eval in `results/hybrid_comparison.md`. These tests pin
the routing logic and cache behaviour, with the underlying retriever
classes monkey-patched so we never spin up Qdrant or load the 212 MB
BM25 pickle.
"""
from __future__ import annotations

import pytest

from src.agent import nodes


@pytest.fixture(autouse=True)
def _reset_cached_retriever(monkeypatch):
    """Each test gets a fresh `_AGENT_RETRIEVER` cache slot."""
    monkeypatch.setattr(nodes, "_AGENT_RETRIEVER", None)
    yield


class _FakeRetriever:
    def __init__(self, *_, **__):
        self.kind = "dense"


class _FakeBM25:
    def __init__(self, kind="bm25"):
        self.kind = kind

    @classmethod
    def load(cls, _path):
        return cls()


class _FakeHybrid:
    def __init__(self, *_, **__):
        self.kind = "hybrid"


def _patch_all(monkeypatch):
    """Stub every retriever import the selector might reach for."""
    import src.index.bm25 as bm25_mod
    import src.index.hybrid as hybrid_mod
    import src.retrieval.retriever as dense_mod

    monkeypatch.setattr(dense_mod, "Retriever", _FakeRetriever)
    monkeypatch.setattr(bm25_mod, "BM25Index", _FakeBM25)
    monkeypatch.setattr(hybrid_mod, "HybridRetriever", _FakeHybrid)


def test_default_is_hybrid(monkeypatch):
    monkeypatch.delenv("CADASTRE_AGENT_RETRIEVER", raising=False)
    _patch_all(monkeypatch)
    r = nodes._agent_retriever()
    assert r.kind == "hybrid"


def test_env_dense(monkeypatch):
    monkeypatch.setenv("CADASTRE_AGENT_RETRIEVER", "dense")
    _patch_all(monkeypatch)
    assert nodes._agent_retriever().kind == "dense"


def test_env_bm25(monkeypatch):
    monkeypatch.setenv("CADASTRE_AGENT_RETRIEVER", "bm25")
    _patch_all(monkeypatch)
    assert nodes._agent_retriever().kind == "bm25"


def test_env_hybrid(monkeypatch):
    monkeypatch.setenv("CADASTRE_AGENT_RETRIEVER", "hybrid")
    _patch_all(monkeypatch)
    assert nodes._agent_retriever().kind == "hybrid"


def test_env_casing_normalised(monkeypatch):
    monkeypatch.setenv("CADASTRE_AGENT_RETRIEVER", "Hybrid")
    _patch_all(monkeypatch)
    assert nodes._agent_retriever().kind == "hybrid"


def test_unknown_value_raises(monkeypatch):
    monkeypatch.setenv("CADASTRE_AGENT_RETRIEVER", "elasticsearch")
    _patch_all(monkeypatch)
    with pytest.raises(ValueError, match="unknown CADASTRE_AGENT_RETRIEVER"):
        nodes._agent_retriever()


def test_cache_persists_across_calls(monkeypatch):
    monkeypatch.setenv("CADASTRE_AGENT_RETRIEVER", "dense")
    _patch_all(monkeypatch)
    r1 = nodes._agent_retriever()
    r2 = nodes._agent_retriever()
    assert r1 is r2


def test_retrieve_docs_uses_selected_retriever(monkeypatch):
    """`_retrieve_docs` calls `.retrieve(query, k)` on whichever the
    selector returned and reshapes to the agent-state format."""
    monkeypatch.delenv("CADASTRE_AGENT_RETRIEVER", raising=False)

    captured = {}

    class _StubHybrid:
        def __init__(self, *_, **__):
            pass

        def retrieve(self, query, k=10):
            captured["query"] = query
            captured["k"] = k
            return [
                ({"chunk_id": "abc", "text": "x"}, 0.42),
                ({"chunk_id": "def", "text": "y"}, 0.31),
            ]

    import src.index.hybrid as hybrid_mod

    monkeypatch.setattr(hybrid_mod, "HybridRetriever", _StubHybrid)

    out = nodes._retrieve_docs("median Sydney house price", k=5)

    assert captured == {"query": "median Sydney house price", "k": 5}
    assert out == [
        {"chunk_id": "abc", "score": 0.42, "payload": {"chunk_id": "abc", "text": "x"}},
        {"chunk_id": "def", "score": 0.31, "payload": {"chunk_id": "def", "text": "y"}},
    ]
