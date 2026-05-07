"""Tests for `_dedupe_and_cap_chunks` (Task 3.29).

v3 averaged 8.93 retrieved_chunks/query because each sub-question and
each reflect cycle appended its own top-k slice — `apply_publisher_boost`
returned them all, ordered by boosted_score, but never deduped or
capped. v4 caps the count at `MAX_TOTAL_CHUNKS` (= 8) after dedupe.
"""
from __future__ import annotations

from src.agent.nodes import _dedupe_and_cap_chunks


def _chunk(cid: str, score: float = 1.0) -> dict:
    return {"chunk_id": cid, "score": score, "payload": {"text": cid}}


def test_under_cap_returns_all():
    chunks = [_chunk(f"c{i}") for i in range(5)]
    out = _dedupe_and_cap_chunks(chunks, cap=8)
    assert len(out) == 5
    assert [c["chunk_id"] for c in out] == ["c0", "c1", "c2", "c3", "c4"]


def test_at_cap_returns_all():
    chunks = [_chunk(f"c{i}") for i in range(8)]
    out = _dedupe_and_cap_chunks(chunks, cap=8)
    assert len(out) == 8


def test_over_cap_truncates_keeping_first():
    chunks = [_chunk(f"c{i}", score=10 - i) for i in range(12)]
    out = _dedupe_and_cap_chunks(chunks, cap=8)
    assert len(out) == 8
    # First 8 by input order are kept — caller is expected to have
    # already sorted by score (apply_publisher_boost does this).
    assert [c["chunk_id"] for c in out] == [f"c{i}" for i in range(8)]


def test_dedupe_keeps_first_occurrence():
    # c0 appears twice — second entry has a lower score (came from a
    # later sub-question retrieval). Keep the higher-scoring first hit.
    chunks = [
        _chunk("c0", score=0.95),
        _chunk("c1", score=0.80),
        _chunk("c0", score=0.55),  # dup
        _chunk("c2", score=0.50),
    ]
    out = _dedupe_and_cap_chunks(chunks, cap=8)
    assert [c["chunk_id"] for c in out] == ["c0", "c1", "c2"]
    assert out[0]["score"] == 0.95


def test_dedupe_then_cap_interaction():
    # Many duplicates compress past the cap when deduped.
    chunks = (
        [_chunk("c0")] * 5
        + [_chunk("c1")] * 5
        + [_chunk(f"c{i}") for i in range(2, 12)]
    )
    out = _dedupe_and_cap_chunks(chunks, cap=8)
    assert len(out) == 8
    assert [c["chunk_id"] for c in out] == [f"c{i}" for i in range(8)]


def test_empty_chunk_id_is_not_a_key():
    # Defensive: chunks missing chunk_id shouldn't collapse onto each
    # other (would happen if we used "" as a dict key for dedupe).
    chunks = [
        {"chunk_id": "", "score": 0.9, "payload": {"text": "a"}},
        {"chunk_id": "", "score": 0.8, "payload": {"text": "b"}},
        _chunk("c1"),
    ]
    out = _dedupe_and_cap_chunks(chunks, cap=8)
    assert len(out) == 3


def test_empty_input():
    assert _dedupe_and_cap_chunks([], cap=8) == []
