"""Live test for `classify_query` — needs ANTHROPIC_API_KEY.

Marked @pytest.mark.network so the unit-test default still runs offline.
We assert structure + plausible routing on three diverse queries; we
don't pin the persona/booleans hard because Haiku has some leeway.
"""
from __future__ import annotations

import os

import pytest

from src.agent.graph import initial_state
from src.agent.nodes import classify_query


@pytest.mark.network
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
@pytest.mark.parametrize(
    "query,expect_data,expect_decomp",
    [
        ("What is the current RBA cash rate?", True, False),
        (
            "Compare median Sydney house prices to Melbourne over the last 12 "
            "months and explain what's driving the difference.",
            True,
            True,
        ),
        ("What is negative gearing in Australia?", False, False),
    ],
)
def test_classify_shape(query, expect_data, expect_decomp):
    out = classify_query(initial_state(query))
    cls = out["classification"]
    # Shape must match the Classification TypedDict.
    assert set(cls.keys()) == {
        "persona",
        "query_type",
        "needs_docs",
        "needs_data",
        "needs_decomposition",
    }
    assert cls["persona"] in {
        "first_home_buyer",
        "investor",
        "policy_researcher",
        "journalist",
        "general",
    }
    assert cls["query_type"] in {
        "factual",
        "comparative",
        "computational",
        "exploratory",
    }
    assert isinstance(cls["needs_docs"], bool)
    assert isinstance(cls["needs_data"], bool)
    assert isinstance(cls["needs_decomposition"], bool)
    # Top-level mirror.
    assert out["query_type"] == cls["query_type"]
    # Loose plausibility checks — we want routing signal, not pinpoint
    # accuracy. Skipping if Haiku disagrees would create flakes.
    if expect_data:
        assert cls["needs_data"] is True, f"expected data routing for: {query}"
    if expect_decomp:
        assert cls["needs_decomposition"] is True, (
            f"expected decomposition for: {query}"
        )
