"""Tests for the `decompose` node (Task 3.14).

Two scenarios:
  1. Pure offline — when the classifier didn't flag
     `needs_decomposition`, decompose must return an empty list and
     never touch the network.
  2. Network-marked — with the flag set, Haiku should return between
     DECOMPOSE_MIN and DECOMPOSE_MAX atomic sub-questions on a known
     compound query. Skipped without ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import os

import pytest

from src.agent.graph import initial_state
from src.agent.nodes import DECOMPOSE_MAX, DECOMPOSE_MIN, decompose


def test_decompose_skips_when_flag_false():
    state = initial_state("What is the cash rate today?")
    state["classification"] = {
        "persona": "general",
        "query_type": "factual",
        "needs_docs": True,
        "needs_data": True,
        "needs_decomposition": False,
    }
    out = decompose(state)
    assert out == {"sub_questions": []}


def test_decompose_skips_when_classification_missing():
    """If something upstream skipped the classifier, decompose stays inert."""
    state = initial_state("anything")
    out = decompose(state)
    assert out == {"sub_questions": []}


@pytest.mark.network
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
def test_decompose_compound_query_live():
    state = initial_state(
        "I'm a first home buyer in NSW looking at a $950k house. What stamp "
        "duty would I pay, what's the monthly repayment with a 20% deposit "
        "at 6%, and how does that compare to current rental yields in Sydney?"
    )
    state["classification"] = {
        "persona": "first_home_buyer",
        "query_type": "computational",
        "needs_docs": False,
        "needs_data": True,
        "needs_decomposition": True,
    }
    out = decompose(state)
    sub_qs = out["sub_questions"]
    assert DECOMPOSE_MIN <= len(sub_qs) <= DECOMPOSE_MAX
    assert all(isinstance(q, str) and q.strip() for q in sub_qs)
    assert len({q.lower() for q in sub_qs}) == len(sub_qs), (
        f"sub-questions not unique: {sub_qs}"
    )
