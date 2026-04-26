"""Smoke tests for the LangGraph agent skeleton (Tasks 3.09–3.11).

Two scenarios:
  1. Happy path — stub `reflect` returns is_complete=True after one
     pass, so the graph terminates with iteration_count == 1.
  2. Loop cap — patch `reflect` to always say is_complete=False and
     confirm `_route_after_reflect` exits when iteration_count >=
     MAX_ITERATIONS, regardless of reflection content.

These tests stub `classify_query` so they don't need ANTHROPIC_API_KEY;
the live classifier has its own dedicated test below.
"""
from __future__ import annotations

import pytest

from src.agent import graph as graph_mod
from src.agent.graph import MAX_ITERATIONS, build_graph, initial_state


def _stub_classifier(state):
    """Replace the live Haiku call with a deterministic shape match."""
    return {
        "classification": {
            "persona": "general",
            "query_type": "factual",
            "needs_docs": True,
            "needs_data": False,
            "needs_decomposition": False,
        },
        "query_type": "factual",
    }


def _stub_planner(_query):
    """Empty plan — keeps retrieve_or_tool offline-safe in graph tests."""
    return {"use_docs": False, "tool_calls": []}


def test_graph_happy_path_terminates_after_one_pass(monkeypatch):
    from src.agent import nodes

    monkeypatch.setattr(nodes, "classify_query", _stub_classifier)
    monkeypatch.setattr(nodes, "_plan_subquestion", _stub_planner)
    g = build_graph()
    out = g.invoke(initial_state("What is the cash rate today?"))
    assert out["query_type"] == "factual"
    assert out["iteration_count"] == 1
    assert out["reflection"]["is_complete"] is True
    assert out["answer_draft"].startswith("[stub answer to:")
    # synthesize appends an assistant message.
    assert any(m["role"] == "assistant" for m in out["messages"])


def test_graph_caps_at_max_iterations(monkeypatch):
    # Force reflect to never declare completion; the cap must take over.
    from src.agent import nodes

    def never_complete(state):
        return {
            "reflection": {
                "is_complete": False,
                "missing": ["everything"],
                "refined_query": "still need more",
            }
        }

    monkeypatch.setattr(nodes, "classify_query", _stub_classifier)
    monkeypatch.setattr(nodes, "_plan_subquestion", _stub_planner)
    monkeypatch.setattr(nodes, "reflect", never_complete)
    # Rebuild the graph so it picks up the patched nodes.
    g = build_graph()
    out = g.invoke(initial_state("Find me everything."))
    assert out["iteration_count"] == MAX_ITERATIONS
    assert out["reflection"]["is_complete"] is False


def test_route_after_reflect_pure():
    """`_route_after_reflect` is the brain of the loop edge — verify directly."""
    # is_complete=True → end
    state = {"iteration_count": 1, "reflection": {"is_complete": True}}
    assert graph_mod._route_after_reflect(state) == "end"
    # is_complete=False under the cap → loop
    state = {"iteration_count": 1, "reflection": {"is_complete": False}}
    assert graph_mod._route_after_reflect(state) == "loop"
    # cap reached even with is_complete=False → end
    state = {
        "iteration_count": MAX_ITERATIONS,
        "reflection": {"is_complete": False},
    }
    assert graph_mod._route_after_reflect(state) == "end"
    # Missing reflection → treated as not-complete, so loop until cap.
    state = {"iteration_count": 0, "reflection": {}}
    assert graph_mod._route_after_reflect(state) == "loop"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
