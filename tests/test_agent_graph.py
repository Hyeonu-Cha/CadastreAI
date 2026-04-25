"""Smoke tests for the LangGraph agent skeleton (Tasks 3.09–3.11).

Two scenarios:
  1. Happy path — stub `reflect` returns is_complete=True after one
     pass, so the graph terminates with iteration_count == 1.
  2. Loop cap — patch `reflect` to always say is_complete=False and
     confirm `_route_after_reflect` exits when iteration_count >=
     MAX_ITERATIONS, regardless of reflection content.
"""
from __future__ import annotations

import pytest

from src.agent import graph as graph_mod
from src.agent.graph import MAX_ITERATIONS, build_graph, initial_state


def test_graph_happy_path_terminates_after_one_pass():
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
                "refined_query": None,
            }
        }

    monkeypatch.setattr(nodes, "reflect", never_complete)
    # Rebuild the graph so it picks up the patched reflect.
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
