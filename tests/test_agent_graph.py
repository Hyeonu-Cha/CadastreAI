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


def _stub_reflect_complete(_state):
    """Always-complete reflect — keeps reflect offline-safe in graph tests."""
    return {
        "reflection": {
            "is_complete": True,
            "missing": [],
            "refined_query": None,
        }
    }


def _stub_synthesize(state):
    """Echo synthesizer — keeps synth offline-safe in graph tests."""
    msgs = state.get("messages", [])
    user_query = next(
        (m["content"] for m in msgs if m.get("role") == "user"),
        "",
    )
    draft = f"[stub answer to: {user_query}]"
    return {
        "answer_draft": draft,
        "messages": msgs + [{"role": "assistant", "content": draft}],
    }


def test_graph_happy_path_terminates_after_one_pass(monkeypatch):
    from src.agent import nodes

    monkeypatch.setattr(nodes, "classify_query", _stub_classifier)
    monkeypatch.setattr(nodes, "_plan_subquestion", _stub_planner)
    monkeypatch.setattr(nodes, "reflect", _stub_reflect_complete)
    monkeypatch.setattr(nodes, "synthesize", _stub_synthesize)
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
    monkeypatch.setattr(nodes, "synthesize", _stub_synthesize)
    # Rebuild the graph so it picks up the patched nodes.
    g = build_graph()
    out = g.invoke(initial_state("Find me everything."))
    assert out["iteration_count"] == MAX_ITERATIONS
    assert out["reflection"]["is_complete"] is False


def test_route_after_reflect_pure():
    """`_route_after_reflect` is the brain of the loop edge — verify directly."""
    # is_complete=True → end (refined_query irrelevant when complete)
    state = {
        "iteration_count": 1,
        "reflection": {"is_complete": True, "refined_query": "ignored"},
    }
    assert graph_mod._route_after_reflect(state) == "end"
    # is_complete=False with a refined_query under the cap → loop
    state = {
        "iteration_count": 1,
        "reflection": {"is_complete": False, "refined_query": "fill the gap"},
    }
    assert graph_mod._route_after_reflect(state) == "loop"
    # cap reached even with is_complete=False → end
    state = {
        "iteration_count": MAX_ITERATIONS,
        "reflection": {"is_complete": False, "refined_query": "still need more"},
    }
    assert graph_mod._route_after_reflect(state) == "end"
    # Incomplete but no refined_query → end (no concrete next pass).
    state = {
        "iteration_count": 1,
        "reflection": {"is_complete": False, "refined_query": None},
    }
    assert graph_mod._route_after_reflect(state) == "end"
    # Empty-string refined_query is also treated as none.
    state = {
        "iteration_count": 1,
        "reflection": {"is_complete": False, "refined_query": "   "},
    }
    assert graph_mod._route_after_reflect(state) == "end"
    # Missing reflection → not-complete + no refined_query → end.
    state = {"iteration_count": 0, "reflection": {}}
    assert graph_mod._route_after_reflect(state) == "end"


def test_loopback_feeds_refined_query_into_router(monkeypatch):
    """End-to-end: reflect says incomplete + refined_query, router sees it on pass 2."""
    from src.agent import nodes

    monkeypatch.setattr(nodes, "classify_query", _stub_classifier)

    seen_queries: list[str] = []

    def stub_plan(q):
        seen_queries.append(q)
        return {"use_docs": False, "tool_calls": []}

    # First reflect call says incomplete + refined; second says complete.
    # That gives us exactly one loop-back, capped well below MAX_ITERATIONS.
    reflect_calls = {"n": 0}

    def stub_reflect(_state):
        reflect_calls["n"] += 1
        if reflect_calls["n"] == 1:
            return {
                "reflection": {
                    "is_complete": False,
                    "missing": ["needs more on Sydney"],
                    "refined_query": "Sydney median price 2025",
                }
            }
        return {
            "reflection": {
                "is_complete": True,
                "missing": [],
                "refined_query": None,
            }
        }

    monkeypatch.setattr(nodes, "_plan_subquestion", stub_plan)
    monkeypatch.setattr(nodes, "reflect", stub_reflect)
    monkeypatch.setattr(nodes, "synthesize", _stub_synthesize)
    g = build_graph()
    out = g.invoke(initial_state("Compare Sydney and Melbourne prices."))

    assert out["iteration_count"] == 2  # one loop-back, then complete
    assert reflect_calls["n"] == 2
    # First pass plans the original question; second pass plans the refined query.
    assert seen_queries == [
        "Compare Sydney and Melbourne prices.",
        "Sydney median price 2025",
    ]
    assert out["reflection"]["is_complete"] is True


def test_loopback_short_circuits_when_refined_query_missing(monkeypatch):
    """Incomplete reflection without a refined_query must NOT loop."""
    from src.agent import nodes

    monkeypatch.setattr(nodes, "classify_query", _stub_classifier)
    monkeypatch.setattr(nodes, "_plan_subquestion", _stub_planner)

    def stub_reflect(_state):
        return {
            "reflection": {
                "is_complete": False,
                "missing": ["everything"],
                "refined_query": None,  # contradiction; router safety kicks in
            }
        }

    monkeypatch.setattr(nodes, "reflect", stub_reflect)
    monkeypatch.setattr(nodes, "synthesize", _stub_synthesize)
    g = build_graph()
    out = g.invoke(initial_state("Find me everything."))
    # Single pass, then ended despite incomplete — no wasted iterations.
    assert out["iteration_count"] == 1
    assert out["reflection"]["is_complete"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
