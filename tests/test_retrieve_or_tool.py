"""Tests for the `retrieve_or_tool` router (Task 3.15).

Offline tests monkey-patch `_plan_subquestion` and `_retrieve_docs` so
the router's dispatch logic is verified without hitting Anthropic or
Qdrant. A single network-marked test exercises the live planner.

Coverage:
  - sub_questions present → one plan per question, both docs and tools
    surface in state.
  - sub_questions empty → router falls back to the user query.
  - loop-back (iteration_count > 0) → only the refined_query is planned.
  - unknown tool name → recorded as an error, doesn't sink the pass.
  - missing required arg → recorded as an error per call.
  - planner exception → recorded against the sub-question, loop continues.
  - iteration_count always increments by exactly one.
"""
from __future__ import annotations

import os

import pytest

from src.agent import nodes
from src.agent.graph import initial_state


def _classify(state, **overrides):
    state["classification"] = {
        "persona": "general",
        "query_type": "factual",
        "needs_docs": True,
        "needs_data": True,
        "needs_decomposition": False,
        **overrides,
    }
    return state


def test_router_dispatches_per_subquestion(monkeypatch):
    state = _classify(initial_state("compound query"))
    state["sub_questions"] = ["What's the cash rate?", "What's NSW stamp duty on $900k?"]

    plans = {
        "What's the cash rate?": {
            "use_docs": False,
            "tool_calls": [{"tool": "rba_cash_rate", "args": {"period": "latest"}}],
        },
        "What's NSW stamp duty on $900k?": {
            "use_docs": True,
            "doc_query": "NSW stamp duty",
            "tool_calls": [
                {
                    "tool": "compute_stamp_duty_nsw",
                    "args": {"purchase_price_aud": 900000, "is_first_home_buyer": False},
                }
            ],
        },
    }
    monkeypatch.setattr(nodes, "_plan_subquestion", lambda q: plans[q])
    monkeypatch.setattr(
        nodes,
        "_retrieve_docs",
        lambda q, k=nodes.DOCS_TOP_K: [
            {"chunk_id": f"c-{q}", "score": 0.9, "payload": {"text": q}}
        ],
    )

    fake_envelope = {"data": {"rate_pct": 4.10}, "source": "RBA F1.1"}
    monkeypatch.setattr(
        nodes,
        "_execute_tool",
        lambda name, args: {**fake_envelope, "tool": name, "args": args},
    )

    out = nodes.retrieve_or_tool(state)

    assert out["iteration_count"] == 1
    # Doc retrieval only fired for the second sub-question.
    assert len(out["retrieved_chunks"]) == 1
    assert out["retrieved_chunks"][0]["chunk_id"] == "c-NSW stamp duty"
    # Two tool calls, one per sub-question.
    assert [t["tool"] for t in out["tool_results"]] == [
        "rba_cash_rate",
        "compute_stamp_duty_nsw",
    ]
    assert out["tool_results"][0]["sub_question"] == "What's the cash rate?"
    assert out["tool_results"][1]["args"]["purchase_price_aud"] == 900000


def test_router_uses_user_query_when_no_subquestions(monkeypatch):
    state = _classify(initial_state("What's the current cash rate?"))
    state["sub_questions"] = []

    seen: list[str] = []

    def plan(q):
        seen.append(q)
        return {"use_docs": False, "tool_calls": []}

    monkeypatch.setattr(nodes, "_plan_subquestion", plan)
    out = nodes.retrieve_or_tool(state)

    assert seen == ["What's the current cash rate?"]
    assert out["iteration_count"] == 1
    assert out["tool_results"] == []
    assert out["retrieved_chunks"] == []


def test_router_loopback_uses_refined_query(monkeypatch):
    state = _classify(initial_state("original question"))
    state["sub_questions"] = ["sub a", "sub b"]
    state["iteration_count"] = 1
    state["reflection"] = {
        "is_complete": False,
        "missing": ["vacancy"],
        "refined_query": "Sydney rental vacancy 2025",
    }

    seen: list[str] = []

    def plan(q):
        seen.append(q)
        return {"use_docs": True, "tool_calls": []}

    monkeypatch.setattr(nodes, "_plan_subquestion", plan)
    monkeypatch.setattr(
        nodes,
        "_retrieve_docs",
        lambda q, k=nodes.DOCS_TOP_K: [
            {"chunk_id": "x", "score": 1.0, "payload": {"text": q}}
        ],
    )

    out = nodes.retrieve_or_tool(state)
    # On loop-back we only re-plan the refined query — sub_questions are
    # NOT replanned (they already produced their evidence the first pass).
    assert seen == ["Sydney rental vacancy 2025"]
    assert out["iteration_count"] == 2
    assert out["retrieved_chunks"][0]["payload"]["text"] == "Sydney rental vacancy 2025"


def test_router_records_unknown_tool_as_error(monkeypatch):
    state = _classify(initial_state("q"))
    state["sub_questions"] = ["q"]
    monkeypatch.setattr(
        nodes,
        "_plan_subquestion",
        lambda q: {
            "use_docs": False,
            "tool_calls": [{"tool": "make_coffee", "args": {}}],
        },
    )
    out = nodes.retrieve_or_tool(state)
    assert len(out["tool_results"]) == 1
    err = out["tool_results"][0]["error"]
    assert "unknown tool" in err
    assert out["iteration_count"] == 1


def test_router_records_missing_required_arg(monkeypatch):
    state = _classify(initial_state("q"))
    state["sub_questions"] = ["q"]
    monkeypatch.setattr(
        nodes,
        "_plan_subquestion",
        lambda q: {
            "use_docs": False,
            "tool_calls": [
                {"tool": "compute_stamp_duty_nsw", "args": {}}  # missing purchase_price_aud
            ],
        },
    )
    out = nodes.retrieve_or_tool(state)
    err = out["tool_results"][0]["error"]
    assert "purchase_price_aud" in err


def test_router_records_planner_failure(monkeypatch):
    state = _classify(initial_state("q"))
    state["sub_questions"] = ["bad", "good"]

    plans = {"good": {"use_docs": False, "tool_calls": []}}

    def plan(q):
        if q == "bad":
            raise RuntimeError("boom")
        return plans[q]

    monkeypatch.setattr(nodes, "_plan_subquestion", plan)
    out = nodes.retrieve_or_tool(state)
    # First sub-question's plan failed; the second still got planned.
    assert out["iteration_count"] == 1
    errors = [t for t in out["tool_results"] if "error" in t]
    assert len(errors) == 1
    assert errors[0]["sub_question"] == "bad"
    assert "plan_failed" in errors[0]["error"]


def test_router_filters_unknown_args_silently(monkeypatch):
    """Planner hallucinations like {hours_per_week: 40} shouldn't crash."""
    state = _classify(initial_state("q"))
    state["sub_questions"] = ["q"]

    captured: dict = {}

    def fake_yield(annual_rent_aud, property_value_aud):
        captured.update(
            annual_rent_aud=annual_rent_aud, property_value_aud=property_value_aud
        )
        return {"data": {"gross_yield_pct": 5.0}, "source": "calc"}

    monkeypatch.setattr(
        nodes,
        "_tool_catalogue",
        lambda: {
            "compute_rental_yield": (
                fake_yield,
                ["annual_rent_aud", "property_value_aud"],
                [],
            )
        },
    )
    monkeypatch.setattr(
        nodes,
        "_plan_subquestion",
        lambda q: {
            "use_docs": False,
            "tool_calls": [
                {
                    "tool": "compute_rental_yield",
                    "args": {
                        "annual_rent_aud": 30000,
                        "property_value_aud": 800000,
                        "hours_per_week": 40,  # bogus
                    },
                }
            ],
        },
    )
    out = nodes.retrieve_or_tool(state)
    assert captured == {"annual_rent_aud": 30000, "property_value_aud": 800000}
    assert "result" in out["tool_results"][0]
    assert "error" not in out["tool_results"][0]


@pytest.mark.network
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
def test_planner_live_returns_valid_structure():
    plan = nodes._plan_subquestion("What is the current RBA cash rate?")
    assert isinstance(plan.get("use_docs"), bool)
    assert isinstance(plan.get("tool_calls"), list)
    # Cash-rate question — Haiku should reach for rba_cash_rate.
    assert any(
        tc.get("tool") == "rba_cash_rate" for tc in plan["tool_calls"]
    ), f"expected rba_cash_rate in plan, got {plan}"
