"""Unit tests for the Streamlit app's pure state helpers (Task 4.01).

The Streamlit entrypoint itself is hard to unit-test (Streamlit needs
its own runtime), so we factor any logic worth testing into
`src.app.state` and keep these tests focused there. Persona mapping,
session-state idempotency, history append/reset, and example-query
shape are the contract surface — anything else is rendering.
"""
from __future__ import annotations

import pytest

from src.app import state as ast_

# ---------- persona mapping ---------------------------------------


def test_ui_persona_list_is_stable():
    """The four user-facing labels are the public contract — pin them."""
    assert ast_.UI_PERSONAS == (
        "Homebuyer",
        "Investor",
        "Researcher",
        "Just exploring",
    )


def test_default_persona_is_just_exploring():
    """Least-commitment default — anyone landing on the app fits here."""
    assert ast_.DEFAULT_PERSONA == "Just exploring"


@pytest.mark.parametrize(
    "ui,agent",
    [
        ("Homebuyer", "first_home_buyer"),
        ("Investor", "investor"),
        ("Researcher", "policy_researcher"),
        ("Just exploring", "general"),
    ],
)
def test_ui_to_agent_persona_known(ui, agent):
    assert ast_.ui_to_agent_persona(ui) == agent


def test_ui_to_agent_persona_unknown_falls_back_to_general():
    """Unknown labels mustn't crash — fallback keeps the app loose."""
    assert ast_.ui_to_agent_persona("Wizard") == "general"
    assert ast_.ui_to_agent_persona("") == "general"


# ---------- session defaults --------------------------------------


def test_init_session_defaults_seeds_empty_session():
    sess: dict = {}
    ast_.init_session_defaults(sess)
    assert sess["persona"] == ast_.DEFAULT_PERSONA
    assert sess["history"] == []
    assert sess["input_buffer"] == ""
    assert sess["show_trace"] is False


def test_init_session_defaults_is_idempotent_and_preserves_existing():
    """Streamlit re-runs the script on every interaction, so this MUST
    NOT clobber state the user has already set."""
    sess = {
        "persona": "Investor",
        "history": [object()],
        "show_trace": True,
        "input_buffer": "draft",
    }
    ast_.init_session_defaults(sess)
    assert sess["persona"] == "Investor"
    assert len(sess["history"]) == 1
    assert sess["show_trace"] is True
    assert sess["input_buffer"] == "draft"


# ---------- turn append / history reset ---------------------------


def test_append_turn_creates_history_if_missing():
    sess: dict = {}
    turn = ast_.TurnRecord(query="hi", persona="Homebuyer")
    ast_.append_turn(sess, turn)
    assert sess["history"] == [turn]


def test_append_turn_preserves_order():
    sess: dict = {"history": []}
    a = ast_.TurnRecord(query="a", persona="Investor")
    b = ast_.TurnRecord(query="b", persona="Investor")
    ast_.append_turn(sess, a)
    ast_.append_turn(sess, b)
    assert [t.query for t in sess["history"]] == ["a", "b"]


def test_reset_history_keeps_persona():
    sess = {"persona": "Researcher", "history": [object(), object()]}
    ast_.reset_history(sess)
    assert sess["history"] == []
    assert sess["persona"] == "Researcher"


# ---------- TurnRecord shape --------------------------------------


def test_turn_record_defaults():
    t = ast_.TurnRecord(query="What's the cash rate?", persona="Homebuyer")
    assert t.answer == ""
    assert t.sub_questions == []
    assert t.tool_results == []
    assert t.retrieved_chunks == []
    assert t.iteration_count == 0
    assert t.error is None
    # ISO-8601 timestamp with timezone — the Z or +00:00 suffix proves
    # we're not getting a naive datetime sneaked in.
    assert "T" in t.created_at and (
        t.created_at.endswith("+00:00") or t.created_at.endswith("Z")
    )


def test_turn_record_can_carry_full_state():
    """Round-trip a fully populated turn — proves no field is read-only."""
    t = ast_.TurnRecord(
        query="Investor question",
        persona="Investor",
        answer="The rental yield is 4.3%.",
        classification={"persona": "investor", "needs_data": True},
        sub_questions=["What is the yield formula?"],
        tool_results=[{"tool": "compute_rental_yield", "args": {"price": 750000}}],
        retrieved_chunks=[{"payload": {"publisher": "RBA", "page": 3}}],
        iteration_count=2,
        elapsed_seconds=1.4,
    )
    assert t.iteration_count == 2
    assert t.tool_results[0]["tool"] == "compute_rental_yield"


# ---------- example queries ---------------------------------------


@pytest.mark.parametrize("persona", list(ast_.UI_PERSONAS))
def test_example_queries_non_empty_per_persona(persona):
    qs = ast_.example_queries_for(persona)
    assert len(qs) >= 3
    assert all(isinstance(q, str) and q.strip() for q in qs)


def test_example_queries_unknown_persona_falls_back():
    """An unmapped persona shouldn't blow up — we fall back to defaults."""
    assert ast_.example_queries_for("Astronaut") == ast_.example_queries_for(
        "Just exploring"
    )


def test_example_queries_are_distinct_per_persona():
    """The whole point of persona-scoped examples is differentiation."""
    h = set(ast_.example_queries_for("Homebuyer"))
    i = set(ast_.example_queries_for("Investor"))
    r = set(ast_.example_queries_for("Researcher"))
    # Each persona's set should differ from at least one other.
    assert h != i or h != r or i != r
