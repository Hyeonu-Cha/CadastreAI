"""Unit tests for persona wiring (Task 4.02).

Three responsibilities to test:

1. **Precedence.** `effective_persona` returns user_persona if set,
   otherwise classifier persona, otherwise "general".
2. **Prompts and hints.** Every supported persona has a non-empty
   addendum + a router hint slot (general's hint may be empty —
   that's the contract).
3. **Publisher boost.** `apply_publisher_boost` reorders chunks by
   `score * boost`, neutral persona is a no-op, and chunks with
   unmapped publishers default to multiplier 1.0.

Plus an integration check that `retrieve_or_tool` threads persona
through `_plan_subquestion` and applies the boost on the way out —
both stubbed so the test stays offline.
"""
from __future__ import annotations

import pytest

from src.agent import persona as p

# ---------- effective_persona precedence -------------------------


def test_effective_persona_user_wins():
    state = {
        "user_persona": "investor",
        "classification": {"persona": "first_home_buyer"},
    }
    assert p.effective_persona(state) == "investor"


def test_effective_persona_falls_back_to_classifier():
    state = {"classification": {"persona": "policy_researcher"}}
    assert p.effective_persona(state) == "policy_researcher"


def test_effective_persona_general_when_nothing_set():
    assert p.effective_persona({}) == "general"


def test_effective_persona_invalid_user_falls_through_to_classifier():
    """Garbage in user_persona shouldn't override a valid classifier."""
    state = {
        "user_persona": "wizard",
        "classification": {"persona": "investor"},
    }
    assert p.effective_persona(state) == "investor"


def test_effective_persona_invalid_classifier_falls_back_to_general():
    state = {"classification": {"persona": "wizard"}}
    assert p.effective_persona(state) == "general"


def test_effective_persona_empty_string_user_falls_through():
    state = {
        "user_persona": "",
        "classification": {"persona": "journalist"},
    }
    assert p.effective_persona(state) == "journalist"


# ---------- prompt addendum / router hints -----------------------


@pytest.mark.parametrize(
    "persona",
    ["first_home_buyer", "investor", "policy_researcher", "journalist", "general"],
)
def test_persona_prompt_addendum_non_empty_for_every_persona(persona):
    addendum = p.persona_prompt_addendum(persona)
    assert isinstance(addendum, str)
    assert len(addendum.strip()) > 0


def test_persona_prompt_addendum_unknown_falls_back_to_general():
    """Unknown persona shouldn't crash — return the general addendum."""
    assert p.persona_prompt_addendum("astronaut") == p.persona_prompt_addendum(  # type: ignore[arg-type]
        "general"
    )


def test_persona_router_hints_non_empty_for_targeted_personas():
    """Targeted personas have a hint; 'general' is intentionally empty."""
    for persona in ("first_home_buyer", "investor", "policy_researcher", "journalist"):
        hint = p.persona_router_hints(persona)
        assert hint and len(hint.strip()) > 10


def test_persona_router_hints_general_is_empty():
    """No hint when persona is 'general' — keeps the router prompt clean."""
    assert p.persona_router_hints("general") == ""


# ---------- publisher boost --------------------------------------


def test_publisher_boost_general_is_empty_dict():
    assert p.persona_publisher_boost("general") == {}


def test_publisher_boost_first_home_buyer_emphasises_consumer_pubs():
    boost = p.persona_publisher_boost("first_home_buyer")
    # NHFIC/PropTrack should be > 1; AHURI should be < 1 for this persona.
    assert boost.get("Housing Australia (NHFIC)", 1.0) > 1.0
    assert boost.get("AHURI", 1.0) < 1.0


def test_publisher_boost_researcher_emphasises_academic_pubs():
    boost = p.persona_publisher_boost("policy_researcher")
    assert boost.get("AHURI", 1.0) > 1.0
    assert boost.get("Productivity Commission", 1.0) > 1.0


def test_apply_publisher_boost_reorders_for_targeted_persona():
    """Investor boost should pull SQM Research above a higher-scored RBA chunk."""
    chunks = [
        {"score": 0.80, "payload": {"publisher": "RBA"}},
        {"score": 0.75, "payload": {"publisher": "SQM Research"}},
    ]
    out = p.apply_publisher_boost(chunks, "investor")
    # SQM (0.75 * 1.20 = 0.90) should win over RBA (0.80 * 1.05 = 0.84)
    assert out[0]["payload"]["publisher"] == "SQM Research"
    assert out[1]["payload"]["publisher"] == "RBA"


def test_apply_publisher_boost_neutral_persona_is_noop():
    """'general' has no boost table — order should be preserved."""
    chunks = [
        {"score": 0.80, "payload": {"publisher": "RBA"}},
        {"score": 0.75, "payload": {"publisher": "AHURI"}},
    ]
    out = p.apply_publisher_boost(chunks, "general")
    assert [c["payload"]["publisher"] for c in out] == ["RBA", "AHURI"]


def test_apply_publisher_boost_unknown_publisher_gets_neutral_multiplier():
    """A publisher not in the boost table gets multiplier 1.0."""
    chunks = [
        {"score": 0.80, "payload": {"publisher": "Some New Publisher"}},
        {"score": 0.70, "payload": {"publisher": "PropTrack"}},
    ]
    out = p.apply_publisher_boost(chunks, "first_home_buyer")
    # PropTrack boost is 1.15 → 0.805; unknown publisher stays at 0.80.
    assert out[0]["payload"]["publisher"] == "PropTrack"
    assert pytest.approx(out[0]["boosted_score"], rel=1e-3) == 0.805
    assert pytest.approx(out[1]["boost"], rel=1e-3) == 1.0


def test_apply_publisher_boost_records_boost_metadata():
    """Inspectable: each chunk gets boosted_score + boost fields."""
    chunks = [{"score": 0.5, "payload": {"publisher": "AHURI"}}]
    out = p.apply_publisher_boost(chunks, "policy_researcher")
    assert "boosted_score" in out[0]
    assert "boost" in out[0]
    # AHURI boost for researcher is 1.20.
    assert pytest.approx(out[0]["boost"], rel=1e-3) == 1.20


def test_apply_publisher_boost_does_not_mutate_input():
    """Defensive — no in-place edits on caller's list of chunks."""
    chunks = [{"score": 0.80, "payload": {"publisher": "AHURI"}}]
    p.apply_publisher_boost(chunks, "policy_researcher")
    assert "boosted_score" not in chunks[0]
    assert "boost" not in chunks[0]


# ---------- end-to-end: retrieve_or_tool threads persona ---------


def test_retrieve_or_tool_passes_persona_to_planner_and_boosts(monkeypatch):
    """End-to-end: retrieve_or_tool reads persona from state, hands it
    to the planner, and applies the boost on the chunks it receives."""
    from src.agent import nodes

    seen_persona: list[str] = []

    def stub_plan(q, persona=None):
        seen_persona.append(persona)
        return {"use_docs": True, "tool_calls": []}

    def stub_retrieve(q, k=10):
        # Return chunks where boost order will differ from raw order.
        return [
            {"chunk_id": "a", "score": 0.80, "payload": {"publisher": "AHURI"}},
            {"chunk_id": "b", "score": 0.70, "payload": {"publisher": "PropTrack"}},
        ]

    monkeypatch.setattr(nodes, "_plan_subquestion", stub_plan)
    monkeypatch.setattr(nodes, "_retrieve_docs", stub_retrieve)

    state = {
        "messages": [{"role": "user", "content": "test query"}],
        "user_persona": "first_home_buyer",
        "iteration_count": 0,
    }
    out = nodes.retrieve_or_tool(state)

    # Planner should have been told the persona.
    assert seen_persona == ["first_home_buyer"]
    # FHB nudges PropTrack up (1.15) and AHURI down (0.90):
    #   AHURI:    0.80 * 0.90 = 0.72
    #   PropTrack:0.70 * 1.15 = 0.805 → wins
    assert out["retrieved_chunks"][0]["payload"]["publisher"] == "PropTrack"
    assert out["retrieved_chunks"][1]["payload"]["publisher"] == "AHURI"


def test_retrieve_or_tool_general_persona_preserves_raw_order(monkeypatch):
    """No nudges for 'general' — raw retriever order is honoured."""
    from src.agent import nodes

    monkeypatch.setattr(
        nodes,
        "_plan_subquestion",
        lambda q, persona=None: {"use_docs": True, "tool_calls": []},
    )
    monkeypatch.setattr(
        nodes,
        "_retrieve_docs",
        lambda q, k=10: [
            {"chunk_id": "a", "score": 0.80, "payload": {"publisher": "AHURI"}},
            {"chunk_id": "b", "score": 0.70, "payload": {"publisher": "PropTrack"}},
        ],
    )

    state = {
        "messages": [{"role": "user", "content": "test"}],
        "user_persona": "general",
        "iteration_count": 0,
    }
    out = nodes.retrieve_or_tool(state)
    assert [c["payload"]["publisher"] for c in out["retrieved_chunks"]] == [
        "AHURI",
        "PropTrack",
    ]


# ---------- initial_state user_persona forwarding ----------------


def test_initial_state_accepts_user_persona():
    from src.agent.graph import initial_state

    state = initial_state("test", user_persona="investor")
    assert state["user_persona"] == "investor"


def test_initial_state_omits_user_persona_when_none():
    from src.agent.graph import initial_state

    state = initial_state("test")
    # Either absent or explicitly None — both fine, but never set to a value.
    assert not state.get("user_persona")
