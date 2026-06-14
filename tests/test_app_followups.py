"""Unit tests for the deterministic follow-up suggester (Task 4.08).

Coverage:
  - tool-driven suggestions take priority and surface the right rule
    per tool name
  - failed tool calls don't trigger their rule (no follow-up that
    references a result that doesn't exist)
  - persona-flavoured suggestions kick in when tool rules don't fill
    the slots
  - generic suggestions cover the always-3 minimum
  - dedupe so persona/generic don't repeat what tools already said
  - cap at MAX_SUGGESTIONS (= 3)
"""
from __future__ import annotations

import src.agent.llm as llm_mod
from src.app.followups import (
    MAX_SUGGESTIONS,
    ai_followups,
    followup_questions,
    suggest_followups,
)

# ---------- tool-driven rules ----------------------------------


def test_rba_cash_rate_tool_suggests_history_followup():
    suggestions = suggest_followups(
        "What's the current cash rate?",
        persona="general",
        tool_results=[{"tool": "rba_cash_rate", "result": {"data": {"rate_pct": 4.10}}}],
    )
    assert any("cash rate" in s.lower() and "12 months" in s for s in suggestions)


def test_stamp_duty_nsw_suggests_vic_qld_compare():
    suggestions = suggest_followups(
        "Stamp duty on $850k in NSW",
        persona="first_home_buyer",
        tool_results=[
            {"tool": "compute_stamp_duty_nsw", "result": {"data": {"duty_aud": 33000}}}
        ],
    )
    assert any(
        "VIC" in s and "QLD" in s for s in suggestions
    ), f"expected NSW vs VIC vs QLD suggestion, got {suggestions}"


def test_failed_tool_call_does_not_emit_its_rule():
    """An errored tool didn't actually deliver a result, so its
    follow-up rule shouldn't fire."""
    suggestions = suggest_followups(
        "Stamp duty",
        persona="general",
        tool_results=[
            {"tool": "compute_stamp_duty_nsw", "error": "missing arg purchase_price_aud"}
        ],
    )
    assert not any(
        "VIC" in s and "QLD" in s for s in suggestions
    ), "expected stamp-duty rule to be suppressed"


def test_compute_mortgage_repayment_suggests_rate_shock():
    suggestions = suggest_followups(
        "Repayment on $700k at today's rates",
        persona="first_home_buyer",
        tool_results=[
            {"tool": "compute_mortgage_repayment", "result": {"data": {}}}
        ],
    )
    assert any("rates rose" in s for s in suggestions)


def test_compute_rental_yield_suggests_vacancy():
    suggestions = suggest_followups(
        "Rental yield on a $750k property",
        persona="investor",
        tool_results=[
            {"tool": "compute_rental_yield", "result": {"data": {}}}
        ],
    )
    assert any("vacancy" in s.lower() for s in suggestions)


# ---------- persona rules --------------------------------------


def test_first_home_buyer_persona_gets_concession_followup():
    suggestions = suggest_followups(
        "Tell me about buying my first place",
        persona="first_home_buyer",
        tool_results=[],
    )
    assert any("grant" in s.lower() or "concession" in s.lower() for s in suggestions)


def test_investor_persona_gets_yield_or_vacancy_followup():
    suggestions = suggest_followups(
        "Tell me about investing in property",
        persona="investor",
        tool_results=[],
    )
    assert any("yield" in s.lower() or "vacancy" in s.lower() for s in suggestions)


def test_researcher_persona_gets_long_run_or_capitals_followup():
    suggestions = suggest_followups(
        "Long-run trends in housing",
        persona="policy_researcher",
        tool_results=[],
    )
    assert any(
        "long-run" in s.lower() or "capital cities" in s.lower() for s in suggestions
    )


def test_general_persona_falls_through_to_generic_set():
    """No persona rules → still 3 generic suggestions."""
    suggestions = suggest_followups(
        "What is housing?",
        persona="general",
        tool_results=[],
    )
    assert len(suggestions) == MAX_SUGGESTIONS


# ---------- dedupe + cap ---------------------------------------


def test_caps_at_max_suggestions():
    suggestions = suggest_followups(
        "Cash rate vacancy stamp duty",
        persona="investor",
        tool_results=[
            {"tool": "rba_cash_rate", "result": {}},
            {"tool": "sqm_vacancy", "result": {}},
            {"tool": "compute_stamp_duty_nsw", "result": {}},
        ],
    )
    assert len(suggestions) == MAX_SUGGESTIONS


def test_dedupes_overlapping_suggestions():
    """If two sources happen to propose the same exact line, it's
    only emitted once."""
    suggestions = suggest_followups(
        "What's vacancy in Sydney?",
        persona="investor",
        tool_results=[
            {"tool": "sqm_vacancy", "result": {}},
        ],
    )
    # The exact list should have no repeats.
    assert len(suggestions) == len(set(suggestions))


# ---------- empties --------------------------------------------


def test_empty_tool_list_and_unknown_persona_still_returns_generic():
    suggestions = suggest_followups(
        "anything",
        persona=None,
        tool_results=None,
    )
    assert len(suggestions) == MAX_SUGGESTIONS


def test_keyword_sweetener_for_cash_rate_query():
    """Cash-rate keyword in the query nudges in an RBA-statement question."""
    suggestions = suggest_followups(
        "What's the cash rate?",
        persona="general",
        tool_results=[],
    )
    assert any("RBA" in s or "statement" in s.lower() for s in suggestions)


def test_keyword_sweetener_for_vacancy_query():
    suggestions = suggest_followups(
        "Sydney rental vacancy?",
        persona="general",
        tool_results=[],
    )
    assert any("tightest rental" in s for s in suggestions)


# ---------- AI-generated layer ---------------------------------


def _stub_call_with_tool(questions):
    """Build a call_with_tool stand-in returning a fixed questions list."""

    def _stub(*, system, user, tool_def, tool_name, max_tokens, model):
        return {"questions": questions}, {}

    return _stub


def test_ai_followups_returns_cleaned_questions(monkeypatch):
    monkeypatch.setattr(
        llm_mod,
        "call_with_tool",
        _stub_call_with_tool(
            ["- What's the yield in Parramatta?", "How has vacancy moved there?"]
        ),
    )
    out = ai_followups(
        "Is Parramatta a good buy?",
        persona="investor",
        answer="Parramatta median is $1.1M with a 3.2% yield.",
        tool_results=[],
    )
    # Leading bullet stripped, both kept, order preserved.
    assert out == [
        "What's the yield in Parramatta?",
        "How has vacancy moved there?",
    ]


def test_ai_followups_empty_answer_skips_call(monkeypatch):
    """No answer text → no LLM call, returns []."""

    def _boom(**kwargs):
        raise AssertionError("call_with_tool should not be invoked")

    monkeypatch.setattr(llm_mod, "call_with_tool", _boom)
    assert ai_followups("q", "investor", answer="", tool_results=[]) == []


def test_ai_followups_swallows_provider_error(monkeypatch):
    def _raise(**kwargs):
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    monkeypatch.setattr(llm_mod, "call_with_tool", _raise)
    assert ai_followups("q", "investor", answer="some answer", tool_results=[]) == []


def test_ai_followups_caps_and_dedupes(monkeypatch):
    monkeypatch.setattr(
        llm_mod,
        "call_with_tool",
        _stub_call_with_tool(["A?", "A?", "B?", "C?", "D?"]),
    )
    out = ai_followups("q", "general", answer="answer", tool_results=[])
    assert out == ["A?", "B?", "C?"]  # deduped + capped at 3


def test_followup_questions_prefers_ai_then_tops_up(monkeypatch):
    """One AI question + deterministic top-up to fill the row."""
    monkeypatch.setattr(
        llm_mod, "call_with_tool", _stub_call_with_tool(["AI-only question?"])
    )
    out = followup_questions(
        "What is the cash rate?",
        persona="general",
        answer="The cash rate is 4.10%.",
        tool_results=[{"tool": "rba_cash_rate", "result": {}}],
    )
    assert out[0] == "AI-only question?"
    assert len(out) == MAX_SUGGESTIONS
    assert len(out) == len(set(out))


def test_followup_questions_falls_back_when_ai_empty(monkeypatch):
    """AI yields nothing → identical to the deterministic suggester."""

    def _raise(**kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(llm_mod, "call_with_tool", _raise)
    out = followup_questions(
        "Stamp duty on $850k in NSW",
        persona="first_home_buyer",
        answer="NSW stamp duty is ~$33,000.",
        tool_results=[{"tool": "compute_stamp_duty_nsw", "result": {}}],
    )
    deterministic = suggest_followups(
        "Stamp duty on $850k in NSW",
        persona="first_home_buyer",
        tool_results=[{"tool": "compute_stamp_duty_nsw", "result": {}}],
    )
    assert out == deterministic


def test_followup_questions_disabled_skips_ai(monkeypatch):
    """CADASTRE_AI_FOLLOWUPS=0 → no LLM call, deterministic only."""
    monkeypatch.setenv("CADASTRE_AI_FOLLOWUPS", "0")

    def _boom(**kwargs):
        raise AssertionError("AI layer should be disabled")

    monkeypatch.setattr(llm_mod, "call_with_tool", _boom)
    out = followup_questions(
        "What is housing?",
        persona="general",
        answer="Housing is shelter.",
        tool_results=[],
    )
    assert len(out) == MAX_SUGGESTIONS
