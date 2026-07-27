"""Tests for the `reflect` node (Task 3.17).

Three tiers:
  1. Pure offline — `_build_reflect_prompt` is a deterministic string
     builder and gets a thorough unit test.
  2. Mocked anthropic — a fake `Anthropic` client returns a synthetic
     tool_use block so we exercise reflect's parsing/normalisation
     without burning tokens.
  3. Network — one live test on a clearly-incomplete state asserts
     reflect returns a usable refined_query.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from src.agent import nodes
from src.agent.graph import initial_state


def _state_with_evidence():
    state = initial_state(
        "What's the current cash rate and median Sydney house price?"
    )
    state["classification"] = {
        "persona": "general",
        "query_type": "comparative",
        "needs_docs": False,
        "needs_data": True,
        "needs_decomposition": True,
    }
    state["sub_questions"] = [
        "What is the current RBA cash rate?",
        "What is the current median Sydney house price?",
    ]
    state["retrieved_chunks"] = [
        {
            "chunk_id": "c1",
            "score": 0.91,
            "payload": {
                "publisher": "RBA",
                "title": "Statement on Monetary Policy",
                "section_heading": "Cash rate decision",
                "text": "The Board left the cash rate unchanged at 4.10%.",
            },
        }
    ]
    state["tool_results"] = [
        {
            "sub_question": "What is the current RBA cash rate?",
            "tool": "rba_cash_rate",
            "args": {"period": "latest"},
            "result": {
                "data": {"rate_pct": 4.10, "frequency": "monthly_average"},
                "source": "RBA F1.1",
            },
        }
    ]
    state["answer_draft"] = "The cash rate is 4.10% [RBA F1.1]."
    state["iteration_count"] = 1
    return state


def test_build_reflect_prompt_renders_all_sections():
    prompt = nodes._build_reflect_prompt(_state_with_evidence())
    assert "ORIGINAL QUESTION:" in prompt
    assert "SUB-QUESTIONS:" in prompt
    assert "1. What is the current RBA cash rate?" in prompt
    assert "RETRIEVED CHUNKS" in prompt
    assert "RBA | Statement on Monetary Policy" in prompt
    assert "TOOL RESULTS" in prompt
    assert "rba_cash_rate" in prompt
    assert "data={'rate_pct': 4.1" in prompt
    assert "DRAFT ANSWER (iteration 1)" in prompt
    assert "4.10%" in prompt


def test_reflect_chunk_summary_shows_date_when_present():
    """Chunk vintage is surfaced to the reflector (Task 5.16) so it can weigh
    whether the evidence is current."""
    out = nodes._summarise_chunks(
        [{"score": 0.5, "payload": {"publisher": "AHURI", "title": "Tax treatment", "date": "2018", "text": "x"}}]
    )
    assert "AHURI (2018) | Tax treatment" in out


def test_reflect_chunk_summary_omits_date_when_absent():
    """No date → no empty `(None)` clutter (keeps the pre-5.16 format)."""
    out = nodes._summarise_chunks(
        [{"score": 0.5, "payload": {"publisher": "RBA", "title": "SMP", "text": "x"}}]
    )
    assert "RBA | SMP" in out
    assert "(None)" not in out


def test_build_reflect_prompt_handles_empty_evidence():
    state = initial_state("anything")
    state["classification"] = {
        "persona": "general",
        "query_type": "factual",
        "needs_docs": True,
        "needs_data": False,
        "needs_decomposition": False,
    }
    state["answer_draft"] = "(empty)"
    prompt = nodes._build_reflect_prompt(state)
    assert "(no chunks retrieved)" in prompt
    assert "(no tool calls)" in prompt
    # No sub-questions → fallback line.
    assert "atomically" in prompt


def test_build_reflect_prompt_caps_evidence(monkeypatch):
    """Avoid blowing the prompt budget with hundreds of chunks."""
    state = initial_state("q")
    monkeypatch.setattr(nodes, "REFLECT_MAX_EVIDENCE_ITEMS", 3)
    state["retrieved_chunks"] = [
        {
            "chunk_id": f"c{i}",
            "score": 0.5,
            "payload": {"publisher": "X", "title": f"t{i}", "text": f"txt{i}"},
        }
        for i in range(10)
    ]
    state["tool_results"] = [
        {"tool": "rba_cash_rate", "args": {}, "result": {"data": {"rate_pct": i}}}
        for i in range(10)
    ]
    state["answer_draft"] = "draft"
    prompt = nodes._build_reflect_prompt(state)
    assert "and 7 more chunks" in prompt
    assert "and 7 more results" in prompt


# ---------- mocked Anthropic client ----------------------------------


class _FakeMessages:
    def __init__(self, tool_input):
        self._tool_input = tool_input

    def create(self, **_kw):
        block = SimpleNamespace(type="tool_use", input=self._tool_input)
        return SimpleNamespace(content=[block])


class _FakeAnthropic:
    def __init__(self, tool_input):
        self.messages = _FakeMessages(tool_input)


def _patch_anthropic(monkeypatch, tool_input):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", lambda **_kw: _FakeAnthropic(tool_input))


def test_reflect_returns_complete_on_success(monkeypatch):
    _patch_anthropic(
        monkeypatch,
        {"is_complete": True, "missing": [], "refined_query": ""},
    )
    out = nodes.reflect(_state_with_evidence())
    refl = out["reflection"]
    assert refl["is_complete"] is True
    assert refl["missing"] == []
    assert refl["refined_query"] is None  # empty string normalised to None


def test_reflect_returns_incomplete_with_refinement(monkeypatch):
    _patch_anthropic(
        monkeypatch,
        {
            "is_complete": False,
            "missing": ["Sydney median price not in draft"],
            "refined_query": "current ABS median Sydney house price",
        },
    )
    out = nodes.reflect(_state_with_evidence())
    refl = out["reflection"]
    assert refl["is_complete"] is False
    assert refl["missing"] == ["Sydney median price not in draft"]
    assert refl["refined_query"] == "current ABS median Sydney house price"


def test_reflect_drops_refined_query_when_complete(monkeypatch):
    """A model that says 'complete' but supplies a refinement contradicts itself."""
    _patch_anthropic(
        monkeypatch,
        {
            "is_complete": True,
            "missing": [],
            "refined_query": "this should be ignored",
        },
    )
    out = nodes.reflect(_state_with_evidence())
    assert out["reflection"]["is_complete"] is True
    assert out["reflection"]["refined_query"] is None


def test_reflect_filters_empty_missing_strings(monkeypatch):
    _patch_anthropic(
        monkeypatch,
        {
            "is_complete": False,
            "missing": ["", "   ", "real gap", None],  # type: ignore[list-item]
            "refined_query": "something",
        },
    )
    out = nodes.reflect(_state_with_evidence())
    assert out["reflection"]["missing"] == ["real gap"]


def test_reflect_falls_back_when_api_errors(monkeypatch):
    """Transient API failure must not strand the agent in a doomed loop."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    import anthropic

    class _Boom:
        @property
        def messages(self):
            class M:
                def create(self_inner, **_kw):
                    raise RuntimeError("API down")

            return M()

    monkeypatch.setattr(anthropic, "Anthropic", lambda **_kw: _Boom())
    out = nodes.reflect(_state_with_evidence())
    refl = out["reflection"]
    assert refl["is_complete"] is True  # fail-safe to ship
    assert refl["missing"] == []


def test_reflect_falls_back_on_missing_tool_use(monkeypatch):
    """Model returned text-only — defensive fallback to is_complete."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    import anthropic

    class _M:
        def create(self, **_kw):
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="hi")])

    monkeypatch.setattr(
        anthropic, "Anthropic", lambda **_kw: SimpleNamespace(messages=_M())
    )
    out = nodes.reflect(_state_with_evidence())
    assert out["reflection"]["is_complete"] is True


def test_reflect_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        nodes.reflect(_state_with_evidence())


# ---------- live -----------------------------------------------------


@pytest.mark.network
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
def test_reflect_live_flags_obvious_gap():
    """Draft answers cash rate but state asks for cash rate AND Sydney prices.

    Reflect should mark incomplete, name the Sydney gap, and produce a
    non-empty refined_query.
    """
    state = _state_with_evidence()
    out = nodes.reflect(state)
    refl = out["reflection"]
    # The draft only addresses one of two sub-questions, so reflect
    # should flag it. We don't pin the exact missing/refined text — just
    # that the structure is plausible.
    assert refl["is_complete"] is False
    assert any("sydney" in m.lower() or "median" in m.lower() for m in refl["missing"]), (
        f"expected a Sydney/median gap, got {refl['missing']}"
    )
    assert refl["refined_query"]
