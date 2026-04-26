"""Tests for `synthesize` + the citation post-processor (Task 3.19).

Layered like reflect's tests:
  1. Unit tests for `_build_synth_prompt` and `_enforce_citations`.
  2. Mocked-anthropic tests for the synthesize path itself.
  3. One network-marked live test that exercises Sonnet end-to-end.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from src.agent import nodes
from src.agent.graph import initial_state


def _state_with_evidence():
    state = initial_state("What's the current cash rate?")
    state["classification"] = {
        "persona": "general",
        "query_type": "factual",
        "needs_docs": True,
        "needs_data": True,
        "needs_decomposition": False,
    }
    state["sub_questions"] = []
    state["retrieved_chunks"] = [
        {
            "chunk_id": "rba-smp-2026-04-1",
            "score": 0.92,
            "payload": {
                "publisher": "RBA",
                "title": "Statement on Monetary Policy — April 2026",
                "section_heading": "Cash rate decision",
                "page": 3,
                "text": "The Reserve Bank Board left the cash rate at 4.10%.",
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
                "retrieved_at": "2026-04-26T12:00:00+00:00",
                "citation": "RBA F1.1 — 2026-03-31",
            },
        }
    ]
    state["iteration_count"] = 1
    return state


# ---------- prompt builder ------------------------------------------


def test_build_synth_prompt_includes_evidence_and_question():
    prompt = nodes._build_synth_prompt(_state_with_evidence())
    assert "USER QUESTION:" in prompt
    assert "current cash rate" in prompt
    assert "EVIDENCE — RETRIEVED CHUNKS:" in prompt
    assert "publisher='RBA'" in prompt
    assert "page=3" in prompt
    assert "EVIDENCE — TOOL RESULTS:" in prompt
    assert "tool='rba_cash_rate'" in prompt
    assert "retrieved='2026-04-26'" in prompt
    assert "(none — answer the original question directly)" in prompt


def test_build_synth_prompt_caps_chunk_count(monkeypatch):
    state = initial_state("q")
    state["retrieved_chunks"] = [
        {
            "chunk_id": f"c{i}",
            "score": 0.5,
            "payload": {"publisher": "X", "title": f"t{i}", "text": "txt"},
        }
        for i in range(20)
    ]
    monkeypatch.setattr(nodes, "SYNTH_MAX_CHUNKS_IN_PROMPT", 4)
    prompt = nodes._build_synth_prompt(state)
    assert "and 16 more chunks" in prompt


# ---------- citation post-processor ---------------------------------


def test_enforce_citations_passes_through_valid():
    state = _state_with_evidence()
    text = (
        "The RBA left the cash rate at 4.10% [tool:rba_cash_rate, retrieved:2026-04-26], "
        "consistent with its April statement [source:RBA, page:3]."
    )
    out, orphans = nodes._enforce_citations(text, state)
    assert orphans == []
    assert "(unverified)" not in out
    assert "Sources (auto):" in out  # auto-footer added


def test_enforce_citations_flags_unknown_publisher():
    state = _state_with_evidence()
    text = "Inflation is sticky [source:Federal Reserve, page:1]."
    out, orphans = nodes._enforce_citations(text, state)
    assert orphans == ["[source:Federal Reserve, page:1]"]
    assert "[source:Federal Reserve, page:1] (unverified)" in out


def test_enforce_citations_flags_unknown_tool():
    state = _state_with_evidence()
    text = "[tool:fred_data, retrieved:2026-04-26]"
    out, orphans = nodes._enforce_citations(text, state)
    assert orphans == ["[tool:fred_data, retrieved:2026-04-26]"]
    assert "(unverified)" in out


def test_enforce_citations_flags_wrong_page():
    state = _state_with_evidence()
    text = "Cash rate [source:RBA, page:99]."
    out, orphans = nodes._enforce_citations(text, state)
    assert orphans == ["[source:RBA, page:99]"]
    assert "(page-unverified)" in out


def test_enforce_citations_skips_failed_tool_calls():
    """A failed tool must not become a valid citation target."""
    state = _state_with_evidence()
    state["tool_results"].append(
        {"tool": "abs_lending_indicators", "args": {}, "error": "TimeoutError: ..."}
    )
    text = "[tool:abs_lending_indicators, retrieved:2026-04-26]"
    out, orphans = nodes._enforce_citations(text, state)
    assert orphans == ["[tool:abs_lending_indicators, retrieved:2026-04-26]"]
    assert "(unverified)" in out


def test_enforce_citations_preserves_existing_sources_block():
    state = _state_with_evidence()
    text = (
        "Cash rate is 4.10% [tool:rba_cash_rate, retrieved:2026-04-26].\n\n"
        "Sources:\n- RBA F1.1 (https://...)"
    )
    out, _ = nodes._enforce_citations(text, state)
    assert "Sources (auto):" not in out  # don't double-up


def test_enforce_citations_handles_unpaged_publisher():
    state = initial_state("q")
    state["retrieved_chunks"] = [
        {
            "chunk_id": "x",
            "score": 0.9,
            "payload": {
                "publisher": "ATO",
                "title": "Negative gearing",
                "page": None,
                "text": "...",
            },
        }
    ]
    state["tool_results"] = []
    text = "Negative gearing rules apply [source:ATO]."
    out, orphans = nodes._enforce_citations(text, state)
    assert orphans == []
    assert "(unverified)" not in out


# ---------- mocked synthesize ---------------------------------------


def _patch_anthropic_text(monkeypatch, text: str):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    import anthropic

    class _M:
        def create(self, **_kw):
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text=text)]
            )

    monkeypatch.setattr(
        anthropic, "Anthropic", lambda **_kw: SimpleNamespace(messages=_M())
    )


def test_synthesize_appends_assistant_message(monkeypatch):
    _patch_anthropic_text(
        monkeypatch,
        "The cash rate is 4.10% [tool:rba_cash_rate, retrieved:2026-04-26].",
    )
    state = _state_with_evidence()
    out = nodes.synthesize(state)
    assert "answer_draft" in out
    assert "4.10%" in out["answer_draft"]
    assert any(m["role"] == "assistant" for m in out["messages"])
    assert out["messages"][-1]["content"] == out["answer_draft"]


def test_synthesize_post_processes_orphans(monkeypatch):
    _patch_anthropic_text(
        monkeypatch,
        "Pulled from Bloomberg [source:Bloomberg, page:1] today.",
    )
    state = _state_with_evidence()
    out = nodes.synthesize(state)
    assert "(unverified)" in out["answer_draft"]


def test_synthesize_falls_back_on_api_error(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    import anthropic

    class _M:
        def create(self, **_kw):
            raise RuntimeError("API down")

    monkeypatch.setattr(
        anthropic, "Anthropic", lambda **_kw: SimpleNamespace(messages=_M())
    )
    state = _state_with_evidence()
    out = nodes.synthesize(state)
    # Graph must still terminate with a usable answer_draft.
    assert "synthesis-stage error" in out["answer_draft"]
    assert any(m["role"] == "assistant" for m in out["messages"])


def test_synthesize_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        nodes.synthesize(_state_with_evidence())


# ---------- live ----------------------------------------------------


@pytest.mark.network
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
def test_synthesize_live_produces_grounded_answer():
    state = _state_with_evidence()
    out = nodes.synthesize(state)
    draft = out["answer_draft"]
    assert "4.10" in draft
    # Sonnet should pick up at least one valid citation marker.
    has_citation = (
        nodes._DOC_CITE_RE.search(draft) is not None
        or nodes._TOOL_CITE_RE.search(draft) is not None
    )
    assert has_citation, f"no citation marker in draft:\n{draft}"
