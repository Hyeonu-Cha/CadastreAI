"""Current-date injection into the agent's node prompts (Task 5.15).

The agent had no notion of "today", so it couldn't reason about recency
or whether cited evidence predates a change (e.g. the 2026 tax reform —
docs/regime_change_gap_analysis.md, F-5). Each node now prepends a dated
context line to its USER message (kept out of the cached system prompt).
These tests pin the helper and that classify_query actually forwards the
dated user; the LLM call is stubbed, so no API key is needed.
"""
from __future__ import annotations

from datetime import datetime

from src.agent import llm as llm_mod
from src.agent import nodes
from src.agent.graph import initial_state


def test_today_context_carries_iso_date_and_label():
    ctx = nodes._today_context()
    today = datetime.now(nodes._AGENT_TZ).strftime("%Y-%m-%d")
    assert "[Context] Today's date is" in ctx
    assert today in ctx
    assert nodes._AGENT_TZ_LABEL in ctx


def test_with_today_prepends_and_preserves_body():
    out = nodes._with_today("ORIGINAL BODY")
    assert out.endswith("ORIGINAL BODY")
    assert out.startswith("[Context] Today's date is")
    # Body is separated from the context by a blank line.
    assert "\n\nORIGINAL BODY" in out


def test_classify_query_forwards_dated_user(monkeypatch):
    captured: dict[str, str] = {}

    def _stub(*, system, user, tool_def, tool_name, max_tokens, model):
        captured["user"] = user
        return (
            {
                "query_type": "factual",
                "persona": "general",
                "needs_docs": False,
                "needs_data": True,
                "needs_decomposition": False,
            },
            {},
        )

    monkeypatch.setattr(llm_mod, "call_with_tool", _stub)
    monkeypatch.setattr(nodes, "log_cost", lambda *a, **k: None)

    out = nodes.classify_query(initial_state("What is the current cash rate?"))

    assert captured["user"].startswith("[Context] Today's date is")
    assert "What is the current cash rate?" in captured["user"]
    assert out["query_type"] == "factual"
