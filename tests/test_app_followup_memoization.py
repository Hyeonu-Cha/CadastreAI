"""Memoization guard for AI follow-up chips (PR #146 follow-on).

`_render_followup_chips` makes a live Haiku call (followup_questions →
ai_followups → call_with_tool). Streamlit re-runs the whole script on
every interaction, so without memoization that call re-fires on every
unrelated rerun (toggling the trace, submitting the next query, clicking
a chip) for the same turn. The fix caches the result on
`TurnRecord.followups`; these tests pin that:

  - a repeated render of the same turn does NOT re-invoke the LLM, and
  - an empty result is still cached (`[]`, not `None`) so a turn with no
    chips doesn't retry the suggester on every rerun. The guard uses
    `is None`, not falsiness, precisely so this holds.

Streamlit is shimmed with a MagicMock before importing streamlit_app, so
the test stays free of the streamlit runtime — same approach as
test_app_streamlit_chart_render.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

# Inject the mock streamlit module BEFORE importing streamlit_app.
sys.modules.setdefault("streamlit", MagicMock())

import src.agent.llm as llm_mod  # noqa: E402
from src.app import streamlit_app  # noqa: E402
from src.app.state import TurnRecord  # noqa: E402


@pytest.fixture
def st_mock(monkeypatch):
    """Replace `st` in streamlit_app with a mock whose `columns()` returns
    an iterable of column mocks that report "not clicked"."""
    m = MagicMock()

    def _columns(n, *args, **kwargs):
        cols = []
        for _ in range(n):
            c = MagicMock()
            c.button.return_value = False  # never "clicked" → no rerun
            cols.append(c)
        return cols

    m.columns.side_effect = _columns
    monkeypatch.setattr(streamlit_app, "st", m)
    return m


def test_render_followup_chips_memoizes_llm_call(monkeypatch, st_mock):
    """Three renders of one turn → exactly one LLM follow-up call."""
    monkeypatch.delenv("CADASTRE_AI_FOLLOWUPS", raising=False)  # default on
    calls = {"n": 0}

    def _stub(**kwargs):
        calls["n"] += 1
        return {"questions": ["A?", "B?", "C?"]}, {}

    monkeypatch.setattr(llm_mod, "call_with_tool", _stub)

    turn = TurnRecord(
        query="Is Parramatta a good buy?",
        persona="Investor",
        answer="Parramatta's median is $1.1M with a 3.2% yield.",
    )
    assert turn.followups is None

    streamlit_app._render_followup_chips(turn)
    streamlit_app._render_followup_chips(turn)
    streamlit_app._render_followup_chips(turn)

    assert calls["n"] == 1, "LLM follow-up call must fire once, then be memoized"
    assert turn.followups == ["A?", "B?", "C?"]


def test_render_followup_chips_caches_empty_result(monkeypatch, st_mock):
    """A turn that yields no chips caches `[]` so it isn't recomputed.

    Guards the `is None` check: a naive `if not turn.followups:` would
    recompute an empty result on every rerun forever.
    """
    calls = {"n": 0}

    def _empty(*args, **kwargs):
        calls["n"] += 1
        return []

    monkeypatch.setattr(streamlit_app, "followup_questions", _empty)

    turn = TurnRecord(query="hi", persona="Investor", answer="hello")
    streamlit_app._render_followup_chips(turn)
    streamlit_app._render_followup_chips(turn)

    assert calls["n"] == 1, "empty result must be cached, not recomputed"
    assert turn.followups == []


def test_render_followup_chips_skips_errored_turn(monkeypatch, st_mock):
    """An errored turn never computes follow-ups (and never calls out)."""

    def _boom(*args, **kwargs):
        raise AssertionError("errored turns must not compute follow-ups")

    monkeypatch.setattr(streamlit_app, "followup_questions", _boom)

    turn = TurnRecord(query="hi", persona="Investor")
    turn.error = "RuntimeError: boom"
    streamlit_app._render_followup_chips(turn)

    assert turn.followups is None
