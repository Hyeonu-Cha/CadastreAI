"""Unit tests for chart-embed rendering path in streamlit_app (Task 4.07, PR #131).

The render path inside `_render_citation_card` branches on the tool
envelope's shape:

  - chart envelope (`mime_type == "image/png"` + `png_base64`) → `st.image`
  - any other tool envelope with `result.data` → `st.json`
  - no matching tool result for the citation → `st.caption("...overreached...")`

These assertions guard the regression PR #131 introduced — a careless
edit that drops the `mime_type` discriminator would silently re-route
chart PNGs into `st.json`, rendering the base64 blob as a wall of text
in the source card.

Streamlit isn't imported normally so the test stays free of the
streamlit runtime dep; we shim `sys.modules["streamlit"]` with a
`MagicMock` before importing the target module, and replace
`streamlit_app.st` with a fresh mock per test via monkeypatch.
"""
from __future__ import annotations

import base64
import sys
from unittest.mock import MagicMock

import pytest

# Inject the mock streamlit module BEFORE importing streamlit_app.
# `setdefault` preserves an already-loaded streamlit if a future test
# decides to use the real package, keeping the suite order-independent.
sys.modules.setdefault("streamlit", MagicMock())

from src.app import streamlit_app  # noqa: E402
from src.app.citations import Citation  # noqa: E402
from src.app.state import TurnRecord  # noqa: E402


# A 1×1 transparent PNG so the base64 → bytes decode produces something
# real to assert against. Hex-literal to avoid pulling Pillow into the
# test dep graph.
_TINY_PNG = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000d49444154789c63600100000005000178b41a0000000049454e44ae426082"
    )
).decode("ascii")


def _tool_citation(name: str = "render_chart", date: str = "2026-04-26") -> Citation:
    return Citation(kind="tool", key=(name, date), raw=f"[tool: {name}, retrieved: {date}]")


def _chart_envelope(date: str = "2026-04-26") -> dict:
    return {
        "tool": "render_chart",
        "result": {
            "data": {
                "png_base64": _TINY_PNG,
                "mime_type": "image/png",
                "title": "cash rate over time",
            },
            "source": "Local matplotlib render",
            "retrieved_at": f"{date}T00:00:00+00:00",
        },
    }


def _generic_tool_envelope(date: str = "2026-04-26") -> dict:
    return {
        "tool": "rba_cash_rate",
        "result": {
            "data": {"2026-01": 4.35, "2026-04": 4.10},
            "source": "RBA F1.1",
            "retrieved_at": f"{date}T00:00:00+00:00",
        },
    }


@pytest.fixture
def st_mock(monkeypatch):
    """Replace `st` inside streamlit_app with a fresh MagicMock per test."""
    m = MagicMock()
    monkeypatch.setattr(streamlit_app, "st", m)
    return m


def test_chart_envelope_renders_via_st_image(st_mock):
    cite = _tool_citation()
    turn = TurnRecord(query="show me cash rate over time", persona="Investor")
    turn.tool_results = [_chart_envelope()]

    streamlit_app._render_citation_card(1, cite, turn)

    assert st_mock.image.called, "chart envelopes must render through st.image"
    png_bytes = st_mock.image.call_args.args[0]
    assert isinstance(png_bytes, bytes)
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n"), \
        "base64 was not decoded into a real PNG byte stream"
    # The regression PR #131 fixed: chart envelopes used to fall through
    # to st.json and render the b64 string as a wall of text.
    assert not st_mock.json.called, "chart envelopes must NOT fall through to st.json"


def test_generic_tool_envelope_renders_via_st_json(st_mock):
    cite = _tool_citation(name="rba_cash_rate")
    turn = TurnRecord(query="cash rate?", persona="Investor")
    turn.tool_results = [_generic_tool_envelope()]

    streamlit_app._render_citation_card(1, cite, turn)

    assert st_mock.json.called, "non-chart envelopes must still render via st.json"
    assert not st_mock.image.called, "non-chart envelopes must NOT render via st.image"


def test_no_matching_tool_result_shows_overreach_caption(st_mock):
    cite = _tool_citation()
    turn = TurnRecord(query="show me a chart", persona="Investor")
    turn.tool_results = []  # no evidence — the citation can't be matched

    streamlit_app._render_citation_card(1, cite, turn)

    captions = [c.args[0] for c in st_mock.caption.call_args_list]
    assert any("overreached" in s for s in captions), \
        f"expected an 'overreached' caption, got: {captions}"
    assert not st_mock.image.called
    assert not st_mock.json.called


def test_source_caption_emitted_after_render(st_mock):
    cite = _tool_citation(name="rba_cash_rate")
    turn = TurnRecord(query="cash rate?", persona="Investor")
    turn.tool_results = [_generic_tool_envelope()]

    streamlit_app._render_citation_card(1, cite, turn)

    captions = [c.args[0] for c in st_mock.caption.call_args_list]
    assert any(s.startswith("source: ") for s in captions), \
        f"expected a 'source: ...' caption, got: {captions}"
