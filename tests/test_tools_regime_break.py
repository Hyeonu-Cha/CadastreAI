"""Structural-break annotation on live-data envelopes (Task 5.24).

The three investor-driven tools (ABS prices, ABS lending, SQM vacancy)
wrap their envelope in `annotate_structural_break`, which flags any
figure dated on/after the 12 May 2026 reform announcement so the source
card warns against extrapolating investor series across the break (F-9).
The note rides the `citation` string — an existing field — so it doesn't
trip the strict `extra_forbidden` tool schemas.

The tools themselves are network-bound; here we pin the pure annotator.
"""
from __future__ import annotations

from src.tools.regime_break import REFORM_ANNOUNCEMENT_DATE, annotate_structural_break


def _env(as_of: str | None) -> dict:
    env = {"data": {"x": 1}, "citation": "ABS Lending Indicators — 2026 (url)"}
    if as_of is not None:
        env["as_of"] = as_of
    return env


def test_annotates_after_announcement():
    out = annotate_structural_break(_env("2026-06-30"))
    assert "structural break" in out["citation"]
    # data is left strictly untouched (strict tool schemas forbid extras).
    assert out["data"] == {"x": 1}


def test_annotates_exactly_on_announcement_date():
    assert REFORM_ANNOUNCEMENT_DATE.isoformat() == "2026-05-12"
    out = annotate_structural_break(_env("2026-05-12"))
    assert "structural break" in out["citation"]


def test_noop_before_announcement():
    out = annotate_structural_break(_env("2026-03-31"))
    assert "structural break" not in out["citation"]


def test_noop_when_as_of_missing_or_unparseable():
    assert "structural break" not in annotate_structural_break(_env(None))["citation"]
    assert "structural break" not in annotate_structural_break(_env("n/a"))["citation"]


def test_handles_datetime_style_as_of():
    """`as_of` carrying a time component still parses off the date prefix."""
    out = annotate_structural_break(_env("2026-07-01T00:00:00+00:00"))
    assert "structural break" in out["citation"]


def test_idempotent_no_double_append():
    once = annotate_structural_break(_env("2026-07-01"))
    citation_after_one = once["citation"]
    twice = annotate_structural_break(once)
    assert twice["citation"] == citation_after_one
    assert twice["citation"].count("structural break") == 1
