"""Unit tests for citation parsing and evidence matching (Task 4.03).

Coverage:
  - parse_citations finds doc + tool markers, in reading order.
  - duplicate citations dedupe to one Citation entry (chip).
  - renumber_answer rewrites every occurrence (incl. duplicates) to [N].
  - markers tagged "(unverified)" or absent from the dedupe list are
    left alone — those aren't real evidence.
  - match_doc_evidence handles publisher+page, falls back to publisher
    only when page absent or unmatched.
  - match_tool_evidence handles tool+retrieved date, falls back to
    tool name only when date doesn't line up.
  - citation_excerpt truncates and ellipsises.
"""
from __future__ import annotations

from src.app.citations import (
    Citation,
    citation_excerpt,
    match_doc_evidence,
    match_tool_evidence,
    parse_citations,
    renumber_answer,
)

# ---------- parse_citations -------------------------------------


def test_parse_citations_finds_doc_and_tool_markers_in_order():
    answer = (
        "Cash rate is 4.10% [tool: rba_cash_rate, retrieved: 2026-04-26]. "
        "AHURI [source: AHURI, page: 12] notes long-run pressures."
    )
    cites = parse_citations(answer)
    assert len(cites) == 2
    assert cites[0].kind == "tool"
    assert cites[0].tool == "rba_cash_rate"
    assert cites[0].retrieved_at == "2026-04-26"
    assert cites[1].kind == "doc"
    assert cites[1].publisher == "AHURI"
    assert cites[1].page == "12"


def test_parse_citations_dedupes_repeats():
    answer = (
        "First [source: RBA, page: 4]. Second [source: RBA, page: 4]. "
        "Third [source: RBA, page: 4]."
    )
    cites = parse_citations(answer)
    assert len(cites) == 1
    assert cites[0].publisher == "RBA"
    assert cites[0].page == "4"


def test_parse_citations_treats_different_pages_as_distinct():
    answer = "[source: ABS, page: 1] and [source: ABS, page: 2]"
    cites = parse_citations(answer)
    assert len(cites) == 2
    assert {c.page for c in cites} == {"1", "2"}


def test_parse_citations_handles_pageless_doc_citation():
    answer = "Per [source: NHFIC]."
    cites = parse_citations(answer)
    assert len(cites) == 1
    assert cites[0].publisher == "NHFIC"
    assert cites[0].page == ""


def test_parse_citations_strips_whitespace_around_fields():
    """Two markers identical except for whitespace should dedupe."""
    answer = "[source:  AHURI , page:  3  ] [source: AHURI, page: 3]"
    cites = parse_citations(answer)
    assert len(cites) == 1


def test_parse_citations_empty_input_returns_empty_list():
    assert parse_citations("") == []
    assert parse_citations(None) == []  # type: ignore[arg-type]


def test_parse_citations_no_citations_in_answer():
    assert parse_citations("Plain text with no citations.") == []


def test_parse_citations_orders_by_position_in_text():
    """Doc citations and tool citations interleaved keep reading order."""
    answer = (
        "[source: A, page: 1] then [tool: t1, retrieved: 2026-01-01] "
        "then [source: B, page: 2] then [tool: t2, retrieved: 2026-01-02]"
    )
    cites = parse_citations(answer)
    assert [c.kind for c in cites] == ["doc", "tool", "doc", "tool"]
    assert cites[0].publisher == "A"
    assert cites[2].publisher == "B"


# ---------- renumber_answer --------------------------------------


def test_renumber_answer_replaces_markers_with_numbers():
    answer = (
        "Rate is 4.1% [tool: rba_cash_rate, retrieved: 2026-04-26]. "
        "AHURI [source: AHURI, page: 12]."
    )
    cites = parse_citations(answer)
    out = renumber_answer(answer, cites)
    assert "[1]" in out
    assert "[2]" in out
    assert "[tool:" not in out
    assert "[source:" not in out


def test_renumber_answer_reuses_number_for_repeated_citation():
    answer = "[source: RBA, page: 4] then [source: RBA, page: 4] again."
    cites = parse_citations(answer)
    out = renumber_answer(answer, cites)
    assert out.count("[1]") == 2
    assert "[2]" not in out


def test_renumber_answer_leaves_unknown_markers_alone():
    """An (unverified)-tagged orphan that's not in the cite list stays raw."""
    answer = "Cited [source: AHURI, page: 1] and orphan [source: Wikipedia, page: 7]."
    # Only AHURI passes parse_citations because we kept both — but to test
    # the "leave unknown alone" branch we manually construct a list with
    # only AHURI in it.
    only_ahuri = [Citation(kind="doc", key=("AHURI", "1"), raw="[source: AHURI, page: 1]")]
    out = renumber_answer(answer, only_ahuri)
    assert "[1]" in out
    assert "[source: Wikipedia, page: 7]" in out  # untouched


def test_renumber_answer_empty_inputs():
    assert renumber_answer("", []) == ""
    assert renumber_answer("text", []) == "text"


# ---------- match_doc_evidence -----------------------------------


def _doc(publisher: str, page=None, text: str = "body") -> dict:
    return {
        "chunk_id": f"{publisher}-{page}",
        "score": 0.8,
        "payload": {"publisher": publisher, "page": page, "text": text},
    }


def test_match_doc_evidence_matches_publisher_and_page():
    chunks = [_doc("AHURI", page=12, text="long-run housing pressure")]
    cite = Citation(kind="doc", key=("AHURI", "12"), raw="[source: AHURI, page: 12]")
    out = match_doc_evidence(cite, chunks)
    assert out is not None
    assert out["payload"]["publisher"] == "AHURI"


def test_match_doc_evidence_falls_back_to_publisher_when_page_unmatched():
    chunks = [_doc("AHURI", page=99, text="other page")]
    cite = Citation(kind="doc", key=("AHURI", "12"), raw="[source: AHURI, page: 12]")
    out = match_doc_evidence(cite, chunks)
    assert out is not None
    assert out["payload"]["page"] == 99  # the only AHURI chunk we had


def test_match_doc_evidence_returns_none_when_publisher_missing():
    chunks = [_doc("RBA", page=1)]
    cite = Citation(kind="doc", key=("AHURI", "12"), raw="[source: AHURI, page: 12]")
    assert match_doc_evidence(cite, chunks) is None


def test_match_doc_evidence_pageless_citation_takes_first_publisher_match():
    chunks = [_doc("NHFIC", page=2), _doc("NHFIC", page=5)]
    cite = Citation(kind="doc", key=("NHFIC", ""), raw="[source: NHFIC]")
    out = match_doc_evidence(cite, chunks)
    assert out is not None
    assert out["payload"]["page"] == 2  # first match wins


def test_match_doc_evidence_skips_tool_citation():
    cite = Citation(kind="tool", key=("rba_cash_rate", "2026-04-26"), raw="…")
    assert match_doc_evidence(cite, [_doc("RBA", page=1)]) is None


# ---------- match_tool_evidence ----------------------------------


def _tool(name: str, retrieved_at: str, data=None) -> dict:
    return {
        "tool": name,
        "args": {},
        "result": {"data": data or {"value": 1}, "retrieved_at": retrieved_at},
    }


def test_match_tool_evidence_matches_name_and_date():
    envs = [_tool("rba_cash_rate", "2026-04-26T10:00:00Z")]
    cite = Citation(
        kind="tool",
        key=("rba_cash_rate", "2026-04-26"),
        raw="[tool: rba_cash_rate, retrieved: 2026-04-26]",
    )
    out = match_tool_evidence(cite, envs)
    assert out is not None
    assert out["tool"] == "rba_cash_rate"


def test_match_tool_evidence_falls_back_to_tool_when_date_off():
    envs = [_tool("rba_cash_rate", "2026-03-01T00:00:00Z")]
    cite = Citation(
        kind="tool",
        key=("rba_cash_rate", "2026-04-26"),
        raw="[tool: rba_cash_rate, retrieved: 2026-04-26]",
    )
    out = match_tool_evidence(cite, envs)
    assert out is not None  # fallback to tool name match


def test_match_tool_evidence_returns_none_when_tool_missing():
    envs = [_tool("rba_cash_rate", "2026-04-26")]
    cite = Citation(
        kind="tool", key=("compute_stamp_duty_nsw", "2026-04-26"), raw="…"
    )
    assert match_tool_evidence(cite, envs) is None


def test_match_tool_evidence_skips_envelopes_with_errors():
    envs = [{"tool": "rba_cash_rate", "args": {}, "error": "boom"}]
    cite = Citation(
        kind="tool", key=("rba_cash_rate", "2026-04-26"), raw="…"
    )
    assert match_tool_evidence(cite, envs) is None


def test_match_tool_evidence_skips_doc_citation():
    cite = Citation(kind="doc", key=("RBA", "1"), raw="…")
    assert match_tool_evidence(cite, [_tool("rba_cash_rate", "2026-04-26")]) is None


# ---------- citation_excerpt -------------------------------------


def test_citation_excerpt_returns_full_text_when_short():
    chunk = _doc("RBA", page=1, text="Short body.")
    assert citation_excerpt(chunk) == "Short body."


def test_citation_excerpt_truncates_with_ellipsis():
    long_text = "a" * 500
    chunk = _doc("RBA", page=1, text=long_text)
    out = citation_excerpt(chunk, max_chars=100)
    assert len(out) == 100
    assert out.endswith("…")


def test_citation_excerpt_handles_missing_text():
    chunk = {"payload": {"publisher": "RBA"}}
    assert citation_excerpt(chunk) == "(no excerpt available)"


def test_citation_excerpt_handles_empty_payload():
    assert citation_excerpt({}) == "(no excerpt available)"
