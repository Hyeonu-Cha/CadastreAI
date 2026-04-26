"""Unit tests for ambiguous-place-name detection (Task 4.04).

Coverage:
  - bare ambiguous name → flagged
  - same name with state code or country qualifier nearby → not flagged
  - non-ambiguous queries → empty list
  - case insensitivity, word-boundary matching (no false hits inside
    longer words)
  - multiple ambiguities reported in reading order, deduped
  - apply_disambiguation appends qualifiers as a parenthetical hint
"""
from __future__ import annotations

from src.app.disambiguation import (
    Ambiguity,
    apply_disambiguation,
    detect_ambiguity,
)

# ---------- detect_ambiguity ------------------------------------


def test_bare_richmond_is_ambiguous():
    out = detect_ambiguity("What's median price in Richmond?")
    assert len(out) == 1
    assert out[0].place.lower() == "richmond"
    assert {opt.label for opt in out[0].options} >= {
        "Richmond, VIC (Melbourne inner suburb)",
        "Richmond, NSW (outer Sydney)",
    }


def test_richmond_with_vic_qualifier_is_not_ambiguous():
    assert detect_ambiguity("Median price in Richmond VIC") == []


def test_richmond_with_state_code_far_away_still_ambiguous():
    """State code far past the qualifier window doesn't count as a hint."""
    long_query = (
        "Tell me a long story about Richmond and at the very end mention "
        "that I live in NSW for unrelated reasons related to my taxes"
    )
    out = detect_ambiguity(long_query)
    assert out  # NSW is too far away to disambiguate


def test_richmond_australia_qualifier_is_not_ambiguous():
    assert detect_ambiguity("Richmond, Australia housing trends") == []


def test_sydney_alone_is_ambiguous():
    out = detect_ambiguity("How is Sydney's rental market?")
    assert any(a.place.lower() == "sydney" for a in out)


def test_sydney_with_nsw_is_not_ambiguous():
    assert detect_ambiguity("Sydney NSW vacancy rate") == []


def test_perth_alone_is_ambiguous():
    out = detect_ambiguity("Perth median house price.")
    assert len(out) == 1
    assert out[0].place.lower() == "perth"


def test_perth_with_wa_is_not_ambiguous():
    assert detect_ambiguity("Perth WA median house price") == []


def test_brighton_alone_is_ambiguous():
    out = detect_ambiguity("Brighton apartment trends")
    assert len(out) == 1
    assert out[0].place.lower() == "brighton"


def test_st_kilda_alone_is_ambiguous():
    out = detect_ambiguity("Tell me about St Kilda")
    assert any(a.place.lower() == "st kilda" for a in out)


def test_newcastle_alone_is_ambiguous():
    out = detect_ambiguity("Newcastle median price.")
    assert len(out) == 1


def test_newcastle_uk_qualifier_is_not_ambiguous():
    """User explicitly named the UK city — leave them be."""
    assert detect_ambiguity("Newcastle upon Tyne housing") == []


def test_unrelated_query_returns_empty():
    assert detect_ambiguity("What's the RBA cash rate?") == []
    assert detect_ambiguity("") == []


def test_case_insensitive_matching():
    assert detect_ambiguity("RICHMOND prices") != []
    assert detect_ambiguity("richmond prices") != []
    assert detect_ambiguity("Richmond prices") != []


def test_word_boundary_no_false_positives():
    """No matches inside longer words like 'Richmondshire' or 'Brightonbeach'."""
    assert detect_ambiguity("Richmondshire is in the UK") == []
    assert detect_ambiguity("Visit brightonbeach.com") == []


def test_multiple_ambiguities_in_one_query():
    out = detect_ambiguity("Compare Richmond and Brighton apartment yields")
    places = [a.place.lower() for a in out]
    assert places == ["richmond", "brighton"]


def test_repeated_place_collapses_to_single_ambiguity():
    out = detect_ambiguity("Richmond is great. Tell me more about Richmond.")
    assert len(out) == 1


def test_qualifier_does_not_match_inside_other_words():
    """'NSW' is a qualifier; 'answer' contains 'nsw' but should not count."""
    # If word-boundary matching were broken, 'answer' would disqualify
    # the 'Richmond' below. We expect the ambiguity to survive.
    out = detect_ambiguity("Give me an answer about Richmond market")
    assert len(out) == 1


# ---------- apply_disambiguation --------------------------------


def test_apply_disambiguation_appends_parenthetical():
    out = apply_disambiguation(
        "Richmond house prices", {"Richmond": "Richmond VIC"}
    )
    assert "Richmond house prices" in out
    assert "Richmond VIC" in out
    assert out.endswith(")")


def test_apply_disambiguation_handles_multiple_choices():
    out = apply_disambiguation(
        "Compare Richmond and Brighton",
        {"Richmond": "Richmond VIC", "Brighton": "Brighton VIC"},
    )
    assert "Richmond VIC" in out
    assert "Brighton VIC" in out


def test_apply_disambiguation_no_choices_returns_query_unchanged():
    assert apply_disambiguation("Richmond house prices", {}) == "Richmond house prices"


def test_apply_disambiguation_strips_trailing_whitespace():
    out = apply_disambiguation(
        "Richmond house prices    ", {"Richmond": "Richmond VIC"}
    )
    assert "  (" not in out  # no double space before the parenthetical


# ---------- shape sanity ---------------------------------------


def test_ambiguity_options_are_non_empty_tuples():
    """Every catalogued place needs at least 2 options — that's the
    whole point. A 1-option entry is a bug, not a clarification."""
    for query in ("Richmond?", "Brighton?", "Sydney?", "Perth?", "Newcastle?", "St Kilda?"):
        out = detect_ambiguity(query)
        assert out, f"expected ambiguity for {query!r}"
        for amb in out:
            assert isinstance(amb, Ambiguity)
            assert len(amb.options) >= 2
            for opt in amb.options:
                assert opt.label.strip()
                assert opt.qualifier.strip()
