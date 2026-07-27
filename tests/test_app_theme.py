"""Unit tests for the CadastreAI theme helpers (design-system pass).

Coverage:
  - style_citation_markers wraps every [N] marker in a monospace chip.
  - chips are coloured by source kind: navy for retrieved documents,
    amber for live tool calls (the product-wide doc/tool split).
  - markers without a backing citation fall back to the neutral (doc) chip
    rather than raising or being dropped.
  - THEME_CSS pulls in the three brand webfonts and the chip classes.
"""
from __future__ import annotations

from src.app.citations import Citation
from src.app.theme import THEME_CSS, notice, style_citation_markers


def _doc(pub: str = "RBA", page: str = "12") -> Citation:
    return Citation(kind="doc", key=(pub, page), raw=f"[source: {pub}, page: {page}]")


def _tool(name: str = "rba_cash_rate", date: str = "2026-04-26") -> Citation:
    return Citation(kind="tool", key=(name, date), raw=f"[tool: {name}]")


def test_chips_coloured_by_source_kind():
    out = style_citation_markers("Rate 4.35% [1], vacancy 1.4% [2].", [_tool(), _doc()])
    assert '<span class="cad-cite cad-cite-tool">[1]</span>' in out
    assert '<span class="cad-cite cad-cite-doc">[2]</span>' in out


def test_every_marker_is_wrapped():
    out = style_citation_markers("[1] then [2] then [1] again.", [_doc(), _tool()])
    assert out.count('class="cad-cite') == 3  # duplicates wrapped too


def test_marker_without_citation_falls_back_to_doc():
    # Model emitted [3] but only two citations deduped — render, don't crash.
    out = style_citation_markers("Claim [3].", [_doc(), _tool()])
    assert '<span class="cad-cite cad-cite-doc">[3]</span>' in out


def test_text_without_markers_is_unchanged():
    assert style_citation_markers("No citations here.", []) == "No citations here."


def test_theme_css_loads_brand_fonts_and_chip_classes():
    for token in ("Source+Serif+4", "Hanken+Grotesk", "JetBrains+Mono"):
        assert token in THEME_CSS
    assert ".cad-cite-doc" in THEME_CSS
    assert ".cad-cite-tool" in THEME_CSS


def test_theme_css_defines_notice_callout():
    assert ".cad-notice" in THEME_CSS


class _FakeSt:
    """Minimal st stand-in that records markdown() calls."""

    def __init__(self) -> None:
        self.markdown_calls: list[tuple[str, bool]] = []

    def markdown(self, body, unsafe_allow_html=False):  # noqa: ANN001, FBT002
        self.markdown_calls.append((body, unsafe_allow_html))


def test_notice_renders_branded_callout_as_raw_html():
    st = _FakeSt()
    notice(st, "Corpus predates <em>the Act</em>.", icon="⚠")

    assert len(st.markdown_calls) == 1
    body, unsafe = st.markdown_calls[0]
    assert unsafe is True, "notice must render as raw HTML"
    assert 'class="cad-notice"' in body
    assert 'class="cad-notice-icon"' in body
    assert "⚠" in body
    # Body HTML is passed through untouched (caller owns the markup).
    assert "Corpus predates <em>the Act</em>." in body
