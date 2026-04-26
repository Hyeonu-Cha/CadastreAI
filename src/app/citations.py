"""Citation parsing + evidence matching for the Streamlit UI (Task 4.03).

The synthesizer emits two inline citation forms:

    [source: PUBLISHER, page: N]      → matches a retrieved chunk
    [tool: TOOL_NAME, retrieved: DT]  → matches a tool result envelope

This module parses those markers, dedupes them in order of appearance,
matches each one to the underlying evidence, and renumbers the answer
text so the UI can display compact "[1]", "[2]" chips beside cards
that show the actual source excerpt or tool envelope.

Kept Streamlit-free so it can be unit-tested without `streamlit`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

DOC_CITE_RE = re.compile(
    r"\[source:\s*([^,\]]+?)\s*(?:,\s*page:\s*([^\]]+?)\s*)?\]"
)
TOOL_CITE_RE = re.compile(
    r"\[tool:\s*([^,\]]+?)\s*,\s*retrieved:\s*([^\]]+?)\s*\]"
)

CitationKind = Literal["doc", "tool"]


@dataclass(frozen=True)
class Citation:
    """One unique citation found in an answer.

    `key` is what we dedupe on — same publisher+page (or same tool+date)
    collapses to a single chip number. `raw` is the original marker text
    so the renumbering pass can find and replace every occurrence.
    """

    kind: CitationKind
    key: tuple[str, str]
    raw: str

    @property
    def publisher(self) -> str:
        return self.key[0] if self.kind == "doc" else ""

    @property
    def page(self) -> str:
        return self.key[1] if self.kind == "doc" else ""

    @property
    def tool(self) -> str:
        return self.key[0] if self.kind == "tool" else ""

    @property
    def retrieved_at(self) -> str:
        return self.key[1] if self.kind == "tool" else ""


def parse_citations(answer: str) -> list[Citation]:
    """Extract unique citations from `answer` in order of first appearance.

    Whitespace around publisher / page / tool name is stripped so two
    formattings of the same citation collapse to one chip.
    """
    if not answer:
        return []

    seen: set[tuple[str, tuple[str, str]]] = set()
    out: list[Citation] = []

    # Walk the string left-to-right with both regexes interleaved so the
    # numbering reflects the order the user actually reads.
    matches: list[tuple[int, CitationKind, re.Match]] = []
    for m in DOC_CITE_RE.finditer(answer):
        matches.append((m.start(), "doc", m))
    for m in TOOL_CITE_RE.finditer(answer):
        matches.append((m.start(), "tool", m))
    matches.sort(key=lambda x: x[0])

    for _start, kind, m in matches:
        if kind == "doc":
            key = (m.group(1).strip(), (m.group(2) or "").strip())
        else:
            key = (m.group(1).strip(), m.group(2).strip())
        sig = (kind, key)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(Citation(kind=kind, key=key, raw=m.group(0)))
    return out


def renumber_answer(answer: str, citations: list[Citation]) -> str:
    """Replace inline markers with `[N]` superscripts matching `citations`.

    Each *unique* citation gets one number; repeated occurrences of the
    same citation reuse it. Markers not present in `citations` (e.g.
    "(unverified)" tagged orphans the post-processor injected) are left
    alone — they're not real evidence so we don't want to chip them.
    """
    if not answer or not citations:
        return answer

    numbering: dict[tuple[CitationKind, tuple[str, str]], int] = {
        (c.kind, c.key): i + 1 for i, c in enumerate(citations)
    }

    def _doc_repl(m: re.Match) -> str:
        key = (m.group(1).strip(), (m.group(2) or "").strip())
        n = numbering.get(("doc", key))
        return f"[{n}]" if n else m.group(0)

    def _tool_repl(m: re.Match) -> str:
        key = (m.group(1).strip(), m.group(2).strip())
        n = numbering.get(("tool", key))
        return f"[{n}]" if n else m.group(0)

    out = DOC_CITE_RE.sub(_doc_repl, answer)
    out = TOOL_CITE_RE.sub(_tool_repl, out)
    return out


def match_doc_evidence(
    citation: Citation, chunks: list[dict]
) -> dict | None:
    """Find the retrieved chunk that backs a doc citation, or None.

    Matching is publisher + page (string-equal). When the citation has
    no page, we accept any chunk for that publisher (the synthesizer
    sometimes drops the page when the source is page-less).
    """
    if citation.kind != "doc":
        return None
    pub = citation.publisher
    page = citation.page
    fallback: dict | None = None
    for chunk in chunks or []:
        payload = chunk.get("payload") or {}
        if (payload.get("publisher") or "").strip() != pub:
            continue
        chunk_page = payload.get("page")
        chunk_page_str = "" if chunk_page is None else str(chunk_page).strip()
        if page and chunk_page_str == page:
            return chunk
        if not page:
            return chunk
        if fallback is None:
            fallback = chunk
    return fallback


def match_tool_evidence(
    citation: Citation, tool_results: list[dict]
) -> dict | None:
    """Find the tool envelope that backs a tool citation, or None."""
    if citation.kind != "tool":
        return None
    name = citation.tool
    date = citation.retrieved_at[:10]
    fallback: dict | None = None
    for env in tool_results or []:
        if (env.get("tool") or "").strip() != name:
            continue
        if "result" not in env:
            continue
        retrieved_at = (env["result"].get("retrieved_at") or "")[:10]
        if retrieved_at == date:
            return env
        if fallback is None:
            fallback = env
    return fallback


def citation_excerpt(chunk: dict, max_chars: int = 320) -> str:
    """Truncate a chunk's text body for the source-excerpt card."""
    payload = chunk.get("payload") or {}
    text = (payload.get("text") or "").strip()
    if not text:
        return "(no excerpt available)"
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


__all__ = [
    "Citation",
    "CitationKind",
    "DOC_CITE_RE",
    "TOOL_CITE_RE",
    "citation_excerpt",
    "match_doc_evidence",
    "match_tool_evidence",
    "parse_citations",
    "renumber_answer",
]
