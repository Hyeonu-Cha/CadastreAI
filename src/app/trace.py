"""Build a structured step-by-step trace of an agent turn (Task 4.05).

The Streamlit "Reasoning steps" expander used to be a flat dump of
classification / sub-questions / tool-results / retrieved-chunks. That
told the user *what* the agent had, but not *what it did*. This module
turns a `TurnRecord` into an ordered list of `TraceStep` cards that
mirror the actual graph traversal:

    1. classify_query  → persona + query_type + flags
    2. decompose       → sub-questions, or "atomic" if none
    3. retrieve_or_tool → docs + tool calls (per sub-question grouping)
    4. synthesize      → iteration count + answer length
    5. reflect         → complete / loop / capped

The renderer lives in `src/app/streamlit_app.py`; this module is
deliberately Streamlit-free so it can be unit-tested directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

StepKind = Literal["info", "ok", "warn"]


@dataclass(frozen=True)
class TraceStep:
    """One row in the Reasoning-steps panel."""

    name: str
    summary: str
    details: list[str] = field(default_factory=list)
    kind: StepKind = "info"


def _classify_step(turn) -> TraceStep:
    cls = turn.classification or {}
    persona = cls.get("persona") or "general"
    qtype = cls.get("query_type") or "?"
    summary = f"persona = {persona} · type = {qtype}"
    flags = []
    if cls.get("needs_decomposition"):
        flags.append("decompose")
    if cls.get("needs_docs"):
        flags.append("docs")
    if cls.get("needs_data"):
        flags.append("data")
    details = [f"flags: {', '.join(flags) or '(none)'}"]
    return TraceStep(name="Classify query", summary=summary, details=details, kind="ok")


def _decompose_step(turn) -> TraceStep:
    subs = list(turn.sub_questions or [])
    if not subs:
        return TraceStep(
            name="Decompose",
            summary="treated as a single atomic question",
            details=[],
            kind="info",
        )
    summary = f"{len(subs)} sub-question{'s' if len(subs) != 1 else ''}"
    details = [f"{i}. {q}" for i, q in enumerate(subs, start=1)]
    return TraceStep(name="Decompose", summary=summary, details=details, kind="ok")


def _retrieve_or_tool_step(turn) -> TraceStep:
    chunks = list(turn.retrieved_chunks or [])
    tools = list(turn.tool_results or [])
    chunk_summary = f"{len(chunks)} chunk{'s' if len(chunks) != 1 else ''}"
    tool_summary = f"{len(tools)} tool call{'s' if len(tools) != 1 else ''}"
    summary = f"{chunk_summary} · {tool_summary}"

    details: list[str] = []
    for i, c in enumerate(chunks[:5], start=1):
        payload = c.get("payload") or {}
        publisher = payload.get("publisher", "?")
        score = c.get("boosted_score") or c.get("score")
        score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "?"
        title = (payload.get("title") or "").strip()
        title_bit = f" — {title[:60]}" if title else ""
        details.append(f"chunk {i}: {publisher} (score {score_str}){title_bit}")
    if len(chunks) > 5:
        details.append(f"… and {len(chunks) - 5} more chunk(s)")

    n_errors = 0
    for t in tools:
        name = t.get("tool", "?")
        if "error" in t:
            n_errors += 1
            details.append(f"tool {name}: ERROR — {t['error']}")
            continue
        result = t.get("result") or {}
        source = result.get("source") or "?"
        details.append(f"tool {name}: ok ({source})")

    kind: StepKind = "warn" if n_errors else "ok"
    return TraceStep(
        name="Retrieve / tools", summary=summary, details=details, kind=kind
    )


def _synthesize_step(turn) -> TraceStep:
    answer = turn.answer or ""
    iters = turn.iteration_count or 0
    summary = (
        f"{iters} iteration{'s' if iters != 1 else ''} · "
        f"{len(answer)} chars in draft"
    )
    return TraceStep(name="Synthesize", summary=summary, kind="ok")


def _reflect_step(turn) -> TraceStep:
    """Reflection step is best-effort: TurnRecord doesn't carry the
    raw reflection dict (we lose it after the graph returns), so we
    inspect what we *can* see — iteration_count vs MAX_ITERATIONS — to
    explain why the loop ended."""
    iters = turn.iteration_count or 0
    if iters >= 4:  # mirrors graph.MAX_ITERATIONS
        return TraceStep(
            name="Reflect",
            summary="iteration cap reached — answered with what we had",
            details=["the loop budget (4 passes) was exhausted"],
            kind="warn",
        )
    if iters > 1:
        return TraceStep(
            name="Reflect",
            summary=f"looped {iters - 1} time(s) before settling",
            details=[
                "reflection asked for another pass and the synthesizer agreed"
            ],
            kind="info",
        )
    return TraceStep(
        name="Reflect",
        summary="answered confidently in one pass",
        details=[],
        kind="ok",
    )


def build_trace_steps(turn) -> list[TraceStep]:
    """Build the ordered Reasoning-steps card list for one turn.

    The turn doesn't need to be a strict `TurnRecord` — anything with
    the same attribute names works (helps tests stay light)."""
    return [
        _classify_step(turn),
        _decompose_step(turn),
        _retrieve_or_tool_step(turn),
        _synthesize_step(turn),
        _reflect_step(turn),
    ]


def chunks_table_rows(chunks: list[dict]) -> list[dict]:
    """Flatten retrieved chunks into ranked rows for a `st.dataframe`.

    Sorting prefers `boosted_score` when the persona-boost pass added
    one — that's the order the synthesizer actually saw. We always
    surface both columns so the user can see how much the boost moved
    a row. `score` and `boosted` are rounded to three decimals so the
    table doesn't render scientific notation, which Streamlit will do
    for raw floats by default.
    """
    if not chunks:
        return []

    def _sort_key(c: dict) -> float:
        return float(c.get("boosted_score") or c.get("score") or 0.0)

    ranked = sorted(chunks, key=_sort_key, reverse=True)
    rows: list[dict] = []
    for i, c in enumerate(ranked, start=1):
        payload = c.get("payload") or {}
        title = (payload.get("title") or "").strip()
        if len(title) > 80:
            title = title[:79] + "…"
        page = payload.get("page")
        score = c.get("score")
        boosted = c.get("boosted_score")
        rows.append(
            {
                "rank": i,
                "publisher": payload.get("publisher") or "?",
                "page": page if page is not None else "",
                "score": round(float(score), 3) if isinstance(score, (int, float)) else None,
                "boosted": (
                    round(float(boosted), 3)
                    if isinstance(boosted, (int, float))
                    else None
                ),
                "title": title,
            }
        )
    return rows


__all__ = [
    "TraceStep",
    "StepKind",
    "build_trace_steps",
    "chunks_table_rows",
]
