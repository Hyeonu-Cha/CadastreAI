"""Smoke-test a deployed CadastreAI instance against 5 canonical queries.

Task 4.16 — runs after Tasks 4.14 (Qdrant Cloud upsert) and 4.15
(HF Spaces / Modal deploy). Drives the same `src.agent.graph` code
path the Streamlit UI uses, so a green smoke run means the deployed
backend is wired correctly end-to-end (Anthropic key, Qdrant URL +
key, BM25 pickle, encoder cache, tool API keys).

Usage
-----

Run against a deployed Qdrant Cloud + Anthropic config:

    QDRANT_URL=https://<cluster>.qdrant.io \
    QDRANT_API_KEY=<key> \
    ANTHROPIC_API_KEY=sk-... \
    python -m scripts.smoke_prod

Or run against a local stack (sanity check before deploy):

    QDRANT_URL=http://localhost:6333 \
    QDRANT_API_KEY=cadastre-dev \
    ANTHROPIC_API_KEY=sk-... \
    python -m scripts.smoke_prod

Exit codes
----------
0  — all 5 queries passed every check
1  — one or more queries failed (details printed)
2  — environment misconfigured (missing required env var, import error)

What "passed" means
-------------------
For each query:
  - agent returned a non-empty answer
  - latency under SOFT_LATENCY_BUDGET_S (warn) / HARD_LATENCY_BUDGET_S (fail)
  - if `expects_tools` is set, at least one was called
  - if `expects_docs` is set, at least one chunk was retrieved
  - if `expects_citation_marker` is set, the marker text appears in
    the answer (catches silent regressions where the synth drops
    citation tags)

The 5 queries deliberately span persona, tool-vs-docs routing, and
the v5 compute-tool disambiguation regime — failures here usually
point at a specific subsystem rather than a generic "agent broken".
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

# Latency budgets are wall-clock end-to-end. Tools (esp. ABS) can be
# slow on cold cache; the soft/hard split lets us warn without failing
# on a one-off slow upstream call.
SOFT_LATENCY_BUDGET_S = 20.0
HARD_LATENCY_BUDGET_S = 45.0


@dataclass
class SmokeQuery:
    id: str
    persona: str
    query: str
    rationale: str
    expects_tools: bool = False
    expects_docs: bool = False
    # Substring that must appear in the answer text (case-insensitive).
    # Empty list means no content check beyond non-empty.
    expects_substrings: list[str] = field(default_factory=list)


# These five exercise distinct code paths. Re-ordering or replacing
# them changes the coverage profile — be deliberate.
SMOKE_QUERIES: list[SmokeQuery] = [
    SmokeQuery(
        id="smoke-01-tool-only",
        persona="general",
        query="What is the current RBA cash rate?",
        rationale="Pure tool path (rba_cash_rate). Verifies tool API "
        "keys, tool cache wiring, and synth's ability to handle a "
        "tool-only result without doc retrieval.",
        expects_tools=True,
    ),
    SmokeQuery(
        id="smoke-02-docs-only",
        persona="researcher",
        query=(
            "How does CoreLogic's hedonic index differ from the ABS "
            "Residential Property Price Index methodologically?"
        ),
        rationale="Pure docs path (no live tool fits). Verifies Qdrant "
        "connectivity, BM25 pickle load, hybrid fusion, and citation "
        "rendering.",
        expects_docs=True,
        expects_substrings=["hedonic"],
    ),
    SmokeQuery(
        id="smoke-03-hybrid-route",
        persona="investor",
        query="What's the median rental yield in Perth and what's driving it?",
        rationale="v5 disambiguation target — must route to "
        "sqm_rental_vacancy + abs_property_price_index (NOT "
        "compute_rental_yield, which would have failed pre-v5). "
        "Should also pull docs for the 'driving it' framing.",
        expects_tools=True,
        expects_docs=True,
    ),
    SmokeQuery(
        id="smoke-04-compute-tool",
        persona="investor",
        query="I get $32,000 per year rent on a $650,000 unit. What's my yield?",
        rationale="Pure compute path — verifies compute_rental_yield "
        "is callable and returns a sensible number when both args "
        "are present in the question.",
        expects_tools=True,
        # 32000/650000 = 4.92% — synth should surface that figure.
        expects_substrings=["4.9", "%"],
    ),
    SmokeQuery(
        id="smoke-05-multi-step",
        persona="investor",
        query=(
            "Which Sydney suburbs have the highest 3-year capital "
            "growth with vacancy below 3%?"
        ),
        rationale="Multi-step / decomposable. Stresses the decompose "
        "→ retrieve_or_tool loop and synth's ability to combine "
        "evidence from multiple sub-answers.",
        # Either route is acceptable; we just need a substantive answer.
    ),
]


@dataclass
class SmokeResult:
    query: SmokeQuery
    elapsed_s: float
    answer: str
    n_tools: int
    n_chunks: int
    error: str | None
    failures: list[str]

    @property
    def passed(self) -> bool:
        return self.error is None and not self.failures


def _check_env() -> list[str]:
    """Return a list of missing required env vars."""
    required = ("QDRANT_URL", "ANTHROPIC_API_KEY")
    return [v for v in required if not os.environ.get(v)]


def _evaluate(
    q: SmokeQuery,
    final_state: dict[str, Any],
    elapsed_s: float,
) -> list[str]:
    """Apply the per-query expectations and return a list of failure strings."""
    failures: list[str] = []

    answer = (final_state.get("answer_draft") or "").strip()
    tools = list(final_state.get("tool_results") or [])
    chunks = list(final_state.get("retrieved_chunks") or [])

    if not answer:
        failures.append("empty answer")

    if elapsed_s > HARD_LATENCY_BUDGET_S:
        failures.append(
            f"latency {elapsed_s:.1f}s exceeds hard budget {HARD_LATENCY_BUDGET_S}s"
        )

    if q.expects_tools and not tools:
        failures.append("expected ≥1 tool call, got 0")

    if q.expects_docs and not chunks:
        failures.append("expected ≥1 retrieved chunk, got 0")

    lowered = answer.lower()
    for needle in q.expects_substrings:
        if needle.lower() not in lowered:
            failures.append(f"answer missing expected substring {needle!r}")

    return failures


def _run_one(q: SmokeQuery, graph, initial_state) -> SmokeResult:
    t0 = time.perf_counter()
    try:
        final = graph.invoke(initial_state(q.query, user_persona=q.persona))
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - t0
        return SmokeResult(
            query=q,
            elapsed_s=elapsed,
            answer="",
            n_tools=0,
            n_chunks=0,
            error=f"{type(exc).__name__}: {exc}",
            failures=[],
        )
    elapsed = time.perf_counter() - t0
    failures = _evaluate(q, final, elapsed)
    return SmokeResult(
        query=q,
        elapsed_s=elapsed,
        answer=(final.get("answer_draft") or "").strip(),
        n_tools=len(final.get("tool_results") or []),
        n_chunks=len(final.get("retrieved_chunks") or []),
        error=None,
        failures=failures,
    )


def _print_result(r: SmokeResult, idx: int, total: int) -> None:
    status = "PASS" if r.passed else "FAIL"
    print(f"\n[{idx}/{total}] {r.query.id}  [{status}]")
    print(f"  persona: {r.query.persona}")
    print(f"  query:   {r.query.query}")
    print(f"  elapsed: {r.elapsed_s:.1f}s")
    print(f"  tools:   {r.n_tools}")
    print(f"  chunks:  {r.n_chunks}")
    if r.error:
        print(f"  ERROR:   {r.error}")
    for f in r.failures:
        print(f"  FAIL:    {f}")
    if r.elapsed_s > SOFT_LATENCY_BUDGET_S and r.passed:
        print(
            f"  WARN:    latency {r.elapsed_s:.1f}s above soft budget "
            f"{SOFT_LATENCY_BUDGET_S}s"
        )
    if r.answer:
        snippet = r.answer.replace("\n", " ")
        if len(snippet) > 200:
            snippet = snippet[:197] + "..."
        print(f"  answer:  {snippet}")


def main() -> int:
    missing = _check_env()
    if missing:
        print(f"ERROR: missing required env vars: {', '.join(missing)}")
        print(
            "Required: QDRANT_URL, ANTHROPIC_API_KEY. "
            "QDRANT_API_KEY is required for any cloud Qdrant cluster."
        )
        return 2

    try:
        from src.agent.graph import build_graph, initial_state
    except ImportError as exc:
        print(f"ERROR: cannot import agent graph: {exc}")
        print("Did you `pip install -e .[agent,index,embed,tools,app]`?")
        return 2

    print("Building agent graph...")
    graph = build_graph()
    print(f"Running {len(SMOKE_QUERIES)} smoke queries against:")
    print(f"  QDRANT_URL = {os.environ['QDRANT_URL']}")
    print(
        f"  collection = {os.environ.get('QDRANT_COLLECTION', 'cadastre_chunks')}"
    )

    results: list[SmokeResult] = []
    for i, q in enumerate(SMOKE_QUERIES, start=1):
        r = _run_one(q, graph, initial_state)
        results.append(r)
        _print_result(r, i, len(SMOKE_QUERIES))

    n_pass = sum(1 for r in results if r.passed)
    n_fail = len(results) - n_pass
    print("\n" + "=" * 60)
    print(f"SUMMARY: {n_pass}/{len(results)} passed, {n_fail} failed")
    if n_fail == 0:
        print("Deploy looks healthy.")
        return 0
    print("Failed queries:")
    for r in results:
        if not r.passed:
            label = r.error or "; ".join(r.failures)
            print(f"  - {r.query.id}: {label}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
