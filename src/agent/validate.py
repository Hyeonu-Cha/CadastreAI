"""10-query validation harness for the CadastreAI agent (Task 3.16).

Runs ten diverse queries through `build_graph().invoke(...)` and writes
a per-query summary plus aggregate stats to `results/agent_validation.json`.
Optionally enables LangSmith tracing (`--trace` or `LANGSMITH_API_KEY`)
so the run shows up in the dashboard for inspection.

The 10 queries cover the routing surface intentionally — at least one
per persona and per query_type, and several that exercise decomposition
and tool composition. They're deliberately not pinned to expected
classifications: this is a smoke/coverage check, not an eval. Pinning
behaviour belongs in Task 3.21.

    python -m src.agent.validate
    python -m src.agent.validate --trace
    python -m src.agent.validate --out results/agent_validation.json -v

Without `ANTHROPIC_API_KEY` the LLM nodes raise — this script does not
auto-stub them, because the whole point of validation is to test the
real graph end-to-end.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from src.agent.graph import build_graph, initial_state
from src.agent.tracing import enable_tracing, traced_run

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

log = logging.getLogger(__name__)

VALIDATION_QUERIES = [
    # Factual single-shot — should hit the doc retriever, no decomposition.
    "What is negative gearing in Australia and how does it work?",
    # Numeric single-shot — should call rba_cash_rate, no docs.
    "What is the current RBA cash rate?",
    # Numeric single-shot — should call abs_property_price_index for Sydney.
    "What's the median house price in Sydney right now?",
    # Numeric single-shot — should call sqm_rental_vacancy.
    "How tight is the rental market in Melbourne at the moment?",
    # Pure compute — should call compute_stamp_duty_nsw with FHB taper.
    "I'm a first home buyer buying a $850,000 apartment in NSW. "
    "What stamp duty do I owe?",
    # Pure compute — should call compute_mortgage_repayment.
    "What's the monthly repayment on a $700,000 loan at 6.10% over 30 years?",
    # Comparative — should decompose into Sydney vs Melbourne legs.
    "Compare median Sydney house prices to Melbourne over the last 12 months "
    "and explain what's driving the difference.",
    # Computational + persona — first-home-buyer scenario, multi-tool.
    "I'm a first home buyer in NSW looking at a $950k house. What stamp duty "
    "would I pay, what's the monthly repayment with a 20% deposit at 6%, "
    "and how does that compare to current rental yields in Sydney?",
    # Investor persona — yield + vacancy combo.
    "As an investor, what gross yield should I expect on a $750,000 "
    "Brisbane property renting for $620 a week, and is vacancy tight there?",
    # Policy/research — exploratory, doc-heavy.
    "How do build-to-rent tax settings in Australia compare to traditional "
    "investor incentives, and what reforms have been proposed?",
]


def _summarise(query: str, final_state: dict, elapsed: float) -> dict:
    cls = final_state.get("classification") or {}
    chunks = final_state.get("retrieved_chunks", [])
    tools = final_state.get("tool_results", [])
    tool_names = sorted({t.get("tool") for t in tools if t.get("tool")})
    tool_errors = [t for t in tools if "error" in t]
    return {
        "query": query,
        "elapsed_seconds": round(elapsed, 2),
        "classification": cls,
        "sub_questions": final_state.get("sub_questions"),
        "iteration_count": final_state.get("iteration_count"),
        "is_complete": (final_state.get("reflection") or {}).get("is_complete"),
        "n_retrieved_chunks": len(chunks),
        "tools_called": tool_names,
        "n_tool_errors": len(tool_errors),
        "answer_draft": final_state.get("answer_draft"),
    }


def run_validation(
    queries: list[str] | None = None,
    *,
    trace: bool = False,
) -> dict:
    queries = queries or VALIDATION_QUERIES
    if trace:
        enable_tracing()
    g = build_graph()
    runs: list[dict] = []
    failures: list[dict] = []
    for i, q in enumerate(queries, start=1):
        log.info("[%d/%d] %s", i, len(queries), q[:80])
        t0 = time.perf_counter()
        try:
            with traced_run("agent.validate", query=q, idx=i):
                final = g.invoke(initial_state(q))
            runs.append(_summarise(q, final, time.perf_counter() - t0))
        except Exception as e:  # noqa: BLE001 — record + continue
            log.exception("query %d failed", i)
            failures.append(
                {"query": q, "error": f"{type(e).__name__}: {e}"}
            )
    return {
        "n_queries": len(queries),
        "n_succeeded": len(runs),
        "n_failed": len(failures),
        "tools_seen": sorted(
            {t for r in runs for t in r["tools_called"]}
        ),
        "runs": runs,
        "failures": failures,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--trace",
        action="store_true",
        help="Enable LangSmith tracing (needs LANGSMITH_API_KEY).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/agent_validation.json"),
        help="Where to write the per-run summary (default: results/...).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    report = run_validation(trace=args.trace)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info(
        "Wrote %s (%d/%d succeeded, %d tools seen)",
        args.out,
        report["n_succeeded"],
        report["n_queries"],
        len(report["tools_seen"]),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
