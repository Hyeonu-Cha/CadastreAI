"""Agent eval harness — score the LangGraph agent against ground truth (Task 3.21).

Reads `data/eval/agent_queries.jsonl` (Task 3.20), runs each query through
`build_graph().invoke(...)`, and scores four metrics against the
annotation:

  * tool_call_accuracy  — Jaccard(actual ∩ expected_tools)
  * tool_call_recall    — fraction of expected tools that were called
  * tool_call_precision — fraction of called tools that were expected
  * publisher_recall    — fraction of expected_doc_publishers actually
                          surfaced in retrieved_chunks
  * trajectory_efficiency — 1 / (1 + extra_iterations + extra_tools);
                            penalises looping past what the gold expects
  * groundedness        — % of numeric claims in `answer_draft` that
                          have a citation marker within 50 chars
  * faithfulness        — optional, network-bound; Claude-as-judge call
                          that asks "is every claim in the answer
                          supported by the supplied evidence?". Off by
                          default; opt in with `--with-judge`.

CLI:
    python -m src.eval.agent_eval \
        --queries data/eval/agent_queries.jsonl \
        --out results/agent_v1.json \
        [--with-judge]      # add Claude-as-judge faithfulness scoring
        [--trace]           # enable LangSmith tracing
        [--limit 10]        # cap queries for smoke runs
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

from src.eval.regime import NEUTRAL_REGIME, is_headline, regime_of

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

log = logging.getLogger(__name__)

DEFAULT_QUERIES = Path("data/eval/agent_queries.jsonl")
DEFAULT_OUTPUT = Path("results/agent_v1.json")

# Numbers, including 4-digit years, percentages, dollar amounts, and
# generic decimals. We use this to find numeric claims in the answer
# draft and check whether each has a nearby citation.
_NUMERIC_RE = re.compile(
    r"(?<!\w)(?:\$\s?\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?\s?%|\d[\d,]*(?:\.\d+)?)(?!\w)"
)
_CITATION_NEAR_RE = re.compile(r"\[(?:source|tool):[^\]]+\]")
_CITATION_WINDOW = 50  # chars before/after the numeric token to scan


# ---------- metric primitives --------------------------------------


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def tool_metrics(actual: list[str], expected: list[str]) -> dict:
    a, e = set(actual), set(expected)
    inter = a & e
    return {
        "tool_call_accuracy": jaccard(a, e),
        "tool_call_recall": (len(inter) / len(e)) if e else (1.0 if not a else 0.0),
        "tool_call_precision": (len(inter) / len(a)) if a else (1.0 if not e else 0.0),
        "extra_tools": sorted(a - e),
        "missing_tools": sorted(e - a),
    }


def publisher_recall(actual_chunks: list[dict], expected_publishers: list[str]) -> dict:
    """Fraction of expected publishers actually surfaced in retrieval."""
    actual_pubs = {
        ((c.get("payload") or {}).get("publisher") or "").strip()
        for c in actual_chunks
    }
    actual_pubs.discard("")
    e = set(expected_publishers)
    inter = actual_pubs & e
    return {
        "publisher_recall": (len(inter) / len(e)) if e else (1.0 if not actual_pubs else 1.0),
        "missing_publishers": sorted(e - actual_pubs),
        "actual_publishers": sorted(actual_pubs),
    }


def trajectory_efficiency(
    actual_tools: list[str],
    expected_tools: list[str],
    iteration_count: int,
    expected_iterations: int = 1,
) -> float:
    """1 / (1 + extra_iterations + extra_tools).

    A perfect run (right tools, one pass) scores 1.0. Each unexpected
    tool call OR extra reflect-loop iteration drops the score
    proportionally. We don't penalise *missing* tools here — that's
    captured by `tool_call_recall`.
    """
    extra_iters = max(0, iteration_count - expected_iterations)
    extra_tools = max(0, len(set(actual_tools) - set(expected_tools)))
    return 1.0 / (1.0 + extra_iters + extra_tools)


def groundedness(answer: str) -> dict:
    """% of numeric claims in `answer` that have a citation within 50 chars.

    Citation markers like `[source:RBA, page:3]` and `[tool:x, retrieved:2026-04-26]`
    contain digits themselves; we redact them with spaces (preserving indices)
    before scanning so they neither inflate `n_numeric_claims` nor
    self-ground their own numbers.
    """
    text = answer or ""
    # Redact citation markers in-place with spaces — preserves offsets so the
    # ±_CITATION_WINDOW lookups still align with the original string.
    redacted = _CITATION_NEAR_RE.sub(lambda m: " " * (m.end() - m.start()), text)
    numbers = list(_NUMERIC_RE.finditer(redacted))
    if not numbers:
        return {"groundedness": 1.0, "n_numeric_claims": 0, "n_grounded": 0}
    grounded = 0
    for m in numbers:
        start = max(0, m.start() - _CITATION_WINDOW)
        end = min(len(text), m.end() + _CITATION_WINDOW)
        if _CITATION_NEAR_RE.search(text[start:end]):
            grounded += 1
    return {
        "groundedness": grounded / len(numbers),
        "n_numeric_claims": len(numbers),
        "n_grounded": grounded,
    }


# ---------- LLM-as-judge (optional, network) -----------------------


_JUDGE_SYSTEM = (
    "You are a strict evaluator for an Australian housing-market research "
    "agent. Given a question, the evidence the agent had, and its answer, "
    "decide whether every factual or numeric claim in the answer is "
    "directly supported by the evidence. Submit via the submit_judgement "
    "tool. Be conservative — if a claim plausibly comes from training "
    "knowledge rather than the evidence, mark it unsupported."
)
_JUDGE_TOOL = {
    "name": "submit_judgement",
    "description": "Submit a faithfulness judgement on the agent's answer.",
    "input_schema": {
        "type": "object",
        "properties": {
            "n_claims": {"type": "integer", "minimum": 0},
            "n_supported": {"type": "integer", "minimum": 0},
            "unsupported": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Verbatim or paraphrased claims that aren't backed.",
            },
        },
        "required": ["n_claims", "n_supported", "unsupported"],
    },
}


def _resolve_judge_model() -> str:
    """Pick the judge model based on the active agent provider.

    Anthropic → Haiku 4.5 (the existing default — strict + cheap).
    OpenAI    → gpt-4o-mini (matches the small-node default in
    ``src/agent/nodes.py``). Both can be overridden via env so an
    eval run can pin a beefier judge without code changes.
    """
    import os

    from src.agent import llm

    if llm.get_provider() == "openai":
        return os.environ.get("CADASTRE_OPENAI_JUDGE_MODEL", "gpt-4o-mini")
    return os.environ.get("CADASTRE_JUDGE_MODEL", "claude-haiku-4-5")


def faithfulness_with_judge(
    query: str,
    evidence_block: str,
    answer: str,
    *,
    model: str | None = None,
) -> dict:
    """LLM-as-judge faithfulness — routes through the active provider.

    Uses ``src.agent.llm.call_with_tool`` so the same eval run that
    drives the agent on OpenAI also judges on OpenAI (and likewise for
    Anthropic). ``model`` defaults to ``_resolve_judge_model()``."""
    from src.agent import llm

    prompt = (
        f"QUESTION:\n{query}\n\n"
        f"EVIDENCE:\n{evidence_block}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Score every factual or numeric claim in the answer."
    )
    raw, _usage = llm.call_with_tool(
        system=_JUDGE_SYSTEM,
        user=prompt,
        tool_def=_JUDGE_TOOL,
        tool_name="submit_judgement",
        max_tokens=512,
        model=model or _resolve_judge_model(),
    )
    if raw is None:
        return {
            "faithfulness": None,
            "n_claims": None,
            "n_supported": None,
            "unsupported": [],
        }
    n_claims = int(raw.get("n_claims", 0))
    n_sup = int(raw.get("n_supported", 0))
    return {
        "faithfulness": (n_sup / n_claims) if n_claims else 1.0,
        "n_claims": n_claims,
        "n_supported": n_sup,
        "unsupported": list(raw.get("unsupported") or []),
    }


def _evidence_block_for_judge(state: dict, max_chars: int = 6000) -> str:
    """Compact evidence pack for the judge — chunks + tool results."""
    lines = []
    for c in (state.get("retrieved_chunks") or [])[:8]:
        p = c.get("payload") or {}
        text = (p.get("text") or "")[:400]
        lines.append(f"[doc] {p.get('publisher')} | {p.get('title','')[:80]}: {text!r}")
    for t in (state.get("tool_results") or []):
        if "result" in t:
            r = t["result"]
            lines.append(f"[tool] {t.get('tool')}({t.get('args')}) → {r.get('data')}")
    block = "\n".join(lines)
    return block[:max_chars]


# ---------- per-query + aggregate ----------------------------------


def evaluate_one(record: dict, final_state: dict, *, with_judge: bool = False) -> dict:
    actual_tools = [
        t.get("tool") for t in (final_state.get("tool_results") or []) if t.get("tool")
    ]
    expected_tools = list(record.get("expected_tools") or [])
    expected_pubs = list(record.get("expected_doc_publishers") or [])

    tm = tool_metrics(actual_tools, expected_tools)
    pr = publisher_recall(final_state.get("retrieved_chunks") or [], expected_pubs)
    teff = trajectory_efficiency(
        actual_tools,
        expected_tools,
        final_state.get("iteration_count", 0),
        expected_iterations=1,
    )
    gnd = groundedness(final_state.get("answer_draft") or "")

    out = {
        "id": record.get("id"),
        "regime": regime_of(record),
        "query": record.get("query"),
        "expected_tools": expected_tools,
        "actual_tools": actual_tools,
        "iteration_count": final_state.get("iteration_count"),
        "n_retrieved_chunks": len(final_state.get("retrieved_chunks") or []),
        **tm,
        **pr,
        "trajectory_efficiency": teff,
        **gnd,
    }
    if with_judge:
        evidence = _evidence_block_for_judge(final_state)
        out.update(
            faithfulness_with_judge(
                record["query"], evidence, final_state.get("answer_draft") or ""
            )
        )
    return out


def aggregate(rows: list[dict]) -> dict:
    """Macro-average the per-query metrics."""
    if not rows:
        return {"n": 0}
    keys = [
        "tool_call_accuracy",
        "tool_call_recall",
        "tool_call_precision",
        "publisher_recall",
        "trajectory_efficiency",
        "groundedness",
    ]
    agg = {f"mean_{k}": sum(r.get(k, 0.0) for r in rows) / len(rows) for k in keys}
    if all("faithfulness" in r and r["faithfulness"] is not None for r in rows):
        agg["mean_faithfulness"] = sum(r["faithfulness"] for r in rows) / len(rows)
    agg["n"] = len(rows)
    return agg


def run_agent_eval(
    queries_path: Path,
    *,
    with_judge: bool = False,
    trace: bool = False,
    limit: int | None = None,
) -> dict:
    from src.agent.graph import build_graph, initial_state
    from src.agent.tracing import enable_tracing, traced_run

    if trace:
        enable_tracing()
    records = [
        json.loads(line) for line in queries_path.read_text(encoding="utf-8").splitlines()
    ]
    if limit:
        records = records[:limit]

    # Pre-warm the agent's configured retriever before the LangGraph/OpenAI
    # stack fully loads. On the 4GB-pagefile Windows host, lazy-loading
    # sentence-transformers mid-run after openai+langchain are resident
    # segfaults during the torch DLL init (same family as the BGE-reranker
    # note in src/retrieval/rerank.py). Goes through the agent's own
    # `_retrieve_docs` so hybrid mode (Task 3.25 default) warms both dense
    # and BM25 in one shot, and dense/bm25 modes still work via env override.
    try:
        from src.agent.nodes import _retrieve_docs as _warm_retrieve_docs

        _warm_retrieve_docs("warmup", k=1)
        log.info("retriever warmed up")
    except Exception as e:  # noqa: BLE001 — eval can still run without docs
        log.warning("retriever warmup failed: %s", e)

    g = build_graph()
    rows: list[dict] = []
    failures: list[dict] = []
    for i, rec in enumerate(records, start=1):
        log.info("[%d/%d] %s", i, len(records), rec["id"])
        t0 = time.perf_counter()
        try:
            with traced_run("agent.eval", id=rec["id"], query=rec["query"]):
                final = g.invoke(initial_state(rec["query"]))
            row = evaluate_one(rec, final, with_judge=with_judge)
            row["elapsed_seconds"] = round(time.perf_counter() - t0, 2)
            rows.append(row)
        except Exception as e:  # noqa: BLE001 — record + continue
            log.exception("eval %s failed", rec["id"])
            failures.append({"id": rec["id"], "error": f"{type(e).__name__}: {e}"})
    # Headline metrics exclude the quarantined pre-reform set (Task 5.02):
    # those queries expect repealed-law answers, so counting them measures
    # fidelity to a superseded statute. Legacy is still scored + reported.
    headline_rows = [r for r in rows if is_headline(r.get("regime", NEUTRAL_REGIME))]
    legacy_rows = [r for r in rows if not is_headline(r.get("regime", NEUTRAL_REGIME))]
    by_regime = {
        reg: aggregate([r for r in rows if r.get("regime", NEUTRAL_REGIME) == reg])
        for reg in sorted({r.get("regime", NEUTRAL_REGIME) for r in rows})
    }
    return {
        "queries_path": str(queries_path),
        "with_judge": with_judge,
        "aggregate": aggregate(headline_rows),
        "aggregate_including_legacy": aggregate(rows),
        "legacy_regime": aggregate(legacy_rows),
        "by_regime": by_regime,
        "rows": rows,
        "failures": failures,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    p.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--with-judge",
        action="store_true",
        help="Add Claude-as-judge faithfulness scoring (network).",
    )
    p.add_argument("--trace", action="store_true", help="Enable LangSmith tracing.")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of queries for a smoke run.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    report = run_agent_eval(
        args.queries,
        with_judge=args.with_judge,
        trace=args.trace,
        limit=args.limit,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(
        "Wrote %s — n=%d mean_tool_acc=%.3f mean_grounded=%.3f",
        args.out,
        report["aggregate"].get("n", 0),
        report["aggregate"].get("mean_tool_call_accuracy", 0.0),
        report["aggregate"].get("mean_groundedness", 0.0),
    )
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
