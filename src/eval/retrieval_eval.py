"""Retrieval-only eval: Recall@5, Recall@10, MRR, nDCG@10 vs gold set.

Runs the dense retriever (src.retrieval.retriever) against each query in
`data/eval/queries_all.jsonl`, collects the top-K ranked chunk_ids, and
scores them against the annotated `gold_chunk_ids` for that query.

Metrics (all binary-relevance, micro-averaged over queries):
- Recall@k: |retrieved[:k] ∩ gold| / |gold|
- MRR@k:    1 / rank of first gold hit in top-k (0 if none)
- nDCG@k:   DCG / IDCG where DCG sums rel_i / log2(i+1) and IDCG
            uses min(k, |gold|) ideal hits at ranks 1..

Binary relevance is right for this dataset: gold annotations are
"is this chunk a correct answer, yes/no"; there's no graded scale.

Writes a results JSON with (a) aggregate metrics, (b) per-persona
breakdowns, and (c) per-query records including retrieved ranks of
each gold chunk so error analysis (Task 1.32) has the raw material.

    python -m src.eval.retrieval_eval --queries data/eval/queries_all.jsonl \
        --out results/baseline.json -k 10
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.retrieval.retriever import Retriever

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

log = logging.getLogger(__name__)


def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    hits = sum(1 for cid in retrieved[:k] if cid in gold)
    return hits / len(gold)


def reciprocal_rank(retrieved: list[str], gold: set[str], k: int) -> float:
    for i, cid in enumerate(retrieved[:k], start=1):
        if cid in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    # Binary relevance, so rel_i ∈ {0, 1}.
    dcg = 0.0
    for i, cid in enumerate(retrieved[:k], start=1):
        if cid in gold:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


@dataclass
class QueryResult:
    query: str
    persona: str
    gold_chunk_ids: list[str]
    retrieved_chunk_ids: list[str]
    gold_ranks: dict[str, int | None]  # gold_id -> 1-indexed rank or None
    recall_5: float
    recall_10: float
    mrr_10: float
    ndcg_10: float


def evaluate_query(
    query: str,
    persona: str,
    gold: list[str],
    retrieved: list[str],
    k_values: tuple[int, ...] = (5, 10),
) -> QueryResult:
    gold_set = set(gold)
    ranks: dict[str, int | None] = {gid: None for gid in gold}
    for i, cid in enumerate(retrieved, start=1):
        if cid in ranks and ranks[cid] is None:
            ranks[cid] = i
    return QueryResult(
        query=query,
        persona=persona,
        gold_chunk_ids=list(gold),
        retrieved_chunk_ids=retrieved,
        gold_ranks=ranks,
        recall_5=recall_at_k(retrieved, gold_set, 5),
        recall_10=recall_at_k(retrieved, gold_set, 10),
        mrr_10=reciprocal_rank(retrieved, gold_set, 10),
        ndcg_10=ndcg_at_k(retrieved, gold_set, 10),
    )


def aggregate(results: list[QueryResult]) -> dict[str, float]:
    if not results:
        return {"recall@5": 0.0, "recall@10": 0.0, "mrr@10": 0.0, "ndcg@10": 0.0, "n": 0}
    n = len(results)
    return {
        "recall@5": sum(r.recall_5 for r in results) / n,
        "recall@10": sum(r.recall_10 for r in results) / n,
        "mrr@10": sum(r.mrr_10 for r in results) / n,
        "ndcg@10": sum(r.ndcg_10 for r in results) / n,
        "n": n,
    }


def _load_queries(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def run(
    queries_path: Path,
    out_path: Path,
    top_k: int,
    retriever: Retriever | None = None,
    limit: int | None = None,
) -> dict:
    retriever = retriever or Retriever()
    queries = _load_queries(queries_path)
    if limit is not None:
        queries = queries[:limit]
    log.info("Evaluating %d queries with top_k=%d", len(queries), top_k)

    results: list[QueryResult] = []
    t0 = time.time()
    for i, rec in enumerate(queries, start=1):
        hits = retriever.retrieve(rec["query"], k=top_k)
        retrieved_ids = [payload.get("chunk_id") for payload, _ in hits]
        results.append(
            evaluate_query(
                query=rec["query"],
                persona=rec.get("persona") or "?",
                gold=rec.get("gold_chunk_ids") or [],
                retrieved=retrieved_ids,
            )
        )
        if i % 10 == 0 or i == len(queries):
            elapsed = time.time() - t0
            log.info("  %d/%d queries (%.1fs, %.2f q/s)", i, len(queries), elapsed, i / elapsed)

    overall = aggregate(results)
    by_persona: dict[str, dict] = {}
    buckets: dict[str, list[QueryResult]] = defaultdict(list)
    for r in results:
        buckets[r.persona].append(r)
    for persona, rs in buckets.items():
        by_persona[persona] = aggregate(rs)

    out = {
        "queries_path": str(queries_path),
        "top_k": top_k,
        "retriever": {
            "model": retriever.model_name,
            "collection": retriever.collection,
        },
        "overall": overall,
        "by_persona": by_persona,
        "per_query": [
            {
                "query": r.query,
                "persona": r.persona,
                "gold_chunk_ids": r.gold_chunk_ids,
                "retrieved_chunk_ids": r.retrieved_chunk_ids,
                "gold_ranks": r.gold_ranks,
                "recall@5": r.recall_5,
                "recall@10": r.recall_10,
                "mrr@10": r.mrr_10,
                "ndcg@10": r.ndcg_10,
            }
            for r in results
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    log.info(
        "Wrote %s | R@5=%.3f R@10=%.3f MRR@10=%.3f nDCG@10=%.3f",
        out_path,
        overall["recall@5"],
        overall["recall@10"],
        overall["mrr@10"],
        overall["ndcg@10"],
    )
    print(
        f"n={overall['n']} "
        f"R@5={overall['recall@5']:.3f} "
        f"R@10={overall['recall@10']:.3f} "
        f"MRR@10={overall['mrr@10']:.3f} "
        f"nDCG@10={overall['ndcg@10']:.3f} "
        f"out={out_path}"
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queries", type=Path, default=Path("data/eval/queries_all.jsonl"))
    p.add_argument("--out", type=Path, default=Path("results/baseline.json"))
    p.add_argument("-k", "--top-k", type=int, default=10)
    p.add_argument("--limit", type=int, default=None, help="evaluate only first N queries")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    run(queries_path=args.queries, out_path=args.out, top_k=args.top_k, limit=args.limit)


if __name__ == "__main__":
    main()
