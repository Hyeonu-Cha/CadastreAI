# Cross-encoder reranker vs hybrid baseline

Hybrid (RRF over dense + BM25) shortlists 30 chunks per query;
the cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~22M params,
CPU) scores each `(query, chunk)` pair and resorts to top-10. Raw
per-query numbers in `results/hybrid.json` and `results/reranked.json`.

## Numbers

### Synthetic split (41 queries, honest — independent gold)

| Retriever       |   R@5 |  R@10 | MRR@10 | nDCG@10 |
|-----------------|------:|------:|-------:|--------:|
| hybrid          | 0.805 | **0.878** |  0.640 |   0.698 |
| **+ rerank**    | **0.829** | 0.854 | **0.680** | **0.723** |
| Δ               | +2.4 pts | −2.4 pts | +3.9 pts | +2.5 pts |

The reranker pushes correct chunks **higher** in the list — R@5 +2.4,
MRR@10 +3.9, nDCG@10 +2.5 — exactly the failure mode Task 1.32's error
analysis flagged: 60% of dense misses had the right doc/topic in the
shortlist but the wrong chunk. R@10 drops 2.4 pts because the reranker
occasionally reorders an in-shortlist gold below position 10. Net: rank
quality clearly improves on the honest split.

### BM25-annotated split (59 queries, circular — annotators chose from BM25 top-10)

| Retriever       |   R@5 |  R@10 | MRR@10 | nDCG@10 |
|-----------------|------:|------:|-------:|--------:|
| hybrid          | 0.395 | 0.655 |  0.616 |   0.506 |
| + rerank        | 0.356 | 0.548 |  0.477 |   0.407 |
| Δ               | −3.9 pts | −10.7 pts | −13.9 pts | −9.9 pts |

Reranker loses across the board, as expected. The gold here was hand-
picked from BM25's top-10 candidates, so the gold chunks are exactly
the lexically-salient ones BM25 already surfaces. The cross-encoder's
semantic re-scoring rewards paraphrase matches that aren't in this
gold set, so it pushes annotated golds down. This split was already
flagged as anti-correlated with semantic depth in `hybrid_comparison.md`.

### Overall (pooled, for reference only)

| Retriever      |   R@5 |  R@10 | MRR@10 | nDCG@10 |
|----------------|------:|------:|-------:|--------:|
| hybrid         | 0.563 | 0.747 |  0.626 |   0.585 |
| + rerank       | 0.550 | 0.673 |  0.560 |   0.537 |

Pooled numbers are dominated by the 59 circular queries — read the
synthetic split for the real signal.

## Latency

| Retriever      | median | p95 | mean | max (cold start) |
|----------------|-------:|----:|-----:|-----------------:|
| hybrid         |  389ms | 490ms |  ~440ms | ~few sec |
| + rerank       | 2851ms | 2955ms | ~2.9s warm | ~27 min (model load) |

Reranker overhead: **+2462ms median**, ~7.3× slower per query. Almost
all of that is the CrossEncoder forward pass over 30 pairs on CPU; the
hybrid retrieve itself is unchanged. Cold start (first call) is much
worse on this dev host (Windows, 4GB GPU, small pagefile) — model
warm-up and Windows page-fault thrashing dominate. On a beefier host
with the cross-encoder on GPU, expect ≪ 1s/query.

## Reading the result

The reranker does what Task 2.04 was designed to do: it pulls the right
chunk *higher* on queries where the bi-encoder retriever already had it
in the shortlist but couldn't pick the best one (the 60% near-miss
class). On the only split that isn't gamed by construction, R@5 +2.4,
MRR@10 +3.9, nDCG@10 +2.5 — small but consistent gains.

The R@10 drop is the trade-off: when the reranker sees an ambiguous
query it sometimes pushes a borderline gold below 10. That's inherent
to any reranker that doesn't preserve base-retriever order — it can
help the *ranking* of correct hits while occasionally sacrificing the
*set* of top-10 hits.

The latency is the real cost. 2.9s/query is workable for a research
RAG (LLM call dominates total response time) but punishing for any
high-throughput use. Two levers if we want to keep the gains and cut
latency: smaller shortlist (e.g. 15 instead of 30 ⇒ ~halve the cross-
encoder cost) or move the cross-encoder to GPU.

## Next steps

- **Smaller shortlist sweep.** 15-vs-30-vs-50: does quality plateau
  at 15? If so, half the cost.
- **Reranker on a real model.** Cross-encoders trained for the AU
  housing domain, or `BAAI/bge-reranker-v2-m3` once we move off this
  dev host, should narrow the latency gap by improving R@5 enough
  that we can drop top-k.
- **End-to-end RAG re-eval.** The retrieval-only metrics here don't
  capture whether the LLM citation quality improves — the actual user-
  visible signal is whether grounded answers cite the right chunks.
