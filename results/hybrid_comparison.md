# Hybrid retrieval vs dense vs BM25

Comparison of three retrievers over the 100-query eval set, top-k=10.
Raw per-query numbers in `baseline.json` (dense), `bm25.json`, `hybrid.json`.

## Why two views

The 100-query eval set has two groups with different provenance:

- **59 BM25-annotated** queries (Tasks 1.24–1.27): human-picked gold
  from BM25 top-10 candidate lists.
- **41 synthetic** queries (Task 1.28): each has a single gold chunk,
  and the query was authored *for* that chunk, with no retriever
  involvement.

The BM25-annotated split is **circular** for pure BM25 — the gold was
literally drawn from BM25's top-10 — so BM25 scores near-perfectly on
that split by construction. The synthetic split is the honest head-to-
head: the gold is independent of every retriever.

## Numbers

### Overall (pooled, for reference only)

| Retriever | R@5 | R@10 | MRR@10 | nDCG@10 |
|-----------|-----|------|--------|---------|
| dense     | 0.383 | 0.450 | 0.352 | 0.349 |
| bm25      | 0.800 | 0.907 | 0.732 | 0.760 |
| **hybrid** | 0.563 | 0.747 | 0.626 | 0.585 |

The BM25 and hybrid overall numbers are inflated by the circular split;
these row is only here so someone tracking `results/baseline.json`
numbers can see how the same metric reads.

### Synthetic split (41 queries, honest — independent gold)

| Retriever | R@5 | R@10 | MRR@10 | nDCG@10 |
|-----------|-----|------|--------|---------|
| dense     | 0.756 | 0.829 | 0.602 | 0.656 |
| bm25      | 0.585 | 0.780 | 0.452 | 0.528 |
| **hybrid** | **0.805** | **0.878** | **0.640** | **0.698** |

**Hybrid wins every metric on the honest split.** R@5 +5 pts and R@10
+5 pts over dense; more than BM25 alone on every metric. This is the
signal that matters for Phase 2.

### BM25-annotated split (59 queries, circular — BM25 should win)

| Retriever | R@5 | R@10 | MRR@10 | nDCG@10 |
|-----------|-----|------|--------|---------|
| dense     | 0.124 | 0.186 | 0.178 | 0.136 |
| bm25      | 0.949 | 0.994 | 0.927 | 0.922 |
| hybrid    | 0.395 | 0.655 | 0.616 | 0.506 |

BM25 R@10 = 0.994 is a consistency check: it confirms the annotation
step preserved BM25's ranking. Dense numbers on this split are
misleadingly low because the annotators chose lexically-salient chunks
— the precise failure mode we were trying to diagnose. Hybrid recovers
most of that gap.

## Reading the result

On the only split that isn't cooked in someone's favour — the synthetic
40 — hybrid beats BGE alone. Dense R@10 of 0.83 was already high on
synth queries (BGE handles paraphrased semantic queries well); the
extra 5 points from RRF fusion comes from cases where the question uses
a rare domain term (Division 43, FHG, NASHH, NFIP) that dominates BM25
but is diluted in dense embeddings.

## Next steps

- **Fix the eval set circularity.** Add a `source: {"annotation" |
  "synthetic"}` field to `queries_all.jsonl` so metrics are reported
  per-split by default, not by heuristic gold-size inference.
- **Reranker on top of hybrid (Task 2.04).** A cross-encoder over
  hybrid top-30 should pick off the intra-document near-misses that
  made up 35% of dense failures; RRF doesn't help with those because
  the gold chunk simply isn't in either shortlist when the query is
  genuinely ambiguous at the sub-document level.
