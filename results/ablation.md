# Task 2.16 — Retriever ablation (queries_all.jsonl, n=100)

All variants scored against the same 100-query gold set (`data/eval/queries_all.jsonl`). Bold indicates the best value per metric across measured rows.

| variant | n | R@5 | R@10 | MRR@10 | nDCG@10 | median lat | notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| base BGE (dense) | 100 | 0.383 | 0.450 | 0.352 | 0.349 | — |  |
| BM25 only | 100 | **0.800** | **0.907** | **0.732** | **0.760** | — |  |
| base BGE + BM25 (hybrid) | 100 | 0.563 | 0.747 | 0.626 | 0.585 | 389 ms |  |
| hybrid + cross-encoder | 100 | 0.550 | 0.673 | 0.560 | 0.537 | 2851 ms |  |
| fine-tuned BGE (dense) | 100 | 0.177 | 0.210 | 0.128 | 0.146 | 38 ms | offline matmul; Qdrant unavailable |
| ft + hybrid + cross-encoder | 100 | 0.617 | 0.770 | 0.614 | 0.611 | 585 ms | Task 2.16 measured (PR #116) |

## What jumps out

- **BM25 alone is the strongest measured variant** — R@10=0.91, MRR@10=0.73 — well ahead of every dense or hybrid configuration. The eval queries lean heavily on named entities ("First Home Owner Grant", "Home Guarantee Scheme", specific publishers) and BM25's lexical bias matches that distribution.
- **Hybrid hurts BM25.** Fusing dense BGE into BM25 via RRF *drops* R@10 from 0.91 → 0.75. The dense retriever is pulling well-ranked BM25 hits down. Worth revisiting the RRF shortlist + k_rrf parameters, or weighting BM25 higher in the fusion.
- **Cross-encoder rerank also regresses** on top of hybrid (R@10 0.75 → 0.67, median latency 0.4s → 2.9s). On this eval set, the reranker is reordering the BM25-favoured passages downward. Revisit the reranker model / shortlist.
- **Fine-tuned BGE collapses** (R@10 0.45 → 0.21) — see Task 2.14/2.15 reports for the publisher-collapse failure mode.

## ft + hybrid + cross-encoder — surprise upset (Task 2.16, PR #116)

The last cell finally landed once Docker was back up. Process:
`reembed_finetuned --skip-embed --collection cadastre_chunks_ft`
upserted the existing 41,959-vector `.npy` to a new Qdrant collection;
`retrieval_eval --retriever hybrid --rerank --dense-collection cadastre_chunks_ft`
ran the eval.

**Result:** R@10=0.770, beating both base hybrid (0.747) and base
hybrid+rerank (0.673) on the pooled n=100 set. R@5=0.617 also clears
both. The publisher-collapsed FT dense vectors *on their own* are
useless (R@10=0.21), but slotting them into the hybrid stack *plus*
the cross-encoder rescues them: BM25 + RRF gives the shortlist
diverse coverage, the cross-encoder rerank then over-rules the FT
dense ranks where they're wrong, and net the FT contribution is
slightly positive on R@10 (vs base hybrid+rerank).

### Honest synthetic split (n=40, independent gold)

For parity with `hybrid_comparison.md`'s synthetic-only table:

| variant                       | R@5      | R@10     | MRR@10   | nDCG@10  |
|-------------------------------|----------|----------|----------|----------|
| base hybrid (from PR #110)    | 0.805    | **0.878**| 0.640    | 0.698    |
| ft + hybrid + cross-encoder   | **0.850**| 0.850    | **0.711**| **0.746**|

On the honest split, ft+hybrid+rerank wins R@5 / MRR@10 / nDCG@10 but
trails base hybrid on R@10 by 0.028. The reranker is tightening the
top-5 (good for the search-result UX) at a small cost to recall at
K=10. For the agent default we keep base hybrid (R@10 is what the
synth has to work with after `_dedupe_and_cap_chunks(cap=8)`); for a
search-results UI the ft+hybrid+rerank stack is worth a follow-up.

## Implications for Task 2.17 / 2.18

- The Recall@K curve plot (Task 2.17) should put BM25 on top so the curves tell the right story; the previous mental model of "hybrid > BM25 > dense" doesn't hold on this eval set.
- The Week 2 blog section (Task 2.18) should be honest about the fine-tune regression and the BM25/hybrid/rerank inversion. The interesting story is *why* BM25 dominates on this corpus, not a clean monotonic ablation, plus the late-breaking ft+hybrid+rerank result that shows the FT vectors aren't dead weight when the stack absorbs them.
- Latency for hybrid+rerank dropped from 2851 ms (Phase 2 first pass, slow CPU) to 585 ms on the FT collection — same eval, faster machine. Don't compare the two latencies directly across rows.
