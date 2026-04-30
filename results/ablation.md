# Task 2.16 — Retriever ablation (queries_all.jsonl, n=100)

All variants scored against the same 100-query gold set (`data/eval/queries_all.jsonl`). Bold indicates the best value per metric across measured rows.

| variant | n | R@5 | R@10 | MRR@10 | nDCG@10 | median lat | notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| base BGE (dense) | 100 | 0.383 | 0.450 | 0.352 | 0.349 | — |  |
| BM25 only | 100 | **0.800** | **0.907** | **0.732** | **0.760** | — |  |
| base BGE + BM25 (hybrid) | 100 | 0.563 | 0.747 | 0.626 | 0.585 | 389 ms |  |
| hybrid + cross-encoder | 100 | 0.550 | 0.673 | 0.560 | 0.537 | 2851 ms |  |
| fine-tuned BGE (dense) | 100 | 0.177 | 0.210 | 0.128 | 0.146 | 38 ms | offline matmul; Qdrant unavailable |
| ft + hybrid + cross-encoder | — | — | — | — | — | — | not run — needs Qdrant collection cadastre_chunks_ft |

## What jumps out

- **BM25 alone is the strongest measured variant** — R@10=0.91, MRR@10=0.73 — well ahead of every dense or hybrid configuration. The eval queries lean heavily on named entities ("First Home Owner Grant", "Home Guarantee Scheme", specific publishers) and BM25's lexical bias matches that distribution.
- **Hybrid hurts BM25.** Fusing dense BGE into BM25 via RRF *drops* R@10 from 0.91 → 0.75. The dense retriever is pulling well-ranked BM25 hits down. Worth revisiting the RRF shortlist + k_rrf parameters, or weighting BM25 higher in the fusion.
- **Cross-encoder rerank also regresses** on top of hybrid (R@10 0.75 → 0.67, median latency 0.4s → 2.9s). On this eval set, the reranker is reordering the BM25-favoured passages downward. Revisit the reranker model / shortlist.
- **Fine-tuned BGE collapses** (R@10 0.45 → 0.21) — see Task 2.14/2.15 reports for the publisher-collapse failure mode.

## Implications for Task 2.17 / 2.18

- The Recall@K curve plot (Task 2.17) should put BM25 on top so the curves tell the right story; the previous mental model of "hybrid > BM25 > dense" doesn't hold on this eval set.
- The Week 2 blog section (Task 2.18) should be honest about the fine-tune regression and the BM25/hybrid/rerank inversion. The interesting story is *why* BM25 dominates on this corpus, not a clean monotonic ablation.
- The `ft + hybrid + cross-encoder` cell is left unmeasured; once Docker is back up we can run `reembed_finetuned --skip-embed` to upsert the existing `.npy` and then re-run `retrieval_eval --retriever hybrid --rerank --dense-collection cadastre_chunks_ft`. Realistic expectation given the publisher collapse upstream: it won't rescue the FT model.
