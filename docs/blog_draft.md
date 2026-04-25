# CadastreAI blog draft

Working draft of the public write-up for CadastreAI. Each H2 is a section
of the final post; sections get filled in as the project progresses.
This file is deliberately verbose — the final post will trim.

---

## Baseline & Problems

With a corpus (41,959 chunks across 10 Australian housing publishers —
AHURI, RBA, Productivity Commission, PropTrack, NHFIC, SQM, APRA,
CoreLogic, Treasury, Grattan) and a 100-query eval set in place, the
first question to answer is: how far does a naive RAG get us, and where
does it break?

### The pipeline

Nothing clever yet. Queries are encoded with `BAAI/bge-base-en-v1.5`
using the canonical query prefix (`"Represent this sentence for searching
relevant passages: "`), Qdrant returns the top-10 chunks by cosine
similarity against the same model's embeddings of every chunk, and
those chunks are stuffed into a single Claude Sonnet 4 prompt with
`<source id="N">` wrappers. The model is told to cite inline as
`[1][2]` and to refuse if the sources don't contain the answer.

The eval set is 100 queries: 60 with human-picked gold from BM25
candidate pools (to avoid circularity with the BGE retriever we're
evaluating) and 40 synthetic queries each written *for* a randomly
sampled chunk stratified across the 9 publishers. Personas are roughly
balanced: 24 homebuyer, 25 investor, 51 researcher. Gold is binary
relevance — each chunk is either "a correct answer for this query"
or not.

### Headline numbers

| Metric   | Overall | Researcher (51) | Homebuyer (24) | Investor (25) |
|----------|---------|-----------------|----------------|----------------|
| R@5      | 0.383   | 0.529           | 0.292          | 0.173          |
| R@10     | 0.450   | 0.556           | 0.403          | 0.280          |
| MRR@10   | 0.352   | 0.435           | 0.329          | 0.206          |
| nDCG@10  | 0.349   | 0.450           | 0.299          | 0.193          |

Researcher queries fare best (the corpus is heavy on academic and
regulatory text, which is also the register BGE was trained on), and
investor queries fare worst — nearly triple the recall gap between
researcher and investor at rank 5. Forty percent of queries returned
no gold chunk at all in the top 10.

### Where it breaks: a 6-category taxonomy

Reviewing 20 of the 40 zero-recall queries surfaced six distinct
failure patterns. They're ranked here by share of the sample.

**1. Near-miss within target document (35%).** BGE retrieves the right
AHURI / RBA / NHFIC paper but picks a neighbouring chunk instead of the
annotated gold paragraph. *"What methodology do AHURI researchers use
to model landlord behaviour?"* — BGE lands on the 2025 landlord-modelling
paper, but returns the introduction rather than the methods section. A
cross-encoder reranker sees the full chunk text and can pick the right
paragraph.

**2. Topic-cluster miss — adjacent document (25%).** Retrieved chunks
are in the right topic cluster but from a different paper than the
gold. Depreciation on investment property → AHURI income-tax-treatment
of housing assets chunks instead of the changing-institutions or
incentivising-small-scale-investors chunks. Reranking helps, but harder
cases here need query expansion that names the specific regulatory
concept (e.g. "Division 43 depreciation").

**3. Abstraction / genre mismatch (15%).** Consumer-facing questions
land on academic policy chunks. *"What upfront and ongoing costs beyond
the purchase price should a first home buyer budget for?"* → AHURI
international policy review on first-home-buyer schemes and a
community-land-trusts chunk. The gold (NHFIC stamp-duty-reform,
PropTrack budget-bonuses) lives in consumer-facing publishers that BGE
down-weights because their chunks read less like academic prose. Persona-
aware publisher boosting should fix most of these.

**4. Temporal mismatch (10%).** Queries with "current", "recent", "past
decade" — BGE ignores the time signal. *"Is it financially better to
keep renting or buy a first home in Sydney given current prices and
rates?"* returns three 2008 SQM Research blog posts from a 17-year-old
bear-market commentary. Fix: a temporal phrase parser + recency boost,
or a date filter over the candidate set.

**5. Multi-concept decomposition needed (10%).** The query has 2–3
conjuncts; BGE matches one. *"Grants or stamp duty concessions for
newly built homes versus existing dwellings"* retrieves the
first-home-buyer cluster but misses the new-build-vs-existing contrast.
Query decomposition with union-then-rerank is the standard answer;
ColBERT-style late interaction would also do it.

**6. Abbreviation / program disambiguation (5%).** Australian housing
scheme acronyms collide. *"How does the Family Home Guarantee help
single parent first home buyers?"* — BGE conflates FHG (Family Home
Guarantee) with FHLDS (First Home Loan Deposit Scheme); both are
children of the HGS umbrella, both have NHFIC trends-and-insights
reports, and their embeddings sit close together. Fix: a small glossary
of scheme acronyms piped into query rewriting.

### What the errors tell us about the Phase 2 plan

The first two categories together account for 60% of failures and both
yield to the same technique: cross-encoder reranking over BGE's top-20.
A reranker sees full chunk text and scores each `(query, chunk)` pair
directly, which is exactly what these near-misses need. That's the
highest-ROI move and the first Phase 2 ticket.

Beyond that, three items compound on top of a reranker:
- **Persona-aware source boosting** to fix genre mismatch. With 24
  homebuyer / 25 investor / 51 researcher labels we already have enough
  signal to learn per-persona publisher priors.
- **Temporal phrase parsing** and a recency boost when the query
  implies "current" or "past N years."
- **Acronym glossary for query expansion** — low-effort, high-precision
  fix for a narrow but annoying failure mode.

Query decomposition is the biggest lift and probably waits until later.

### What the baseline doesn't tell us yet

Two caveats worth flagging.

First, 25% of queries have 0 < R@10 < 1 — partial recall, not
included in the failure review above. Those queries usually have two
gold chunks and BGE found one of them; the interesting question is
whether the missing gold is annotation noise or a real retrieval miss.
Spot-check during Phase 2.

Second, the error analysis revealed at least one gold-annotation that
looks wrong: *"How do long-term capital growth rates compare between
detached houses, units and townhouses?"* has Productivity Commission
housing-construction as gold, but BGE's rank-1 hit
(`rba/2015 long-run-trends-in-housing-price-growth`) is a stronger
answer. Eval sets get better over time; we'll batch these back into
the annotation set before we start claiming absolute numbers.

---

## Week 2: closing the retrieval gap

The Phase 1 baseline left a clear punch list: dense BGE missed 60% of
near-target queries (right doc, wrong chunk or right cluster, wrong
doc), 25% on genre/temporal/decomposition issues, and the rest on
acronym collisions. Week 2 worked through the highest-ROI items in
order: **(1)** add lexical signal with hybrid retrieval, **(2)** add a
cross-encoder reranker, **(3)** fine-tune the bi-encoder on the
domain.

### Two splits, two stories

Before reading numbers, the eval set has a known wart: 59 of the 100
queries had gold hand-picked from BM25's top-10 candidate lists, so any
BM25-driven retriever scores near-perfectly on those by construction.
The remaining 41 are synthetic — each query was authored *for* a
randomly sampled chunk, with no retriever involvement, so the gold is
independent of every retriever. **All Week 2 wins/losses below are
read off the synthetic 41**; pooled 100-query numbers are kept around
for tracking but are dominated by the circular split.

### Step 1 — Hybrid (RRF over BGE + BM25)

Reciprocal Rank Fusion over BGE top-30 and BM25 top-30, then trim to
top-10. BM25 contributes the rare-term lifeline (Division 43, FHG,
NASHH) that dense embeddings dilute; BGE keeps the paraphrase
robustness BM25 lacks.

| Retriever (synthetic 41) | R@5 | R@10 | MRR@10 | nDCG@10 |
|--------------------------|----:|-----:|-------:|--------:|
| dense BGE                | 0.756 | 0.829 | 0.602 | 0.656 |
| BM25                     | 0.585 | 0.780 | 0.452 | 0.528 |
| **hybrid (RRF)**         | **0.805** | **0.878** | **0.640** | **0.698** |

Hybrid wins every metric on the honest split: R@5 +4.9 pts and R@10
+4.9 pts over dense BGE, beating BM25 on every metric. Latency stays
in the same regime as dense alone (median ~390ms, p95 ~490ms) since
both retrievers run in parallel and RRF is O(k log k).

### Step 2 — Cross-encoder reranker

Hybrid top-30 → `cross-encoder/ms-marco-MiniLM-L-6-v2` (~22M params,
CPU) scores each `(query, chunk)` pair → resort to top-10. The
reranker sees full chunk text rather than a vector, which is exactly
what the 35% "right doc, wrong chunk" failures need.

| Retriever (synthetic 41) | R@5 | R@10 | MRR@10 | nDCG@10 |
|--------------------------|----:|-----:|-------:|--------:|
| hybrid                   | 0.805 | **0.878** | 0.640 | 0.698 |
| **+ rerank**             | **0.829** | 0.854 | **0.680** | **0.723** |
| Δ                        | +2.4 | −2.4 | +3.9 | +2.5 |

The reranker pulls correct chunks **higher** in the list — R@5 +2.4,
MRR@10 +3.9, nDCG@10 +2.5 — exactly the near-miss class. R@10 drops
2.4 pts because an in-shortlist gold occasionally gets reordered below
position 10; that's inherent to any reranker that doesn't preserve
base-retriever order.

The cost is latency. Median jumps from 390ms (hybrid) to 2851ms
(+rerank) — about 7.3× — almost all of it the cross-encoder forward
pass over 30 pairs on CPU. Workable for a research RAG (the LLM call
still dominates total response time) but punishing for high-throughput
serving. Two levers: smaller shortlist (15 instead of 30, ~halve the
cost) or move the cross-encoder to GPU.

### Step 3 — Fine-tuning the bi-encoder

A reranker fixes near-misses by re-scoring; fine-tuning attacks the
same problem one layer earlier by pushing the right chunks into the
shortlist to begin with. The pipeline (Tasks 2.06–2.13):

1. **Pair generation (2.06).** Claude Haiku read each chunk and
   produced 2–3 plausible search queries — ~4,000 (anchor, positive)
   pairs at ~$3 in API spend. Two passes (homebuyer-style and
   investor-style prompts) so the training set isn't all
   researcher-register.
2. **Filter (2.07).** Drop pairs where BGE cosine of (anchor, positive)
   is too low (Haiku hallucinated) or too high (anchor is just a
   restatement, no learning signal).
3. **Hard negatives (2.08).** For each pair, BM25-mine 50 candidates
   from the corpus, exclude the positive's source document (same-doc
   chunks are too topically close to be useful contrastive signal),
   keep the top BM25 scorers as hard negatives.
4. **Triplets (2.09).** Build `(anchor, positive, neg1..neg5)`
   records; top-5 hardest negatives per anchor.
5. **Train (2.10–2.12).** Fine-tune `BAAI/bge-base-en-v1.5` with
   `MultipleNegativesRankingLoss` (symmetric InfoNCE) — explicit hard
   negatives plus in-batch negatives. 3 epochs, batch 64, lr 2e-5,
   warmup 10% on a single Colab T4. A 10% dev split feeds a per-epoch
   `RerankingEvaluator` for MAP/MRR@10 curves.
6. **Re-embed and A/B (2.13).** Re-embed the 41,959-chunk corpus and
   upsert to `cadastre_chunks_ft` (parallel to the live
   `cadastre_chunks`). The retriever points at either collection via
   `--collection`, giving a clean A/B harness with no live-traffic
   risk.

**End-to-end pipeline:**

| Variant (synthetic 41) | R@5 | R@10 | MRR@10 | nDCG@10 |
|------------------------|----:|-----:|-------:|--------:|
| base BGE               | 0.756 | 0.829 | 0.602 | 0.656 |
| + hybrid               | 0.805 | 0.878 | 0.640 | 0.698 |
| + hybrid + rerank      | 0.829 | 0.854 | 0.680 | 0.723 |
| ft + hybrid + rerank   | *(pending Task 2.14)* | | | |

`docs/figures/recall_curves.png` plots Recall@K for K=1..10 across
every variant; `results/persona_breakdown.md` compares per-persona
deltas (homebuyer / investor / researcher) so we can read which user
class the fine-tune actually helps.

### What changed about how we evaluate

Two pieces of tooling came out of Week 2 that are useful beyond this
ablation:

- **`src.eval.persona_breakdown`** — diff two retrieval-eval JSONs by
  persona, emit a Markdown table plus a "biggest lift" pointer. Useful
  whenever an intervention might not lift uniformly across user types.
- **`src.eval.ablation`** — aggregate per-variant retrieval-eval
  JSONs into one summary table with end-to-end deltas. Designed to
  gracefully skip variants whose JSON doesn't exist yet, so the table
  still renders while the next experiment is in flight.

### Reading Week 2

On the synthetic split, the BGE → +hybrid → +rerank chain moved R@5
from 0.756 to 0.829 (+7.3 pts) and MRR@10 from 0.602 to 0.680
(+7.8 pts). Hybrid is the bigger of the two contributions; the
reranker pulls about a third of the total lift but at 7× the latency.
Fine-tune numbers land separately — the training script is checked in
and the corpus re-embed orchestrator is ready, but the actual training
run waits on Colab compute.

The two unsolved failure classes from Phase 1 — temporal queries and
multi-concept decomposition — are still unsolved. Both are explicitly
Week 3+ territory: temporal needs a phrase parser and recency boost
on the retrieval side, decomposition wants either query rewriting or
ColBERT-style late interaction. Neither is a fine-tuning problem.

---

## [Future sections — Week 3 onwards]

- Agentic extensions: when to call tools vs when to retrieve
- End-to-end answer quality: citation faithfulness, coverage, hallucinations
