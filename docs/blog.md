# CadastreAI: building a domain-tuned RAG agent for the Australian housing market

> Four weeks. One corpus of ~42,000 chunks across ten Australian housing
> publishers. One agent that has to answer first-home-buyer, investor,
> researcher, and journalist questions with cited evidence — and an
> honest accounting of where it works and where it doesn't.

## Why this exists

Most public RAG demos answer questions about *generic* corpora —
Wikipedia, arXiv, the company handbook. The interesting failure modes
only show up when the corpus is **specialised**, the queries are
**personal**, and the user can sanity-check the output. Australian
residential property fits all three: every claim is checkable against
RBA cash rate data, ABS lending indicators, or the actual paper from
AHURI / NHFIC / Productivity Commission, and a wrong answer costs
someone real money.

The bet is that a domain-tuned retriever + a structured agent + a
disciplined citation contract beats off-the-shelf chat for this kind of
question. The four weeks below test that bet end-to-end:

1. **Phase 1 — baseline and a failure taxonomy.** Naive dense RAG, then
   an honest categorisation of where it breaks.
2. **Phase 2 — retrieval engineering.** Hybrid + reranker + (planned)
   bi-encoder fine-tune, ablated against the baseline.
3. **Phase 3 — from RAG to agent.** A LangGraph agent with reflection,
   live-data tools, and a citation post-processor.
4. **Phase 4 — serving.** Streamlit UI, persona wiring, prompt caching,
   per-query cost telemetry, and a production container.

What follows is what the work actually looked like, what the numbers
were on each step, and which problems are still open.

---

## Phase 1 — Baseline and a six-category failure taxonomy

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

## Phase 2 — Closing the retrieval gap

The Phase 1 baseline left a clear punch list: dense BGE missed 60% of
near-target queries (right doc, wrong chunk or right cluster, wrong
doc), 25% on genre/temporal/decomposition issues, and the rest on
acronym collisions. Phase 2 worked through the highest-ROI items in
order: **(1)** add lexical signal with hybrid retrieval, **(2)** add a
cross-encoder reranker, **(3)** fine-tune the bi-encoder on the
domain.

### Two splits, two stories

Before reading numbers, the eval set has a known wart: 59 of the 100
queries had gold hand-picked from BM25's top-10 candidate lists, so any
BM25-driven retriever scores near-perfectly on those by construction.
The remaining 41 are synthetic — each query was authored *for* a
randomly sampled chunk, with no retriever involvement, so the gold is
independent of every retriever. **All Phase 2 wins/losses below are
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

**End-to-end pipeline (synthetic 41 split):**

| Variant (synthetic 41) | R@5 | R@10 | MRR@10 | nDCG@10 |
|------------------------|----:|-----:|-------:|--------:|
| base BGE               | 0.756 | 0.829 | 0.602 | 0.656 |
| + hybrid               | 0.805 | 0.878 | 0.640 | 0.698 |
| + hybrid + rerank      | 0.829 | 0.854 | 0.680 | 0.723 |
| ft + hybrid + rerank   | *not run — see below* | | | |

**End-to-end pipeline (full 100-query eval set):**

| Variant (n=100)              | R@5 | R@10 | MRR@10 | nDCG@10 | median lat |
|------------------------------|----:|-----:|-------:|--------:|----------:|
| base BGE (dense)             | 0.383 | 0.450 | 0.352 | 0.349 | — |
| BM25 only                    | **0.800** | **0.907** | **0.732** | **0.760** | — |
| BGE + BM25 (hybrid, RRF)     | 0.563 | 0.747 | 0.626 | 0.585 | 389 ms |
| hybrid + cross-encoder       | 0.550 | 0.673 | 0.560 | 0.537 | 2851 ms |
| fine-tuned BGE (dense)       | 0.177 | 0.210 | 0.128 | 0.146 | 38 ms |

Two splits, two stories. On the synthetic-41 split (designed to be
retriever-independent), the hybrid + rerank chain monotonically
improves over base BGE — the original Phase 2 thesis. On the full
100-query set (which folds in the 59 BM25-pooled queries), **BM25
alone dominates every measured variant**, the dense retriever
*regresses* it under RRF, and the cross-encoder pulls it down further.

This isn't "the synthetic split is wrong and the full split is right"
— each measures something different. The synthetic split tests how
well a retriever generalises to queries it had no hand in producing.
The full split tests how the system performs on the eval set we
actually shipped, BM25-pooling and all. Phase 2 wins on the first;
the second is what surfaces when you stop hiding behind your own
sampling distribution.

`docs/figures/recall_curves.png` plots Recall@K for K=1..10 across
every measured variant on the full 100-query set; the curves are
parallel to the table above (BM25 on top, FT on the floor).
`results/finetune_persona_breakdown.md` slices the FT regression by
persona — homebuyer −76% MRR, investor −71% (zero query-level wins),
researcher −58%.

### Step 3.5 — Reading the fine-tune regression

The fine-tune was meant to be the headline win of Phase 2. It wasn't.
Across every metric, persona, and almost every individual query, the
fine-tuned `bge-au-housing-v1` checkpoint is *worse* than the base
encoder — MRR@10 0.352 → 0.128 on the full eval set, R@10 0.450 →
0.210. On dev metrics during training (held-out 10% of the synthetic
triplets) MRR@10 hit 0.79; on the real eval queries that signal
disappeared.

The failure mode is consistent across queries. The FT model retrieves
AHURI papers for almost every question, including ones where AHURI
has no relevant content. A Victoria-stamp-duty query pulls "changing
geography of homelessness", "filtering as a source of low-income
housing", and "demand-side assistance in Australia's rental market" —
all AHURI working papers, all wrong. The model has collapsed onto the
publisher style that dominates the training distribution.

Three plausible upstream causes (in order of likely impact):

1. **Pair-generation distribution skew.** AHURI is the largest single
   publisher in the chunk corpus, so synthetic pairs sampled from
   chunks were AHURI-heavy. The fine-tune learns "good answers look
   like AHURI."
2. **Synthetic-anchor style ≠ real query style.** Anchors generated
   by an LLM from a passage tend to read as academic paraphrases. Real
   eval queries (`"How does the First Home Owner Grant work in NSW
   and who qualifies?"`) are conversational. The model fits the
   training register.
3. **Catastrophic forgetting.** 3,943 pairs at lr=2e-5 for 3 epochs
   over a narrow distribution is enough to overwrite the broad
   retrieval prior the base BGE encodes.

The next experiments, in rough cost order: (a) stratified per-publisher
pair sampling so AHURI doesn't dominate; (b) bias the LLM
pair-generation prompt toward natural questions instead of paraphrases;
(c) lower learning rate and fewer epochs (lr=5e-6, 1 epoch); (d) a
small held-out portion of the eval queries themselves as supplementary
training pairs (handle the contamination split carefully); (e) skip
encoder fine-tuning and fine-tune a cross-encoder reranker instead —
better signal-to-noise per training example.

The `ft + hybrid + rerank` row in the table is left explicitly "not
run" rather than silently dropped: it needs Qdrant for the
`cadastre_chunks_ft` collection (Docker was unavailable for this
write-up). Once it runs, the realistic expectation given the
upstream collapse is that downstream BM25 fusion and the reranker
won't rescue the FT signal — `ft + hybrid + rerank` will likely sit
between `+ hybrid + rerank` and `fine-tuned BGE (dense)`.

### What changed about how we evaluate

Two pieces of tooling came out of Phase 2 that are useful beyond this
ablation:

- **`src.eval.persona_breakdown`** — diff two retrieval-eval JSONs by
  persona, emit a Markdown table plus a "biggest lift" pointer. Useful
  whenever an intervention might not lift uniformly across user types.
- **`src.eval.ablation`** — aggregate per-variant retrieval-eval
  JSONs into one summary table with end-to-end deltas. Designed to
  gracefully skip variants whose JSON doesn't exist yet, so the table
  still renders while the next experiment is in flight.

### Reading Phase 2

Two readings, depending on which split you trust. On the synthetic
41-query split — designed to be independent of the BM25 pooling that
produced 59 of the gold sets — the BGE → +hybrid → +rerank chain moves
R@5 from 0.756 to 0.829 (+7.3 pts) and MRR@10 from 0.602 to 0.680
(+7.8 pts). Hybrid is the bigger contribution; the reranker pulls
about a third of the total lift at 7× the latency.

On the full 100-query set, the reading inverts. **BM25 alone wins by
a wide margin** (R@10 0.91, MRR 0.73), and adding dense BGE via RRF
*regresses* BM25 to R@10 0.75 — the dense retriever is pulling
correct lexical hits down in the fused ranking. The cross-encoder
rerank regresses again to R@10 0.67 at 7× the latency. The
fine-tuned dense encoder collapses to R@10 0.21.

The honest takeaway: hybrid + rerank produces real gains on
queries that aren't already in BM25's wheelhouse, but on this
specific eval set BM25's wheelhouse covers most of the ground.
Two follow-ups for Phase 3+:

- **Re-tune the fusion.** RRF with equal weighting is the simplest
  thing that could have worked; on this eval set it doesn't. Worth
  exploring `k_rrf` values, BM25-weighted RRF, or a learned linear
  combination on a held-out tune split.
- **Diagnose the rerank regression.** The cross-encoder
  (`ms-marco-MiniLM-L-6-v2`) was trained on web-search-style data;
  on Australian housing-policy passages it may be miscalibrated.
  Worth trying a domain-adjacent reranker or a domain-tuned one
  (Phase 2's failed encoder fine-tune story argues for fine-tuning
  a cross-encoder *instead* — better signal-to-noise per training
  example).

The two unsolved failure classes from Phase 1 — temporal queries and
multi-concept decomposition — are still unsolved. Both are explicitly
Phase 3 territory: temporal needs a phrase parser and recency boost on
the retrieval side, decomposition wants either query rewriting or
ColBERT-style late interaction. Neither is a retrieval-stack tuning
problem.

---

## Phase 3 — From RAG to agent

The Phase 2 chain (BGE → hybrid → rerank, optionally fine-tuned) closed
most of the *retrieval* gaps from Phase 1. Two of the original six
failure classes survive every retrieval improvement we threw at them:

- **Temporal queries.** *"Is renting still cheaper than buying in
  Sydney given current rates?"* — there is no chunk in the corpus that
  contains today's cash rate, today's median price, and a buy-vs-rent
  computation. Even a perfect retriever can't synthesise an answer
  from data the corpus doesn't contain.
- **Multi-concept decomposition.** *"How have ABS lending indicators
  for first home buyers and the RBA cash rate co-moved over the past
  decade?"* — gold lives in *two* sources that need to be cross-
  referenced. A single shortlist returns either the lending series or
  the rate, not both.

Both call for an agent: something that can hit live data sources
(RBA / ABS / SQM rate-and-volume APIs), break compound questions into
sub-questions, and reflect on whether the evidence actually answers
the user. Phase 3 builds that agent on top of the Phase 2 retriever.

### The graph

The agent is a LangGraph `StateGraph` with five nodes and one
conditional edge:

```
classify_query → decompose → retrieve_or_tool → synthesize → reflect ⇄ retrieve_or_tool
                                                                       ↘ END
```

Each node returns a partial state; the typed `AgentState` carries
`messages`, `classification`, `sub_questions`, `retrieved_chunks`,
`tool_results`, `iteration_count`, `reflection`, and `answer_draft`
through the run. The reflect→retrieve loop is gated by
`_route_after_reflect` with four exit conditions (cap reached, marked
complete, marked incomplete with no `refined_query`, or empty refined
string) so a confused reflector can never spin forever. `MAX_ITERATIONS
= 4` is the hard ceiling.

Models are chosen for cost-vs-stakes: routing, decomposition, planning,
and reflection all run on `claude-haiku-4-5` (cheap, fast, structured-
output-friendly via forced tool use); synthesis runs on
`claude-sonnet-4-6` because that's where prose quality and citation
discipline matter. Both can be overridden per-node via env vars
(`CADASTRE_REFLECTOR_MODEL`, etc.) for ablation runs.

### Structured output via forced tool use

Every Haiku call uses Anthropic's `tool_choice={"type": "tool", "name":
...}` pattern instead of asking the model to "return JSON in this
shape." The classifier returns a `submit_classification` tool call;
the decomposer returns `submit_subquestions`; the per-question planner
returns `submit_routing_plan` with `{use_docs, doc_query, tool_calls}`;
the reflector returns `submit_reflection` with `{is_complete, missing,
refined_query}`. The Anthropic SDK validates the tool input against the
schema before it ever reaches our code, which kills the "the model
forgot a closing brace" failure mode and lets us treat the tool input
dict as already-typed.

### The tool catalogue

Nine tools, registered in `src.agent.nodes._tool_catalogue()` with
`(callable, required_keys, optional_keys)` triples so the per-question
planner can validate args before calling:

- **Live data (6).** `rba_cash_rate`, `rba_mortgage_rates`,
  `abs_property_price_index`, `abs_building_approvals`,
  `abs_lending_indicators`, `sqm_rental_vacancy`. Each wraps a
  publisher's API or scrape and returns
  `{data, source, retrieved_at}` so downstream synthesis can render the
  `[tool:<name>, retrieved:<YYYY-MM-DD>]` citation grammar.
- **Compute (3).** `compute_rental_yield`,
  `compute_mortgage_repayment`, `compute_stamp_duty_nsw`. Pure
  functions over user-supplied numbers — no network, fully testable.

`retrieve_or_tool` calls a per-question planner (Haiku) that emits a
single `submit_routing_plan` tool call deciding whether to retrieve
docs, call tools, or both. On loop iterations (`iteration_count > 0`)
the planner runs only on the reflector's `refined_query`, so each pass
is scoped to the gap the reflector named — no re-planning the original
question.

### Reflection self-consistency

The reflector's job is to read the current evidence (chunks + tool
results) and decide whether the agent should loop. Self-contradiction
is the failure mode to design against — *"is_complete: true, missing:
[everything], refined_query: still need..."* — so the prompt is
structured to make the contract explicit and the post-processor drops
`refined_query` whenever `is_complete=True`. Combined with
`_route_after_reflect`'s short-circuit when `refined_query` is null /
empty, the loop edge can't be tricked into wasting an iteration.

If the reflector API call fails, the fallback is `is_complete=True`
with a noted error — better to ship the current draft than spin until
the iteration cap.

### Synthesize — citation enforcement post-process

The synthesizer (Sonnet) is prompted with a strict citation grammar:

- `[source:<publisher>, page:<N>]` for retrieved-doc claims, drawn
  from the indexed chunk metadata.
- `[tool:<tool_name>, retrieved:<YYYY-MM-DD>]` for tool-data claims.

Sonnet generally follows it, but "generally" isn't good enough when
the eval scores citation discipline. So `_enforce_citations` runs
after generation:

1. Build the **valid citation set** from current state — only tools
   that actually returned data (failed calls excluded), only
   publishers + pages that appear in `retrieved_chunks`.
2. Walk every emitted `[source:..]` / `[tool:..]` marker. Markers
   that don't resolve to the valid set get tagged `(unverified)` in
   place — *not* deleted, so the eval harness can score how often the
   model hallucinates citations.
3. If Sonnet didn't emit a `Sources:` footer, append one auto-generated
   from the valid citation set so the user-visible answer always shows
   provenance.

If the synthesis call fails, the fallback is a stub answer that lists
"used: <tool/doc>..." rather than nothing — graph termination always
takes priority over output quality.

### Eval: how do you score an agent?

Retrieval has well-known metrics (R@K, MRR, nDCG). Agents do not. The
agent eval harness (`src.eval.agent_eval` against
`data/eval/agent_queries.jsonl`, 30 queries with annotated expected
tools and publishers) scores six things offline plus one optional
network metric:

- **`tool_call_accuracy`** — Jaccard(actual, expected_tools). Plus
  `tool_call_recall` (did we call all the expected tools?) and
  `tool_call_precision` (did we avoid extras?). The Jaccard captures
  both directions in a single number; the split metrics localise where
  a regression came from.
- **`publisher_recall`** — fraction of `expected_doc_publishers` that
  actually appeared in `retrieved_chunks`. Tests retrieval *coverage*,
  not relevance — did the agent at least look at the right corpora?
- **`trajectory_efficiency = 1 / (1 + extra_iters + extra_tools)`** —
  penalises wandering. A perfect run (right tools, one pass) scores
  1.0. Each unexpected tool call OR each extra reflect-loop iteration
  proportionally drops the score. Missing tools are deliberately
  *not* penalised here — that's `tool_call_recall`'s job.
- **`groundedness`** — % of numeric claims in the answer draft that
  have a `[source:..]` or `[tool:..]` citation marker within 50 chars.
  The 50-char window is tight enough that one number's citation can't
  accidentally credit the next number. Citation markers are redacted
  before the numeric regex runs, so the digits inside `[tool:rba_cash_
  rate, retrieved:2026-04-26]` aren't themselves scored as ungrounded.
- **`faithfulness`** *(optional, `--with-judge`)* — Claude-as-judge
  reads `(question, evidence pack, answer)` and submits an
  `{n_claims, n_supported, unsupported[]}` judgement. Off by default
  so the standard eval is fully offline; opt in for end-of-cycle
  scoring runs.

A stubbed-graph test path (`_StubGraph` in `tests/test_agent_eval.py`)
exercises `run_agent_eval` end-to-end without an API key, so the eval
harness itself stays under unit-test pressure even when the live agent
isn't reachable.

### From v1 to v5: five iterations against the same eval set

The agent eval got run five times — `agent_v1.json` through
`agent_v5.json` — with each iteration committing a single targeted
change and re-running the harness. The numbers below are over the
same 30 annotated queries, OpenAI provider (`gpt-4o-mini` for
classify/route/reflect/judge, `gpt-4o` for synth), faithfulness graded
by LLM judge.

| Metric                  | v1     | v2     | v3     | v4     | v5     |
|-------------------------|--------|--------|--------|--------|--------|
| tool_call_accuracy      | 0.894  | 0.919  | 0.925  | 0.917  | 0.913  |
| trajectory_efficiency   | 0.299  | 0.672  | **0.722** | 0.717 | 0.719 |
| faithfulness (judge)    | 0.570  | 0.648  | 0.630  | 0.637  | **0.664** |
| publisher_recall        | 0.789  | 0.778  | 0.764  | 0.772  | **0.825** |
| groundedness (regex)    | 0.373  | 0.296  | 0.240  | **0.368** | 0.316 |

What each iteration changed, and what the diff said:

- **v2 (PR #106): reflect-loop convergence + "use the evidence."**
  v1 had 27/30 queries pegged at the iteration ceiling. Two changes:
  `MAX_ITERATIONS` 4 → 2; the reflect prompt's `is_complete` default
  flipped from "False unless certain" to "True unless there's a
  specific gap." Trajectory efficiency 0.299 → 0.672 (+0.37) — the
  single biggest agent-quality lift in the project. Plus a "USE THE
  EVIDENCE" paragraph in the synth prompt that fixed v1's "I cannot
  provide…" refusals when the tool result above contained the answer.
- **v3 (PR #112): hybrid retriever (Task 3.25 default).** Aggregate
  looked flat. Slicing by `n_retrieved_chunks > 0` showed the truth:
  doc-using queries pulled +19% more chunks (8.93 vs 7.50) and
  faithfulness slipped -0.076 because synth's citation discipline
  cracked under the wider context. Tool-only queries are a no-op for
  the retriever swap. **Aggregate eval can hide route-conditional
  impact.**
- **v4 (PR #114): citation adjacency + chunk cap.** Two surgical
  fixes targeted at v3's doc-using regression. `_CITATION_RULES`
  added "PLACEMENT IS LOAD-BEARING" with a worked GOOD/BAD example;
  `_dedupe_and_cap_chunks(cap=8)` ran after `apply_publisher_boost`
  to undo the multi-sub-question chunk pile-up. Doc-using subset
  groundedness 0.268 → 0.485 (+0.217), faithfulness +0.055.
- **v5 (PR #117): compute-tool routing disambiguation.** `_ROUTER_SYSTEM`
  now says explicitly that `compute_*` tools require user-provided
  numbers, with worked GOOD/BAD examples. Closes the canonical
  `compute_rental_yield` vs market-lookup confusion (agent-019:
  tool_acc 0.25 → 1.00). The worked example also nudged 12 queries
  from tool-only to doc-using; on that switched cohort faithfulness
  jumped +0.110 because the judge approves of doc-supported tool
  answers. The aggregate groundedness regex dip is the same v3-era
  regex/judge mismatch surfacing on a different cohort — a follow-up
  for any future v6, but not gating launch.

The interesting meta-pattern: tool accuracy is mostly a planner-prompt
problem; trajectory efficiency is mostly a reflector-contract problem;
groundedness is mostly a synthesis-prompt problem; faithfulness
reflects all three. Decomposing the dashboard into those four levers
is what makes the agent debuggable — same idea as the six-category
retrieval taxonomy from Phase 1, one layer up the stack.

A second meta-pattern that only became visible across iterations:
**a worked example in a routing prompt is itself a routing change.**
v5's example showed `use_docs=True`, and unrelated queries shifted
into the doc-using regime. Faithfulness on that switched cohort
jumped, but a different metric (groundedness regex) slipped because
adjacent-citation discipline is harder under more chunks. When a
prompt edit nudges the planner's regime, expect downstream metrics
to redistribute — not just the metric you targeted.

---

## Phase 4 — Serving

A research RAG that lives only in a notebook isn't a product. Phase 4
wraps the agent in a UI a real user can drive, and adds the operational
plumbing — caching at three different layers, per-query cost
telemetry, a non-root container — that makes the difference between
"it ran on my machine" and "this could be deployed."

### The UI: persona, citations, and an audit trail

The Streamlit app exposes four things the demo needs to *look* like a
product rather than a chat box:

- **Persona selector.** Four options (Homebuyer / Investor /
  Researcher / Journalist) wired through to the classifier, the
  retriever (publisher-priority boosts per persona), and the
  synthesiser (tone addendums in the system prompt). The classifier's
  guess is a fallback; the user's explicit pick wins.
- **Disambiguation prompt.** Place names like *"Newtown"* or
  *"Richmond"* match multiple suburbs across NSW / VIC / QLD; a small
  qualifier-window detector flags the ambiguity before the agent runs
  and asks the user to pick.
- **Inline citations + a sources panel.** The synthesizer's
  `[source:..]` / `[tool:..]` markers are parsed, deduplicated, and
  renumbered to `[1]…[N]` in the rendered answer. Each citation chip
  is clickable; the sidebar shows the matching chunk excerpt or the
  tool's `retrieved_at` timestamp.
- **Reasoning trace.** A collapsible 5-step timeline (classify →
  decompose → retrieve_or_tool → synthesize → reflect), each step
  marked `ok` / `info` / `warn` so a debugging user can see exactly
  where a wrong tool got picked or a sub-question went off-topic.
- **Suggested follow-ups.** Three persona-flavoured chips per answer,
  built from a small rules table (e.g. `compute_stamp_duty_nsw` → "How
  does VIC compare?"; investor persona → "Which suburbs match this
  yield profile?").

### Caching at three layers

The cost of running this thing live is dominated by two things: LLM
calls and embedding inference. Both are cacheable:

- **Embedding cache (disk-backed).** Sha-256 of `(model, prefix,
  query)` keys a `.npy` on disk. A repeat query — common in eval
  reruns and in the agent's reflection loop — skips the
  sentence-transformers forward pass entirely. Roughly a 1–2 second
  saving per warm query on CPU.
- **Tool-result cache (in-memory, TTL).** Per-tool TTLs reflect how
  often the upstream actually moves: 24h for RBA / ABS / SQM
  (monthly-cadence data), 7d for the deterministic compute tools
  (stamp duty, mortgage repayment, rental yield), 5min for chart
  artefacts. A `skip_on=lambda r: 'error' in r` predicate keeps
  transient upstream blips out of the cache.
- **Anthropic prompt caching.** Every system prompt is wrapped in a
  `cache_control: ephemeral` breakpoint. Render order is tools →
  system → messages, so the breakpoint caches both the tool list and
  the system text together; only the per-call user message stays on
  the hot path. Documented hit yields ~90% input-token discount; the
  cache TTL is 5 minutes by default which comfortably covers a single
  agent run's 5 messages.create() calls.

### Cost telemetry

Every `messages.create()` response is fed through a small `cost.py`
module that computes a `CostBreakdown` across four meters: input,
output, `cache_read_input_tokens` (0.1× input rate), and
`cache_creation_input_tokens` (1.25× input rate). Each call emits a
structured log line —

```
cost node=classify model=claude-haiku-4-5 in=512 out=128 cache_r=0 cache_w=2048 total=$0.000893
```

— that an eval harness or a Streamlit sidebar widget can grep for to
attribute $/query back to nodes. Pricing is a one-table lookup keyed
by `model_id`, so when Anthropic moves prices it's a one-line patch.
The decoupling matters for the cache-hit story: without per-meter
attribution, you can't tell whether a 90% input-token discount is
landing or whether something is silently busting the cache prefix.

### Container

Production runtime is a two-stage Dockerfile:

- **Builder** (`python:3.11-slim` + `build-essential`) compiles wheels
  for the runtime extras (`agent,index,embed,tools,app,chunk`) into a
  venv. The `parse` extras (docling + torch, ~1.5 GB) are deliberately
  skipped — they're only needed for offline ingestion, not the
  runtime.
- **Runtime** (`python:3.11-slim` + `libgomp1`) copies the prebuilt
  venv, runs Streamlit as a non-root `cadastre` user (UID 1000), and
  exposes 8501 with a `/_stcore/health` healthcheck.

A matching `.dockerignore` keeps the build context small (~10 MB) so
source-only changes don't bust the dep cache layer.

---

## Lessons & open threads

What four weeks of this kind of work look like, in retrospect:

**Don't trust the hybrid retrieval consensus on lexical-heavy corpora.**
The Phase 2 numbers say the obvious thing — hybrid + rerank is best —
on the synthetic split. On the *full* 100-query held-out set, BM25
alone (recall@10 = 0.907) beats hybrid (0.747) and reranked (0.673)
because 60% of the queries share heavy lexical overlap with the source
documents. The published "always use hybrid" advice is right on
average; on a corpus where users naturally type the same phrasing the
documents use, it's wrong. The right move would have been to label the
two query-style buckets up front and ablate retrievers per-bucket.

**Forced tool use is the structured-output story.** Every Haiku call
in the agent uses `tool_choice={"type": "tool", "name": ...}` instead
of "return JSON in this shape." The Anthropic SDK validates the tool
input against the schema before it reaches user code, which kills the
*"the model forgot a closing brace"* failure mode entirely. Treating
tool input as already-typed turns out to be a much more durable
contract than retry-and-parse loops.

**Citation discipline is a post-process, not a prompt.** Sonnet
*usually* follows a strict citation grammar when asked. "Usually" is a
bug when the eval scores citation discipline. The post-processor walks
every emitted `[source:..]` / `[tool:..]` marker, reconciles it
against state (only tools that *actually returned data*, only
publishers that *actually appear* in `retrieved_chunks`), and tags
unverified ones in place — *not* deleted, so the eval harness can
still score them. This pattern generalises: any structural property
you want to *measure* should be enforced after generation, not during.

**Reflection loops need a forced exit.** The reflector can confuse
itself into self-contradiction (`is_complete: true, missing:
[everything]`). Four exit conditions in `_route_after_reflect`
(iteration cap, marked complete, marked incomplete with no
`refined_query`, or empty refined string) plus a hard 4-iteration cap
mean a confused reflector can't spin forever. The hard cap is the one
that actually saves you in production — the others can all be tricked
by a stubborn model.

**Cost only matters if it's per-meter.** Aggregate `total_usd` per
query is a vanity number; the input/output/cache_read/cache_write
split is what tells you whether prompt caching is actually working,
whether the agent is over-decomposing (output tokens in the planner
spike), or whether a regression added a per-call timestamp that's
silently busting the cache prefix. The telemetry has to expose the
levers it's measuring.

**The fine-tune isn't dead, it's just lonely.** The Phase 2 BGE
fine-tune collapsed onto a publisher dimension when used as a
standalone dense retriever (R@10 dropped from 0.450 → 0.210). For a
long time the table sat with that as the headline failure. The
deferred ablation cell — `ft + hybrid + cross-encoder` — finally
landed in PR #116 and showed the FT vectors actually beat both base
hybrid (R@10 = 0.747) and base hybrid+rerank (R@10 = 0.673) on the
pooled 100-query set, scoring 0.770. The reranker absorbs the
publisher-collapse and reorders cleanly. Two lessons compound: a
single-component eval can falsely declare a component dead, and a
fine-tune can be useful as a *signal* in a stack even when it's
useless as a *retriever* alone.

**Open threads.** Cloud deployment (Qdrant Cloud + Modal/HF Spaces)
is the remaining hands-on item before the v1.0.0 cut — the local
docker-compose runs cleanly, but the demo URL is what makes the
README clickable for someone arriving cold. Once the cloud upsert
runs, the Phase 4 final-results table (`Tasks 4.17–4.18`) gets one
last refresh and the launch is mechanical.

---

*Code: [github.com/Hyeonu-Cha/CadastreAI](https://github.com/Hyeonu-Cha/CadastreAI).
Architecture diagram and full eval JSONs are in the repo. Built with
Claude (Haiku 4.5 routing, Sonnet 4.6 synthesis), LangGraph, Qdrant,
sentence-transformers, and Streamlit.*
