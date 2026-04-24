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

## [Future sections — Week 2 onwards]

- Reranking results: how much does BGE + reranker close the gap?
- Fine-tuning the retriever on our domain: does it beat reranking?
- Agentic extensions: when to call tools vs when to retrieve
- End-to-end answer quality: citation faithfulness, coverage, hallucinations
