# CadastreAI — Ticket List

> Detailed implementation tickets derived from `project_plan.md` and `product.md`.
> Workflow per ticket: create branch `Development/Task{N}{NN}` → implement → update `progress.md` → commit → push.
> ID format: `Task W.NN` where `W` = week number, `NN` = zero-padded sequence.

---

## WEEK 1 — Foundation, Ingestion, Baseline RAG

### Day 1 — Repo setup + source discovery

- [x] **Task 1.01** — Initialize Python project (`uv init` or `poetry init`), create `pyproject.toml`, add `.env.example`, set Python 3.11+
- [x] **Task 1.02** — Create repository directory skeleton (`src/ingest`, `src/index`, `src/training`, `src/retrieval`, `src/tools`, `src/agent`, `src/eval`, `src/app`, `data/{raw,processed,eval,training}`, `notebooks/`, `scripts/`, `docs/`)
- [x] **Task 1.03** — Write `docker-compose.yml` with Qdrant service (+ optional Phoenix tracing)
- [x] **Task 1.04** — Implement `src/ingest/scrapers/rba.py` — discover all RDP PDF links from `rba.gov.au/publications/rdp/`, filter by housing keywords (housing, dwelling, property, mortgage, rent)
- [x] **Task 1.05** — Extend RBA scraper to also cover Bulletin housing articles and Financial Stability Review (FSR) PDFs
- [x] **Task 1.06** — Implement `src/ingest/scrapers/ahuri.py` — scrape AHURI final reports index at `ahuri.edu.au/research/final-reports`
- [x] **Task 1.07** — Implement `src/ingest/scrapers/grattan.py` — scrape Grattan Institute housing topic page
- [x] **Task 1.08** — Define `sources.jsonl` schema (`{title, publisher, date, url, category}`) and merge all scraper outputs into a single `data/raw/sources.jsonl` with ~100 entries

### Day 2 — Remaining scrapers + bulk download

- [x] **Task 1.09** — Implement `src/ingest/scrapers/corelogic.py` — scrape CoreLogic/Cotality news & research index
- [x] **Task 1.10** — Implement `src/ingest/scrapers/sqm.py` — scrape SQM Research monthly reports index
- [x] **Task 1.11** — Implement `src/ingest/scrapers/domain_proptrack.py` — scrape Domain research + PropTrack insights quarterly reports
- [x] **Task 1.12** — Implement `src/ingest/scrapers/treasury_pc.py` — Treasury + Productivity Commission housing pages
- [x] **Task 1.13** — Implement `src/ingest/scrapers/nhfic_apra.py` — NHFIC/Housing Australia + APRA quarterly stats (P2)
- [x] **Task 1.14** — Write polite concurrent downloader (`src/ingest/download.py`) with rate limiting, retries, resume-on-failure; output `data/raw/{publisher}/{yyyy-mm}_{slug}.pdf`
- [x] **Task 1.15** — Run bulk download targeting 250–350 PDFs (~2–5 GB) and verify counts per publisher

### Day 3 — Parsing + chunking

- [x] **Task 1.16** — Install `docling`; implement `src/ingest/parse.py` converting PDFs → markdown with headings and tables preserved
- [x] **Task 1.17** — Add `pymupdf4llm` fallback path for docs that fail `docling`; log failures
- [x] **Task 1.18** — Implement `src/ingest/chunk.py` — recursive heading-aware chunking (500–800 tokens, 100-token overlap)
- [x] **Task 1.19** — Attach chunk metadata (`publisher, title, date, section_heading, page, url, chunk_id`) and emit `data/processed/chunks.jsonl` (final corpus: 41,959 chunks)

### Day 4 — Baseline indexing

- [x] **Task 1.20** — Implement `src/index/embed.py` — embed all chunks with `BAAI/bge-base-en-v1.5` via sentence-transformers (GPU if available)
- [x] **Task 1.21** — Upsert embeddings + metadata payload into Qdrant collection `cadastre_chunks`
- [x] **Task 1.22** — Implement `src/retrieval/retriever.py` exposing `retrieve(query, k=10)` returning `(chunk, score)` tuples
- [x] **Task 1.23** — Write sanity-check script (`scripts/sanity_retrieve.py`) running 5 manual queries and printing top-5 results

### Day 5 — Eval set construction

- [x] **Task 1.24** — Author 20 homebuyer-style eval queries (practical, local, decision-oriented) — `data/eval/queries_homebuyer.jsonl`
- [x] **Task 1.25** — Author 20 investor-style eval queries (numeric, comparative, yield/growth) — `data/eval/queries_investor.jsonl`
- [x] **Task 1.26** — Author 20 researcher-style eval queries (methodology, cross-report, definitional) — `data/eval/queries_researcher.jsonl`
- [x] **Task 1.27** — Manually annotate 2–5 gold chunks per query by inspecting corpus; store in `data/eval/queries.jsonl` (`{query, persona, gold_chunk_ids}`)
- [x] **Task 1.28** — Generate 40 additional synthetic queries via Claude over random chunks, spot-check, and merge into eval set — `data/eval/queries_synth.jsonl` + `queries_all.jsonl` (100 total)

### Day 6 — Baseline RAG end-to-end + metrics

- [x] **Task 1.29** — Build naive RAG pipeline: retrieve top-5 → stuff into Claude prompt → generate answer with inline citations
- [x] **Task 1.30** — Implement `src/eval/retrieval_eval.py` computing Recall@5, Recall@10, MRR, nDCG@10 against gold set
- [x] **Task 1.31** — Run baseline eval and save metrics to `results/baseline.json`; commit results file

### Day 7 — Error analysis + blog section

- [x] **Task 1.32** — Review 20 failure cases; categorize into error taxonomy (jargon mismatch, temporal, abbreviations, numeric-needs-tools)
- [x] **Task 1.33** — Write "Baseline & Problems" section in `docs/blog.md` including the error taxonomy

---

## WEEK 2 — Hybrid Retrieval + Embedding Fine-Tuning

### Day 8 — Hybrid search

- [x] **Task 2.01** — Implement BM25 index over the same chunk set — `src/index/bm25.py` (`rank_bm25`, pickled to `data/processed/bm25.pkl`)
- [x] **Task 2.02** — Implement Reciprocal Rank Fusion combining top-50 BM25 + top-50 dense in `src/index/hybrid.py`
- [x] **Task 2.03** — Re-run retrieval eval on hybrid pipeline; record metrics vs baseline — see `results/hybrid_comparison.md`

### Day 9 — Reranking

- [x] **Task 2.04** — Integrate cross-encoder reranker in `src/retrieval/rerank.py`; pipeline: retrieve top-30 → rerank → return top-5. *Used `cross-encoder/ms-marco-MiniLM-L-6-v2` (smaller, faster) instead of originally-planned `bge-reranker-v2-m3` — sufficient quality at meaningfully lower latency.*
- [x] **Task 2.05** — Re-run eval with reranker; record latency overhead and updated metrics

### Day 10 — Training data generation

- [x] **Task 2.06** — Implement `src/training/generate_pairs.py` — prompt Claude per chunk to produce persona-tagged queries
- [x] **Task 2.07** — Quality-filter pairs via `src/training/filter_pairs.py`; save `data/training/pairs.jsonl`

### Day 11 — Hard negative mining

- [x] **Task 2.08** — Implement `src/training/mine_hard_negatives.py` — retrieve top-20 via BM25, filter out positives and same-section chunks
- [x] **Task 2.09** — Keep 5 hardest negatives per query; emit `data/training/triplets.jsonl` via `src/training/build_triplets.py`

### Day 12 — Fine-tune embedding model

- [x] **Task 2.10** — Implement `src/training/train_embeddings.py` with `MultipleNegativesRankingLoss`
- [x] **Task 2.11** — Hold out validation split; log loss + val metrics
- [x] **Task 2.12** — Train on consumer GPU; save checkpoint `models/bge-au-housing-v1/`

### Day 13 — Re-embed + benchmark

- [x] **Task 2.13** — Re-embed entire corpus with fine-tuned model and upsert to new Qdrant collection `cadastre_chunks_ft`
- [x] **Task 2.14** — Run full retrieval eval on fine-tuned embeddings; target MRR +15–25% over base BGE; save `results/finetuned.json`
- [x] **Task 2.15** — Produce per-category breakdown (homebuyer/investor/researcher) to identify where fine-tune helps most

### Day 14 — Ablation + blog

- [x] **Task 2.16** — Build ablation comparison (base BGE / +hybrid / +hybrid+rerank / ft+hybrid+rerank); emit `results/ablation.json` + Markdown table
- [x] **Task 2.17** — Plot Recall@K curves for all variants; save to `docs/figures/recall_curves.png`
- [x] **Task 2.18** — Write Week 2 section of `docs/blog_draft.md` covering hybrid, rerank, fine-tuning story

---

## WEEK 3 — Agentic Layer

### Day 15 — Tool implementations

- [x] **Task 3.01** — Implement `src/tools/rba_stats.py::rba_cash_rate(period)` pulling from RBA F1.1
- [x] **Task 3.02** — Extend `rba_stats.py` with `rba_mortgage_rates(period)` from F6
- [x] **Task 3.03** — Implement `src/tools/abs_stats.py::abs_property_price_index(capital_city, period)` (cat. 6432.0)
- [x] **Task 3.04** — Extend `abs_stats.py` with `abs_building_approvals(state, period)` (cat. 8731.0) and `abs_lending_indicators` (cat. 5601.0)
- [x] **Task 3.05** — Implement `src/tools/sqm.py::sqm_rental_vacancy(postcode_or_city)` via SQM chart-data scraping
- [x] **Task 3.06** — Implement `src/tools/chart.py::render_chart(series_dict, title)` producing base64 PNG via matplotlib
- [x] **Task 3.07** — Implement numeric helpers in `src/tools/compute.py`: `compute_rental_yield`, `compute_mortgage_repayment`, `compute_stamp_duty_nsw`
- [x] **Task 3.08** — Standardize all tool returns as `{data, source, retrieved_at, citation}` and define Pydantic schemas for each (`src/tools/schemas.py`)

### Day 16 — LangGraph agent skeleton

- [x] **Task 3.09** — Define agent state (`messages, query_type, sub_questions, retrieved_chunks, tool_results, answer_draft, reflection, iteration_count`) in `src/agent/graph.py`
- [x] **Task 3.10** — Implement node stubs in `src/agent/nodes.py`: `classify_query`, `decompose`, `retrieve_or_tool`, `reflect`, `synthesize`
- [x] **Task 3.11** — Wire LangGraph edges with conditional routing + `max_iterations=4` safety cap (later tightened to 2 in Task 3.27)
- [x] **Task 3.12** — Smoke-test happy path on 3 queries and export graph diagram (`src/agent/smoke.py`)

### Day 17 — Query decomposition + routing

- [x] **Task 3.13** — Implement `classify_query` with structured output `{persona, needs_docs, needs_data, needs_decomposition}`
- [x] **Task 3.14** — Implement `decompose` producing 2–4 atomic sub-questions when flagged complex
- [x] **Task 3.15** — Implement per-subquestion router deciding doc retrieval vs tool calls vs both
- [x] **Task 3.16** — Integrate LangSmith or Phoenix tracing and validate on 10 diverse queries (`src/agent/tracing.py`)

### Day 18 — Self-reflection + citation discipline

- [x] **Task 3.17** — Implement `reflect` node checking coverage of sub-questions, citation presence, and gaps
- [x] **Task 3.18** — Add loop-back edge feeding refined queries when reflection flags gaps (respecting iteration cap)
- [x] **Task 3.19** — Enforce strict citation format `[source:publisher, page:N]` / `[tool:name, retrieved:date]` via prompt + post-processor

### Day 19 — Agent evaluation harness

- [x] **Task 3.20** — Author `data/eval/agent_queries.jsonl` with 30 complex multi-step queries annotated with expected tool calls and doc sources
- [x] **Task 3.21** — Implement `src/eval/agent_eval.py` measuring tool-call accuracy, trajectory efficiency, faithfulness (RAGAS), groundedness (% numeric claims cited)
- [x] **Task 3.22** — Run agent eval and save `results/agent_v1.json`

### Day 20 — Agent error analysis + iteration

- [x] **Task 3.23** — Analyze 15 agent failures; categorize (over-decomposition, tool confusion, citation drops)
- [x] **Task 3.24** — Fix top 2–3 issues via prompt tuning or graph restructuring; re-run eval to `results/agent_v2.json` (PR #106)

### Day 21 — Buffer + blog

- [x] **Task 3.25** — Catch up on any slipped Week 3 tasks (hybrid retriever wired as agent default)
- [x] **Task 3.26** — Write Week 3 agent section in `docs/blog_draft.md` with decomposition + reflection examples (now `docs/blog.md` Phase 3)
- [x] **Task 3.27** — Agent v2: tighten reflect/synth prompts, drop `MAX_ITERATIONS` 4 → 2 (PR #106)
- [x] **Task 3.28** — Agent v3 eval against hybrid retriever; route-conditional analysis (PR #112)
- [x] **Task 3.29** — Agent v4: citation-adjacency rules + chunk cap with dedupe (PR #114)
- [x] **Task 3.30** — Agent v5: compute-tool routing disambiguation + worked examples (PR #117)

---

## WEEK 4 — Polish, Deploy, Write-up

### Day 22 — Persona UX

- [x] **Task 4.01** — Scaffold Streamlit app `src/app/streamlit_app.py` with persona selector (Homebuyer / Investor / Researcher / Just exploring)
- [x] **Task 4.02** — Wire persona into system prompt tone + retrieval weighting + tool-selection hints
- [x] **Task 4.03** — Render citations as clickable chips with sidebar source-excerpt panel
- [x] **Task 4.04** — Implement disambiguation flow for ambiguous place names (e.g., "Newtown")

### Day 23 — Trace & transparency UI

- [x] **Task 4.05** — Add collapsible "Reasoning steps" panel showing sub-questions, tool calls, tool results
- [x] **Task 4.06** — Display top-K retrieved chunks with relevance scores in trace panel
- [x] **Task 4.07** — Embed `render_chart` PNGs inline with source caption underneath (`src/app/streamlit_app.py::_render_citation_card`, PR #131)
- [x] **Task 4.08** — Add suggested follow-up question chips (3 per answer)

### Day 24 — Caching + cost optimization

- [x] **Task 4.09** — Implement disk-based embedding cache keyed on chunk hash (`src/retrieval/embedding_cache.py`)
- [x] **Task 4.10** — Implement tool-result cache with TTL (24h for market data, 7d for compute, 5min for charts)
- [x] **Task 4.11** — Enable Anthropic prompt caching (`cache_control`) on system prompt + retrieved context
- [x] **Task 4.12** — Instrument cost-per-query logging and report before/after in `README.md` (`src/agent/cost.py`)

### Day 25 — Deployment

- [x] **Task 4.13** — Write production `Dockerfile` bundling app + dependencies (multi-stage; bm25.pkl + pre-warmed encoder cache baked in via PRs #121, #122)
- [x] **Task 4.14** — Provision Qdrant Cloud (GCP australia-southeast1); `cadastre_chunks` collection upserted with 41,959 points, JWT-auth verified
- [x] **Task 4.15** — Deploy to HuggingFace Spaces (Docker SDK) at `https://huggingface.co/spaces/ericcha901/cadastreai`; deploy branch `Development/Task415-hf-deploy` carries bm25.pkl via Git LFS; secrets wired (ANTHROPIC_API_KEY, QDRANT_URL, QDRANT_API_KEY)
- [x] **Task 4.16** — Public URL live (HTTP 200, Streamlit `_stcore/health=ok`); 5-query smoke against the deploy config: 3/5 PASS on content + latency, 2/5 produced correct cited answers but breached the 45s hard latency budget on local CPU (HF Spaces basic-CPU latency tracked separately)

### Day 26 — Final eval report

- [x] **Task 4.17** — Final eval against deployed config: retrieval re-run on n=40 synth split (`results/synth_{dense,bm25,hybrid,hybrid_rerank}.json`) and agent v6 (n=30, judge on) at `results/agent_v6_final.json` — faithfulness 0.664 → 0.793, tool-call accuracy 0.913 → 0.980
- [x] **Task 4.18** — Headline results table consolidated into `results/final_metrics.json`; README/`docs/report_v2.md` refreshed with current numbers

### Day 27 — README polish

- [ ] **Task 4.19** — Record UI demo GIF and add to README hero section *(README hero pre-staged with one-line swap-in — PR #127; needs recording)*
- [x] **Task 4.20** — Write problem statement, architecture diagram, quickstart (`docker compose up`), results table, tech choices in `README.md` (uses `architecture.svg` rather than mermaid)

### Day 28 — Blog post finalization

- [x] **Task 4.21** — Consolidate `docs/blog_draft.md` into final post (now `docs/blog.md`; problem → baseline → retrieval engineering → agent → eval → lessons)
- [ ] **Task 4.22** — Publish to personal blog + cross-post (Medium, LinkedIn, r/LocalLLaMA, r/MachineLearning, r/AusFinance) *(per-channel drafts ready in `docs/launch_posts.md` — PR #124; needs publishing + placeholders filled)*

### Day 29 — Demo video + social

- [ ] **Task 4.23** — Record 3-minute Loom walkthrough (one query per persona with reasoning trace)
- [ ] **Task 4.24** — Embed video in README, blog, LinkedIn post

### Day 30 — Buffer + retrospective

- [ ] **Task 4.25** — Write private retrospective (what worked, cut, surprised) *(skeleton in `docs/retrospective.md` — PR #126; needs filling in)*
- [ ] **Task 4.26** — Clean up branches, tag `v1.0.0`, flip repo to public

---

## WEEK 5 — Regime-change remediation (2026 housing tax reform)

> Trigger: Treasury Laws Amendment (Tax Reform No. 1) Act 2026, enacted 26 Jun 2026.
> Blocks a *clean* `v1.0.0`; see `docs/regime_change_gap_analysis.md` for findings F-1..F-10.
> **Interim disclosure shipped ahead of v1.0.0** (banner, README caveat, synthesizer currency caveat, investor-persona reword, `compute.py` vintage consistency). The items below are the full post-launch remediation.

### Day 31 — Stop the bleeding (eval + disclosure)

- [x] **Task 5.01** — Add a `regime` (`pre_2026_reform` | `post_2026_reform` | `regime_neutral`) field to the eval schema (`src/eval/regime.py`); backfill the regime-dependent queries in `queries_all.jsonl` + `agent_queries.jsonl` (F-3). *`as_of` and the secondary files (candidates/synth/persona) deferred — absent field defaults to `regime_neutral`, so they simply count as headline until tagged.*
- [x] **Task 5.02** — Audited the tax-dependent queries (5 quarantined: retrieval NG after-tax cash flow / 12-month CGT cliff / abolishing-NG modelling; agent agent-010 + agent-026) and excluded them from headline metrics in `src/eval/retrieval_eval.py` and `agent_eval.py`. *Deviation: tagged in place + harness filters (with `legacy_regime` / `by_regime` / `*_including_legacy` reported) rather than moving to a separate file — no data duplication, one flag to flip. Covered by `tests/test_eval_regime.py`.* (F-3)
- [ ] **Task 5.03** — Re-word the three malformed queries whose *premise* is now false (`queries_all.jsonl` "abolishing"; the 12-month CGT-cliff item; `agent_queries.jsonl:26` "proposed reforms") and re-annotate gold chunks — **blocked on F-1 ingestion** (no post-reform chunks exist yet to re-point at); exclusion via 5.02 already removes their headline harm (F-3)
- [x] **Task 5.04** — Interim disclosure: corpus-vintage banner in `src/app/streamlit_app.py` + README §Disclaimer caveat stating the corpus predates the 2026 reform (F-6) — *done in this PR; supersede when 5.01–5.14 land.*

### Day 32 — Primary-source ingestion

- [ ] **Task 5.05** — Implement `src/ingest/scrapers/ato.py` — ATO new-legislation guidance, filtered to housing/CGT/rental topics (F-1)
- [ ] **Task 5.06** — Implement `src/ingest/scrapers/legislation.py` — Federal Register of Legislation + APH bills pages; capture Act text and explanatory memoranda (F-1)
- [ ] **Task 5.07** — Close the known Treasury scraper gap (`progress.md:485`) — follow `/publications?topic=...` pages out of the `/policy-topics/housing` hub (F-1)
- [ ] **Task 5.08** — Add Budget-paper ingestion (2026-27 BP1/BP2 housing + revenue measures) to `treasury_pc.py` (F-1)
- [ ] **Task 5.09** — Extend chunk metadata with `regime` and `supersedes` / `superseded_by`; backfill and re-upsert (F-1, F-2)

### Day 33 — Date-aware retrieval

- [ ] **Task 5.10** — Normalise the free-form `date` payload to a sortable ISO value at index time (F-2)
- [ ] **Task 5.11** — Add a configurable recency prior to `src/index/hybrid.py` RRF scoring; off for `regime_neutral`, on for policy/tax (F-2)
- [ ] **Task 5.12** — Implement supersession handling in `src/retrieval/retriever.py`: force a `superseded_by` successor into context and demote the predecessor (F-2)
- [ ] **Task 5.13** — Add `as_of` / `regime` filter params to `retrieve()` and thread through `retrieve_or_tool` (F-2, F-10)
- [ ] **Task 5.14** — Regression-test date-aware retrieval against the `current_regime` split; target post-reform gold chunk in top-3 for all tax queries (F-2, F-3)

### Day 34 — Prompt stack + guardrails

- [ ] **Task 5.15** — Inject current date into classifier / decomposer / router / reflector / synthesizer prompts in `src/agent/nodes.py` (F-5)
- [ ] **Task 5.16** — Surface chunk publication date and `regime` in `_summarise_chunks_for_synth` evidence blocks (F-5)
- [ ] **Task 5.17** — Amend `_REFLECTOR_SYSTEM` to admit "all retrieved evidence predates a known regime change" as a nameable gap (F-5)
- [ ] **Task 5.18** — Add a `temporal_currency` guardrail category to `src/agent/guardrails.py`; action is **preamble injection**, not refusal (needs a new `GuardrailAction` "annotate") (F-7)
- [~] **Task 5.19** — Regime-aware investor persona + currency caveat. *Partially done in this PR: reworded the investor addendum in `persona.py` and added a CURRENCY clause to `_DISCLAIMER_BASELINE`. Remaining: fuller `_PERSONA_DISCLAIMER` treatment.* (F-6)
- [ ] **Task 5.20** — Add `acquisition_date` and `is_new_build` to `AgentState` + `Classification`; wire disambiguation in `src/app/disambiguation.py` (F-10)

### Day 35 — Tools

- [~] **Task 5.21** — NSW bracket vintage. *Partially done in this PR: aligned the four conflicting vintage strings in `compute.py` to 2024-25. Remaining: move brackets to a dated config keyed by financial year and refresh to current-year values.* (F-8)
- [ ] **Task 5.22** — Implement `compute_cgt_indexed()` — cost-base indexation for gains from 1 Jul 2027; expose in `src/tools/schemas.py` (F-8)
- [ ] **Task 5.23** — Implement `compute_gearing_position()` — deductible-vs-quarantined split keyed on `acquisition_date` / `is_new_build`; refuse when unknown (F-8, F-10)
- [ ] **Task 5.24** — Structural-break annotation on `rba_stats` / `abs_stats` / `sqm` envelopes when the period spans 12 May 2026 (F-9)

### Day 36 — Re-train + re-baseline

- [ ] **Task 5.25** — Re-run `generate_pairs.py` post-ingestion with per-publisher **and** per-era caps (F-4)
- [ ] **Task 5.26** — Re-run retrieval + agent eval on the corrected split; publish `results/regime_change_v1.md`; refresh README / `docs/report_v2.md` (F-3, F-4)
- [ ] **Task 5.27** — CI check in `.github/workflows/tests.yml` asserting no `pre_2026_reform` query contributes to headline metrics (F-3)

---

## Cross-cutting / Ongoing

- [x] **Task X.01** — Maintain `progress.md` with status + notes per completed ticket (most recent refresh: PR #120)
- [x] **Task X.02** — Keep `.env.example` in sync with any new secrets (ANTHROPIC_API_KEY, QDRANT_URL, etc.)
- [x] **Task X.03** — Enforce persistent "not financial advice" disclaimer in system prompt (proportional by persona)
      - `src/agent/persona.py::persona_disclaimer()` returns a persona-specific DISCLAIMER POLICY block; `nodes.py:1218` appends it to the synthesizer system prompt every turn. Homebuyer/investor get the most explicit "consult a licensed professional" language; researcher/journalist get audience-appropriate caveats; `general` fallback always non-empty. Covered by `tests/test_persona_disclaimer.py` (passing).
- [x] **Task X.04** — Guardrails: block financial-product recommendations (mortgage/insurance); log refusals
      - `src/agent/guardrails.py::screen_query()` is a pure pattern-matching screen across 4 categories (mortgage_product, insurance_product, super_or_managed_fund, specific_security_pick). `guardrail_screen` is wired as the first node in `graph.py` (START → guardrail_screen → END on refusal, no LLM call on the hot path). Structured WARNING log via `log_refusal()`. Covered by `tests/test_guardrails.py` (passing).
- [ ] **Task X.06** — Regime-change watch: quarterly re-run of `scripts/collect_sources.py` against ATO + legislation scrapers, with a diff report on any chunk whose `regime` tag would change (see `docs/regime_change_gap_analysis.md`)
- [ ] **Task X.07** — Document the corpus `as_of` date prominently in README, `product.md`, and the Streamlit footer; treat "corpus vintage" as a first-class released artifact alongside `final_metrics.json`
