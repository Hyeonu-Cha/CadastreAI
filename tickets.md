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
- [ ] **Task 1.18** — Implement `src/ingest/chunk.py` — recursive heading-aware chunking (500–800 tokens, 100-token overlap)
- [ ] **Task 1.19** — Attach chunk metadata (`publisher, title, date, section_heading, page, url, chunk_id`) and emit `data/processed/chunks.jsonl` (target 30–80k chunks)

### Day 4 — Baseline indexing

- [ ] **Task 1.20** — Implement `src/index/embed.py` — embed all chunks with `BAAI/bge-base-en-v1.5` via sentence-transformers (GPU if available)
- [ ] **Task 1.21** — Upsert embeddings + metadata payload into Qdrant collection `cadastre_chunks`
- [ ] **Task 1.22** — Implement `src/retrieval/retriever.py` exposing `retrieve(query, k=10)` returning `(chunk, score)` tuples
- [ ] **Task 1.23** — Write sanity-check script (`scripts/sanity_retrieve.py`) running 5 manual queries and printing top-5 results

### Day 5 — Eval set construction

- [ ] **Task 1.24** — Author 20 homebuyer-style eval queries (practical, local, decision-oriented)
- [ ] **Task 1.25** — Author 20 investor-style eval queries (numeric, comparative, yield/growth)
- [ ] **Task 1.26** — Author 20 researcher-style eval queries (methodology, cross-report, definitional)
- [ ] **Task 1.27** — Manually annotate 2–5 gold chunks per query by inspecting corpus; store in `data/eval/queries.jsonl` (`{query, persona, gold_chunk_ids}`)
- [ ] **Task 1.28** — Generate 40 additional synthetic queries via Claude over random chunks, spot-check, and merge into eval set (~100 total)

### Day 6 — Baseline RAG end-to-end + metrics

- [ ] **Task 1.29** — Build naive RAG pipeline: retrieve top-5 → stuff into Claude Sonnet 4.5 prompt → generate answer with inline citations
- [ ] **Task 1.30** — Implement `src/eval/retrieval_eval.py` computing Recall@5, Recall@10, MRR, nDCG@10 against gold set
- [ ] **Task 1.31** — Run baseline eval and save metrics to `results/baseline.json`; commit results file

### Day 7 — Error analysis + blog section

- [ ] **Task 1.32** — Review 20 failure cases; categorize into error taxonomy (jargon mismatch, temporal, abbreviations, numeric-needs-tools)
- [ ] **Task 1.33** — Write "Baseline & Problems" section in `docs/blog_draft.md` including the error taxonomy

---

## WEEK 2 — Hybrid Retrieval + Embedding Fine-Tuning

### Day 8 — Hybrid search

- [ ] **Task 2.01** — Implement BM25 index over the same chunk set (`rank_bm25` or Qdrant sparse vectors)
- [ ] **Task 2.02** — Implement Reciprocal Rank Fusion combining top-50 BM25 + top-50 dense in `src/index/hybrid.py`
- [ ] **Task 2.03** — Re-run retrieval eval on hybrid pipeline; record metrics vs baseline

### Day 9 — Reranking

- [ ] **Task 2.04** — Integrate `BAAI/bge-reranker-v2-m3` in `src/retrieval/rerank.py`; pipeline: retrieve top-30 → rerank → return top-5
- [ ] **Task 2.05** — Re-run eval with reranker; record latency overhead and updated metrics

### Day 10 — Training data generation

- [ ] **Task 2.06** — Implement `src/training/generate_pairs.py` — prompt Claude per chunk to produce 2 persona-tagged queries; target ~4,000 pairs
- [ ] **Task 2.07** — Quality-filter pairs: drop cosine < 0.3 (unrelated) and > 0.95 (trivial) using baseline embeddings; save `data/training/pairs.jsonl`

### Day 11 — Hard negative mining

- [ ] **Task 2.08** — Implement `src/training/mine_hard_negatives.py` — for each (query, positive), retrieve top-20 via BM25, filter out positives and same-section chunks
- [ ] **Task 2.09** — Keep 5 hardest negatives per query; emit `data/training/triplets.jsonl` (`{anchor, positive, negatives[5]}`)

### Day 12 — Fine-tune embedding model

- [ ] **Task 2.10** — Implement `src/training/train_embeddings.py` with `MultipleNegativesRankingLoss` (batch 64, lr 2e-5, 3 epochs, warmup 10%, AdamW)
- [ ] **Task 2.11** — Hold out 10% of pairs for training-time validation; log loss + val metrics
- [ ] **Task 2.12** — Train on single consumer GPU / Colab T4; save checkpoint `models/bge-au-housing-v1/` with training curves

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

- [ ] **Task 3.01** — Implement `src/tools/rba_stats.py::rba_cash_rate(period)` pulling from RBA F1.1
- [ ] **Task 3.02** — Extend `rba_stats.py` with `rba_mortgage_rates(period)` from F6
- [ ] **Task 3.03** — Implement `src/tools/abs_stats.py::abs_property_price_index(capital_city, period)` (cat. 6432.0)
- [ ] **Task 3.04** — Extend `abs_stats.py` with `abs_building_approvals(state, period)` (cat. 8731.0) and `abs_lending_indicators` (cat. 5601.0)
- [ ] **Task 3.05** — Implement `src/tools/sqm.py::sqm_rental_vacancy(postcode_or_city)` via SQM chart-data scraping
- [ ] **Task 3.06** — Implement `src/tools/chart.py::render_chart(series_dict, title)` producing base64 PNG via matplotlib
- [ ] **Task 3.07** — Implement numeric helpers in `src/tools/compute.py`: `compute_rental_yield`, `compute_mortgage_repayment`, `compute_stamp_duty_nsw`
- [ ] **Task 3.08** — Standardize all tool returns as `{data, source, retrieved_at, citation}` and define Pydantic schemas for each

### Day 16 — LangGraph agent skeleton

- [ ] **Task 3.09** — Define agent state (`messages, query_type, sub_questions, retrieved_chunks, tool_results, answer_draft, reflection, iteration_count`) in `src/agent/graph.py`
- [ ] **Task 3.10** — Implement node stubs in `src/agent/nodes.py`: `classify_query`, `decompose`, `retrieve_or_tool`, `reflect`, `synthesize`
- [ ] **Task 3.11** — Wire LangGraph edges with conditional routing + `max_iterations=4` safety cap
- [ ] **Task 3.12** — Smoke-test happy path on 3 queries and export graph diagram

### Day 17 — Query decomposition + routing

- [ ] **Task 3.13** — Implement `classify_query` with structured output `{persona, needs_docs, needs_data, needs_decomposition}`
- [ ] **Task 3.14** — Implement `decompose` producing 2–4 atomic sub-questions when flagged complex
- [ ] **Task 3.15** — Implement per-subquestion router deciding doc retrieval vs tool calls vs both
- [ ] **Task 3.16** — Integrate LangSmith or Phoenix tracing and validate on 10 diverse queries

### Day 18 — Self-reflection + citation discipline

- [ ] **Task 3.17** — Implement `reflect` node checking coverage of sub-questions, citation presence, and gaps
- [ ] **Task 3.18** — Add loop-back edge feeding refined queries when reflection flags gaps (respecting iteration cap)
- [ ] **Task 3.19** — Enforce strict citation format `[source:publisher, page:N]` / `[tool:name, retrieved:date]` via prompt + post-processor

### Day 19 — Agent evaluation harness

- [ ] **Task 3.20** — Author `data/eval/agent_queries.jsonl` with 30 complex multi-step queries annotated with expected tool calls and doc sources
- [ ] **Task 3.21** — Implement `src/eval/agent_eval.py` measuring tool-call accuracy, trajectory efficiency, faithfulness (RAGAS), groundedness (% numeric claims cited)
- [ ] **Task 3.22** — Run agent eval and save `results/agent_v1.json`

### Day 20 — Agent error analysis + iteration

- [ ] **Task 3.23** — Analyze 15 agent failures; categorize (over-decomposition, tool confusion, citation drops)
- [ ] **Task 3.24** — Fix top 2–3 issues via prompt tuning or graph restructuring; re-run eval to `results/agent_v2.json`

### Day 21 — Buffer + blog

- [ ] **Task 3.25** — Catch up on any slipped Week 3 tasks
- [ ] **Task 3.26** — Write Week 3 agent section in `docs/blog_draft.md` with decomposition + reflection examples

---

## WEEK 4 — Polish, Deploy, Write-up

### Day 22 — Persona UX

- [ ] **Task 4.01** — Scaffold Streamlit app `src/app/streamlit_app.py` with persona selector (Homebuyer / Investor / Researcher / Just exploring)
- [ ] **Task 4.02** — Wire persona into system prompt tone + retrieval weighting + tool-selection hints
- [ ] **Task 4.03** — Render citations as clickable chips with sidebar source-excerpt panel
- [ ] **Task 4.04** — Implement disambiguation flow for ambiguous place names (e.g., "Newtown")

### Day 23 — Trace & transparency UI

- [ ] **Task 4.05** — Add collapsible "Reasoning steps" panel showing sub-questions, tool calls, tool results
- [ ] **Task 4.06** — Display top-K retrieved chunks with relevance scores in trace panel
- [ ] **Task 4.07** — Embed `render_chart` PNGs inline with source caption underneath
- [ ] **Task 4.08** — Add suggested follow-up question chips (3 per answer)

### Day 24 — Caching + cost optimization

- [ ] **Task 4.09** — Implement disk-based embedding cache keyed on chunk hash
- [ ] **Task 4.10** — Implement tool-result cache with TTL (24h for market data)
- [ ] **Task 4.11** — Enable Anthropic prompt caching (`cache_control`) on system prompt + retrieved context
- [ ] **Task 4.12** — Instrument cost-per-query logging and report before/after in `README.md`

### Day 25 — Deployment

- [ ] **Task 4.13** — Write production `Dockerfile` bundling app + dependencies
- [ ] **Task 4.14** — Provision Qdrant Cloud free tier; slim corpus if >1GB by keeping top-priority publishers only
- [ ] **Task 4.15** — Deploy to Modal or HuggingFace Spaces with secrets configured
- [ ] **Task 4.16** — Smoke-test public URL with 5 canonical queries across personas

### Day 26 — Final eval report

- [ ] **Task 4.17** — Run full eval suite on deployed pipeline (retrieval + agent + latency + cost)
- [ ] **Task 4.18** — Produce headline results table (Baseline / +Hybrid+Rerank / +Fine-tuned / Full Agent) and save `results/final_metrics.json`

### Day 27 — README polish

- [ ] **Task 4.19** — Record UI demo GIF and add to README hero section
- [ ] **Task 4.20** — Write problem statement, mermaid architecture diagram, quickstart (`docker compose up`), results table, tech choices in `README.md`

### Day 28 — Blog post finalization

- [ ] **Task 4.21** — Consolidate `docs/blog_draft.md` into final post (problem → baseline → retrieval engineering → agent → eval → lessons)
- [ ] **Task 4.22** — Publish to personal blog + cross-post (Medium, LinkedIn, r/LocalLLaMA, r/MachineLearning, r/AusFinance)

### Day 29 — Demo video + social

- [ ] **Task 4.23** — Record 3-minute Loom walkthrough (one query per persona with reasoning trace)
- [ ] **Task 4.24** — Embed video in README, blog, LinkedIn post

### Day 30 — Buffer + retrospective

- [ ] **Task 4.25** — Write private retrospective (what worked, cut, surprised)
- [ ] **Task 4.26** — Clean up branches, tag `v1.0.0`, flip repo to public

---

## Cross-cutting / Ongoing

- [ ] **Task X.01** — Maintain `progress.md` with status + notes per completed ticket
- [ ] **Task X.02** — Keep `.env.example` in sync with any new secrets (ANTHROPIC_API_KEY, QDRANT_URL, etc.)
- [ ] **Task X.03** — Enforce persistent "not financial advice" disclaimer in system prompt (proportional by persona)
- [ ] **Task X.04** — Guardrails: block financial-product recommendations (mortgage/insurance); log refusals
