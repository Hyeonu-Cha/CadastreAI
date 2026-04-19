# 🏠 CadastreAI — AU Housing Market Research Agent
### A 4-Week AI Engineering Project: Domain-Specialized RAG + Fine-tuned Retriever + Agentic Reasoning

---

## 📌 TL;DR

**CadastreAI** is a production-grade AI research agent for the **Australian housing market**, serving three personas (homebuyers, investors, researchers) by combining:

1. **RAG** over ~300 research reports (RBA, CoreLogic/Cotality, AHURI, Grattan, SQM, Domain, PropTrack)
2. **Fine-tuned embedding model** specialized on AU housing terminology (gains demonstrable on held-out eval)
3. **LangGraph agent** with tool-use over live data (RBA stats, ABS, AIHW dashboard)
4. **Evaluation harness** with before/after metrics (Recall@K, MRR, Answer Faithfulness, Agent Trajectory)

**End state**: deployed demo + GitHub repo + technical blog post with quantitative results.

---

## 🎯 Target Users & Example Queries

The agent must handle three personas with different query styles:

### Homebuyer
- "Is now a good time to buy in Parramatta? What's the price trend and rental yield?"
- "Explain stamp duty in NSW for a $900K first home."
- "Compare Blacktown vs Penrith for a family with two kids."

### Investor
- "Which Sydney suburbs had highest capital growth over 3 years with sub-3% vacancy?"
- "What does CoreLogic's latest Pain & Gain report say about loss-making resales?"
- "Current gross rental yield in Brisbane units vs Melbourne units."

### Researcher / Analyst
- "Summarise key findings from the 2024 AHURI report on build-to-rent."
- "What's the methodological difference between CoreLogic's hedonic index and ABS residential property prices?"
- "How has the housing shortage estimate evolved across RBA discussion papers since 2019?"

---

## 🗂️ Data Sources (All Free or Freemium)

### Text Corpus (for RAG — target ~300 docs)

| Source | Content | Access | Priority |
|---|---|---|---|
| **RBA Research Discussion Papers** | Academic-quality housing papers | `rba.gov.au/publications/rdp/` PDF download | P0 |
| **RBA Bulletin articles** | Short quarterly housing analysis | `rba.gov.au/publications/bulletin/` | P0 |
| **RBA Financial Stability Review** | Biannual, dedicated housing chapter | `rba.gov.au/publications/fsr/` | P0 |
| **AHURI Final Reports** | Deep research on AU housing policy | `ahuri.edu.au/research/final-reports` | P0 |
| **CoreLogic/Cotality Monthly Indices & Pain & Gain** | Market commentary + methodology | `corelogic.com.au/news-research` | P0 |
| **Grattan Institute housing reports** | Policy analysis (neg gearing, supply, tax) | `grattan.edu.au/topics/housing/` | P0 |
| **Treasury Intergenerational & Housing papers** | Macro & policy | `treasury.gov.au` | P1 |
| **Productivity Commission housing inquiries** | Long-form policy | `pc.gov.au` | P1 |
| **SQM Research monthly reports** | Vacancy, distressed listings, rents | `sqmresearch.com.au` | P1 |
| **Domain & PropTrack quarterly reports** | Buyer demand, listings | `domain.com.au/research`, `realestate.com.au/insights` | P1 |
| **NHFIC / Housing Australia reports** | First home buyer, social housing | `housingaustralia.gov.au` | P2 |
| **APRA quarterly stats commentary** | Lending standards, LVR, DTI | `apra.gov.au` | P2 |

### Structured Data (for agent's tool-use)

| Source | What | Access |
|---|---|---|
| **RBA Statistical Tables** | Cash rate, mortgage rates, credit aggregates, housing debt/income | CSV download, `rba.gov.au/statistics/tables/` |
| **ABS Residential Property Price Indexes** | Quarterly capital city prices (cat. 6432.0) | `abs.gov.au`, ABS.Stat Beta API |
| **ABS Building Approvals (8731.0)** | Monthly new dwelling approvals | Same |
| **ABS Lending Indicators (5601.0)** | New housing loan commitments | Same |
| **AIHW Housing Data Dashboard** | 36 national housing datasets | `housingdata.gov.au` |
| **SQM Research chart data** | Rental vacancy, asking prices/rents by postcode (CSV) | `sqmresearch.com.au/buychartdata.php` |
| **CoreLogic Home Value Index** | Daily/monthly capital city index | Free summary on website; full data paid |

### Python Helpers
- `rba` package or `raustats` (R → port logic to Python) for RBA/ABS downloads
- Direct `requests` to ABS.Stat API
- BeautifulSoup for scraping PDF links from research pages

---

## 🛠️ Tech Stack

| Layer | Choice | Why |
|---|---|---|
| **LLM (generation)** | Claude Sonnet 4.5 primary, Claude Haiku 4.5 for cheap tasks | Strong reasoning, good citation behaviour |
| **Embedding (fine-tuned)** | `BAAI/bge-base-en-v1.5` base → fine-tuned | 768-dim, good quality/speed balance, widely used |
| **Reranker** | `BAAI/bge-reranker-v2-m3` | Strong cross-encoder, open weights |
| **Vector DB** | Qdrant (local Docker for dev → Qdrant Cloud free tier for prod) | Hybrid search native, good Python SDK |
| **Sparse retrieval** | BM25 via `rank_bm25` or Qdrant native | For hybrid fusion |
| **Agent framework** | LangGraph | Explicit graph = debuggable; better than opaque abstractions |
| **PDF parsing** | `docling` (primary) + `pymupdf4llm` (fallback) | Docling handles tables well — critical for housing reports |
| **Training** | `sentence-transformers` 3.x | Clean API for the loss functions we need |
| **Evaluation** | Custom harness + `ragas` for answer metrics | Most housing-specific eval has to be custom |
| **Tracing** | LangSmith (free tier) or Phoenix (OSS) | See agent trajectories |
| **Frontend** | Streamlit (fast) or Next.js (polished) | Depends on time remaining in Week 4 |
| **Deployment** | Modal or HuggingFace Spaces | Free tier sufficient for demo |

---

## 📁 Repository Structure

```
cadastreai/
├── README.md
├── pyproject.toml                 # uv or poetry
├── .env.example
├── docker-compose.yml             # Qdrant + optional Phoenix
│
├── data/
│   ├── raw/                       # downloaded PDFs (gitignored)
│   ├── processed/                 # parsed JSONL chunks
│   ├── eval/                      # eval set (committed)
│   └── training/                  # fine-tuning triplets (committed)
│
├── src/
│   ├── ingest/
│   │   ├── scrapers/              # one per source (rba.py, ahuri.py, ...)
│   │   ├── parse.py               # PDF → structured chunks
│   │   └── chunk.py               # chunking strategies
│   ├── index/
│   │   ├── embed.py               # embed + upsert to Qdrant
│   │   └── hybrid.py              # BM25 + dense fusion
│   ├── training/
│   │   ├── generate_pairs.py      # synthetic (query, chunk) generation
│   │   ├── mine_hard_negatives.py # BM25-based mining
│   │   └── train_embeddings.py    # sentence-transformers training
│   ├── retrieval/
│   │   ├── retriever.py           # unified retrieve() interface
│   │   └── rerank.py              # bge reranker
│   ├── tools/                     # agent tools
│   │   ├── rba_stats.py
│   │   ├── abs_stats.py
│   │   ├── corelogic_index.py
│   │   └── chart.py               # matplotlib chart generator
│   ├── agent/
│   │   ├── graph.py               # LangGraph state machine
│   │   ├── nodes.py               # decompose, retrieve, reflect, answer
│   │   └── prompts.py
│   ├── eval/
│   │   ├── retrieval_eval.py      # Recall@K, MRR, nDCG
│   │   ├── answer_eval.py         # faithfulness, groundedness
│   │   └── agent_eval.py          # trajectory + tool-use correctness
│   └── app/
│       └── streamlit_app.py
│
├── notebooks/                     # exploration + ablations
├── scripts/                       # one-shot CLI entrypoints
└── docs/
    ├── blog_draft.md              # grows through the project
    └── architecture.md
```

---

# 📅 Week-by-Week Plan

Each day is ~4–6 focused hours. Adjust to your pace; the checkpoints matter more than the day numbers.

---

## **WEEK 1 — Foundation, Ingestion, Baseline RAG**

### Goal
A working (mediocre) baseline you can measure everything else against.

### Day 1 — Repo setup + source discovery
- [ ] Create repo with structure above; `uv init` or `poetry init`
- [ ] Docker-compose with Qdrant
- [ ] Write `src/ingest/scrapers/rba.py`: discover all RDP PDF links from `rba.gov.au/publications/rdp/` + FSR + Bulletin housing articles (filter by title keywords: "housing", "dwelling", "property", "mortgage", "rent")
- [ ] Write equivalent for AHURI (`ahuri.edu.au/research/final-reports` — has a search/filter UI)
- [ ] Write scraper for Grattan housing topic page
- [ ] **Deliverable**: a `sources.jsonl` with ~100 PDF URLs + metadata (title, publisher, date, url)

### Day 2 — Remaining scrapers + bulk download
- [ ] CoreLogic news/research index, SQM Research reports index, Domain/PropTrack research pages
- [ ] Treasury + Productivity Commission housing pages
- [ ] Polite downloader: concurrent but respectful, retries, resume
- [ ] Target **250–350 PDFs** total (~2–5 GB)
- [ ] **Deliverable**: PDFs on disk at `data/raw/{publisher}/{yyyy-mm}_{slug}.pdf`

### Day 3 — Parsing + chunking
- [ ] Install `docling`; parse all PDFs → markdown with structure preserved (headings, tables)
- [ ] Fallback to `pymupdf4llm` for docs that fail
- [ ] Chunk strategy: **recursive with heading-aware splits**, target 500–800 tokens, 100 token overlap
- [ ] Preserve metadata per chunk: `{publisher, title, date, section_heading, page, url}`
- [ ] **Deliverable**: `data/processed/chunks.jsonl` — expect 30–80k chunks

### Day 4 — Baseline indexing
- [ ] Embed all chunks with `bge-base-en-v1.5` (no fine-tune yet) via sentence-transformers. Use GPU if available — ~15–30 min for 50k chunks
- [ ] Upsert to Qdrant with metadata payload
- [ ] Write `retrieve(query, k=10)` returning chunks + scores
- [ ] Sanity-check with 5 manual queries
- [ ] **Deliverable**: working dense retrieval

### Day 5 — Eval set construction (most important day of Week 1)
- [ ] Create `data/eval/queries.jsonl` with **60 eval queries** across:
  - 20 homebuyer-style (practical, local)
  - 20 investor-style (numeric, comparative)
  - 20 researcher-style (methodology, cross-report)
- [ ] For each query, manually identify **2–5 gold chunks** from your corpus (the ones that SHOULD be retrieved). This is tedious but non-negotiable — everything downstream depends on it.
- [ ] Also generate 40 additional queries via Claude using chunk content, then spot-check. Gives you ~100 total.
- [ ] **Deliverable**: gold-annotated eval set

### Day 6 — Baseline RAG end-to-end + metrics
- [ ] Build naive RAG: retrieve top-5 → stuff into Claude prompt → generate answer with citations
- [ ] Implement `src/eval/retrieval_eval.py`: Recall@5, Recall@10, MRR, nDCG@10
- [ ] Run against eval set. **Record baseline numbers.** These become the number you beat.
- [ ] Typical baseline on AU housing with vanilla BGE: Recall@10 around 0.55–0.70
- [ ] **Deliverable**: `results/baseline.json`

### Day 7 — Error analysis + write Week 1 section of blog
- [ ] Look at 20 failure cases. Categorize: jargon mismatch (e.g., "LVR" vs "loan-to-value"), temporal confusion ("latest" queries), abbreviation problems (AHURI, HILDA, NHFIC), numeric queries that need tools not text
- [ ] Write the "Baseline & Problems" section of blog post
- [ ] **Deliverable**: `docs/blog_draft.md` with first section + error taxonomy

### Week 1 Checkpoint
✅ ~300 PDFs parsed and indexed
✅ 100-query gold eval set
✅ Baseline Recall@10 and MRR recorded
✅ Error taxonomy documenting what fails and why

---

## **WEEK 2 — Hybrid Retrieval + Embedding Fine-Tuning**

### Goal
Demonstrably better retrieval through engineering + training. This is the ML-heavy week.

### Day 8 — Hybrid search (BM25 + dense with RRF)
- [ ] BM25 index over same chunks (`rank_bm25` or Qdrant sparse vectors)
- [ ] Reciprocal Rank Fusion combining top-50 from each
- [ ] Re-run eval — expect +5 to +10 Recall@10 from hybrid alone
- [ ] **Deliverable**: hybrid retriever + updated metrics

### Day 9 — Reranking
- [ ] Add `bge-reranker-v2-m3` as second stage: retrieve top-30 → rerank → return top-5
- [ ] Re-run eval — expect another +5 to +15 improvement
- [ ] Measure latency impact (will add ~200–500ms; decide if acceptable)
- [ ] **Deliverable**: reranker integrated, metrics updated

### Day 10 — Training data generation
- [ ] Use Claude to synthesize (query, positive_chunk) pairs from each chunk. Prompt: "Given this chunk from a housing report, write 2 questions a {homebuyer|investor|researcher} might ask that this chunk answers."
- [ ] Generate **~4,000 pairs**
- [ ] Quality filter: embed query + chunk with baseline model, drop pairs where cosine < 0.3 (model thinks they're unrelated — likely bad synthetic query) or > 0.95 (too trivial, essentially paraphrase)
- [ ] **Deliverable**: `data/training/pairs.jsonl`

### Day 11 — Hard negative mining
- [ ] For each (query, positive) pair: retrieve top-20 chunks with BM25. Any chunk that's NOT the positive and NOT from the same section becomes a candidate hard negative.
- [ ] Keep 5 hardest per query (highest BM25 score among non-positives)
- [ ] Final format: `(anchor, positive, [neg_1..neg_5])`
- [ ] **Deliverable**: `data/training/triplets.jsonl`

### Day 12 — Fine-tune embedding model
- [ ] `sentence-transformers` script with `MultipleNegativesRankingLoss`
- [ ] Training config: batch size 64, lr 2e-5, 3 epochs, warmup 10%, AdamW
- [ ] Can train on a single consumer GPU (RTX 3090/4090) in 1–3 hours, or Colab free T4 in ~4–6h
- [ ] Hold out 10% of pairs as training-time validation
- [ ] Save as `models/bge-au-housing-v1/`
- [ ] **Deliverable**: trained checkpoint + training curves

### Day 13 — Re-embed + benchmark
- [ ] Re-embed entire corpus with fine-tuned model (~15–30 min)
- [ ] Re-run full eval suite. Target: **MRR improvement of +15–25%** over base BGE
- [ ] Diagnostic breakdown: which query categories improved most? (Researcher queries often improve most because jargon-heavy)
- [ ] **Deliverable**: before/after table with statistical breakdown

### Day 14 — Ablation + write-up
- [ ] Ablation table in blog: {base-BGE, base+hybrid, base+hybrid+rerank, ft+hybrid+rerank}
- [ ] Plot: Recall@K curves for each variant
- [ ] Write Week 2 blog section — this is the meat of your technical story
- [ ] **Deliverable**: `docs/blog_draft.md` with retrieval story complete

### Week 2 Checkpoint
✅ Fine-tuned AU-housing embedding model
✅ Ablation table showing +15-25% MRR over baseline
✅ Hybrid + rerank pipeline in production retriever
✅ Blog draft of retrieval engineering

---

## **WEEK 3 — Agentic Layer**

### Goal
Transform "RAG chatbot" into a genuine multi-step research agent that reasons, plans, and uses tools.

### Day 15 — Tool implementations
Implement each as a standalone function with a clear Pydantic schema:
- [ ] `rba_cash_rate(period)` — pull from RBA F1.1 table
- [ ] `rba_mortgage_rates(period)` — standard variable + fixed (table F6)
- [ ] `abs_property_price_index(capital_city, period)` — cat. 6432.0
- [ ] `abs_building_approvals(state, period)` — cat. 8731.0
- [ ] `sqm_rental_vacancy(postcode_or_city)` — scrape SQM chart data
- [ ] `render_chart(series_dict, title)` — matplotlib → base64 PNG
- [ ] `compute_rental_yield(price, weekly_rent)` — simple math
- [ ] All tools return `{data, source, retrieved_at, citation}` so agent can cite
- [ ] **Deliverable**: `src/tools/` with ~7 working tools

### Day 16 — LangGraph agent skeleton
- [ ] State: `messages`, `query_type`, `sub_questions`, `retrieved_chunks`, `tool_results`, `answer_draft`, `reflection`, `iteration_count`
- [ ] Nodes: `classify_query` → `decompose` → `retrieve_OR_tool` (router) → `reflect` → loop OR `synthesize`
- [ ] Edges: conditional routing based on query classification + reflection
- [ ] Max iteration cap: 4 (prevent infinite loops)
- [ ] **Deliverable**: graph renders + happy path works

### Day 17 — Query decomposition + routing logic
- [ ] `classify_query` node: LLM with structured output → `{persona, needs_docs: bool, needs_data: bool, needs_decomposition: bool}`
- [ ] `decompose` node: for complex queries, split into 2–4 atomic sub-questions
- [ ] Router: for each sub-question, decide doc retrieval vs tool call vs both
- [ ] Test on 10 diverse queries, trace execution with LangSmith or Phoenix
- [ ] **Deliverable**: decomposition working on hybrid queries

### Day 18 — Self-reflection + citation discipline
- [ ] `reflect` node after answer draft: "Does the draft answer all sub-questions? Are all claims cited? Are there obvious gaps?"
- [ ] If gaps identified → loop back with refined query
- [ ] Strict citation format: every numeric claim gets `[source:publisher, page:N]` or `[tool:tool_name, retrieved:date]`
- [ ] **Deliverable**: reflection working; fewer hallucinations vs Week 1 baseline

### Day 19 — Agent evaluation harness
- [ ] Build `data/eval/agent_queries.jsonl` — 30 complex multi-step queries with annotated expected tool calls + expected doc sources
- [ ] Metrics:
  - **Tool-call accuracy**: did it call the right tool?
  - **Trajectory efficiency**: did it avoid unnecessary steps?
  - **Faithfulness** (via RAGAS): are answer claims supported by retrieved context?
  - **Groundedness**: % of numeric claims with a citation
- [ ] **Deliverable**: agent eval script + initial numbers

### Day 20 — Error analysis + iteration
- [ ] Look at 15 failures. Common patterns: over-decomposition (asking 1 thing as 5 sub-questions), tool confusion (uses doc retrieval when data is needed), citation drops
- [ ] Fix 2–3 highest-impact issues via prompt tuning or graph restructuring
- [ ] **Deliverable**: v2 metrics showing improvement

### Day 21 — Buffer day + blog write-up
- [ ] Catch up on anything incomplete
- [ ] Write agent section of blog: decomposition examples, reflection examples, failure modes
- [ ] **Deliverable**: blog draft complete for Weeks 1–3

### Week 3 Checkpoint
✅ ~7 working tools covering live AU housing data
✅ LangGraph agent with decompose → retrieve/tool → reflect → synthesize
✅ Agent eval harness with trajectory + faithfulness metrics
✅ Citations on every quantitative claim

---

## **WEEK 4 — Polish, Deploy, Write-up**

### Goal
Turn the working system into portfolio-quality material. This week's output determines whether recruiters care.

### Day 22 — Persona UX
- [ ] Streamlit app with persona selector (Homebuyer / Investor / Researcher)
- [ ] Persona adjusts:
  - System prompt tone ("explain like I'm new to this" vs "be technical and concise")
  - Retrieval weighting (homebuyers want practical docs; researchers want methodology papers)
  - Tool selection hints
- [ ] Citations rendered as clickable chips → sidebar with source excerpt
- [ ] **Deliverable**: clean 3-persona UI

### Day 23 — Trace & transparency UI
- [ ] Expandable "Reasoning steps" panel showing:
  - Sub-questions the agent generated
  - Which tools were called + what they returned
  - Top retrieved chunks with relevance scores
- [ ] Embedded charts (from `render_chart`) inline in answer
- [ ] **Deliverable**: transparent agent UX — this is what makes it a portfolio piece vs a toy

### Day 24 — Caching + cost optimization
- [ ] Embedding cache (disk-based) keyed on chunk hash
- [ ] Tool result cache with TTL (RBA data updates monthly — cache 24h is fine)
- [ ] Prompt caching for Claude API (use `cache_control` on system prompt + retrieved context)
- [ ] Measure cost per query before/after: target under AUD $0.02 per typical query
- [ ] **Deliverable**: cost dashboard in `README.md`

### Day 25 — Deployment
- [ ] Dockerfile with Qdrant + app
- [ ] Deploy to Modal (easiest for Python) or HF Spaces (simplest for Streamlit)
- [ ] Qdrant Cloud free tier (1GB) — may need to slim index if corpus is large; solve by keeping only top-quality publishers in prod
- [ ] Secrets management via deployment platform
- [ ] **Deliverable**: public URL anyone can try

### Day 26 — Final eval report
- [ ] Run **full eval suite one more time** on the deployed pipeline
- [ ] Produce the headline table for your README + blog:

| Configuration | Recall@10 | MRR | Faithfulness | Avg Latency | Cost/query |
|---|---|---|---|---|---|
| Baseline BGE | X | X | X | X | X |
| + Hybrid + Rerank | X | X | X | X | X |
| + Fine-tuned | X | X | X | X | X |
| Full Agent | — | — | X | X | X |

- [ ] **Deliverable**: `results/final_metrics.json` + README table

### Day 27 — README polish
- [ ] Hero section: one-line description + GIF of the UI
- [ ] Problem statement (why generic RAG fails on AU housing)
- [ ] Architecture diagram (mermaid)
- [ ] Quickstart: `docker compose up` → working in <5 min
- [ ] Results table
- [ ] Tech choices + what you'd do differently
- [ ] **Deliverable**: README that recruiters can read in 90 seconds and get the full picture

### Day 28 — Blog post finalization
Structure:
1. The problem: AU housing research is fragmented
2. Baseline RAG and why it fails (jargon, hybrid-data queries)
3. Retrieval engineering: hybrid + rerank + **fine-tuning an AU-housing embedding model** ← hero section
4. Agentic layer: decomposition, tools, reflection
5. Eval methodology & results (the ablation table)
6. Lessons learned + what's next

Post on: personal blog + cross-post to Medium + LinkedIn + relevant subreddits (r/LocalLLaMA, r/MachineLearning, r/AusFinance)

- [ ] **Deliverable**: published blog post with diagrams and code snippets

### Day 29 — Demo video + social
- [ ] 3-minute Loom walkthrough: three queries (one per persona) showing the agent's reasoning trace
- [ ] Embed in README, blog, LinkedIn
- [ ] **Deliverable**: shareable video

### Day 30 — Buffer + retrospective
- [ ] Write a personal retrospective (private): what worked, what you'd cut, what surprised you
- [ ] Clean up branches, tag `v1.0.0`
- [ ] **Deliverable**: you're done. Show it off.

---

## 🎯 Success Criteria

You've nailed this project if all of these are true at the end:

1. ✅ **Quantitative retrieval win**: fine-tuned model beats base BGE by ≥15% MRR on your eval set
2. ✅ **Working multi-step agent**: handles queries requiring both doc retrieval and live data
3. ✅ **Public, reproducible**: anyone can `git clone` and get it running in under 10 minutes
4. ✅ **Portfolio artifacts**: blog post + video + README + live demo, all linked
5. ✅ **Honest eval**: you can walk a recruiter through before/after numbers and explain the engineering tradeoffs

---

## ⚠️ Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Data licensing — CoreLogic/SQM PDFs may be restricted | Stick to publicly posted research reports; cite sources clearly; don't redistribute the corpus publicly. The index and model weights are yours. |
| Fine-tuning doesn't show big gains | Likely causes: (a) training data too synthetic → spot-check harder, generate from multiple personas; (b) hard negatives not hard enough → increase to top-30 BM25 candidates. Have a fallback: document the negative result honestly — showing a rigorous evaluation that reveals "fine-tuning helped on queries X but not Y" is also a great portfolio story. |
| Agent loops or explodes cost | Hard iteration cap (4) + per-query budget check + circuit breaker on tool calls |
| PDF parsing issues with tables | Use docling (specifically designed for this); fallback to screenshot + vision model for the handful of critical but unparseable docs |
| Scope creep | Explicitly cut: multi-turn memory, user accounts, property-level valuations, live MLS listings. These are v2. |

---

## 📚 Key References to Read Before Starting

- Sentence-Transformers docs on `MultipleNegativesRankingLoss`
- LangGraph tutorial on agentic RAG
- RAGAS paper on evaluation metrics
- BGE paper (Xiao et al.) for the embedding approach
- Anthropic's cookbook on tool use + prompt caching
- Cotality (CoreLogic AU) methodology paper on hedonic index — essential domain reading

---

## 🏁 Day-One Checklist

Before you start coding:

- [ ] Confirm GPU access (Colab Pro, RunPod, or local) — needed Week 2
- [ ] Anthropic API key with sufficient credit (~$30 AUD will cover whole project comfortably)
- [ ] Qdrant running locally via Docker
- [ ] Python 3.11+, `uv` installed
- [ ] GitHub repo created (private initially, flip to public at end of Week 4)
- [ ] Decide your blog platform (personal site, Medium, Substack)

Good luck. The fine-tuning + agent combo on a niche AU domain is genuinely uncommon in portfolios — this is a strong signal to employers.
