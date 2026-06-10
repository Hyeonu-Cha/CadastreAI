<!--
  HF Spaces frontmatter (Task 4.15 prep). Picked up automatically when this
  repo is mirrored to a Docker-mode Space on huggingface.co/spaces. GitHub
  hides this block from the rendered README. Three secrets need to be set
  in the Space settings before the agent will run:
      ANTHROPIC_API_KEY, QDRANT_URL, QDRANT_API_KEY
-->
---
title: CadastreAI
emoji: 🏠
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8501
pinned: false
short_description: AI research agent for the Australian housing market
---

<div align="center">
  <img src="./logo.svg" alt="CadastreAI" width="400"/>

  <h3>The property register, intelligent.</h3>

  <p>
    <strong>AI research agent for the Australian housing market</strong><br/>
    Domain-specialized RAG · Fine-tuned retriever · Agentic reasoning over RBA, AHURI, CoreLogic & live data
  </p>

  <p>
    <a href="#-demo">Demo</a> ·
    <a href="#-quickstart">Quickstart</a> ·
    <a href="#-results">Results</a> ·
    <a href="#-architecture">Architecture</a> ·
    <a href="./docs/blog.md">Blog post</a>
  </p>
</div>

---

## 🎬 Demo

> A 20-second walkthrough GIF of the Streamlit UI — persona switcher,
> streaming answer with inline citations, and the 5-step reasoning
> trace panel — will live here once recorded. Run it locally for now
> (see [Quickstart](#-quickstart)).

<!--
  Task 4.19: when the GIF is recorded, save it as `assets/demo.gif`
  and replace the blockquote above with this single line:

      ![CadastreAI walkthrough](./assets/demo.gif)

  Loom embed (Task 4.24): paste below the GIF —

      [![CadastreAI walkthrough (3 min)](./assets/demo.gif)](LOOM_URL)
-->


---

## 🏗️ What it does

CadastreAI answers questions about the Australian residential property market by combining a curated corpus of research reports with live data from government APIs. Four personas (first-home buyer, investor, policy researcher, journalist) get tailored responses, every claim is cited, and the reasoning trajectory is auditable in the UI.

**Example queries it handles well**:

- *"Is Parramatta a good bet for a family on $1.2M? What's the trend and yield?"* *(homebuyer)*
- *"Sydney suburbs with highest 3yr capital growth and vacancy under 3%"* *(investor)*
- *"How does CoreLogic's hedonic index differ from ABS 6432.0 methodologically?"* *(researcher)*
- *"What's the 5-year story on housing supply in NSW that I can cite in a piece?"* *(journalist)*

---

## 📊 Results

### Retrieval

The 100-query eval set has two groups with different provenance:

- **60 BM25-annotated** queries (Tasks 1.24–1.27): gold chunks were
  picked by humans from BM25 top-10 candidate lists, so any pure-BM25
  metric on this split is **circular by construction**.
- **40 synthetic** queries (Task 1.28): each gold chunk came first,
  the query was authored *for* that chunk with no retriever
  involvement. This is the honest head-to-head.

Reporting both splits matters: pooled numbers make BM25 look like the
clear winner, but that's the BM25-annotated split inflating the mean.
The honest split tells a different story.

**Honest split (40 synthetic queries, top-10):**

| Configuration                  | Recall@5  | Recall@10 | MRR@10    | nDCG@10   |
|--------------------------------|-----------|-----------|-----------|-----------|
| Dense (BAAI/bge-base-en-v1.5)  | 0.750     | 0.825     | 0.592     | 0.648     |
| BM25 only                      | 0.575     | 0.775     | 0.455     | 0.529     |
| **Hybrid (BM25 + Dense) + RRF** | **0.825** | **0.900** | 0.637     | 0.700     |
| Hybrid + cross-encoder rerank  | **0.850** | 0.875     | **0.697** | **0.741** |

Hybrid wins R@K on its own; adding the cross-encoder reranker shifts
the win from R@10 to MRR/nDCG (it pulls the right hit *higher*, even
when the top-10 set itself is comparable). The agent ships with
hybrid + rerank as the default retriever.

At n=40, one query is worth 0.025 of recall — the close calls in this
table (0.900 vs 0.875, 0.825 vs 0.850) are within a single query and
should be read as ties. The orderings that survive that noise floor:
hybrid > either component alone, and rerank > no-rerank on MRR/nDCG.

**Pooled (all 100 queries, for reference):**

| Configuration                    | Recall@5  | Recall@10 | MRR@10    | nDCG@10   |
|----------------------------------|-----------|-----------|-----------|-----------|
| Dense (BAAI/bge-base-en-v1.5)    | 0.383     | 0.450     | 0.352     | 0.349     |
| BM25 only                        | 0.800     | **0.907** | **0.732** | **0.760** |
| Hybrid (BM25 + Dense) + RRF      | 0.563     | 0.747     | 0.626     | 0.585     |
| Hybrid + cross-encoder rerank    | 0.550     | 0.673     | 0.560     | 0.537     |

The agent ships with hybrid as the default retriever (per Task 3.25)
based on the honest split. Full discussion of the circularity, the
fine-tune publisher-collapse failure mode, and per-persona breakdowns
in [`results/hybrid_comparison.md`](./results/hybrid_comparison.md)
and [`results/ablation.md`](./results/ablation.md).

### Agent

Agent eval over 30 queries with judge-graded faithfulness, six runs
(`results/agent_v1_vs_v2.md`, `results/agent_v2_vs_v3.md`,
`results/agent_v3_vs_v4.md`, `results/agent_v4_vs_v5.md`,
`results/agent_v6_final.json`):

| Metric                  | v1     | v2     | v3     | v4     | v5     | **v6 final** |
|-------------------------|--------|--------|--------|--------|--------|--------------|
| tool_acc                | 0.894  | 0.919  | 0.925  | 0.917  | 0.913  | **0.980**    |
| tool_recall             | 0.956  | 0.978  | 0.944  | 0.944  | 0.956  | **0.989**    |
| trajectory_efficiency   | 0.299  | 0.672  | 0.722  | 0.717  | 0.719  | **0.811**    |
| faithfulness (judge)    | 0.570  | 0.648  | 0.630  | 0.637  | 0.664  | **0.793**    |
| publisher_recall        | 0.789  | 0.778  | 0.764  | 0.772  | **0.825** | 0.781    |
| groundedness (regex)\*  | 0.373  | 0.296  | 0.240  | 0.368  | 0.316  | 0.124        |

\* *groundedness is a strict formatting-contract check — it requires a
citation literal within 50 characters of every numeric token — not an
answer-quality measure. Faithfulness (judge) is the quality number;
see the v6 note below for why the two diverge. n=30 throughout, so
single-query swings move any metric by ~0.03 — read deltas under that
as noise.*

v2 dropped `MAX_ITERATIONS` 4 → 2 and tightened the reflect/synth
prompts (+0.37 trajectory, +0.08 faith). v3 swapped the retriever to
hybrid (Task 3.25 default) — aggregate looked flat because the impact
was route-conditional: doc-using queries pulled +19% more chunks
and synth's citation discipline slipped.

v4 (Task 3.29) added citation-adjacency rules + a chunk cap, lifting
groundedness regex 0.240 → 0.368 (+0.128 aggregate, +0.217 on the
doc-using subset).

v5 (Task 3.30) added compute-tool disambiguation rules to the router
prompt — `compute_*` tools now explicitly require user-provided
numbers, with worked GOOD/BAD examples. Closes the canonical
`compute_rental_yield` vs market-lookup confusion (agent-019:
tool_call_accuracy 0.25 → 1.00). The worked example also nudged 12
queries from tool-only to doc-using; on that switched cohort
faithfulness jumped +0.110 because the judge now sees doc-supported
tool answers. Aggregate faithfulness +0.027, publisher_recall +0.053.

**v6 (Task 4.17)** is the final-pass eval against the deployed config
(Cloud Qdrant + Anthropic). Faithfulness +0.129 (0.664 → 0.793),
trajectory efficiency +0.092, tool-call accuracy reached 0.980 with
recall at 0.989. Publisher recall slipped 0.044 — fewer doc routes
when tool answers were sufficient. The groundedness regex (0.124)
reads low because it requires a citation literal within 50 characters
of every numeric token; the judge-graded faithfulness (0.793) is the
load-bearing number for whether the answer is actually supported.

See [`docs/blog.md`](./docs/blog.md) for the full write-up and
[`results/final_metrics.json`](./results/final_metrics.json) for the
shipping numbers.

---

## 🏛️ Architecture

<div align="center">
  <img src="./architecture.svg" alt="CadastreAI system architecture" width="900"/>
</div>

Three layers:

1. **Agent** (LangGraph, 5 nodes) — classifies query, decomposes complex questions, routes to retrieval or tools, reflects on gaps, synthesizes a cited answer. Loop-back capped at 4 iterations. Anthropic prompt caching on every system prompt for ~90% input-token discount on warm calls.
2. **Retrieval** — BM25 + dense BGE-base-en-v1.5 with Reciprocal Rank Fusion + optional cross-encoder reranker. Disk-cached query embeddings for sub-millisecond repeat lookups.
3. **Live data** — tool-use over RBA, ABS, SQM public endpoints + deterministic compute tools (mortgage, stamp duty, yield). Per-tool TTL cache (24h for upstream data, 7d for pure math).

Full design in [`product.md`](./product.md); engineering plan in [`project_plan.md`](./project_plan.md).

---

## 🚀 Quickstart

### Local (recommended for development)

```bash
git clone https://github.com/Hyeonu-Cha/CadastreAI
cd CadastreAI

# Environment — set ANTHROPIC_API_KEY and (optionally) QDRANT_API_KEY
cp .env.example .env

# Start Qdrant in the background
docker compose up -d qdrant

# Python env
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[agent,index,embed,tools,app,chunk]"

# Ingest corpus (one-off, ~30 min on a laptop)
python -m src.ingest.run_all
python -m src.index.upsert

# Launch UI
streamlit run src/app/streamlit_app.py
```

### Docker (single-image runtime)

```bash
docker compose up -d qdrant         # Qdrant on :6333
docker build -t cadastreai:latest .
docker run --rm -p 8501:8501 \
    -e ANTHROPIC_API_KEY=sk-... \
    -e QDRANT_URL=http://host.docker.internal:6333 \
    cadastreai:latest
```

Open `http://localhost:8501`, pick a persona, ask a question.

### Verify a fork (or a deploy)

`scripts/deploy_check.sh` runs three checks in fail-fast order: env
vars set, Qdrant collection reachable with the expected point count,
and a 5-query smoke test exercising the tool-only / docs-only /
hybrid / compute / multi-step routes:

```bash
QDRANT_URL=http://localhost:6333 \
QDRANT_API_KEY=cadastre-dev \
ANTHROPIC_API_KEY=sk-... \
./scripts/deploy_check.sh
```

Exit 0 = healthy, 1 = a smoke query failed, 2 = environment or
Qdrant misconfigured. Same script verifies a Qdrant Cloud + HF
Spaces deploy by swapping `QDRANT_URL` to the cluster URL.

---

## 🗂️ Repo structure

```
CadastreAI/
├── src/
│   ├── ingest/          # scrapers + PDF parsing (docling, PyMuPDF)
│   ├── index/           # Qdrant upsert + BM25 + hybrid + rerank
│   ├── retrieval/       # query-side retriever + on-disk embedding cache
│   ├── training/        # BGE fine-tuning pipeline (synthetic Q/A pairs)
│   ├── tools/           # RBA / ABS / SQM / compute / chart tool impls
│   ├── agent/           # LangGraph graph, nodes, persona, tool cache, cost
│   ├── eval/            # retrieval + agent eval harnesses
│   └── app/             # Streamlit UI + citation/disambiguation/trace
├── data/                # gitignored: raw PDFs, processed chunks, caches
├── docs/                # blog draft, architecture diagrams
├── results/             # eval JSON dumps for each retrieval config
├── tests/               # pytest unit + integration tests
├── Dockerfile           # multi-stage production image
├── docker-compose.yml   # Qdrant + optional Phoenix tracing
├── product.md           # product document (what & why)
└── project_plan.md      # engineering plan (how & when)
```

---

## 📝 Design docs

- **[`product.md`](./product.md)** — product requirements, personas, features, user flows, success metrics
- **[`project_plan.md`](./project_plan.md)** — 4-week engineering plan with daily tasks and checkpoints
- **[`docs/blog.md`](./docs/blog.md)** — technical deep-dive on fine-tuning and ablations
- **[`tickets.md`](./tickets.md)** — full ticket-level breakdown of the 4 weeks

---

## ⚠️ Disclaimer

CadastreAI is a research and education tool. Nothing it produces constitutes financial advice. Property decisions should involve a licensed buyer's agent, mortgage broker, solicitor, and financial adviser. Data accuracy depends on upstream sources (RBA, ABS, etc.) and may be stale between cache refreshes.

---

## 📜 License

Code: MIT. Corpus data is indexed for research use only; all reports remain the property of their original publishers (RBA, AHURI, CoreLogic/Cotality, Grattan Institute, SQM Research, etc.).

---

<div align="center">
  Built with Claude (Haiku 4.5 for routing/reflection, Sonnet 4.6 for synthesis), Qdrant, sentence-transformers, and LangGraph.<br/>
  <sub>A 4-week portfolio project</sub>
</div>
