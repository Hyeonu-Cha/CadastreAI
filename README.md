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

Retrieval eval over a 100-query held-out set across the four personas
(homebuyer / investor / researcher / policy). Numbers are top-10:

| Configuration                    | Recall@5 | Recall@10 | MRR@10  | nDCG@10 |
|----------------------------------|----------|-----------|---------|---------|
| Dense only (BAAI/bge-base-en-v1.5) | 0.383    | 0.450     | 0.352   | 0.349   |
| BM25 only                          | 0.800    | **0.907** | **0.732** | **0.760** |
| Hybrid (BM25 + Dense) + RRF        | 0.563    | 0.747     | 0.626   | 0.585   |
| Hybrid + cross-encoder rerank      | 0.550    | 0.673     | 0.560   | 0.537   |

*Headline finding:* on a domain-specific Australian housing corpus
where queries share heavy lexical overlap with the source documents
(e.g. *"first home buyer grant NSW"* lands directly on the relevant
section heading), **BM25 outperforms dense and hybrid retrieval** out
of the box. Hybrid + reranker make the dense channel less useful here,
not more. This motivates Week 2's BGE fine-tune on synthetic
(query, chunk) pairs to close the gap on more abstractive queries —
that work is in flight; numbers will land in this table when the run
completes.

See [`docs/blog.md`](./docs/blog.md) for the full write-up,
ablations, and per-persona breakdowns.

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
