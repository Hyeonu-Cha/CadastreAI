<div align="center">
  <img src="./assets/logo.svg" alt="CadastreAI" width="400"/>

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

> [Insert 20-second GIF of the Streamlit UI here — showing a query, streaming answer with citations, and the reasoning trace panel]

Live demo: **[cadastreai.modal.run](https://example.com)** *(link goes live at end of Week 4)*

---

## 🏗️ What it does

CadastreAI answers questions about the Australian residential property market by combining a curated corpus of ~300 research reports with live data from government APIs. Three personas (homebuyer, investor, researcher) get tailored responses, every claim is cited, and the reasoning is auditable.

**Example queries it handles well**:

- *"Is Parramatta a good bet for a family on $1.2M? What's the trend and yield?"* *(homebuyer)*
- *"Sydney suburbs with highest 3yr capital growth and vacancy under 3%"* *(investor)*
- *"How does CoreLogic's hedonic index differ from ABS 6432.0 methodologically?"* *(researcher)*

---

## 📊 Results

| Configuration | Recall@10 | MRR | Faithfulness | Avg Latency | Cost/query |
|---|---|---|---|---|---|
| Baseline BGE | 0.XX | 0.XX | 0.XX | X.Xs | $0.0X |
| + Hybrid (BM25 + Dense) + RRF | 0.XX | 0.XX | 0.XX | X.Xs | $0.0X |
| + Reranker (bge-reranker-v2-m3) | 0.XX | 0.XX | 0.XX | X.Xs | $0.0X |
| **+ Fine-tuned BGE (AU housing)** | **0.XX** | **0.XX** | **0.XX** | **X.Xs** | **$0.0X** |
| Full Agent (retrieval + tools) | — | — | 0.XX | X.Xs | $0.0X |

*Headline result*: fine-tuning the embedding model on ~4k (query, chunk) pairs generated from the AU housing corpus improved MRR by **X%** over base BGE-base-en-v1.5 on a 100-query held-out eval set.

See [`docs/blog.md`](./docs/blog.md) for the full write-up and methodology.

---

## 🏛️ Architecture

<div align="center">
  <img src="./assets/architecture.svg" alt="CadastreAI system architecture" width="900"/>
</div>

Three layers:

1. **Agent** (LangGraph) — classifies query, decomposes complex questions, routes to retrieval or tools, reflects on gaps, synthesizes a cited answer.
2. **Retrieval** — hybrid BM25 + dense (fine-tuned BGE) → Reciprocal Rank Fusion → cross-encoder reranker.
3. **Live data** — tool-use over RBA, ABS, SQM public endpoints, with 24h cache and timestamped results.

Full design in [`product.md`](./product.md); engineering plan in [`project_plan.md`](./project_plan.md).

---

## 🚀 Quickstart

```bash
# Clone and enter
git clone https://github.com/yourname/cadastreai
cd cadastreai

# Environment
cp .env.example .env  # add ANTHROPIC_API_KEY

# Start Qdrant + app
docker compose up -d

# Install deps
uv sync

# Ingest corpus (takes ~30 min)
uv run python scripts/ingest_all.py

# Launch UI
uv run streamlit run src/app/streamlit_app.py
```

Open `http://localhost:8501` — pick a persona and ask a question.

---

## 🗂️ Repo structure

```
cadastreai/
├── assets/              # logo, architecture diagram
├── src/
│   ├── ingest/          # scrapers + PDF parsing
│   ├── index/           # Qdrant + hybrid retrieval
│   ├── training/        # embedding fine-tuning pipeline
│   ├── retrieval/       # retriever + reranker
│   ├── tools/           # live-data tool implementations
│   ├── agent/           # LangGraph graph + nodes
│   ├── eval/            # retrieval + agent eval harnesses
│   └── app/             # Streamlit UI
├── data/                # gitignored: raw PDFs + processed chunks
├── docs/                # blog post + architecture notes
├── product.md           # product document (what & why)
└── project_plan.md      # engineering plan (how & when)
```

---

## 📝 Design docs

- **[`product.md`](./product.md)** — product requirements, personas, features, user flows, success metrics
- **[`project_plan.md`](./project_plan.md)** — 4-week engineering plan with daily tasks and checkpoints
- **[`docs/blog.md`](./docs/blog.md)** — technical deep-dive on fine-tuning and ablations

---

## ⚠️ Disclaimer

CadastreAI is a research and education tool. Nothing it produces constitutes financial advice. Property decisions should involve a licensed buyer's agent, mortgage broker, solicitor, and financial adviser. Data accuracy depends on upstream sources (RBA, ABS, etc.) and may be stale between cache refreshes.

---

## 📜 License

Code: MIT. Corpus data is indexed for research use only; all reports remain the property of their original publishers (RBA, AHURI, CoreLogic/Cotality, Grattan Institute, SQM Research, etc.).

---

<div align="center">
  Built with Claude Sonnet 4.5, Qdrant, sentence-transformers, and LangGraph.<br/>
  <sub>A 4-week portfolio project · [Your name] · [Your link]</sub>
</div>
