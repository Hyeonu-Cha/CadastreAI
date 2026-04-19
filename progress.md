# CadastreAI — Progress Log

Tracks completed tickets with short notes. See `tickets.md` for the full backlog.

## Week 1 — Foundation, Ingestion, Baseline RAG

### Day 1

- [x] **Task 1.01** — Initialize Python project (2026-04-19)
  - Added `pyproject.toml` (PEP 621, hatchling build backend, `requires-python = ">=3.11"`, optional `dev` deps: pytest + ruff)
  - Added `.env.example` with `ANTHROPIC_API_KEY`, `QDRANT_URL/API_KEY`, LangSmith + Phoenix placeholders
  - Extended `.gitignore` to cover Python artifacts, `.env`, and `data/raw/`
  - Note: `uv` / `poetry` not installed on this machine; kept `pyproject.toml` tool-agnostic so either can adopt it later

- [x] **Task 1.02** — Create repository directory skeleton (2026-04-19)
  - Added `src/` package with subpackages: `ingest/` (+ `scrapers/`), `index/`, `training/`, `retrieval/`, `tools/`, `agent/`, `eval/`, `app/` — each with `__init__.py` so they import as Python packages
  - Added `data/{raw,processed,eval,training}/` (raw is gitignored; others carry `.gitkeep`)
  - Added top-level `notebooks/`, `scripts/`, `docs/`, `results/` with `.gitkeep` where empty
  - `docs/` left empty for now (blog draft / architecture diagrams land here later)

- [x] **Task 1.03** — Docker Compose with Qdrant + optional Phoenix (2026-04-19)
  - `docker-compose.yml` defines `qdrant` (v1.12.4) on ports 6333/6334 with persistent `qdrant_storage` volume and optional API-key auth via `QDRANT_API_KEY`
  - TCP healthcheck on 6333
  - `phoenix` service gated behind `tracing` profile (run with `docker compose --profile tracing up`) for agent-trajectory debugging later in Week 3
