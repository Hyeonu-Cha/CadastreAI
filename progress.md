# CadastreAI — Progress Log

Tracks completed tickets with short notes. See `tickets.md` for the full backlog.

## Week 1 — Foundation, Ingestion, Baseline RAG

### Day 1

- [x] **Task 1.01** — Initialize Python project (2026-04-19)
  - Added `pyproject.toml` (PEP 621, hatchling build backend, `requires-python = ">=3.11"`, optional `dev` deps: pytest + ruff)
  - Added `.env.example` with `ANTHROPIC_API_KEY`, `QDRANT_URL/API_KEY`, LangSmith + Phoenix placeholders
  - Extended `.gitignore` to cover Python artifacts, `.env`, and `data/raw/`
  - Note: `uv` / `poetry` not installed on this machine; kept `pyproject.toml` tool-agnostic so either can adopt it later
