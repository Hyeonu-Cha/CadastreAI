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

- [x] **Task 1.04** — RBA RDP scraper (2026-04-19)
  - `src/ingest/scrapers/common.py`: shared `Source` dataclass, `HOUSING_KEYWORDS`, `matches_housing()`, `slugify()`, `write_jsonl()`, polite `USER_AGENT`
  - `src/ingest/scrapers/rba.py`: fetches the RDP index at `rba.gov.au/publications/rdp/` via `httpx`, parses year headings + paper list items, keeps only housing-keyword-matching titles, emits JSONL of `Source` records
  - CLI entrypoint: `python -m src.ingest.scrapers.rba --out data/raw/rba_rdp.jsonl`
  - Added runtime deps to `pyproject.toml`: `httpx`, `beautifulsoup4`, `lxml`
  - Bulletin + FSR scrapers tracked separately as Task 1.05 (not touched here)
  - Syntax-checked with `py_compile`; not run live yet (deps not installed on dev machine)

- [x] **Task 1.09** — CoreLogic / Cotality news-research scraper (2026-04-20)
  - `src/ingest/scrapers/corelogic.py`: handles the CoreLogic → Cotality rebrand with `discover_base()` which tries `cotality.com/au/news-research`, `cotality.com/news-research`, `corelogic.com.au/au/news-research`, `corelogic.com.au/news-research` in order and uses the first 200 response
  - No keyword filter (news-research listing is wholly housing-relevant); dedupes by URL
  - `_is_article_path()` excludes nav/product/contact/about paths while accepting `/au/<slug>/` style article URLs
  - 0.8s delay between page fetches, `--max-pages` default 25, `/page/N/` pagination convention
  - Gemini code review (via gemini CLI) flagged no blocking issues; noted future improvements: tighter CSS selectors over `find_all("a")`, more precise date regex
  - Syntax-checked; not run live (no network/deps on dev machine yet)
  - **Workflow change:** from this ticket onward, merging via GitHub PR instead of local fast-forward, with gemini code review on each PR

- [x] **Task 1.08** — Unified sources.jsonl collector (2026-04-19)
  - `scripts/collect_sources.py`: orchestrates `rba.scrape_all`, `ahuri.scrape`, `grattan.scrape`; merges + dedupes by URL; writes `data/raw/sources.jsonl`
  - Schema documented at top of script: `{title, publisher, url, date?, category?, extra?}`
  - Also emits `sources.stats.json` with totals + per-publisher counts for README generation later
  - `--only rba ahuri grattan` flag to subset scrapers; per-scraper failures logged + skipped
  - Target ~100 entries after Day-1 scrapers; full 250–350 comes in Day 2 once remaining sources land

- [x] **Task 1.07** — Grattan Institute housing scraper (2026-04-19)
  - `src/ingest/scrapers/grattan.py`: walks `grattan.edu.au/topics/housing/` and subsequent `page/N/` pagination, captures publication titles, URLs, and any date string found in the surrounding card text
  - Skips obvious non-publication URLs (`/topics/`, `/people/`, `/about/`, `/category/`, `/tag/`, `/news/`, `/events/`, `/contact/`)
  - 0.8s delay between pages; max 30 pages by default
  - Syntax-checked

- [x] **Task 1.06** — AHURI final-reports scraper (2026-04-19)
  - `src/ingest/scrapers/ahuri.py`: walks paginated listing at `ahuri.edu.au/research/final-reports` (`?page=N`) until an empty page, extracts report detail URLs matching `/research/final-reports/<id>`, captures title + year from surrounding card text
  - No keyword filter — AHURI is wholly housing-relevant
  - Polite 0.8s delay between page fetches; configurable `--max-pages` cap (default 60)
  - Syntax-checked; live HTML structure not verified

- [x] **Task 1.05** — RBA Bulletin + FSR scrapers (2026-04-19)
  - Extended `rba.py` with `scrape_bulletin()` and `scrape_fsr()` alongside `scrape_rdp()`
  - Bulletin: walks the main index → each `YYYY/{mon}` issue page → collects article links matching `bulletin/YYYY/mon/*.html`, filters titles via `matches_housing`
  - FSR: walks the main index → each issue page → collects all `.pdf` links under `/fsr/` (the whole review is housing-relevant, so no title filter)
  - Refactored CLI to `python -m src.ingest.scrapers.rba {rdp|bulletin|fsr|all} --out …`; `all` runs every scraper and dedupes
  - Date normalization: `YYYY-mon` for Bulletin/FSR, `YYYY` for RDPs
  - Syntax-checked; live HTML structure not verified (dev machine lacks deps/network)
