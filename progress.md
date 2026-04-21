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

- [x] **Follow-up (Task 1.14)** — Filename collision guard in downloader (2026-04-21)
  - After the HTML-fallback landed and the full download re-ran, 40 stems collided across 292 tasks, silently dropping **252 URLs** on disk — 206 of them RBA FSR chapters (each issue's 22 chapter anchors all read "Download PDF", slugifying identically to `{year}-00_download-pdf`). PropTrack 36, APRA 5, PC 3, AHURI 2 rounded out the loss
  - `src/ingest/download.py`: new `_compute_unique_stems()` pre-scans all tasks, detects stems claimed by more than one task, appends an 8-char MD5-of-URL suffix to every colliding member (non-colliding stems stay short for readability). `_download_one` now takes its resolved `stem` as a parameter instead of recomputing from the task; `_run` logs the collision count at startup
  - Verified on the real `sources.jsonl`: 1430 tasks → 1430 unique stems (292 got a hash suffix). Sample RBA FSR: `2025-00_download-pdf_de1b1545.pdf`, `2025-00_download-pdf_79e8fc1f.pdf`, …
  - Without this, anything fed to the Task 1.16/1.17 parsers was built on a corpus that had silently lost 17% of its RBA FSR content — parse metrics would have looked fine but downstream retrieval recall on financial-stability topics would have been quietly hollowed out

- [x] **Follow-up (Task 1.14/1.15/1.16)** — HTML-article download + parse path (2026-04-21)
  - External review flagged the "no pdf link found" failures (293/646 on the first bulk run, ~20% of sources) as a corpus-shape problem, not a bug: PropTrack/CoreLogic/Domain publish a large share of their content as pure HTML articles with no companion PDF. Dropping them skewed the corpus towards RBA research papers and undermined the homebuyer/investor personas
  - `src/ingest/download.py`: `_target_path` → `_target_stem` (returns Path without suffix). `_fetch_and_save` now returns `(saved_path, error)` so the caller knows whether a `.pdf` or `.html` landed. New fallback branch: if the landing page response is HTML and no in-body PDF link is found, save the HTML body as `.html` instead of failing with "no pdf link found". Guard: skip saving if body is <1 KB (login walls / JS-only shells would produce zero useful markdown downstream)
  - `_download_one` skip-check widened to cover both `.pdf` and `.html` targets so a partial rerun doesn't re-download either form
  - `src/ingest/parse.py`: `_discover_pdfs` → `_discover_docs` (globs both `*.pdf` and `*.html`). `convert_pdf` → `convert_doc`. docling's `DocumentConverter.convert()` dispatches on extension, so PDF and HTML go through the same code path
  - `src/ingest/parse_fallback.py`: skip non-PDF entries in the failures file (pymupdf4llm is PDF-only; HTML docling failures need a different retry path, which is out of scope for now — they pass through to `parse.failures.final.jsonl` unchanged)
  - Smoke test on 5 PropTrack sources: 5/5 OK — one URL was a landing page with a real PDF link (followed and saved as `.pdf`), four were pure HTML articles (saved as `.html`). docling converted the HTML article cleanly to 12.9k chars of structured markdown with headings, images-as-placeholders, and author/date metadata intact
  - Also fixed a pre-existing `UP035` lint warning (`from typing import AsyncIterator` → `from collections.abc import AsyncIterator`) while I was in the file
  - Deferred: the reviewer's related note about `_date_prefix → "unknown"` filename collisions (two undated PDFs with the same slug silently overwrite each other). Real concern — the PropTrack smoke test had three `unknown_*.html` files in one directory — but easier to fix as a one-liner hash-suffix patch separately so this PR stays tightly scoped

- [x] **Task 1.17** — `pymupdf4llm` fallback parser for docling failures (2026-04-21)
  - `src/ingest/parse_fallback.py`: reads `data/processed/parse.failures.jsonl` produced by `src.ingest.parse`, retries each failed PDF with `pymupdf4llm.to_markdown()`, writes successful conversions to the same `data/processed/{publisher}/{stem}.md` layout as the docling path
  - `convert_pdf()` lazy-imports `pymupdf4llm` so `--help` / other code paths don't pay the PyMuPDF load cost
  - Skips targets that already have a non-empty `.md` (consistent with `parse.py`); `--force` flag to overwrite
  - Empty-markdown output is treated as a failure (pymupdf4llm occasionally returns `""` on heavily scan-only PDFs)
  - Remaining failures written to `data/processed/parse.failures.final.jsonl` for manual review — that's the end of the automated parsing funnel
  - CLI: `python -m src.ingest.parse_fallback --failures data/processed/parse.failures.jsonl --raw-dir data/raw --out-dir data/processed [--limit N] [--force]`
  - Added `pymupdf4llm>=0.0.17` to the `[parse]` optional-dependencies alongside docling; pulls in PyMuPDF + onnxruntime + pymupdf-layout but no torch/vision models
  - Smoke test: 3/3 PDFs converted, 104k–524k chars each, headings and paragraph structure preserved (tables often degrade to plain paragraphs — expected trade-off for recovering scan/encrypted/malformed-stream PDFs that docling can't process)
  - Motivation: docling's `standard_pdf_pipeline` crashed with `std::bad_alloc` on an image-heavy multi-page PDF during the full parse run, killing the process before any failures file could be written; pymupdf4llm is lightweight enough to serve as a no-torch backstop

- [x] **Task 1.16** — Install docling; implement `src/ingest/parse.py` (2026-04-20)
  - `src/ingest/parse.py`: walks `data/raw/{publisher}/*.pdf`, converts each via docling's `DocumentConverter.convert()` + `export_to_markdown()`, writes `data/processed/{publisher}/{stem}.md` preserving headings (`##`), paragraphs, and GitHub-flavored tables
  - Single converter instance reused across PDFs (model load is the bottleneck); lazy-imported at call time so `--help` stays instant
  - Resume: skips existing non-empty `.md` targets; `--force` flag to reparse
  - Failures logged to `data/processed/parse.failures.jsonl` (JSON lines of `{pdf_path, publisher, error}`) for the Task 1.17 pymupdf4llm fallback to retry
  - Added `docling>=2.0` as a `[project.optional-dependencies].parse` extra — heavy (pulls torch + vision models, ~1–2 GB on first run), kept optional so scraper-only users aren't forced into it
  - Smoke test on 3 PDFs: 3/3 OK, ~25s/PDF, output markdown preserves `## headings` + paragraph flow; images are emitted as `<!-- image -->` placeholders (docling default)
  - **Full parsing run deferred** — at ~25 s/PDF × 351 PDFs ≈ 2.5 h, that's a long unattended run; coded the pipeline now, will kick off the full run in a follow-up session or let Task 1.17 (pymupdf4llm fallback) land first so the whole pipeline can run end-to-end
  - Gemini review: no blocking issues; minor suggestions (`rglob` instead of level-by-level iteration, append-mode for failures log) deferred

- [x] **Task 1.15** — Bulk download run: 351 PDFs collected (2026-04-20)
  - Extended `scripts/collect_sources.py` to include the 5 new scrapers (corelogic, sqm, domain_proptrack, treasury_pc, nhfic_apra)
  - Live scraper run produced `data/raw/sources.jsonl` with **646 unique source URLs** across 6 working publishers: RBA 304, PropTrack 218, CoreLogic/Cotality 72, Housing Australia (NHFIC) 29, Treasury 15, Grattan 8
  - Five scrapers returned 0 in this run and need investigation (AHURI, Domain, SQM, APRA, PC — site structures shifted since scrapers were written); filed as follow-up debugging tickets
  - Full download run: **292 OK + 59 skipped = 351 PDFs saved** (hits the 250–350 target). 295 failures, 293 of which are "no pdf link found" — these are HTML-only web articles (mostly PropTrack & CoreLogic/Cotality) that need a different parsing path in Task 1.16
  - **Critical bug found + fixed during the run:** one PropTrack page contained a leaked local filesystem path (`/C:/Users/eleanor.creagh/Downloads/...`) as a PDF href. `urljoin` produced an unparseable URL and the raised `ValueError` propagated out of the async gather, crashing the whole batch. Fixed `_download_one` to catch `ValueError` and `OSError` (Windows path-length, malformed URL) as per-task failures. Rerunning resumed from 40/646 thanks to the skip-existing logic
  - On-disk layout at `data/raw/{publisher-slug}/{yyyy-mm}_{slug}.pdf` matches the Task 1.14 spec
  - `data/raw/` is gitignored; the commit only carries the collector update + the downloader resilience fix

- [x] **Task 1.14** — Polite concurrent downloader (2026-04-20)
  - `src/ingest/download.py`: async downloader that reads `data/raw/sources.jsonl` and writes `data/raw/{publisher_slug}/{yyyy-mm}_{slug}.pdf`
  - Global concurrency cap via `asyncio.Semaphore` (default 4); per-host serialization + 1.0s min-delay via `_HostGate` async context manager — every outgoing request (both HTML landing page and follow-up PDF fetch) passes through the gate
  - Streams responses directly to a `.part` sidecar via `client.stream()` + `aiter_bytes()`; file writes offloaded via `asyncio.to_thread` so the event loop stays responsive; atomic rename on success
  - Direct-PDF URLs stream from the initial response without buffering or double-downloading. HTML landing pages: body is consumed once (`resp.aread()`), parsed for the first `.pdf` link, followed in a second streamed request
  - Retries with exponential backoff + jitter (default 3 attempts, 1.5s base); 404s short-circuit without retry
  - Resume: skips targets that already exist non-empty; emits `{in_path}.download.json` report with ok/skipped/failed counts + failure list
  - CLI: `python -m src.ingest.download --in data/raw/sources.jsonl --out-dir data/raw --concurrency 4 --per-host-delay-s 1.0`
  - Gemini review caught three real issues pre-merge: host-gate leak on the second request, blocking file I/O inside the event loop, and full-body buffering before content-type check. Each fixed and re-reviewed clean

- [x] **Task 1.13** — NHFIC / Housing Australia + APRA scraper (2026-04-20)
  - `src/ingest/scrapers/nhfic_apra.py`: two P2 publishers covering mortgage-market + guarantee-scheme material
  - Housing Australia (formerly NHFIC): tries `/research`, `/publications`, `/resources` — wholly housing agency, no keyword filter
  - APRA: tries `/publications`, `/statistics`, `/authorised-deposit-taking-institutions` — covers banking/insurance/super too, so `matches_housing` filter IS applied
  - Drupal 0-indexed pagination (same pattern as Task 1.12): page 1 → landing, page N≥2 → `?page=N-1`
  - **Pagination guard strengthened over Task 1.12:** now requires ≥2 new URLs per page to continue walking. If only 0–1 new items show up on a given page, the scraper assumes the main pager is exhausted and only a sidebar is rotating — it captures the lone new item (if any) and stops. Caught by gemini review as HIGH — without this guard a rotating "Recent Publications" sidebar would drag every run to `max_pages`
  - Also tightened parent-block to `[article, li, tr]` to avoid header-level keyword leakage (same fix as Task 1.12)

- [x] **Task 1.12** — Treasury + Productivity Commission housing scraper (2026-04-20)
  - `src/ingest/scrapers/treasury_pc.py`: scrapes Treasury (`treasury.gov.au/policy-topics/housing`) and Productivity Commission (`pc.gov.au/topics/housing` with `/inquiries/completed` fallback) — both Drupal sites
  - Applies `matches_housing` keyword filter (unlike specialist scrapers) because both publishers cover far more than housing
  - **Drupal 0-indexed pagination:** loop page 1 → landing URL, page 2 → `?page=1`, page 3 → `?page=2`, etc. Initial version incorrectly jumped from page 1 to `?page=2`, skipping the real second page (caught by gemini review and fixed pre-merge)
  - Parent-block narrowed to `[article, li, tr]` (dropped `div`/`section`) so a single "housing" mention in a page header can't leak unrelated links through the keyword filter
  - Known limitation: Treasury `/policy-topics/housing` is a topic hub, not a full paginated list — the hub links out to search-result pages at `/publications?topic=...` which this scraper does NOT follow. Flagged for a future improvement ticket
  - Gemini review: initial pagination bug fixed on second review; remaining notes (date regex breadth, segment-count permissiveness) logged for cross-cutting cleanup

- [x] **Task 1.11** — Domain + PropTrack quarterly reports scraper (2026-04-20)
  - `src/ingest/scrapers/domain_proptrack.py`: covers two publishers in one module since they follow the same structural pattern
  - Domain: `domain.com.au/research/` with `/page/N/` pagination; required-prefix filter pins article paths to `/research/`
  - PropTrack: tries `proptrack.com.au/insights/` first, falls back to `realestate.com.au/insights/proptrack/` (mirror for some reports); required-prefix is dynamically derived from the resolved listing path
  - Shared `_walk_pages()` helper factors pagination + dedupe between the two sub-scrapers
  - `scrape()` runs both sub-scrapers and tolerates either failing (logs + continues)
  - CLI supports `python -m src.ingest.scrapers.domain_proptrack {domain|proptrack|all} --out …`
  - Gemini review: no blocking issues; noted cross-cutting follow-ups (promote `_dedupe` helper from `rba.py` to `common.py`, tighten `_DATE_RE`, narrow the date-search block)

- [x] **Task 1.10** — SQM Research press-releases scraper (2026-04-20)
  - `src/ingest/scrapers/sqm.py`: scrapes sqmresearch.com.au press-releases index (tries `/press-releases.php`, `/news.php`, `/media-releases.php` in order)
  - `_looks_like_report()` accepts direct PDFs, press/media-release slugs, and ISO-date-bearing paths; rejects nav/account/contact prefixes and static assets
  - Title fallback: for anchor-less PDF links (common on SQM), derives title from the URL stem
  - Pagination guard: stops when `?page=N` returns identical URLs to page 1 (SQM's PHP template ignores the query param in most layouts) — prevents infinite loops
  - No keyword filter (SQM output is wholly housing-focused)
  - Gemini review: no blocking issues; noted fragility in `discover_listing` length heuristic and broad `_DATE_RE`
  - Syntax-checked

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
