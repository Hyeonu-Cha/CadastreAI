# CadastreAI — Progress Log

Tracks completed tickets with short notes. See `tickets.md` for the full backlog.

> **Log gap (Day 2 → Phase 3 close).** Per-ticket notes in this file
> stopped after Day 1; the ~65 tickets between Task 1.05 and Task 4.10
> are documented via squash-merged PRs on `main` instead. Run
> `git log --oneline --no-merges main | grep "Task "` for the
> chronological ledger; each commit's body has the design notes that
> would otherwise live here. This file resumes detailed logging from
> Phase 4 onwards.

## Phase 4 — Serving, caching, packaging

### Day — Streamlit + UX

(Tasks 4.01–4.08 are documented in their respective squash-merged
PRs #69–#75. Notes resume below from the operational tickets I drove
in this session.)

### Day — Cache layers, cost telemetry, packaging

- [x] **Task 4.09** — Disk-backed embedding cache (PR #76)
- [x] **Task 4.10** — In-memory tool-result cache with per-tool TTLs (PR #77)
- [x] **Task 4.11** — Anthropic prompt caching on every system prompt (PR #78, 2026-04-27)
  - Wrapped each of the 5 `messages.create()` system strings in a
    list-of-dicts with `cache_control: {"type": "ephemeral"}` —
    classifier, decomposer, router, reflector, synthesizer
  - Render order is tools → system → messages, so a breakpoint on the
    last system block caches tools + system together; per-call user
    content stays in `messages[]` after the breakpoint
  - Yields ~90% input-token discount on warm calls within the 5-minute
    TTL; verified shape via the `claude-api` skill before patching
  - Tests: 254 passed (pre-existing stub clients don't care about the
    new cache_control marker)

- [x] **Task 4.12** — Per-query cost accounting + node hooks (PR #79, 2026-04-27)
  - New `src/agent/cost.py`: `calculate_cost(usage, model)` →
    `CostBreakdown` across 4 meters (input full-rate, output, cache_read
    0.1×, cache_write 1.25×); `CostTracker` class for per-run
    accumulation; `log_cost()` one-shot helper for nodes
  - Pricing is a one-table lookup keyed by model_id (cached 2026-04-15)
    — when Anthropic changes prices it's a one-line patch
  - Wired `log_cost()` into all 5 `messages.create()` callsites in
    `nodes.py`. Each emits a structured line:
    `cost node=X model=Y in=N out=N cache_r=N cache_w=N total=$0.000XYZ`
  - 15 unit tests cover the pricing maths, dict-vs-attr usage shapes,
    unknown-model fallback, missing-cache-fields default, tracker
    accumulation, and the structured log format
  - Decoupling per-meter is what makes the cache-hit story checkable
    — without `cache_read_input_tokens` separated out, you can't tell
    whether Task 4.11's prompt cache is actually landing

- [x] **Task 4.13** — Multi-stage production Dockerfile (PR #80, 2026-04-27)
  - Two stages: builder (`python:3.11-slim` + `build-essential`)
    compiles wheels for runtime extras (`agent,index,embed,tools,app,
    chunk`) into `/opt/venv`; runtime (`python:3.11-slim` + `libgomp1`)
    copies the prebuilt venv and runs as a non-root `cadastre` user
    (UID 1000)
  - `parse` extras (docling + torch ~1.5 GB) deliberately skipped —
    only needed for offline ingestion, not the runtime
  - Streamlit defaults: headless, 0.0.0.0:8501, browser-stats off
  - `/_stcore/health` healthcheck; `.dockerignore` mirrors `.gitignore`
    + excludes notebooks/docs/data so source-only changes don't bust
    the dep cache layer
  - Build context shrinks from ~repo size to ~10 MB
  - Not built locally (Docker daemon wasn't running on dev machine);
    syntax is conventional buildkit, deferred to first deploy/CI

- [x] **Task 4.20** — README finalised for launch (PR #81, 2026-04-27)
  - Real eval numbers in the results table from `results/{baseline,
    bm25,hybrid,reranked}.json` — recall@5/10, MRR@10, nDCG@10 across
    the full 100-query held-out set
  - Honest framing: BM25 currently leads (R@10 = 0.907) on this
    lexical-heavy AU housing corpus; the dense fine-tune is what's
    expected to close the gap on more abstractive queries — flagged
    as in-flight rather than presented as done
  - Dual Quickstart paths: local pip-editable install + Docker
    single-image runtime (using the new Task 4.13 Dockerfile)
  - Asset paths fixed (`./logo.svg` / `./architecture.svg` at repo
    root, not the non-existent `assets/` directory)
  - Repo structure refreshed to match actual `src/` subdirs and call
    out top-level `Dockerfile` / `docker-compose.yml` / `results/`
  - Architecture blurb mentions prompt caching (4.11), embedding
    cache (4.09), tool cache (4.10), 5-node LangGraph + 4-iter cap
  - Persona count corrected: 4 (the journalist persona was added in
    Phase 4, the README was still saying 3); model attribution
    refreshed to Haiku 4.5 + Sonnet 4.6
  - Dropped the dead `cadastreai.modal.run` placeholder link

- [x] **Task 4.21** — Blog draft consolidated into final post (PR #82, 2026-04-27)
  - Renamed `docs/blog_draft.md` → `docs/blog.md` (preserves
    `git log --follow` history)
  - Replaced the meta-header ("working draft, will trim") with a
    proper "Why this exists" intro framing the problem (specialised
    corpus, personal queries, checkable answers) and the 4-phase
    arc
  - Renamed Week-N → Phase-N section headers; rewrote internal
    "Week 2/3" references to phase language. Drops the sprint cadence
    from the public post
  - Added **Phase 4 — Serving** section covering the UI surfaces
    (persona / disambiguation / citations / trace / follow-ups), the
    three caching layers, per-meter cost telemetry, and the Dockerfile
  - Added **Lessons & open threads** close: BM25 vs hybrid on
    lexical-heavy corpora, forced tool use as the structured-output
    story, citation discipline as a post-process, reflection-loop
    forced exits, why cost has to be per-meter; plus the open Phase 2
    fine-tune and Phase 3 agent eval as named gating items
  - README refreshed (3 refs `docs/blog_draft.md` → `docs/blog.md`)
  - Historical references in `project_plan.md` / `tickets.md` left as
    "blog_draft.md" intentionally — those are planning artefacts, not
    live links

### Day — Compliance / safety layer

- [x] **Task X.03** — Persistent "not financial advice" disclaimer in
  the synthesizer system prompt, proportional by persona
  (PR #85, 2026-04-27)
  - Two layers, both prefixed so the model recognises them in-prompt:
    a universal `COMPLIANCE` baseline directly in `_SYNTHESIZER_SYSTEM`
    (so even unrecognised persona keys still get the "not a licensed
    professional" floor), and a per-persona `DISCLAIMER POLICY` block
    via the new `persona_disclaimer()` helper, layered on top of the
    existing tone addendum
  - Proportional intensity: homebuyer + investor get the strongest
    "consult a licensed mortgage broker / buyer's agent / financial
    adviser" closing-line policy because their queries directly inform
    personal financial decisions; researcher gets a research-only +
    primary-sources caveat; journalist gets a re-verify-before-
    publication caveat; general gets the middle-ground language
  - `persona_disclaimer()` exported from `src.agent.persona`; wired
    into the synth `system_prompt` as a third paragraph alongside
    `_SYNTHESIZER_SYSTEM` and `persona_prompt_addendum`
  - 24 new tests (`tests/test_persona_disclaimer.py`) cover every
    persona's coverage, the unknown-persona fallback, proportional
    intensity assertions per persona, and end-to-end synth-prompt
    assembly

- [x] **Task X.04** — Block financial-product recommendations + log
  refusals (PR #86, 2026-04-27)
  - New `src/agent/guardrails.py` — pure regex library that screens
    queries for product-pick phrasings across four buckets: mortgage
    products / lenders / brokers, insurance products, super funds and
    SMSFs / managed funds / ETFs, and specific listed-security picks
  - Pure function `screen_query(query)` returns a frozen
    `GuardrailDecision`; pattern library is module-level and ordered
    so the first match wins for category labelling. No LLM call on
    the hot path — guardrails must be deterministic and fast
  - Refusal text per category names what we won't do, redirects the
    user to a licensed professional, and offers a phrasing we *can*
    help with (e.g. "I CAN help with average mortgage rates from RBA
    data, stamp-duty calculations, the First Home Guarantee scheme")
  - New `guardrail_screen` graph node placed *before* `classify_query`;
    refused queries short-circuit straight to END (zero Anthropic
    tokens spent), allowed queries return `{}` and pass through
  - New `_route_after_guardrail` conditional edge in `graph.py` reads
    `state["guardrail"]["blocked"]` to decide END vs continue
  - `log_refusal()` emits a structured WARNING line per refused query
    (`category=...`, `reason=...`, full query) — fuel for tuning the
    regex when false positives surface
  - Allow-list deliberately includes the README's headline example
    queries ("Is Parramatta a good bet for a family on $1.2M?", etc.)
    — the X.03 disclaimer is what carries those, not a guardrail
    refusal
  - 41 new tests (`tests/test_guardrails.py`) cover the full pattern
    library (parametrized allow/refuse cases), the logging side-
    effect, the node behaviour on both paths, the conditional-edge
    logic, and end-to-end graph short-circuit + pass-through

### Day — Week 2 fine-tune A/B + ablation revisit

- [x] **Tasks 2.06–2.10** — Pair generation, hard-negative mining,
  training pipeline merged across PRs #88–#93 (Gemini + OpenAI
  providers for `generate_pairs.py`, cost-projection helper, Colab
  T4 notebook, fp16 + max-seq-length flags + CSV-glob fix on
  `train_embeddings.py`). Bi-encoder `bge-au-housing-v1` trained
  on Colab T4 against ~3.7k synthetic pairs.
- [x] **Task 2.13** — Re-embedded entire corpus with the fine-tuned
  model (offline `.npy` matmul; Qdrant collection upsert
  `cadastre_chunks_ft` deferred — Docker daemon outage on dev box).
- [x] **Task 2.14** — `src/eval/retrieval_eval_offline.py`
  (PR #94, 2026-04-29) — Qdrant-free brute-force-cosine eval over
  `data/processed/finetuned/embeddings.npy`. Reuses
  `retrieval_eval.run()` for scoring/aggregation. **Result:
  fine-tune regressed** vs. base BGE on every metric
  (R@10 0.45 → 0.21, MRR 0.36 → 0.13). Diagnosed as
  publisher-collapse: the FT model retrieves AHURI papers for
  almost any query regardless of topic.
- [x] **Task 2.15** — `scripts/finetune_persona_breakdown.py`
  (PR #95, 2026-04-29) — pure JSON-in / Markdown-out helper.
  Confirms FT regresses on every persona; investor has 0 query-
  level wins (FT worse on all 25 investor queries). Single
  largest FT win is ΔMRR=+1.00 on a researcher query — there is
  *some* salvageable signal but not enough to ship the checkpoint.
- [x] **Task 2.16** — `scripts/build_ablation.py` (PR #96,
  2026-04-29) — emits `results/ablation.{json,md}` from the five
  measured variants. **BM25 alone wins all four metrics** on this
  100-query eval set (R@10=0.91, MRR=0.73). Hybrid (BM25+BGE via
  RRF) drops R@10 to 0.75; cross-encoder rerank drops it further
  to 0.67. The `ft+hybrid+rerank` cell is left explicitly "not
  run" — needs the deferred `cadastre_chunks_ft` Qdrant collection.
- [x] **Task 2.17** — Refreshed `src/eval/plot_recall_curves.py`
  (PR #97, 2026-04-29) — DEFAULT_VARIANTS now includes BM25 and
  fixes a prior mislabeling where "ft + hybrid + rerank" pointed
  at `results/finetuned.json` (which was the FT-dense run).
  Regenerates `docs/figures/recall_curves.png` with five lines:
  BM25 on top throughout K=1..10, FT on the bottom.
- [x] **Task 2.18** — Week 2 blog write-up (PR #98, 2026-04-30).
  New "End-to-end pipeline (full 100-query eval set)" table next
  to the synthetic-41 one; new "Reading the fine-tune regression"
  subsection covering publisher-collapse and five candidate
  next-experiments (stratified pair sampling, anchor-style
  alignment, lower lr, eval-query mix-in, cross-encoder
  fine-tune); rewrote "Reading Phase 2" to be honest about the
  two-splits-two-stories outcome and surface follow-ups
  (re-tune RRF weighting, diagnose cross-encoder regression on
  this corpus).

### Still gating launch

- **Tasks 3.22 / 3.23 / 3.24** — agent eval v1, failure analysis, v2.
  Need `ANTHROPIC_API_KEY` + a populated Qdrant. Harness is
  unit-tested end-to-end against a stub graph (`tests/test_agent_eval`).
- **`ft+hybrid+rerank` ablation cell** — needs Docker back up so
  we can upsert `cadastre_chunks_ft` and re-run
  `retrieval_eval --retriever hybrid --rerank --dense-collection
  cadastre_chunks_ft`. Realistic expectation given the
  publisher-collapse failure mode upstream: it won't rescue the
  FT model. Tracked under Task 2.16 follow-up.
- **Tasks 4.14 / 4.15 / 4.16** — Qdrant Cloud + Modal-or-HF deploy +
  smoke test. Need cloud accounts.
- **Tasks 4.17 / 4.18** — full eval suite + final results table.
  Depends on the agent eval and the deploy.
- **Tasks 4.19 / 4.23 / 4.24** — demo GIF + Loom walkthrough +
  embedded video. Manual screen recording.
- **Task 4.22** — publish blog/socials. Manual.
- **Task 4.26** — tag `v1.0.0`, flip repo public. Held for explicit
  user sign-off.

---

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
