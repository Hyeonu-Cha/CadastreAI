# CadastreAI — Project Report v2

> Generated: 2026-05-06 | Branch: `docs/report-v2-refresh` | Replaces the
> closed PR #88 draft (2026-04-28), which was superseded by Tasks
> 3.22–3.25 and the README correction in #110.

---

## 1. Executive Summary

CadastreAI is a domain-specialized AI research agent for the Australian
residential property market. The build is feature-complete on every
phase except the cloud deploy and the manual launch artifacts; the rest
is operations.

| Phase                               | Status              | Gating remainder                                         |
|-------------------------------------|---------------------|----------------------------------------------------------|
| **1 — Ingestion + baseline RAG**    | Complete            | —                                                        |
| **2 — Hybrid retrieval + FT A/B**   | Complete            | `ft+hybrid+rerank` cell deferred (needs Qdrant up)       |
| **3 — Agentic layer**               | Complete (v2)       | v3 re-eval against hybrid retriever                      |
| **4 — Polish, deploy, write-up**    | ~85%                | Cloud deploy (4.14–4.16), demo recording, blog publish   |

**Today's headline numbers:**

| Surface                             | Metric                       | Value         |
|-------------------------------------|------------------------------|---------------|
| Retrieval — honest split (n=41)     | Hybrid R@10                  | **0.878**     |
| Agent eval v2 (n=30)                | Faithfulness (judge)         | 0.648         |
| Agent eval v2 vs v1                 | Trajectory efficiency Δ      | **+0.373**    |
| Codebase                            | Tests passing                | ~385 / 25 files |
| Codebase                            | Merged PRs on `main`         | 110           |

---

## 2. Retrieval — what we measured and why both views matter

The 100-query eval set has two groups with different provenance, and
they tell opposite stories. **Reporting either one alone would be
misleading.**

- **59 BM25-annotated queries** (Tasks 1.24–1.27): gold chunks were
  picked by humans from BM25 top-10 candidate lists. Pure-BM25 metrics
  on this split are **circular by construction** — BM25 was the source
  of the candidates the annotators chose from.
- **41 synthetic queries** (Task 1.28): each gold chunk came first,
  then the query was authored *for* that chunk with no retriever
  involvement. This is the honest head-to-head.

### 2a. Honest split (41 synthetic queries, top-10)

| Configuration                       | R@5       | R@10      | MRR@10    | nDCG@10   |
|-------------------------------------|-----------|-----------|-----------|-----------|
| Dense (BAAI/bge-base-en-v1.5)       | 0.756     | 0.829     | 0.602     | 0.656     |
| BM25 only                           | 0.585     | 0.780     | 0.452     | 0.528     |
| **Hybrid (BM25 + Dense) + RRF**     | **0.805** | **0.878** | **0.640** | **0.698** |

Hybrid wins every metric. Dense was already strong here (BGE handles
paraphrased semantic queries well); the +5 R@10 from RRF fusion comes
from queries that hinge on rare domain tokens — *Division 43*, *FHG*,
*NASHH*, *NFIP* — that BM25 nails but dense dilutes.

### 2b. Pooled (all 100, for reference only)

| Configuration                       | R@5       | R@10      | MRR@10    | nDCG@10   |
|-------------------------------------|-----------|-----------|-----------|-----------|
| Dense (BAAI/bge-base-en-v1.5)       | 0.383     | 0.450     | 0.352     | 0.349     |
| BM25 only                           | 0.800     | **0.907** | **0.732** | **0.760** |
| Hybrid (BM25 + Dense) + RRF         | 0.563     | 0.747     | 0.626     | 0.585     |
| Hybrid + cross-encoder rerank       | 0.550     | 0.673     | 0.560     | 0.537     |

The earlier project narrative — "BM25 outperforms dense and hybrid out
of the box" — was reading off this table without controlling for the
circularity. It's left here so anyone tracking
`results/{baseline,bm25,hybrid,reranked}.json` can reconcile the
numbers, but **the honest 41 is what we believe and what the agent
ships against.**

### 2c. The fine-tune A/B that didn't pan out

`bge-au-housing-v1` was trained on Colab T4 against ~3.7k synthetic
pairs (Task 2.10). Offline brute-force-cosine eval (Task 2.14) showed
**regression on every metric** vs base BGE: R@10 0.45 → 0.21,
MRR 0.36 → 0.13. Per-persona breakdown (Task 2.15): the FT model
collapses onto AHURI papers regardless of query topic — *publisher
collapse*, not under-training. Surfaced honestly in the blog post; the
checkpoint is not shipped. Retraining strategies are queued in
`docs/blog.md` (stratified pair sampling, anchor-style alignment,
lower lr, eval-query mix-in, cross-encoder fine-tune) for a future
pass.

### 2d. The cross-encoder regression on this corpus

Adding a cross-encoder rerank on top of hybrid drops R@10 from 0.75 →
0.67 (pooled split) and median latency from 0.4s → 2.9s. On a corpus
this lexical, the reranker is reordering BM25-favoured passages
*downward*. The agent ships hybrid without the reranker by default
until we either swap models or change the shortlist composition.

---

## 3. Agent — v1 → v2

Eval over 30 annotated queries, OpenAI provider (`gpt-4o-mini` for
classify/route/reflect/judge, `gpt-4o` for synth), faithfulness graded
by LLM judge.

| Metric                    | v1 (initial) | v2 (PR #106)  | Δ          |
|---------------------------|--------------|---------------|------------|
| Tool-call accuracy        | 0.894        | 0.919         | +0.025     |
| Tool-call recall          | 0.956        | 0.978         | +0.022     |
| Tool-call precision       | 0.900        | 0.928         | +0.028     |
| Trajectory efficiency     | 0.299        | **0.672**     | **+0.373** |
| Faithfulness (judge)      | 0.570        | **0.648**     | **+0.078** |
| Publisher recall          | 0.789        | 0.778         | -0.011     |
| Groundedness (regex)      | 0.373        | 0.296         | -0.077     |

### 3a. What v2 changed

Two prompt/graph fixes from `results/agent_failure_analysis.md`:

1. **Reflect convergence.** `MAX_ITERATIONS` 4 → 2; reflect prompt
   flipped from *"default to False when in doubt"* to
   *"default to True unless there is a SPECIFIC, concrete gap"*.
   v1 had 27/30 queries pegged at the ceiling; v2 has 12/30
   finishing in a single pass.
2. **Synth must use the evidence.** New "USE THE EVIDENCE" paragraph
   in `_SYNTHESIZER_SYSTEM`. v1 had 6/30 queries refusing to answer
   while the tool result above contained the data — e.g. agent-002
   (median Sydney price): tool returned `$1,515,000`, synth said
   *"I cannot provide…"*. v2 cites it.

### 3b. Regressions to flag, honestly

- **Groundedness -0.08** is a *measurement* bug, not a content bug.
  Sixteen v2 rows score `groundedness=0` while `faithfulness>0`
  because the synth now writes numeric answers but doesn't place the
  `[tool:..., retrieved:...]` citation *immediately adjacent* to each
  number. The judge (reading the whole answer) sees the support; the
  regex doesn't. Fix in v3: tighten the synth prompt for citation
  adjacency.
- **Two ≥0.2 faithfulness regressions** (agent-004, agent-019) —
  same root cause: planner picks `compute_rental_yield` instead of
  `abs_property_price_index`. Tool-confusion was a known v1 problem;
  the lower iteration cap removed v1's accidental recovery rounds.
  Fix in v3: explicit tool-routing examples in `_ROUTER_SYSTEM`.

### 3c. Task 3.25 — configurable retriever, hybrid default

v1 and v2 ran on dense-only retrieval. The honest split says hybrid
wins. PR #108 added `_agent_retriever()` with a process-wide cache,
env-driven routing (`CADASTRE_AGENT_RETRIEVER=dense|bm25|hybrid`,
default hybrid), and lazy backend imports so test/CLI users don't pay
the 212 MB BM25 pickle when they don't need it. The hybrid branch
pre-imports torch ahead of the BM25 pickle to dodge a cuBLAS DLL
load-order segfault on the 4 GB-pagefile Windows host.

**8 new unit tests cover** default selection, env routing, casing,
unknown-value error, cache persistence, `_retrieve_docs` reshape
contract — all retriever classes monkey-patched, no Qdrant or pickle
touched.

The v3 re-eval against v2 was blocked on this host by an environmental
Python-SDK hang (`import openai` / `client.create()` hang while raw
curl works). Code shipped; v3 vs v2 numbers + `results/agent_v2_vs_v3.md`
deferred to the next stable env.

---

## 4. Compliance + safety (Phase 4)

Two layers landed before launch (Tasks X.03, X.04):

- **Persistent disclaimer** in the synth system prompt, proportional
  by persona. Homebuyer / investor get the strongest "consult a
  licensed mortgage broker / buyer's agent / financial adviser"
  closing-line policy; researcher gets a research-only +
  primary-sources caveat; journalist gets a re-verify-before-
  publication caveat. 24 unit tests.
- **Guardrail node** before `classify_query` blocks
  product-recommendation queries (specific lenders, funds,
  securities) with a deterministic regex library. Refused queries
  short-circuit to `END` (zero Anthropic tokens spent) and emit a
  structured WARNING log line for tuning. 41 unit tests.

Allow-list deliberately includes the README's headline example
queries — the disclaimer is what carries those, not a refusal.

---

## 5. What's open before `v1.0.0`

| Item                                              | Blocker                          | Effort  |
|---------------------------------------------------|----------------------------------|---------|
| `ft+hybrid+rerank` ablation cell                  | Docker daemon down on dev host   | 30 min  |
| Tasks 4.14–4.16 — Qdrant Cloud + Modal/HF deploy  | Cloud accounts                   | 2–3 hrs |
| Tasks 4.17–4.18 — full eval + final results table | Depends on 4.14–4.16 + agent v3  | 1 hr    |
| Tasks 4.19 / 4.23 / 4.24 — demo GIF + Loom        | Manual screen recording          | 1 hr    |
| Task 4.22 — publish blog + socials                | Manual                           | 30 min  |
| Task 4.26 — tag `v1.0.0`, flip repo public        | User sign-off                    | 5 min   |

Open environmental issue: Python SDK calls hang on the dev host
(`import openai` / `client.create()`), most likely Windows Defender
real-time scanning + 4 GB pagefile thrashing on first-run `.pyd`/
`.dll` loads. Curl works. Workarounds tried: source-then-export of
`.env`, pre-warming the BGE retriever before any OpenAI import,
hybrid-mode torch pre-import. Cleaner unblock paths: WSL2, Defender
exclusions for `%USERPROFILE%\AppData\Local\Programs\Python` and
the project venv, or a clean `python:3.11-slim` Docker container.

---

## 6. Key learnings worth carrying forward

1. **Eval-set provenance matters more than eval-set size.** A
   100-query set with mixed annotation pedigree gave a misleading
   pooled headline; the honest 41-query split flipped the conclusion.
   Future eval design: every query carries a `source` field
   (`annotation` | `synthetic`) and metrics are reported per-split
   by default.
2. **Fine-tuning a retriever can collapse onto a publisher dimension
   that doesn't show up in aggregate metrics until you slice by
   query topic.** The single ΔMRR=+1.00 win on a researcher query
   showed there's *some* salvageable signal — not enough to ship.
3. **Reflection loops want a default-True is_complete signal.** v1's
   "default False when in doubt" pegged 27/30 queries at the ceiling.
   Inverting the default was the single largest agent-quality fix.
4. **Per-meter cost telemetry is what makes prompt caching checkable.**
   Without separating `cache_read_input_tokens` from full-rate input
   tokens, you can't tell whether the ~90% warm-call discount is
   landing.
5. **Citation adjacency is a contract between the synth prompt and
   the regex grader, not a quality attribute.** The v2 groundedness
   "regression" was the synth deciding to put citations at end of
   sentence instead of mid-sentence; the judge had no problem with
   the answers.

---

## 7. Reproducibility pointers

- Per-ticket merge log: `git log --oneline --no-merges main | grep "Task "`
- Retrieval numbers: `results/{baseline,bm25,hybrid,reranked}.json`,
  `results/hybrid_comparison.md`, `results/ablation.md`
- Agent eval: `results/agent_v1.json`, `results/agent_v2.json`,
  `results/agent_v1_vs_v2.md`, `results/agent_failure_analysis.md`
- Per-task design notes: squash-merge commit bodies on `main`
  (Task numbers in subject)
- Phase 4 progress.md notes: lines 12–319 of `progress.md` (gap
  notice + Phase 4 entries; per-ticket Phase 1–3 notes are in
  PRs #69–#107 commit bodies, not this file)
