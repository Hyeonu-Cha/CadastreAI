# CadastreAI — Project Report v2

> Generated: 2026-05-07 | Branch: `Development/Task330-DocsRefresh` |
> Updated with v5 agent eval numbers from PR #117 + ft+hybrid+rerank
> ablation cell from PR #116; supersedes the v4 snapshot (PR #115).

---

## 1. Executive Summary

CadastreAI is a domain-specialized AI research agent for the Australian
residential property market. The build is feature-complete on every
phase except the cloud deploy and the manual launch artifacts; the rest
is operations.

| Phase                               | Status              | Gating remainder                                         |
|-------------------------------------|---------------------|----------------------------------------------------------|
| **1 — Ingestion + baseline RAG**    | Complete            | —                                                        |
| **2 — Hybrid retrieval + FT A/B**   | Complete            | —                                                        |
| **3 — Agentic layer**               | Complete (v5)       | —                                                        |
| **4 — Polish, deploy, write-up**    | ~85%                | Cloud deploy (4.14–4.16), demo recording, blog publish   |

**Today's headline numbers:**

| Surface                             | Metric                       | Value         |
|-------------------------------------|------------------------------|---------------|
| Retrieval — honest split (n=41)     | Hybrid R@10                  | **0.878**     |
| Retrieval — ft+hybrid+rerank n=100  | R@10                         | **0.770**     |
| Agent eval v5 (n=30)                | Faithfulness (judge)         | **0.664**     |
| Agent eval v5 (n=30)                | Publisher recall             | **0.825**     |
| Agent eval v5 (n=30)                | Trajectory efficiency        | 0.719         |
| Agent eval v4 → v5                  | Switched-cohort faith Δ      | **+0.110**    |
| Codebase                            | Tests passing                | 397 / 27 files |
| Codebase                            | Merged PRs on `main`         | 117           |

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

## 3. Agent — v1 → v2 → v3 → v4 → v5

Eval over 30 annotated queries, OpenAI provider (`gpt-4o-mini` for
classify/route/reflect/judge, `gpt-4o` for synth), faithfulness graded
by LLM judge.

| Metric                    | v1     | v2 (PR #106)  | v3 (PR #112)  | v4 (PR #114)  | v5 (PR #117)  |
|---------------------------|--------|---------------|---------------|---------------|---------------|
| Tool-call accuracy        | 0.894  | 0.919         | 0.925         | 0.917         | 0.913         |
| Tool-call recall          | 0.956  | 0.978         | 0.944         | 0.944         | 0.956         |
| Tool-call precision       | 0.900  | 0.928         | 0.933         | 0.925         | 0.917         |
| Trajectory efficiency     | 0.299  | 0.672         | **0.722**     | 0.717         | 0.719         |
| Faithfulness (judge)      | 0.570  | 0.648         | 0.630         | 0.637         | **0.664**     |
| Publisher recall          | 0.789  | 0.778         | 0.764         | 0.772         | **0.825**     |
| Groundedness (regex)      | 0.373  | 0.296         | 0.240         | **0.368**     | 0.316         |

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

### 3c. v2 → v3 — hybrid retriever (Task 3.25 + PR #112)

v1 and v2 ran on dense-only retrieval. The honest split says hybrid
wins, so PR #108 (Task 3.25) added `_agent_retriever()` with a
process-wide cache, env-driven routing
(`CADASTRE_AGENT_RETRIEVER=dense|bm25|hybrid`, default hybrid), and
lazy backend imports so test/CLI users don't pay the 212 MB BM25
pickle. Hybrid branch pre-imports torch ahead of the BM25 pickle to
dodge a cuBLAS DLL load-order segfault on the 4 GB-pagefile Windows
host. 8 new unit tests; retriever classes monkey-patched.

**v3 aggregate looks flat. The honest story is route-conditional.**

Splitting by retrieval status surfaces two opposing effects (full
table in `results/agent_v2_vs_v3.md`):

| Subset                       | n  | Faith Δ    | Traj Δ   | Read                                 |
|------------------------------|----|------------|----------|--------------------------------------|
| Tool-only (no retrieval)     | 14 | +0.048     | +0.083   | Within noise — retriever is no-op    |
| Doc-using (apples-to-apples) | 14 | **-0.076** | +0.012   | Hybrid pulls +19% chunks; synth slips |

Hybrid retrieves 8.93 chunks/query vs dense's 7.50 on the doc-using
subset. The retrieval benchmark's R@10 said hybrid is strictly better;
at the agent level the +19% chunk count is *diluting* synth's citation
discipline. agent-022 (1.00 → 0.60), agent-023 (0.75 → 0.50), and
agent-024 (0.83 → 0.57) all show the same pattern — answers cite more
sources but each citation is less tightly grounded.

**Hybrid stays the ship default** (the retrieval benchmark says it
should, and the tool-only path confirms no regression on the bulk of
agent queries), but at v3 the agent's downstream synth wasn't yet
capitalising on the improved retrieval. v4 (next section) closes this.

### 3d. v3 → v4 — citation adjacency + chunk cap (Task 3.29 + PR #114)

Two targeted fixes to the v3 doc-using regression. Full breakdown in
`results/agent_v3_vs_v4.md`.

1. **Citation adjacency in `_CITATION_RULES`.** The synth prompt now
   demands `[tool:..., retrieved:...]` *immediately after* each
   numeric value, with a worked GOOD/BAD example. The regex grader
   checks adjacency; v3's prompt tolerated trailing-citation patterns
   the judge accepted but the regex missed. (Carry-over from v1 → v2
   → v3 — promised three times, finally landed.)
2. **`_dedupe_and_cap_chunks(cap=8)`** in `retrieve_or_tool` after
   `apply_publisher_boost`. v3 averaged 8.93 chunks/query (max 25 —
   five sub-questions × top-5 each, no dedupe across reflect cycles),
   diluting synth context. Cap at the v2 average. 7 new unit tests
   in `tests/test_chunk_dedupe_cap.py`.

| Subset                          | n  | Faith Δ    | Grounded Δ  | n_chunks Δ        |
|---------------------------------|----|------------|-------------|-------------------|
| Aggregate                       | 30 | +0.007     | **+0.128**  | (n/a — see below) |
| Doc-using (apples-to-apples)    | 14 | **+0.055** | **+0.217**  | 8.93 → 5.64       |
| Tool-only (no retrieval)        | 15 | -0.043     | +0.052      | 0 → 0             |

The doc-using subset is where the v2 → v3 regression lived; v4 lands
both planned wins on it (faithfulness +0.055, groundedness +0.217).
Tool-only swings ±0.05 between any two judge runs — read it as a
noise floor, not as evidence v4 hurt tool answers. Max chunks per
query went 25 → 8; the cap is hitting cleanly.

**One known issue stays open for v5.** The v2/v3 tool-confusion class
(`compute_rental_yield` vs `abs_property_price_index` on agent-019)
is a routing-prompt fix and v4 only touched citation rules + chunk
cap. Closed in v5 (next section).

### 3e. v4 → v5 — compute-tool routing disambiguation (Task 3.30 + PR #117)

`_ROUTER_SYSTEM` now states explicitly that `compute_*` tools require
user-provided numbers, with worked GOOD/BAD examples for the
canonical confusion. Full breakdown in `results/agent_v4_vs_v5.md`.

The headline target — agent-019 ("If I take out a $850k loan...
median rental yields in Perth?") — is closed cleanly:

| Metric              | v4    | v5    |
|---------------------|-------|-------|
| tool_call_accuracy  | 0.250 | **1.000** |
| faithfulness        | 0.000 | 0.400 |
| groundedness        | 0.000 | 0.222 |

Routing now picks `sqm_rental_vacancy + abs_property_price_index` for
"median rental yields in Perth" instead of hallucinating
`compute_rental_yield` args. The carry-over from v2 → v3 → v4 is
done.

The single prompt change had two effects: it fixed agent-019 *and*
nudged 12 other queries from tool-only to doc-using (the worked
example shows `use_docs=True` for the lookup case). On the switched
cohort:

| Metric                | v4 (tool-only) | v5 (doc-using) | Δ          |
|-----------------------|----------------|----------------|------------|
| Faithfulness (judge)  | 0.567          | **0.677**      | **+0.110** |
| Publisher recall      | 0.917          | **1.000**      | **+0.083** |
| Groundedness (regex)  | 0.354          | 0.352          | -0.002     |

The +0.110 faith on this cohort is what carries the aggregate
faithfulness +0.027 and publisher_recall +0.053 wins.

The aggregate groundedness regex slipped 0.368 → 0.316. The drop is
on the both-doc-using cohort (-0.102) — same regex/judge mismatch v4
fixed, surfaced again on a different cohort because v5's routing
moved more queries into the doc-using regime. The judge sees
faithfulness +0.110 on the switched cohort; the regex doesn't get
credit until adjacent-citation discipline holds at higher chunk
counts. A future v6 could revisit citation adjacency on multi-tool
+ multi-doc synth, but it's not gating launch.

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
| Tasks 4.14–4.16 — Qdrant Cloud + Modal/HF deploy  | Cloud accounts                   | 2–3 hrs |
| Tasks 4.17–4.18 — full eval + final results table | Depends on 4.14–4.16             | 1 hr    |
| Tasks 4.19 / 4.23 / 4.24 — demo GIF + Loom        | Manual screen recording          | 1 hr    |
| Task 4.22 — publish blog + socials                | Manual                           | 30 min  |
| Task 4.26 — tag `v1.0.0`, flip repo public        | User sign-off                    | 5 min   |

The earlier-flagged dev-host environmental issue (Python SDK hangs
on `import openai` / `client.create()`) cleared once Docker Desktop
was running — the Qdrant container coming up may have settled DLL
load order or pagefile pressure. v3 eval (PR #112) ran cleanly end-
to-end through OpenAI. If it recurs, mitigations queued: WSL2,
Windows Defender exclusions for `%USERPROFILE%\AppData\Local\Programs\Python`
and the project venv, or a clean `python:3.11-slim` Docker container.

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
   the answers. The v3 retriever swap made this worse (more chunks
   to weave, looser inline-citation discipline). v4's prompt-level
   fix (GOOD/BAD example + "PLACEMENT IS LOAD-BEARING") clawed back
   the drop — aggregate groundedness 0.240 → 0.368, doc-using
   subset 0.268 → 0.485. Confirmed: the regression was a contract
   bug, not a quality bug.
6. **Aggregate eval numbers can hide route-conditional impact.** v3's
   flat aggregate split into a clean tool-only no-op vs a doc-using
   regression once subset by `n_retrieved_chunks > 0`. A retriever
   change is necessarily route-conditional; eval reporting should
   subset by route by default.
7. **A worked example in a routing prompt is a routing change.** v5
   added GOOD/BAD examples to `_ROUTER_SYSTEM` to disambiguate
   `compute_rental_yield`. The GOOD example happened to show
   `use_docs=True`, and 12 unrelated queries shifted from tool-only
   to doc-using as a result. Faithfulness on that switched cohort
   jumped +0.110 (judge approves), but groundedness regex slipped
   because adjacent-citation discipline is harder under more chunks.
   When a prompt edit nudges the planner's regime, expect downstream
   metrics to redistribute — not just the metric you targeted.
8. **The retrieval ft cell isn't dead weight.** ft-only dense
   collapsed (R@10=0.21), but slotted into the hybrid+rerank stack
   the FT vectors actually beat both base hybrid (0.747) and base
   hybrid+rerank (0.673) on R@10 (0.770). The reranker absorbs the
   FT publisher-collapse and reorders cleanly. Worth a follow-up for
   any future search-results UI; the agent default stays base hybrid
   because synth caps chunks at 8.

---

## 7. Reproducibility pointers

- Per-ticket merge log: `git log --oneline --no-merges main | grep "Task "`
- Retrieval numbers: `results/{baseline,bm25,hybrid,reranked}.json`,
  `results/hybrid_comparison.md`, `results/ablation.md`
- Agent eval: `results/agent_v{1,2,3,4,5}.json`,
  `results/agent_v1_vs_v2.md`, `results/agent_v2_vs_v3.md`,
  `results/agent_v3_vs_v4.md`, `results/agent_v4_vs_v5.md`,
  `results/agent_failure_analysis.md`
- Per-task design notes: squash-merge commit bodies on `main`
  (Task numbers in subject)
- Phase 4 progress.md notes: lines 12–319 of `progress.md` (gap
  notice + Phase 4 entries; per-ticket Phase 1–3 notes are in
  PRs #69–#107 commit bodies, not this file)
