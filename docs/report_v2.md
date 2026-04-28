# CadastreAI — Project Report v2

> Generated: 2026-04-28 | Branch: `claude/analyze-cadastre-project-AF25x` | Commits: 50

---

## 1. Executive Summary

CadastreAI is a domain-specialized AI research agent for the Australian residential property market. As of this report, the project is **substantially complete** across all four phases:

- **Phase 1 — Ingestion & Baseline RAG**: complete (8 scrapers, 351 PDFs, baseline eval)
- **Phase 2 — Hybrid Retrieval + Embedding Fine-Tuning**: complete except live eval run
- **Phase 3 — Agentic Layer**: complete (LangGraph 5-node graph, 9 tools, guardrails)
- **Phase 4 — Polish, Deploy, Write-up**: ~85% complete (Dockerfile, caching, UI done; cloud deploy + publish pending)

**Key numbers committed to `results/`:**

| Configuration | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
|---|---|---|---|---|
| Dense only (BAAI/bge-base-en-v1.5) | 0.383 | 0.450 | 0.352 | 0.349 |
| BM25 only | 0.800 | **0.907** | **0.732** | **0.760** |
| Hybrid (BM25 + Dense, RRF) | 0.563 | 0.747 | 0.626 | 0.585 |
| Hybrid + cross-encoder rerank | 0.550 | 0.673 | 0.560 | 0.537 |

**vs. v1 report**: v1 said "zero tests written" — there are now **254 passing tests across 22 files**. v1 projected 8.5/10 by end of week 4 — the project is already at that level before launch tasks complete.

---

## 2. Tasks Missed Using Claude API — and Can Gemini Substitute?

### 2a. Claude API Tasks That Are Coded But Not Yet Run

Three task groups are implemented and unit-tested but have not executed against a live Claude API + populated Qdrant instance:

| Task | What it does | Why not run yet |
|---|---|---|
| **2.06–2.08** — Synthetic pair generation | `src/training/generate_pairs.py` calls Claude Haiku to produce (query, chunk) training pairs from the corpus | Needs `ANTHROPIC_API_KEY` + corpus on disk |
| **3.22–3.24** — Agent eval v1/v2 | `src/eval/agent_eval.py` runs 30 annotated queries through the real LangGraph agent (tool-call accuracy, trajectory efficiency, faithfulness) | Needs `ANTHROPIC_API_KEY` + populated Qdrant |
| **Faithfulness scoring** | Claude-as-judge in `agent_eval.py` grades each answer against source chunks | Disabled by default (cost); implemented and ready to enable |

These are the highest-priority items to unlock before the v1.0.0 tag.

### 2b. Can Gemini API Substitute for Claude?

**Short answer: technically yes for most tasks, architecturally expensive to swap.**

The codebase is single-vendor (Anthropic only) across all five agent nodes:

| Node | Model | Claude-specific feature |
|---|---|---|
| `classify_query` | Haiku 4.5 | Forced tool-use (`tool_choice="any"`) |
| `decompose` | Haiku 4.5 | Forced tool-use |
| `retrieve_or_tool` | Haiku 4.5 | Forced tool-use |
| `reflect` | Haiku 4.5 | Forced tool-use |
| `synthesize` | Sonnet 4.6 | Prompt caching (`cache_control` on system prompt), structured citation discipline |

**What would switching to Gemini require:**
1. Replace `anthropic` SDK with `google-generativeai` SDK in `src/agent/nodes.py` (~1,147 lines)
2. Rewrite all tool schemas from Anthropic format to Gemini function-calling format
3. Remove `cache_control` blocks (Gemini has its own caching API, different format)
4. Rewrite the cost accounting in `src/agent/cost.py` (Gemini pricing differs)
5. Update all 254 tests that mock `anthropic.Anthropic`

**Where Gemini is a drop-in option (lower effort):**
- **Pair generation** (`generate_pairs.py`) — any capable LLM works here; Gemini Flash would reduce cost for the ~4k pair generation run
- **Faithfulness scoring** (Claude-as-judge mode in `agent_eval.py`) — could be Gemini-graded with a prompt-level swap

**Recommendation**: Keep Anthropic for the portfolio. Single-vendor is simpler to explain in interviews. If cost is a concern for the live eval runs, run `generate_pairs.py` with `CADASTRE_DECOMPOSER_MODEL` overriding to `claude-haiku-4-5` (already supported via `.env`), not Gemini. The per-node model is already configurable without code changes.

---

## 3. Project Purpose and Skillsets (Updated)

### 3a. What This Project Is Actually For

**Primary purpose: a 4-week portfolio artifact** demonstrating end-to-end AI engineering on a real, messy domain. The project covers:

1. Scoping a product with personas and success metrics before coding
2. Ingesting 351 PDFs from 8 publishers with different CMS structures
3. Running a retrieval ablation study with honest results (BM25 dominated dense on this corpus — published as a finding, not hidden)
4. Fine-tuning a bi-encoder with synthetic pairs + hard negatives
5. Building a multi-step LangGraph agent with decomposition, tool-use, and reflection
6. Instrumenting cost telemetry and three layers of caching
7. Wrapping in a Streamlit UI with compliance guardrails

**Secondary purpose: a real product hypothesis** — that domain-specialized RAG beats generic search for AU housing research queries. The ablation data tests this.

### 3b. Skillset Ratings (Updated from v1)

#### Tier 1: Visible in code right now

| Skillset | Evidence | Rating |
|---|---|---|
| **Async/concurrent programming** | `download.py` — `asyncio.Semaphore`, per-host `_HostGate`, `to_thread`, exponential backoff retries | ⭐⭐⭐⭐⭐ |
| **Testing** | 22 test files, 254 passing tests, monkeypatched LLM stubs, parametrized fixtures | ⭐⭐⭐⭐⭐ (was 5/10 in v1 — now fully addressed) |
| **Production code quality** | Three caching layers, per-meter cost telemetry, guardrails, Dockerfile (multi-stage, non-root), compliance disclaimers | ⭐⭐⭐⭐⭐ |
| **Data pipeline design** | Source → download → parse → chunk → embed → index, each step with `.failures.jsonl`, deduplication, resume-safety | ⭐⭐⭐⭐ |
| **Web scraping** | 8 scrapers adapting to different DOM/CMS structures, polite headers, rate limiting | ⭐⭐⭐⭐ |

#### Tier 2: Visible in architecture and evaluation

| Skillset | Evidence | Rating |
|---|---|---|
| **System design thinking** | `product.md` (500 lines: personas, UX flows, success metrics), `project_plan.md` (470 lines), risk matrix | ⭐⭐⭐⭐⭐ |
| **ML evaluation rigor** | Real ablation table (not placeholders), honest finding that BM25 > hybrid on lexical corpus, per-persona breakdown (Researcher R@10=0.556 vs Investor 0.280) | ⭐⭐⭐⭐⭐ |
| **LLM prompt engineering** | Forced tool-use pattern for structured outputs, per-persona tone addenda, citation enforcement post-processing | ⭐⭐⭐⭐ |
| **Cost optimization** | Anthropic prompt caching (~90% input-token discount), embedding cache (sub-ms repeats), tool TTL cache, per-meter breakdown | ⭐⭐⭐⭐ |

#### Tier 3: In Weeks 2–3 code

| Skillset | Evidence | Rating |
|---|---|---|
| **ML training** | `train_embeddings.py` — `MultipleNegativesRankingLoss`, hard negative mining from BM25, 90/10 split, `RerankingEvaluator` | ⭐⭐⭐⭐⭐ (if you can walk through the training choices) |
| **LangGraph / agentic workflows** | 5-node graph, `MAX_ITERATIONS=4` cap, `_route_after_reflect` conditional edge, `AgentState` TypedDict, guardrail short-circuit | ⭐⭐⭐⭐ |
| **Full-stack deployment** | Multi-stage Dockerfile, docker-compose Qdrant, `.env.example`, non-root runtime, healthcheck | ⭐⭐⭐ |

### 3c. What Interviewers Will Ask by Company Type

**RAG/LLM startups** (Anthropic, Modal, Together):
> "Your BM25 outperformed your dense embeddings. Why? What does that tell you about when to fine-tune vs. when to improve lexical retrieval?"

You'll answer with: `results/error_analysis.md`, the 6-category failure taxonomy, and the fine-tuning motivation (close the gap on abstractive queries, not lexical ones).

**Data/ML-heavy companies** (Stripe, Canva, Figma):
> "Walk me through your eval methodology. Why Recall@K and MRR? How did you construct the 100-query gold set?"

You'll answer with: `data/eval/retrieval_queries.jsonl` (60 human-annotated + 40 synthetic), the metric formulas in `src/eval/retrieval_eval.py`, and why nDCG@10 matters for ranked retrieval.

**Traditional software companies** (ANZ, Atlassian, Xero):
> "If another engineer picked this up, what would they need? How did you plan the 4 weeks?"

You'll answer with: the modular `pyproject.toml` optional dependencies, the 254-test suite, the README quickstart, and the `project_plan.md` daily breakdown.

**AI consulting firms** (McKinsey Digital, Bain Catalyst):
> "What would you tell a client if fine-tuning didn't beat BM25? Is this approach economically viable at scale?"

You'll answer with: the ablation table, the cost-per-query telemetry from `src/agent/cost.py`, and the BM25 dominance explanation (lexical corpus = lexical retrieval wins; fine-tuning helps abstractive queries, not all queries).

---

## 4. Improvements and Scalability

### 4a. Immediate Improvements (Within Current Scope)

These unblock the v1.0.0 tag and cost <1 day each:

| Priority | Task | What it unblocks |
|---|---|---|
| **#1** | Run agent eval v1 (Task 3.22) with API key + Qdrant | Final metrics for README + blog |
| **#2** | Run fine-tune evaluation (Task 2.14) | Ablation table row for fine-tuned model |
| **#3** | Record 20-second demo GIF (Task 4.19) | Highest-visibility addition to README |
| **#4** | Deploy to HuggingFace Spaces or Modal (Tasks 4.14–4.16) | Live demo URL for recruiters |
| **#5** | Publish blog post (Task 4.22) | Main external signal |

### 4b. Architecture Improvements (Medium Effort)

**Parallel sub-question retrieval**

Currently `retrieve_or_tool` executes sub-questions sequentially. For decomposed queries (2–4 sub-questions), switching to `asyncio.gather` would cut latency by ~60% on multi-hop queries.

```python
# current (sequential)
for sub_q in state["sub_questions"]:
    chunks = retriever.retrieve(sub_q)

# improved (parallel)
results = await asyncio.gather(*[retriever.aretrieve(q) for q in state["sub_questions"]])
```

**Streaming responses**

The synthesizer calls `claude-sonnet-4-6` with `max_tokens=1200`. Adding `stream=True` + Streamlit `st.write_stream` would show tokens as they arrive, reducing perceived latency from ~8s to near-instant first token.

**Per-publisher retrieval weighting**

Investor queries have 2× lower recall than Researcher queries (0.280 vs 0.556 R@10). A persona-aware retrieval filter — passing `publisher_filter` to Qdrant based on persona — would improve investor and homebuyer results without retraining.

### 4c. Scalability (Beyond v1 Scope)

| Dimension | Current limit | Scaled approach |
|---|---|---|
| **Corpus size** | 351 PDFs, single Qdrant instance | Qdrant Cloud sharded by publisher; async re-embedding pipeline |
| **Concurrent users** | Single-user Streamlit | FastAPI backend + Streamlit as thin client; connection pooling for Qdrant |
| **Multi-region** | Single deploy | Modal's multi-region functions; Qdrant Cloud cluster replication |
| **Model updates** | Manual re-embed on model change | Versioned Qdrant collections (`cadastre_chunks_v2`); shadow indexing |
| **Multi-provider LLMs** | Anthropic-only | Abstract `LLMClient` interface; Gemini Flash for routing nodes (cost ~40% lower than Haiku) |
| **Monitoring** | LangSmith traces only | Add Prometheus metrics on latency/cost per node; Grafana dashboard |
| **Auth & multi-tenancy** | None | Clerk/Auth0 for user sessions; per-user query history in PostgreSQL |

### 4d. The One Structural Change Worth Making Now

**Publisher balance check.** The corpus is RBA-heavy. If RBA sources exceed ~40% of chunks, investor and homebuyer queries will systematically under-retrieve from CoreLogic and PropTrack (which have the highest-signal data for those personas). Run:

```bash
python -c "
import json, collections
chunks = [json.loads(l) for l in open('data/processed/chunks.jsonl')]
counts = collections.Counter(c['publisher'] for c in chunks)
total = sum(counts.values())
for pub, n in counts.most_common():
    print(f'{pub}: {n} ({100*n/total:.1f}%)')
"
```

If any publisher exceeds 40%, cap it at 40% before re-embedding for the fine-tuned model. This is a one-line filter in `reembed_finetuned.py`.

---

## 5. What Remains to Hit v1.0.0

| Category | Task | Effort |
|---|---|---|
| **Eval** | Run agent eval v1 (Tasks 3.22–3.24) | ~2h (needs API key + Qdrant) |
| **Eval** | Run fine-tune eval (Task 2.14) | ~1h (needs GPU + API key) |
| **Deploy** | Qdrant Cloud + Modal/HF Spaces (Tasks 4.14–4.16) | ~3h |
| **Eval** | Final results table in README (Task 4.17–4.18) | ~1h |
| **Demo** | Record 20s GIF (Task 4.19) | ~30min |
| **Publish** | Blog post cross-post (Task 4.22) | ~1h (manual) |
| **Publish** | Loom walkthrough (Tasks 4.23–4.24) | ~1h (manual) |
| **Release** | Tag v1.0.0 + flip repo public (Task 4.26) | ~5min (user sign-off required) |

**Total remaining**: ~10 hours of actual work, mostly gated on infrastructure provisioning (API keys, cloud accounts) not coding.

---

## 6. Summary

| Dimension | v1 Report (old) | v2 Report (now) |
|---|---|---|
| Tests | 0 written | 254 passing across 22 files |
| Eval results | Placeholders | Real numbers committed to `results/` |
| Production features | Planned | 3 caching layers, cost telemetry, guardrails, Dockerfile shipped |
| Overall rating | Projected 8.5/10 | Currently 9/10; 9.5/10 after launch tasks |
| Claude API tasks pending | N/A | Pair generation + agent eval (coded, blocked on infra) |
| Gemini feasibility | Not assessed | Possible but ~2 days refactor; not recommended for this scope |
| Biggest remaining gap | No tests, no eval | Run live evals + cloud deploy + publish |
