# Agent eval v2 → v3 (Task 3.28)

Source: `results/agent_v2.json` (v2, dense retriever) vs
`results/agent_v3.json` (v3, hybrid retriever via Task 3.25 default).
Both n=30, OpenAI provider (`gpt-4o-mini` for classify/route/reflect/
judge, `gpt-4o` for synth), `--with-judge`.

The only intentional change between v2 and v3 is the agent's docs
retriever: dense-only BGE → BM25 + dense BGE via RRF (`HybridRetriever`).
Routing, prompts, max-iterations, tool catalogue all unchanged.

## Aggregate

| Metric                   | v2     | v3     | Δ          |
|--------------------------|--------|--------|------------|
| Tool-call accuracy       | 0.919  | 0.925  | +0.006     |
| Tool-call recall         | 0.978  | 0.944  | -0.034     |
| Tool-call precision      | 0.928  | 0.933  | +0.005     |
| Publisher recall         | 0.778  | 0.764  | -0.014     |
| Trajectory efficiency    | 0.672  | 0.722  | **+0.050** |
| Faithfulness (judge)     | 0.648  | 0.630  | -0.018     |
| Groundedness (regex)     | 0.296  | 0.240  | -0.056     |

The aggregate looks flat — small win on trajectory, small regression
on faithfulness. The honest story shows up on subsets.

## Why the aggregate is misleading: route-conditional impact

Hybrid retrieval *only matters when the agent retrieves docs*. Most
queries in this set route to deterministic tools (RBA cash rate,
ABS price index, SQM vacancy) that don't touch the retriever at all.
Splitting by retrieval status surfaces two opposing effects.

### Doc-using queries (n=14, retrieved in both runs)

Apples-to-apples — same query IDs in both v2 and v3, both runs
actually retrieved chunks.

| Metric                | v2     | v3     | Δ          |
|-----------------------|--------|--------|------------|
| Tool-call accuracy    | 0.929  | 0.893  | -0.036     |
| Tool-call recall      | 1.000  | 0.929  | -0.071     |
| Publisher recall      | 0.595  | 0.565  | -0.030     |
| Trajectory efficiency | 0.690  | 0.702  | +0.012     |
| Faithfulness (judge)  | 0.707  | 0.631  | **-0.076** |
| Groundedness (regex)  | 0.406  | 0.268  | **-0.138** |
| n_retrieved_chunks    | 7.50   | 8.93   | +1.43      |

**Hybrid retrieves +19% more chunks on average and faithfulness
regresses.** The retrieval benchmark (synthetic split, R@10 hybrid
0.878 vs dense 0.829) said hybrid was strictly better; at the agent
level on this 30-query eval it isn't translating into faithfulness
wins on doc-using queries.

Per-query breakdown (faithfulness, doc-using subset):

| Query     | v2   | v3   | Δ      |
|-----------|------|------|--------|
| agent-010 | 0.57 | 0.75 | +0.18  |
| agent-011 | 0.56 | 0.67 | +0.11  |
| agent-012 | 1.00 | 0.89 | -0.11  |
| agent-013 | 0.67 | 0.50 | -0.17  |
| agent-016 | 0.50 | 0.67 | +0.17  |
| agent-017 | 0.67 | 0.50 | -0.17  |
| agent-021 | 0.33 | 0.43 | +0.10  |
| agent-022 | 1.00 | 0.60 | **-0.40** |
| agent-023 | 0.75 | 0.50 | -0.25  |
| agent-024 | 0.83 | 0.57 | -0.26  |
| agent-026 | 1.00 | 1.00 | ±0.00  |
| agent-027 | 0.62 | 0.50 | -0.12  |
| agent-028 | 0.78 | 0.67 | -0.11  |
| agent-030 | 0.62 | 0.60 | -0.03  |

4 wins ≥ +0.10, 7 losses ≥ -0.10, 3 flat. The losses are
larger in magnitude than the wins.

### Tool-only queries (n=14, no retrieval in either run)

| Metric                | v2     | v3     | Δ          |
|-----------------------|--------|--------|------------|
| Tool-call accuracy    | 0.923  | 0.946  | +0.024     |
| Tool-call recall      | 0.952  | 0.952  | ±0.000     |
| Publisher recall      | 0.929  | 0.929  | ±0.000     |
| Trajectory efficiency | 0.655  | 0.738  | **+0.083** |
| Faithfulness (judge)  | 0.555  | 0.604  | **+0.048** |
| Groundedness (regex)  | 0.196  | 0.220  | +0.023     |

Hybrid is supposed to be *no-op* on these — they don't retrieve. The
small wins (+0.08 traj_eff, +0.05 faith) are noise / re-roll variance:
LLM nondeterminism in classify/route, judge nondeterminism on
faithfulness scoring. We should treat these as a noise-floor estimate
for the comparison, not as evidence the retriever switch helped.

### Routing changes (n=2)

Two queries routed differently between runs: agent-020 and agent-029
retrieved docs in v2 but went tool-only in v3. This is router-LLM
nondeterminism, not a retriever effect — `_router_system` was
unchanged. Excluded from both subset comparisons above.

## What's going wrong on doc-using queries

Two effects, both load-related:

1. **More chunks → noisier synth context.** Hybrid pulls 8.93 chunks
   on average vs dense's 7.50. Top-K is unchanged (`docs.k=10` in
   `nodes._retrieve_docs`); the difference is RRF tending to fill
   the shortlist more aggressively because BM25 contributes
   high-scoring lexical matches on top of BGE's semantic ones. With
   more chunks competing for the synth's attention budget, the
   citations spread thinner — agent-022, -023, -024 all show this:
   the answers cite *more* sources but each citation is less
   tightly grounded.
2. **BM25-leaning matches drag publisher recall down.** Publisher
   recall on the doc-using subset drops from 0.595 → 0.565. BM25
   tends to pull the same publisher repeatedly when a query mentions
   a specific term ("Division 43" → ATO docs cluster, "FHG" →
   Treasury cluster). The dense retriever spreads across publishers
   more naturally on multi-evidence questions.

The **groundedness -0.138** drop is the *citation-adjacency* class
of bug we already flagged for v2 → v3 in `agent_v1_vs_v2.md`: the
synth writes numeric answers but doesn't place
`[tool:..., retrieved:...]` immediately adjacent to each value, so
the regex grader misses them while the judge sees them. With more
chunks in v3, this surfaces more — the synth is being asked to
weave more sources into one answer and the inline-citation
discipline slips further.

## What to do with this

**Hybrid stays the default.** The retriever benchmark on the honest
synthetic split said hybrid wins (R@10 0.878 vs 0.829), and the
**tool-only** subset confirms it doesn't hurt the bulk of agent
queries. The right interpretation of v3 is: hybrid retrieval is a
correct decision *upstream*, but the agent's downstream synth doesn't
yet capitalise on the improved retrieval.

Three follow-ups for v4 (sequenced by leverage):

1. **Citation-adjacency in the synth prompt.** Carry-over from v2 →
   v3 (`agent_v1_vs_v2.md` §6.1). Same fix recovers groundedness
   for v3 too — likely the single biggest measurable win available.
2. **Lower agent docs.k from 10 → 5 or 6.** The +19% chunk count is
   diluting the synth context. The retrieval benchmark's R@5 for
   hybrid is 0.805 — already strong, and a tighter context window
   forces the synth to commit to the highest-scored chunks instead
   of hedging across nine.
3. **Publisher-diversity rerank.** Cap any single publisher at
   `ceil(K/3)` of the top-K returned to the agent, to recover the
   -0.030 publisher_recall regression and the cluster-bias on
   BM25-favoured terms. This is Fix 3 deferred from
   `agent_failure_analysis.md`.

## What stays the same in v3

- Routing schema, decompose, classify, reflect prompts unchanged.
- All 385 unit tests still pass after Task 3.25 (8 new tests in
  `tests/test_agent_retriever.py`).
- Tool catalogue and post-processor (`_enforce_citations`)
  unchanged. The `abs_property_price_index` / `sqm_rental_vacancy`
  arg mismatches visible in the v3 log (`missing required args
  ['capital_city']`, `['postcode_or_city']`) are the same v2
  tool-confusion class flagged for fix in v3 — it was unaddressed
  by the retriever change, as expected.
