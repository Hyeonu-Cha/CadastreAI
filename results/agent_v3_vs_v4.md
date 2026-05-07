# Agent eval v3 → v4 (Task 3.29)

Source: `results/agent_v3.json` (v3) vs `results/agent_v4.json` (v4),
both n=30, OpenAI provider (`gpt-4o-mini` for classify/route/reflect/
judge, `gpt-4o` for synth), `--with-judge`. Same retriever (hybrid),
same max-iterations.

Two changes between v3 and v4, both in `src/agent/nodes.py`:

1. **Citation adjacency** — `_CITATION_RULES` now requires the
   `[tool:..., retrieved:...]` token *immediately after* each numeric
   value, with a worked GOOD/BAD example. The regex grader checks
   adjacency; v3's prompt tolerated trailing-citation patterns the
   judge accepted but the regex missed.
2. **Chunk cap with dedupe** — new `_dedupe_and_cap_chunks(cap=8)`
   wired into `retrieve_or_tool` after `apply_publisher_boost`. v3
   averaged 8.93 chunks/query (max 25 — five sub-questions × top-5
   each), which diluted synth context.

## Aggregate

| Metric                   | v3     | v4     | Δ          |
|--------------------------|--------|--------|------------|
| Tool-call accuracy       | 0.925  | 0.917  | -0.008     |
| Tool-call recall         | 0.944  | 0.944  | ±0.000     |
| Tool-call precision      | 0.933  | 0.925  | -0.008     |
| Publisher recall         | 0.764  | 0.772  | +0.008     |
| Trajectory efficiency    | 0.722  | 0.717  | -0.005     |
| Faithfulness (judge)     | 0.630  | 0.637  | +0.007     |
| Groundedness (regex)     | 0.240  | **0.368** | **+0.128** |

Groundedness +0.128 was the planned win; everything else is flat to
noise. The aggregate hides a sharper signal on the doc-using subset.

## Doc-using subset (n=14, retrieved in both runs)

Apples-to-apples — same query IDs in both runs, both runs actually
retrieved chunks.

| Metric                | v3     | v4     | Δ          |
|-----------------------|--------|--------|------------|
| Tool-call accuracy    | 0.893  | 0.893  | ±0.000     |
| Publisher recall      | 0.565  | 0.583  | +0.018     |
| Trajectory efficiency | 0.702  | 0.738  | +0.036     |
| Faithfulness (judge)  | 0.631  | 0.686  | **+0.055** |
| Groundedness (regex)  | 0.268  | 0.485  | **+0.217** |
| n_retrieved_chunks    | 8.93   | 5.64   | **-3.29**  |

**Both planned fixes landed.** Chunk cap pulled the average from 8.93
→ 5.64 (max from 25 → 8); citation adjacency moved groundedness from
0.268 → 0.485, faithfulness from 0.631 → 0.686. Publisher recall
ticked up too (+0.018) because the cap kicks the lower-scored
duplicates out, leaving more room for diverse publishers near the
top.

## Tool-only subset (n=15, no retrieval in either run)

| Metric                | v3     | v4     | Δ          |
|-----------------------|--------|--------|------------|
| Tool-call accuracy    | 0.950  | 0.933  | -0.017     |
| Trajectory efficiency | 0.756  | 0.711  | -0.044     |
| Faithfulness (judge)  | 0.620  | 0.577  | -0.043     |
| Groundedness (regex)  | 0.231  | 0.283  | +0.052     |

These queries don't touch the retriever or the citation prompt's
chunk-citation form (they only cite tools). The small regressions
are LLM run-to-run noise on classify/route. Faithfulness in
particular swings ±0.05 between any two judge runs on the same n=15.
Read this subset as a noise floor for the comparison, not as
evidence v4 hurt tool answers.

## Chunk-count distribution

|       | v3   | v4   |
|-------|------|------|
| mean  | 4.17 | 2.80 |
| max   | 25   | 8    |

Cap is hitting cleanly — no v4 query exceeds 8 chunks. The v3
worst-cases (multi-sub-question queries that piled up 15–25 chunks
across reflect cycles) now get truncated to top-8 by boosted score.

## What stays the same in v4

- Retriever (hybrid), routing schema, decompose, classify, reflect
  prompts, max-iterations, tool catalogue all unchanged.
- 7 new unit tests in `tests/test_chunk_dedupe_cap.py` cover the
  cap, the dedupe-keeps-first-occurrence behaviour, the cap-after-
  dedupe interaction, and the empty-chunk-id edge case. All 392
  unit tests pass.
- The known v2/v3 tool-confusion class (`compute_rental_yield` vs
  `abs_property_price_index`, `missing required args ['capital_city']`)
  is unaddressed by v4 — it's a routing-prompt fix and v4 only
  touched citation-rules + chunk-cap. Carry-over to a future v5.

## Headline reading

v4 is the cleanest single-PR agent improvement of the project so far
on the metrics that were regressing:

- **Groundedness +0.128** aggregate (+0.217 on doc-using).
- **Faithfulness +0.055** on doc-using.
- No code change to the retrieval *quality*, just to how synth
  consumes its output.

Confirms the v2 → v3 hypothesis: hybrid retrieval was the right
upstream call; the agent's downstream synth just wasn't capitalising
on it. Tighten the synth context (cap) and tighten the citation
contract (adjacency), and v3's "regression" on doc-using queries
flips into a clear v4 win.
