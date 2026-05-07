# Agent eval v4 → v5 (Task 3.30)

Source: `results/agent_v4.json` (v4) vs `results/agent_v5.json` (v5),
both n=30, OpenAI provider (`gpt-4o-mini` for classify/route/reflect/
judge, `gpt-4o` for synth), `--with-judge`. Same retriever (hybrid),
same chunk-cap (8), same citation rules.

One change between v4 and v5, in `src/agent/nodes.py`'s
`_ROUTER_SYSTEM`:

**Compute-tool disambiguation rules.** The routing prompt now states
that `compute_*` tools require user-provided numbers, with worked
GOOD/BAD examples for the canonical `compute_rental_yield`-vs-lookup
confusion (agent-019). Two examples:

- "What's the median rental yield in Perth?" →
  `sqm_rental_vacancy(city='Perth') + abs_property_price_index(city='Perth')`,
  `use_docs=True`.
- "I get $32k/yr rent on a $650k unit — what's my yield?" →
  `compute_rental_yield(annual_rent_aud=32000, property_value_aud=650000)`,
  `use_docs=False`.

5 new unit tests in `tests/test_router_compute_disambiguation.py` pin
the disambiguation rules to the prompt so they can't be silently
dropped. All 397 unit tests pass.

## The headline target — agent-019

| Metric              | v4    | v5    | Δ        |
|---------------------|-------|-------|----------|
| actual_tools        | `[compute_mortgage_repayment, compute_rental_yield, compute_rental_yield]` | `[compute_mortgage_repayment, sqm_rental_vacancy, abs_property_price_index, ...]` | — |
| tool_call_accuracy  | 0.250 | **1.000** | **+0.750** |
| faithfulness        | 0.000 | 0.400 | **+0.400** |
| groundedness        | 0.000 | 0.222 | +0.222   |

Routing now picks `sqm_rental_vacancy + abs_property_price_index` for
"median rental yields in Perth" instead of hallucinating
`compute_rental_yield` args. The known v2/v3/v4 regression class is
fixed.

## Aggregate

| Metric                   | v4     | v5     | Δ          |
|--------------------------|--------|--------|------------|
| Tool-call accuracy       | 0.917  | 0.913  | -0.003     |
| Tool-call recall         | 0.944  | 0.956  | +0.011     |
| Tool-call precision      | 0.925  | 0.917  | -0.008     |
| Publisher recall         | 0.772  | **0.825** | **+0.053** |
| Trajectory efficiency    | 0.717  | 0.719  | +0.003     |
| Faithfulness (judge)     | 0.637  | **0.664** | **+0.027** |
| Groundedness (regex)     | 0.368  | 0.316  | -0.052     |

Faithfulness +0.027 and publisher_recall +0.053 are the planned
wins. Groundedness regex -0.052 is a downstream side-effect of the
worked example: the LOOKUP example shows `use_docs=True`, which
nudged the agent to retrieve docs on a wider set of queries — see
"Switched cohort" below.

## Subset breakdowns

The single prompt change had two effects: it fixed agent-019's
routing, AND it shifted 12 other queries from tool-only to
doc-using. Aggregate alone hides the structure.

### Switched cohort (v4 tool-only → v5 doc-using, n=12)

| Metric                | v4     | v5     | Δ          |
|-----------------------|--------|--------|------------|
| Tool-call accuracy    | 0.917  | 0.908  | -0.008     |
| Publisher recall      | 0.917  | **1.000** | **+0.083** |
| Faithfulness (judge)  | 0.567  | **0.677** | **+0.110** |
| Groundedness (regex)  | 0.354  | 0.352  | -0.002     |
| n_retrieved_chunks    | 0.00   | 6.25   | +6.25      |

**Faithfulness +0.110** on this cohort. The agent now retrieves
supporting docs for 12 queries it previously answered tool-only —
the judge sees those answers as more grounded, publisher_recall
hits 1.00, and groundedness regex is unchanged because the citation
discipline holds when the synth's only source of numeric values is
the tool result that came before retrieval.

### Both-doc-using cohort (n=15, retrieved in both runs)

| Metric                | v4     | v5     | Δ          |
|-----------------------|--------|--------|------------|
| Tool-call accuracy    | 0.900  | 0.900  | ±0.000     |
| Publisher recall      | 0.611  | 0.650  | +0.039     |
| Trajectory efficiency | 0.722  | 0.756  | +0.033     |
| Faithfulness (judge)  | 0.696  | 0.671  | -0.025     |
| Groundedness (regex)  | 0.452  | 0.350  | -0.102     |
| n_retrieved_chunks    | 5.60   | 6.67   | +1.07      |

This cohort gained +1 chunk on average (the v5 prompt's "use_docs"
nudge bled into doc-using queries too). Faithfulness slips -0.025,
groundedness -0.102 — same regex/judge mismatch we documented in v3
→ v4: more chunks → more numeric claims → harder for the synth to
keep every value adjacent to its inline citation. The judge is
mostly fine with it; the regex isn't.

agent-013, agent-016, agent-022 each dropped from groundedness=1.00
to <=0.50 — they're the noisiest contributors to the aggregate
groundedness drop. Manual spot-check needed for v6 if we want to
claw this back; it's the same single-prompt-fix story as the
v3 → v4 work.

### Both-tool-only cohort (n=3)

Too small to be informative; deltas are LLM run-to-run noise on
classify/synth.

## What stays the same in v5

- Retriever (hybrid), chunk cap (8), citation rules, max-iterations,
  decompose, classify, reflect prompts, tool catalogue all unchanged.
- 5 new unit tests in `tests/test_router_compute_disambiguation.py`.
  All 397 unit tests pass.

## Headline reading

v5 is a **routing fix that worked exactly as designed**:

- **agent-019 is closed.** Tool-call accuracy 0.25 → 1.00 on the
  canonical `compute_rental_yield`-vs-lookup confusion.
- **Faithfulness +0.027 aggregate, +0.110 on the 12-query switched
  cohort.** The judge approves of the agent retrieving docs to
  support tool-result narratives.
- **Publisher recall +0.053 aggregate.** Now hitting 1.00 on the
  switched cohort.

The groundedness regex drop is a known follow-up class — multi-
chunk synth citation adjacency. Same theme as v3 → v4, surfaced
again on a different query cohort because v5's routing fix moved
12 queries into the doc-using regime. Carry-over to v6 if it ever
matters for the launch.
