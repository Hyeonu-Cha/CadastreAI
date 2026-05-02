# Agent eval v1 → v2 (Task 3.24)

Source: `results/agent_v1.json` (v1) vs `results/agent_v2.json` (v2),
both n=30, OpenAI provider (gpt-4o-mini classify/route/reflect/judge,
gpt-4o synthesise).

## Aggregate

| Metric                    | v1     | v2     | Δ       |
|---------------------------|--------|--------|---------|
| Tool-call accuracy        | 0.894  | 0.919  | +0.025  |
| Tool-call recall          | 0.956  | 0.978  | +0.022  |
| Tool-call precision       | 0.900  | 0.928  | +0.028  |
| Publisher recall          | 0.789  | 0.778  | -0.011  |
| Trajectory efficiency     | 0.299  | 0.672  | **+0.373** |
| Groundedness              | 0.373  | 0.296  | -0.077  |
| Faithfulness (LLM judge)  | 0.570  | 0.648  | **+0.078** |

## What changed

Two prompt/graph fixes from `results/agent_failure_analysis.md` landed:

- **Fix 1 — reflect convergence.** `MAX_ITERATIONS` 4 → 2 in
  `src/agent/graph.py`. The reflect tool's `is_complete` description
  flipped from *"Be strict — default to False when in doubt"* to
  *"Default to True unless there is a SPECIFIC, concrete gap"* and
  the reflector system prompt now explicitly says don't loop on
  stylistic concerns or generic "more context would help".
- **Fix 2 — synth must use tool data.** Added a "USE THE EVIDENCE"
  paragraph to `_SYNTHESIZER_SYSTEM` instructing the synth never
  to fall back on "I cannot provide" / "consult a real estate
  website" when the tool result above contains the answer.

Fix 3 (publisher-diversity rerank) was deferred — it's the lowest
leverage of the three and the v1→v2 change set is already large
enough to evaluate cleanly.

## Big wins

- **Trajectory efficiency +0.37 (124% relative).** v1 had 27/30
  queries pegged at the iteration ceiling (4); v2 has 12/30
  finishing in a single pass and only 18/30 hitting the new ceiling
  of 2. The reflect prompt no longer routes single-tool factual
  queries back into the planner.
- **Faithfulness +0.08 (LLM-judge).** The synth now actually quotes
  tool results. v1 had 6/30 queries at faithfulness=0 with answers
  like *"I cannot provide the current median Sydney house price"*
  while the tool had returned `$1,515,000`; in v2 those flip into
  cited answers. agent-002 (median Sydney price): 0.00 → 0.50.
  agent-017 (Victoria building approvals): 0.00 → 0.67. agent-027
  (NSW stamp duty + mortgage): 0.33 → 0.62.
- **Tool precision +0.028.** Lower iteration cap means fewer
  redundant retries; agent-030 (rental vacancy + price index)
  drops from 15 tool calls to 5; agent-013 from 8 to 4.
- **Latency.** Per-query elapsed time roughly halves (median
  ~14s vs ~33s), driven entirely by the iteration-cap drop.

## Regressions to flag

- **Groundedness -0.08.** Sixteen v2 rows have `groundedness=0` even
  though `faithfulness>0`, e.g. agent-008 (26 numeric claims, judge
  ratings 0.83, regex grounded 0.00) and agent-009 (10 numeric
  claims, judge 1.00, regex grounded 0.00). The synth is now writing
  numeric answers but isn't placing the `[tool:..., retrieved:...]`
  citation immediately adjacent to each number, so the regex-based
  groundedness counter misses them. The judge — which reads the
  whole answer — sees them fine. **Class of bug, not actual unsupported
  claims.** Fix in v3: tighten the synth prompt to require the
  citation token *immediately after* every numeric value.
- **Two regressions ≥0.2 faithfulness.** agent-004 (NSW dwelling
  approvals: 1.00 → 0.00) and agent-019 (loan + Perth yield:
  0.33 → 0.00). Same root cause for both — the planner picks the
  wrong tool (`compute_rental_yield` instead of
  `abs_property_price_index`), so synth can't find the data the
  question asks for. Tool-confusion was already a known v1 issue
  for agent-019; the lower iteration cap removes the extra round
  that v1 sometimes used to recover.
- **Publisher recall flat.** Doc-only queries still miss multi-
  publisher coverage on Treasury / Grattan / Productivity Commission.
  Tracked separately as Fix 3 for v3.

## Top fixes for v3 (when we get to it)

1. **Citation adjacency.** Synth prompt: "Place the
   `[tool:..., retrieved:YYYY-MM-DD]` citation immediately after
   the number it supports, e.g. *the median Sydney house price is
   $1,515,000 [tool:abs_property_price_index, retrieved:2026-05-02]*."
   Should recover most of the groundedness drop without changing
   any other behaviour.
2. **Tool-confusion guardrail.** When the user asks about *median
   prices* or *rental yields in city X*, the planner should prefer
   `abs_property_price_index` + `sqm_rental_vacancy` over
   `compute_rental_yield` (which needs `annual_rent_aud` /
   `property_value_aud` numbers the user typically doesn't supply).
   Add explicit examples to `_ROUTER_SYSTEM`.
3. **Publisher-diversity rerank.** Cap any single publisher at
   `ceil(K/3)` of the top-K returned to the agent, to lift
   `publisher_recall` on multi-source policy questions.

## What stays the same in v2

- Tool catalogue and routing schema unchanged — the planner is
  already strong (tool_recall 0.96 → 0.98 in v2).
- Decompose / classify nodes untouched.
- Citation post-processor (`_enforce_citations`) unchanged — it
  still tags orphans and appends the auto-Sources footer.
- All 377 unit tests still pass after the prompt + iteration-cap
  edits.
