# Agent eval v1 — failure analysis (Task 3.23)

Source: `results/agent_v1.json` (n=30, OpenAI provider: gpt-4o-mini for
classify/route/reflect/judge, gpt-4o for synthesise).

## Headline numbers

| Metric                         | Mean   |
|--------------------------------|--------|
| Tool-call accuracy             | 0.894  |
| Tool-call recall               | 0.956  |
| Tool-call precision            | 0.900  |
| Publisher recall (doc queries) | 0.789  |
| Trajectory efficiency          | 0.299  |
| Groundedness                   | 0.373  |
| Faithfulness (LLM-as-judge)    | 0.570  |

The agent picks the right tools (`tool_acc=0.89`, `tool_recall=0.96`),
but it answers poorly: faithfulness is only 0.57 and trajectory
efficiency collapses to 0.30 because **27 of 30 queries hit the
4-iteration ceiling** even when the correct answer needed one tool
call. The bottleneck is the reflect-loop, not retrieval and not tool
selection.

## Failure buckets

### Bucket A — Reflection never converges (27/30 queries)

Symptom: `iteration_count == max_iterations (4)` for 27 of 30 queries,
including queries with a single expected tool. Each iteration replays
the same tool, so the trajectory length explodes:

- `agent-002` ("median Sydney house price") — `abs_property_price_index` called **4×**, single expected tool.
- `agent-013` ("price index for Sydney") — called **8×**.
- `agent-027` (NSW first home buyer, $1M house) — `compute_stamp_duty_nsw` called **5×**, `compute_mortgage_repayment` **4×**.
- `agent-030` ("rental vacancy + price index") — 15 tool calls for two expected tools.

20 of 30 queries call the same tool at least twice; 7 call it ≥4×.
Even queries with no expected tools (doc-only `agent-011`, `agent-023`,
`agent-026`) burn 4 iterations before synthesising.

**Root cause hypothesis.** The reflect prompt is biased toward "needs
more evidence" and routes back to the planner regardless of whether
the synthesised draft actually answers the question. Combined with
`max_iterations=4`, every query maxes out the budget.

### Bucket B — Tool result not reaching synthesis (6/30 queries, faithfulness=0)

Symptom: the right tool fires, but the final answer says "I cannot
provide …" and the judge rates faithfulness 0:

- `agent-002` — calls `abs_property_price_index` 4× yet answers "I cannot provide the current median house price for Sydney … recommend consulting real estate websites or reports."
- `agent-007` (NSW stamp duty, $700k apartment) — calls `compute_stamp_duty_nsw` 4× yet answers "the evidence available does not include specific figures."
- `agent-014` (multi-tool stamp duty + mortgage + yield) — 11 tool calls, answer falls back to generic phrasing ("you would calculate stamp duty based on NSW government rules"), no numbers.
- `agent-017` (Victoria building approvals) — `abs_building_approvals` called 4× yet answer says "Building approvals data specifically for Victoria over the past year was not retrieved."

These are exactly the queries Bucket A loops on: the synthesiser sees
no useful evidence, drafts a refusal, reflect says "still missing
data, try again", planner replays the tool. The pattern is a
plumbing bug — the structured tool result is not surviving the
hand-off into the synthesis context — not a tool-output bug. (The
tools themselves return the right shape; `agent-001` and `agent-006`
prove that on near-identical queries.)

### Bucket C — Publisher recall on doc-only queries (5/7 misses)

Symptom: doc-only queries (`expected_tools=[]`) retrieve plenty of
chunks but from a narrow set of publishers:

- `agent-011` (build-to-rent + tax) — retrieved 20 chunks, hit AHURI + Treasury, missed NHFIC.
- `agent-023` (housing-affordability reform proposals) — 20 chunks, AHURI only; missed Treasury, Grattan, Productivity Commission.
- `agent-026` (negative gearing cost & reforms) — 20 chunks, AHURI only; missed Treasury, Grattan, Productivity Commission.
- `agent-028`, `agent-024` — same shape: AHURI-dominated retrieval, multi-publisher gold.

The retriever returns enough chunks but the top-K is dominated by
one publisher (AHURI is the largest source by volume), so policy /
budget questions miss the cross-publisher coverage the rubric expects.

## Out-of-scope smaller issues (not in the top fixes)

- **Namespace leak — 1/30.** `agent-016` records `functions.rba_cash_rate` alongside `rba_cash_rate` in `actual_tools` — the OpenAI tool-name prefix `functions.` leaked through one row. Cosmetic, doesn't affect correctness or scoring beyond a 0.5 precision penalty on that single row. Strip in `actual_tools` accumulation.
- **Empty-question hand-off** — `agent-002` / `agent-003` / `agent-005` show the agent answering "I do not have current data" even when the tool fires; this is downstream of Bucket B (no evidence reaches synthesis), not a separate bug.

## Top 3 fixes for Task 3.24 (ranked by leverage)

### Fix 1 — Reflect-loop convergence (Bucket A, 27/30 queries)

**Problem.** The reflect node sends the trajectory back to the planner
even when the synthesiser already produced a grounded answer. With
`max_iterations=4`, every query saturates and `trajectory_efficiency`
floors at 0.25–0.30.

**Proposed change.**
1. Tighten the reflect prompt to require an explicit binary judgement
   on whether the *current* synthesised draft answers the question
   ("Does the draft answer every sub-question with a number/citation?
   yes / no"). Only loop on `no`, and require a one-line reason.
2. Drop `max_iterations` from 4 → 2 once Fix 2 lands. The remaining
   2 iterations cover the legit cases (multi-step decomposition that
   needs a second planner pass).

**Expected impact.** `trajectory_efficiency` 0.30 → ~0.70+ (the median
single-tool query collapses from 4 iterations to 1). Faithfulness +0.10
from less synth-context churn. Per-query latency roughly halves.

### Fix 2 — Pipe structured tool results into the synthesiser (Bucket B, 6/30 queries, faithfulness=0)

**Problem.** Tool calls land but the synthesiser doesn't see their
structured output, so it drafts "I cannot provide …" and reflect
loops. The tool catalogue and the right tools are firing — the
context object that synthesis reads is missing the `tool_results`
bag, or formats it in a way the prompt doesn't reference.

**Proposed change.**
1. In `src/agent/state.py` (or wherever the agent state shape lives),
   ensure each tool call appends a structured entry to a
   `tool_results: list[{tool, args, result}]` field on the state.
2. In the synthesise prompt, render those entries verbatim into a
   "Tool outputs" block so the synthesiser must cite them when the
   user's question is computational (stamp duty, mortgage, yield).
3. Add an assertion in `agent_eval` that flags rows where
   `actual_tools != []` but the answer contains "I cannot" / "do not
   have" — current eval misses this regression class.

**Expected impact.** Faithfulness 0.57 → ~0.75. Groundedness 0.37 →
~0.55. The 6 agent-XXX rows currently at faithfulness=0 should jump
to ≥0.6.

### Fix 3 — Publisher-diversity rerank for doc-only queries (Bucket C, 5/7 doc queries)

**Problem.** Top-K retrieval is dominated by AHURI; multi-publisher
policy questions miss Treasury / Grattan / Productivity Commission.

**Proposed change.**
1. Apply a publisher-quota cap during rerank: at most `ceil(K/3)`
   chunks from any single publisher in the top-K returned to the
   agent.
2. Verify the change on the existing retrieval eval to make sure
   it doesn't regress single-publisher queries.

**Expected impact.** Publisher recall 0.79 → ~0.90 on the doc-only
bucket. Marginal effect on faithfulness; main payoff is the
multi-source claim coverage that the eval rubric rewards.

## What stays the same in v2

Tool selection (89% accuracy, 95% recall) is already strong — Fix 1
and Fix 2 leave the planner / classify / decompose nodes untouched.
The cost line (Anthropic-tuned cache-read multiplier on the OpenAI
branch) over-attributes savings on cached tokens — flagged in
`src/agent/cost.py`, deferred until pricing parity is needed.

## Re-eval plan (Task 3.24)

1. Land Fix 1 (reflect prompt + `max_iterations=2`).
2. Land Fix 2 (state plumbing + synth prompt + eval assertion).
3. Optional: land Fix 3 if the cost is low and v1→v2 publisher recall
   regresses unexpectedly.
4. Re-run `agent_eval --out results/agent_v2.json --with-judge`.
5. Diff vs `results/agent_v1.json` — target metrics: tool_acc ≥ 0.89,
   trajectory_efficiency ≥ 0.65, faithfulness ≥ 0.70, groundedness ≥ 0.50.
