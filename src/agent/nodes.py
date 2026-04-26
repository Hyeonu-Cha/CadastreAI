"""LangGraph node implementations for the CadastreAI agent.

Each function takes the full `AgentState` and returns a partial state
update that LangGraph merges in.

Implementation status
---------------------
- `classify_query`   → Task 3.13 (Claude Haiku, structured output)
- `decompose`        → Task 3.14 (Claude Haiku, 2–4 sub-questions)
- `retrieve_or_tool` → Task 3.15 (Claude Haiku planner → docs + tools)
- `reflect`          → Task 3.17 (Claude Haiku, structured output)
- `synthesize`       → Task 3.19 (Claude Sonnet + citation post-processor)
"""
from __future__ import annotations

import json
import logging
import os
import re

from src.agent.graph import AgentState, Classification
from src.agent.persona import (
    apply_publisher_boost,
    effective_persona,
    persona_prompt_addendum,
    persona_router_hints,
)

log = logging.getLogger(__name__)

CLASSIFIER_MODEL = os.environ.get("CADASTRE_CLASSIFIER_MODEL", "claude-haiku-4-5")
CLASSIFIER_MAX_TOKENS = 256

# Single tool definition that pins the JSON schema of the classifier
# output. Forcing tool_choice on this tool means Claude must emit a
# tool_use block whose `input` we can validate as a Classification.
_CLASSIFY_TOOL = {
    "name": "submit_classification",
    "description": (
        "Submit a structured classification of the user's question. "
        "Use this exactly once per call."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "persona": {
                "type": "string",
                "enum": [
                    "first_home_buyer",
                    "investor",
                    "policy_researcher",
                    "journalist",
                    "general",
                ],
                "description": (
                    "Best-fit persona inferred from the question. "
                    "'general' when none clearly applies."
                ),
            },
            "query_type": {
                "type": "string",
                "enum": ["factual", "comparative", "computational", "exploratory"],
            },
            "needs_docs": {
                "type": "boolean",
                "description": (
                    "True when the answer requires retrieving from "
                    "policy/explanatory text (RBA bulletins, ABS notes, "
                    "guidelines, glossaries)."
                ),
            },
            "needs_data": {
                "type": "boolean",
                "description": (
                    "True when the answer requires a numeric series or "
                    "calculation (cash rate, mortgage rate, vacancy, "
                    "median price, repayment, stamp duty)."
                ),
            },
            "needs_decomposition": {
                "type": "boolean",
                "description": (
                    "True when the question contains multiple atomic "
                    "questions that should be answered independently and "
                    "then composed."
                ),
            },
        },
        "required": [
            "persona",
            "query_type",
            "needs_docs",
            "needs_data",
            "needs_decomposition",
        ],
    },
}

_CLASSIFIER_SYSTEM = (
    "You are a routing classifier for an Australian housing-market "
    "research agent. Read the user's question and classify it via the "
    "submit_classification tool. Be decisive — do not ask follow-up "
    "questions. If the question mixes persona signals (e.g. 'as a first "
    "home buyer, what's the stamp duty AND should I wait for rates to "
    "fall?'), pick the dominant persona and set needs_decomposition=true."
)


def _user_query(state: AgentState) -> str:
    for m in state.get("messages", []):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def classify_query(state: AgentState) -> dict:
    """Route the question via Claude Haiku with a forced tool call.

    Returns a partial state with both the full `classification` dict
    and the top-level `query_type` mirror so downstream nodes that
    only care about query_type don't have to reach into the dict.

    Requires `ANTHROPIC_API_KEY` in the environment. Raises
    `RuntimeError` if the key is missing or the model returns an
    unexpected payload — silent fallback to a stub here would mask
    routing bugs.
    """
    import anthropic  # local import — keeps module import light

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set; classify_query needs Claude Haiku."
        )
    query = _user_query(state)
    if not query:
        raise RuntimeError("classify_query: no user message in state")

    client = anthropic.Anthropic(api_key=api_key)
    log.debug("classify_query → %s", CLASSIFIER_MODEL)
    resp = client.messages.create(
        model=CLASSIFIER_MODEL,
        max_tokens=CLASSIFIER_MAX_TOKENS,
        system=_CLASSIFIER_SYSTEM,
        tools=[_CLASSIFY_TOOL],
        tool_choice={"type": "tool", "name": "submit_classification"},
        messages=[{"role": "user", "content": query}],
    )

    tool_use = next(
        (b for b in resp.content if getattr(b, "type", None) == "tool_use"),
        None,
    )
    if tool_use is None:
        raise RuntimeError(
            f"classify_query: model {CLASSIFIER_MODEL} returned no tool_use block "
            f"(content types: {[getattr(b, 'type', None) for b in resp.content]})"
        )
    cls: Classification = tool_use.input  # type: ignore[assignment]
    log.info("classify_query: %s", json.dumps(cls, ensure_ascii=False))
    return {"classification": cls, "query_type": cls["query_type"]}


DECOMPOSER_MODEL = os.environ.get("CADASTRE_DECOMPOSER_MODEL", "claude-haiku-4-5")
DECOMPOSER_MAX_TOKENS = 512
DECOMPOSE_MIN = 2
DECOMPOSE_MAX = 4

_DECOMPOSE_TOOL = {
    "name": "submit_subquestions",
    "description": (
        "Submit a list of 2–4 atomic sub-questions that together cover "
        "the user's original question. Each sub-question must be "
        "answerable independently with retrieval or a single tool call."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "sub_questions": {
                "type": "array",
                "minItems": DECOMPOSE_MIN,
                "maxItems": DECOMPOSE_MAX,
                "items": {"type": "string", "minLength": 1},
                "description": (
                    "Atomic sub-questions, ordered logically. Avoid "
                    "duplicates and questions that depend on each other."
                ),
            },
        },
        "required": ["sub_questions"],
    },
}

_DECOMPOSER_SYSTEM = (
    "You decompose a complex Australian housing-market question into "
    "2–4 atomic sub-questions. Each sub-question must be self-contained, "
    "non-overlapping, and answerable on its own (a doc lookup, a tool "
    "call, or a calculation). Preserve the user's intent — do not add "
    "scope beyond the original question. Submit via the "
    "submit_subquestions tool."
)


def decompose(state: AgentState) -> dict:
    """Split a complex query into 2–4 atomic sub-questions.

    Only runs when the classifier flagged `needs_decomposition=True`.
    Otherwise we leave `sub_questions` empty and the downstream router
    treats the original question atomically. Uses Claude Haiku via the
    same forced-tool-call pattern as `classify_query`.
    """
    cls = state.get("classification") or {}
    if not cls.get("needs_decomposition"):
        log.debug("decompose: needs_decomposition=False, skipping")
        return {"sub_questions": []}

    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set; decompose needs Claude Haiku."
        )
    query = _user_query(state)
    if not query:
        raise RuntimeError("decompose: no user message in state")

    client = anthropic.Anthropic(api_key=api_key)
    log.debug("decompose → %s", DECOMPOSER_MODEL)
    resp = client.messages.create(
        model=DECOMPOSER_MODEL,
        max_tokens=DECOMPOSER_MAX_TOKENS,
        system=_DECOMPOSER_SYSTEM,
        tools=[_DECOMPOSE_TOOL],
        tool_choice={"type": "tool", "name": "submit_subquestions"},
        messages=[{"role": "user", "content": query}],
    )

    tool_use = next(
        (b for b in resp.content if getattr(b, "type", None) == "tool_use"),
        None,
    )
    if tool_use is None:
        raise RuntimeError(
            f"decompose: model {DECOMPOSER_MODEL} returned no tool_use block"
        )
    raw = tool_use.input.get("sub_questions") or []  # type: ignore[union-attr]
    sub_qs = [s.strip() for s in raw if isinstance(s, str) and s.strip()]
    if not (DECOMPOSE_MIN <= len(sub_qs) <= DECOMPOSE_MAX):
        # Anthropic's schema enforces this server-side, but if the model
        # ever drops below 2 we fall back to atomic handling rather than
        # forwarding a degenerate list.
        log.warning(
            "decompose returned %d sub-questions (expected %d–%d); "
            "treating as atomic",
            len(sub_qs),
            DECOMPOSE_MIN,
            DECOMPOSE_MAX,
        )
        return {"sub_questions": []}
    log.info("decompose: %s", json.dumps(sub_qs, ensure_ascii=False))
    return {"sub_questions": sub_qs}


ROUTER_MODEL = os.environ.get("CADASTRE_ROUTER_MODEL", "claude-haiku-4-5")
ROUTER_MAX_TOKENS = 1024
DOCS_TOP_K = 5
MAX_TOOL_CALLS_PER_QUESTION = 4

# Tool catalogue: name → (callable, required arg keys, optional arg keys).
# Keeping this as a module-level dict means tests can monkey-patch entries
# (e.g. swap a real ABS call for a fake) without touching the dispatch
# logic. The arg lists let `_execute_tool` filter unknown keys the planner
# might hallucinate, and verify required keys are present before calling.
def _tool_catalogue() -> dict[str, tuple]:
    from src.tools.abs_stats import (
        abs_building_approvals,
        abs_lending_indicators,
        abs_property_price_index,
    )
    from src.tools.compute import (
        compute_mortgage_repayment,
        compute_rental_yield,
        compute_stamp_duty_nsw,
    )
    from src.tools.rba_stats import rba_cash_rate, rba_mortgage_rates
    from src.tools.sqm import sqm_rental_vacancy

    return {
        "rba_cash_rate": (rba_cash_rate, [], ["period"]),
        "rba_mortgage_rates": (rba_mortgage_rates, [], ["period"]),
        "abs_property_price_index": (
            abs_property_price_index,
            ["capital_city"],
            ["period"],
        ),
        "abs_building_approvals": (
            abs_building_approvals,
            ["state"],
            ["period"],
        ),
        "abs_lending_indicators": (abs_lending_indicators, [], ["period"]),
        "sqm_rental_vacancy": (
            sqm_rental_vacancy,
            ["postcode_or_city"],
            ["period"],
        ),
        "compute_rental_yield": (
            compute_rental_yield,
            ["annual_rent_aud", "property_value_aud"],
            [],
        ),
        "compute_mortgage_repayment": (
            compute_mortgage_repayment,
            ["principal_aud", "annual_rate_pct", "term_years"],
            ["frequency"],
        ),
        "compute_stamp_duty_nsw": (
            compute_stamp_duty_nsw,
            ["purchase_price_aud"],
            ["is_first_home_buyer"],
        ),
    }


_ROUTER_TOOL = {
    "name": "submit_routing_plan",
    "description": (
        "Submit a routing plan for a single sub-question: whether to "
        "retrieve from the document index, and which structured tools "
        "(if any) to call with what arguments. Use exactly once."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "use_docs": {
                "type": "boolean",
                "description": (
                    "True when answering this sub-question requires "
                    "policy/explanatory text from the doc index "
                    "(RBA bulletins, ABS notes, guidelines, glossaries)."
                ),
            },
            "doc_query": {
                "type": "string",
                "description": (
                    "Optional rewritten query for the retriever. "
                    "Defaults to the sub-question itself. Use this to "
                    "strip conversational filler or focus on the "
                    "retrievable concept."
                ),
            },
            "tool_calls": {
                "type": "array",
                "maxItems": MAX_TOOL_CALLS_PER_QUESTION,
                "items": {
                    "type": "object",
                    "properties": {
                        "tool": {
                            "type": "string",
                            "enum": [
                                "rba_cash_rate",
                                "rba_mortgage_rates",
                                "abs_property_price_index",
                                "abs_building_approvals",
                                "abs_lending_indicators",
                                "sqm_rental_vacancy",
                                "compute_rental_yield",
                                "compute_mortgage_repayment",
                                "compute_stamp_duty_nsw",
                            ],
                        },
                        "args": {
                            "type": "object",
                            "description": (
                                "Keyword arguments for the tool. Use "
                                "string 'latest' / 'YYYY' / 'YYYY-MM' "
                                "for `period`. Capital-city names are "
                                "case-insensitive."
                            ),
                        },
                    },
                    "required": ["tool", "args"],
                },
                "description": (
                    "Ordered tool calls to execute for this sub-question. "
                    "Empty list when only retrieval (or nothing) is needed."
                ),
            },
        },
        "required": ["use_docs", "tool_calls"],
    },
}

_ROUTER_SYSTEM = (
    "You route a single Australian housing-market sub-question to the "
    "right evidence: document retrieval, one or more structured tools, "
    "or both. Pick tools deliberately — call each at most once unless "
    "the question genuinely compares periods/locations. Use 'latest' "
    "for `period` unless the question names a specific date or year. "
    "For NSW stamp duty on a first-home-buyer scenario, set "
    "`is_first_home_buyer=true`. Submit via the submit_routing_plan tool."
)


def _plan_subquestion(query: str, persona: str | None = None) -> dict:
    """Ask Claude Haiku for a routing plan for one sub-question.

    Factored out so tests can monkey-patch this without touching the
    Anthropic SDK. Raises RuntimeError on missing API key or malformed
    response — silent fallbacks would hide planning bugs.

    `persona` (optional) appends a one-line tool-selection hint to the
    router system prompt — biases ties without restricting the toolset.
    """
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set; retrieve_or_tool needs Claude Haiku."
        )
    system_prompt = _ROUTER_SYSTEM
    hint = persona_router_hints(persona) if persona else ""
    if hint:
        system_prompt = f"{_ROUTER_SYSTEM}\n\n{hint}"
    client = anthropic.Anthropic(api_key=api_key)
    log.debug("router → %s (persona=%s)", ROUTER_MODEL, persona)
    resp = client.messages.create(
        model=ROUTER_MODEL,
        max_tokens=ROUTER_MAX_TOKENS,
        system=system_prompt,
        tools=[_ROUTER_TOOL],
        tool_choice={"type": "tool", "name": "submit_routing_plan"},
        messages=[{"role": "user", "content": query}],
    )
    tool_use = next(
        (b for b in resp.content if getattr(b, "type", None) == "tool_use"),
        None,
    )
    if tool_use is None:
        raise RuntimeError(
            f"router: model {ROUTER_MODEL} returned no tool_use block"
        )
    plan = tool_use.input  # type: ignore[assignment]
    log.info("router plan for %r: %s", query, json.dumps(plan, ensure_ascii=False))
    return plan


def _execute_tool(name: str, args: dict) -> dict:
    """Dispatch one tool call; filter unknown args, enforce required keys.

    Wraps unexpected exceptions in the result dict so a single bad arg
    doesn't sink the whole router pass. Tool-internal failures still
    surface — we just attribute them rather than propagating.
    """
    catalogue = _tool_catalogue()
    spec = catalogue.get(name)
    if spec is None:
        raise ValueError(f"unknown tool: {name}")
    fn, required, optional = spec
    allowed = set(required) | set(optional)
    filtered = {k: v for k, v in (args or {}).items() if k in allowed}
    missing = [k for k in required if k not in filtered]
    if missing:
        raise ValueError(f"{name}: missing required args {missing}")
    return fn(**filtered)


def _retrieve_docs(query: str, k: int = DOCS_TOP_K) -> list[dict]:
    """Pull top-k chunks via the dense retriever.

    Wrapped so tests can monkey-patch this without spinning up Qdrant.
    Returns the agent-state shape (`{chunk_id, score, payload}`).
    """
    from src.retrieval.retriever import retrieve as do_retrieve

    hits = do_retrieve(query, k=k)
    return [
        {
            "chunk_id": payload.get("chunk_id", ""),
            "score": float(score),
            "payload": payload,
        }
        for payload, score in hits
    ]


def retrieve_or_tool(state: AgentState) -> dict:
    """Per sub-question, decide doc retrieval vs tool calls vs both.

    For each sub-question (or the original query if decomposition was
    skipped), asks Claude Haiku for a `submit_routing_plan` and then
    executes it: dense retrieval into `retrieved_chunks` and zero or
    more structured-tool calls into `tool_results`. On loop-back from
    `reflect`, the refined_query becomes the single sub-question for
    that iteration.

    Always increments `iteration_count` so the cap in `_route_after_reflect`
    is observable even when the planner returns an empty plan.
    """
    iter_count = state.get("iteration_count", 0)
    if iter_count == 0:
        questions = list(state.get("sub_questions") or [])
        if not questions:
            uq = _user_query(state)
            questions = [uq] if uq else []
    else:
        refl = state.get("reflection") or {}
        refined = (refl.get("refined_query") or "").strip()
        questions = [refined] if refined else []

    persona = effective_persona(state)
    chunks = list(state.get("retrieved_chunks", []))
    tool_results = list(state.get("tool_results", []))

    for q in questions:
        try:
            plan = _plan_subquestion(q, persona=persona)
        except Exception as e:  # noqa: BLE001 — log and skip this question
            log.warning("router planning failed for %r: %s", q, e)
            tool_results.append(
                {"sub_question": q, "error": f"plan_failed: {type(e).__name__}: {e}"}
            )
            continue

        if plan.get("use_docs"):
            doc_q = (plan.get("doc_query") or q).strip() or q
            try:
                chunks.extend(_retrieve_docs(doc_q))
            except Exception as e:  # noqa: BLE001 — partial progress > full fail
                log.warning("retrieve failed for %r: %s", doc_q, e)
                tool_results.append(
                    {"sub_question": q, "error": f"retrieve_failed: {type(e).__name__}: {e}"}
                )

        for tc in plan.get("tool_calls", []) or []:
            tname = tc.get("tool")
            targs = tc.get("args") or {}
            entry: dict = {"sub_question": q, "tool": tname, "args": targs}
            try:
                entry["result"] = _execute_tool(tname, targs)
            except Exception as e:  # noqa: BLE001 — record per-tool failure
                log.warning("tool %s(%s) failed: %s", tname, targs, e)
                entry["error"] = f"{type(e).__name__}: {e}"
            tool_results.append(entry)

    chunks = apply_publisher_boost(chunks, persona)

    return {
        "retrieved_chunks": chunks,
        "tool_results": tool_results,
        "iteration_count": iter_count + 1,
    }


REFLECTOR_MODEL = os.environ.get("CADASTRE_REFLECTOR_MODEL", "claude-haiku-4-5")
REFLECTOR_MAX_TOKENS = 768
# Per-chunk text preview length. Big enough to judge coverage; small
# enough that 12 chunks fits in budget.
REFLECT_EVIDENCE_PREVIEW_CHARS = 240
REFLECT_MAX_EVIDENCE_ITEMS = 12

_REFLECT_TOOL = {
    "name": "submit_reflection",
    "description": (
        "Submit a reflection on the current draft answer. Decide whether "
        "the draft fully answers the user's question (and all sub-questions) "
        "using the retrieved evidence and tool results, or whether another "
        "retrieval/tool pass is needed."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "is_complete": {
                "type": "boolean",
                "description": (
                    "True only when every sub-question is answered, "
                    "every numeric claim is backed by a tool result or "
                    "cited chunk, and no obvious gaps remain. Be strict — "
                    "default to False when in doubt."
                ),
            },
            "missing": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Specific gaps when is_complete=false. Each entry "
                    "names one thing the draft is missing or under-supports "
                    "(e.g. 'no source for current cash rate', 'Melbourne "
                    "leg of comparison absent'). Empty when complete."
                ),
            },
            "refined_query": {
                "type": "string",
                "description": (
                    "When is_complete=false, a single concrete query to "
                    "send into the next retrieval/tool pass. Should target "
                    "the most-load-bearing gap, not a paraphrase of the "
                    "original question. Empty string when complete."
                ),
            },
        },
        "required": ["is_complete", "missing", "refined_query"],
    },
}

_REFLECTOR_SYSTEM = (
    "You audit a draft answer for an Australian housing-market research "
    "agent. Your job is to decide whether the draft is good enough to "
    "ship, or whether the agent should run another retrieval/tool pass. "
    "A draft is complete only when (a) every sub-question is addressed, "
    "(b) every numeric or factual claim has supporting evidence in the "
    "retrieved chunks or tool results, and (c) there are no obvious gaps. "
    "When incomplete, name the gaps concretely and propose ONE refined "
    "query that targets the most important missing piece. Submit via "
    "the submit_reflection tool."
)


def _summarise_chunks(chunks: list[dict]) -> str:
    """Compact retrieved-chunk preview so reflect's prompt stays bounded."""
    if not chunks:
        return "(no chunks retrieved)"
    lines = []
    for i, c in enumerate(chunks[:REFLECT_MAX_EVIDENCE_ITEMS], start=1):
        payload = c.get("payload") or {}
        title = (payload.get("title") or "")[:80]
        section = (payload.get("section_heading") or "")[:60]
        text = (payload.get("text") or "")[:REFLECT_EVIDENCE_PREVIEW_CHARS]
        lines.append(
            f"[{i}] {payload.get('publisher', '?')} | {title}"
            + (f" | {section}" if section else "")
            + f"\n    score={c.get('score', 0):.3f}  text={text!r}"
        )
    if len(chunks) > REFLECT_MAX_EVIDENCE_ITEMS:
        lines.append(f"... and {len(chunks) - REFLECT_MAX_EVIDENCE_ITEMS} more chunks")
    return "\n".join(lines)


def _summarise_tools(tool_results: list[dict]) -> str:
    """Compact tool-result preview — name, args, top-level data fields, errors."""
    if not tool_results:
        return "(no tool calls)"
    lines = []
    for i, t in enumerate(tool_results[:REFLECT_MAX_EVIDENCE_ITEMS], start=1):
        if "error" in t:
            lines.append(
                f"[{i}] {t.get('tool', '<plan>')}({t.get('args', {})}) → ERROR: {t['error']}"
            )
            continue
        result = t.get("result") or {}
        data = result.get("data") or {}
        # Show keys + a few representative values; full payload is in state.
        preview = {k: data[k] for k in list(data)[:5]}
        lines.append(
            f"[{i}] {t.get('tool')}({t.get('args', {})}) → "
            f"data={preview}  source={result.get('source', '?')!r}"
        )
    if len(tool_results) > REFLECT_MAX_EVIDENCE_ITEMS:
        lines.append(f"... and {len(tool_results) - REFLECT_MAX_EVIDENCE_ITEMS} more results")
    return "\n".join(lines)


def _build_reflect_prompt(state: AgentState) -> str:
    """Assemble the user-message body for reflect.

    Factored so tests can inspect the rendered prompt directly without
    mocking the Anthropic call.
    """
    user_q = _user_query(state)
    sub_qs = state.get("sub_questions") or []
    chunks = state.get("retrieved_chunks") or []
    tool_results = state.get("tool_results") or []
    draft = state.get("answer_draft") or "(no draft yet)"
    iter_n = state.get("iteration_count", 0)

    sub_block = (
        "\n".join(f"  {i + 1}. {q}" for i, q in enumerate(sub_qs))
        if sub_qs
        else "(none — answer the original question atomically)"
    )

    return (
        f"ORIGINAL QUESTION:\n{user_q}\n\n"
        f"SUB-QUESTIONS:\n{sub_block}\n\n"
        f"RETRIEVED CHUNKS (top {REFLECT_MAX_EVIDENCE_ITEMS}):\n"
        f"{_summarise_chunks(chunks)}\n\n"
        f"TOOL RESULTS (top {REFLECT_MAX_EVIDENCE_ITEMS}):\n"
        f"{_summarise_tools(tool_results)}\n\n"
        f"DRAFT ANSWER (iteration {iter_n}):\n{draft}\n\n"
        "Decide whether to ship the draft or run another pass."
    )


def reflect(state: AgentState) -> dict:
    """Audit the draft and decide whether to ship or loop.

    Returns `{reflection: {is_complete, missing, refined_query}}`. The
    graph's `_route_after_reflect` reads `is_complete` and the iteration
    cap to decide between END and looping back to `retrieve_or_tool`
    (Task 3.18). Live-call failures fall back to is_complete=True so a
    transient API error doesn't trap the agent in a loop until the cap.
    """
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set; reflect needs Claude Haiku."
        )

    prompt = _build_reflect_prompt(state)
    client = anthropic.Anthropic(api_key=api_key)
    log.debug("reflect → %s", REFLECTOR_MODEL)
    try:
        resp = client.messages.create(
            model=REFLECTOR_MODEL,
            max_tokens=REFLECTOR_MAX_TOKENS,
            system=_REFLECTOR_SYSTEM,
            tools=[_REFLECT_TOOL],
            tool_choice={"type": "tool", "name": "submit_reflection"},
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:  # noqa: BLE001 — see fallback rationale above
        log.warning("reflect API call failed (%s: %s); marking complete", type(e).__name__, e)
        return {
            "reflection": {
                "is_complete": True,
                "missing": [],
                "refined_query": None,
            }
        }

    tool_use = next(
        (b for b in resp.content if getattr(b, "type", None) == "tool_use"),
        None,
    )
    if tool_use is None:
        log.warning("reflect: no tool_use block; marking complete")
        return {
            "reflection": {
                "is_complete": True,
                "missing": [],
                "refined_query": None,
            }
        }

    raw = tool_use.input  # type: ignore[assignment]
    is_complete = bool(raw.get("is_complete", True))
    missing = [m for m in (raw.get("missing") or []) if isinstance(m, str) and m.strip()]
    refined = (raw.get("refined_query") or "").strip() or None
    # Self-consistency: a model that says "complete" but ships a refined
    # query is contradicting itself — treat as complete and drop the query.
    if is_complete:
        refined = None
    log.info(
        "reflect: is_complete=%s missing=%d refined=%r",
        is_complete,
        len(missing),
        refined,
    )
    return {
        "reflection": {
            "is_complete": is_complete,
            "missing": missing,
            "refined_query": refined,
        }
    }


SYNTHESIZER_MODEL = os.environ.get("CADASTRE_SYNTHESIZER_MODEL", "claude-sonnet-4-6")
SYNTHESIZER_MAX_TOKENS = 1536
SYNTH_CHUNK_TEXT_LIMIT = 600  # Per chunk in the prompt; full text sits in state.
SYNTH_MAX_CHUNKS_IN_PROMPT = 12

# Citation grammar (Task 3.19):
#   [source:<publisher>, page:<N>]   for retrieved chunks
#   [source:<publisher>]             when page is unknown (acceptable)
#   [tool:<name>, retrieved:<date>]  for structured tool results
# Whitespace is tolerated; everything else must match exactly.
_DOC_CITE_RE = re.compile(r"\[source:\s*([^,\]]+?)\s*(?:,\s*page:\s*([^\]]+?)\s*)?\]")
_TOOL_CITE_RE = re.compile(r"\[tool:\s*([^,\]]+?)\s*,\s*retrieved:\s*([^\]]+?)\s*\]")

_CITATION_RULES = (
    "Cite every factual or numeric claim inline using ONE of these forms:\n"
    "  - Retrieved documents:  [source:<publisher>, page:<N>]\n"
    "    (omit `, page:<N>` only if the chunk has no page; never invent one)\n"
    "  - Tool results:         [tool:<tool_name>, retrieved:<YYYY-MM-DD>]\n"
    "Use ONLY the publishers, page numbers, tool names, and retrieved dates "
    "shown in EVIDENCE below. If you cannot support a claim from the "
    "evidence, OMIT the claim entirely — do not paraphrase from training "
    "knowledge. End with a 'Sources' line that lists each cited publisher "
    "and tool exactly once."
)

_SYNTHESIZER_SYSTEM = (
    "You are a careful Australian housing-market research assistant. "
    "Answer the user's question concisely and ONLY from the supplied "
    "evidence. Numbers must come from tool results; explanatory framing "
    "must come from retrieved chunks. " + _CITATION_RULES
)


def _summarise_chunks_for_synth(chunks: list[dict]) -> str:
    if not chunks:
        return "(no document chunks retrieved)"
    lines = []
    for i, c in enumerate(chunks[:SYNTH_MAX_CHUNKS_IN_PROMPT], start=1):
        payload = c.get("payload") or {}
        publisher = payload.get("publisher", "?")
        title = (payload.get("title") or "")[:120]
        section = (payload.get("section_heading") or "")[:80]
        page = payload.get("page")
        text = (payload.get("text") or "")[:SYNTH_CHUNK_TEXT_LIMIT]
        header = f"[{i}] publisher={publisher!r} page={page!r} title={title!r}"
        if section:
            header += f" section={section!r}"
        lines.append(f"{header}\n    text={text!r}")
    if len(chunks) > SYNTH_MAX_CHUNKS_IN_PROMPT:
        lines.append(f"... and {len(chunks) - SYNTH_MAX_CHUNKS_IN_PROMPT} more chunks")
    return "\n".join(lines)


def _summarise_tools_for_synth(tool_results: list[dict]) -> str:
    if not tool_results:
        return "(no tool results)"
    lines = []
    for i, t in enumerate(tool_results, start=1):
        if "error" in t:
            tool_name = t.get("tool", "<plan>")
            lines.append(
                f"[{i}] tool={tool_name!r} args={t.get('args', {})} "
                f"→ ERROR: {t['error']}"
            )
            continue
        result = t.get("result") or {}
        retrieved_at = result.get("retrieved_at") or ""
        retrieved_date = retrieved_at[:10] if retrieved_at else "?"
        lines.append(
            f"[{i}] tool={t.get('tool')!r} args={t.get('args', {})} "
            f"retrieved={retrieved_date!r} data={result.get('data')} "
            f"source={result.get('source')!r} citation={result.get('citation')!r}"
        )
    return "\n".join(lines)


def _build_synth_prompt(state: AgentState) -> str:
    user_q = _user_query(state)
    sub_qs = state.get("sub_questions") or []
    chunks = state.get("retrieved_chunks") or []
    tools = state.get("tool_results") or []

    sub_block = (
        "\n".join(f"  {i + 1}. {q}" for i, q in enumerate(sub_qs))
        if sub_qs
        else "(none — answer the original question directly)"
    )
    return (
        f"USER QUESTION:\n{user_q}\n\n"
        f"SUB-QUESTIONS:\n{sub_block}\n\n"
        f"EVIDENCE — RETRIEVED CHUNKS:\n{_summarise_chunks_for_synth(chunks)}\n\n"
        f"EVIDENCE — TOOL RESULTS:\n{_summarise_tools_for_synth(tools)}\n\n"
        "Write the answer now."
    )


def _valid_citation_targets(state: AgentState) -> tuple[set[str], dict[str, set[str]]]:
    """Build the allowlist for citation post-processing.

    Returns:
      (valid_tools, valid_publishers_pages) where
        * `valid_tools` is the set of tool names that produced a result
          (errors don't count — we don't want the model citing failed
          calls). The retrieved-date check is done separately.
        * `valid_publishers_pages` maps publisher → set of page strings
          observed in retrieved chunks. A page string of "" means the
          chunk had no page, in which case un-paged citations are OK.
    """
    chunks = state.get("retrieved_chunks") or []
    publishers: dict[str, set[str]] = {}
    for c in chunks:
        payload = c.get("payload") or {}
        pub = (payload.get("publisher") or "").strip()
        if not pub:
            continue
        page = payload.get("page")
        page_str = str(page).strip() if page is not None else ""
        publishers.setdefault(pub, set()).add(page_str)

    tools_by_date: dict[str, set[str]] = {}
    for t in state.get("tool_results") or []:
        if "result" not in t:
            continue
        name = (t.get("tool") or "").strip()
        if not name:
            continue
        retrieved_at = (t["result"].get("retrieved_at") or "")[:10]
        tools_by_date.setdefault(name, set()).add(retrieved_at)
    return set(tools_by_date), publishers


def _enforce_citations(text: str, state: AgentState) -> tuple[str, list[str]]:
    """Validate inline citations and append a Sources footer.

    Returns the (possibly annotated) text plus a list of orphan
    citations that didn't match the evidence. Orphans aren't deleted —
    we tag them `(unverified)` so the user can see where the model
    overreached, and so the eval harness can score citation discipline
    without the post-processor silently fixing things up.
    """
    valid_tools, valid_publishers = _valid_citation_targets(state)
    orphans: list[str] = []

    def _doc_repl(m: re.Match) -> str:
        publisher = m.group(1).strip()
        page = (m.group(2) or "").strip()
        allowed_pages = valid_publishers.get(publisher)
        if allowed_pages is None:
            orphans.append(m.group(0))
            return f"{m.group(0)} (unverified)"
        if page and page not in allowed_pages:
            # Publisher matched but page didn't — partial orphan.
            orphans.append(m.group(0))
            return f"{m.group(0)} (page-unverified)"
        return m.group(0)

    def _tool_repl(m: re.Match) -> str:
        name = m.group(1).strip()
        if name not in valid_tools:
            orphans.append(m.group(0))
            return f"{m.group(0)} (unverified)"
        return m.group(0)

    annotated = _DOC_CITE_RE.sub(_doc_repl, text)
    annotated = _TOOL_CITE_RE.sub(_tool_repl, annotated)

    # If the model already emitted a "Sources" block we leave it; it's
    # part of the prompted format and may hold publisher+page lines we
    # can't confidently rewrite. Append our own machine-built footer
    # only when the model omitted one.
    if "\nSources" not in annotated and "Sources:" not in annotated.split("\n")[-3:][0]:
        footer_lines = []
        if valid_publishers:
            footer_lines.append("- Documents: " + ", ".join(sorted(valid_publishers)))
        if valid_tools:
            footer_lines.append("- Tools: " + ", ".join(sorted(valid_tools)))
        if footer_lines:
            annotated = annotated.rstrip() + "\n\nSources (auto):\n" + "\n".join(footer_lines)
    return annotated, orphans


def synthesize(state: AgentState) -> dict:
    """Produce the user-facing answer with strict inline citations.

    Calls Claude Sonnet (default) on the original question + sub-questions
    + retrieved chunks + tool results, then post-processes the draft to
    flag any citation that doesn't reference real evidence.

    Defensive: API failures fall back to a short error stub so the graph
    can still terminate. Orphan citations are not deleted — they're
    tagged `(unverified)` so the eval harness can score citation
    discipline without the post-processor silently fixing things up.
    """
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set; synthesize needs Claude Sonnet."
        )

    msgs = state.get("messages", [])
    prompt = _build_synth_prompt(state)
    persona = effective_persona(state)
    system_prompt = f"{_SYNTHESIZER_SYSTEM}\n\n{persona_prompt_addendum(persona)}"
    client = anthropic.Anthropic(api_key=api_key)
    log.debug("synthesize → %s (persona=%s)", SYNTHESIZER_MODEL, persona)
    try:
        resp = client.messages.create(
            model=SYNTHESIZER_MODEL,
            max_tokens=SYNTHESIZER_MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:  # noqa: BLE001 — graph must still terminate
        log.warning("synthesize API call failed (%s: %s)", type(e).__name__, e)
        draft = (
            "I couldn't generate a final answer due to a synthesis-stage "
            f"error ({type(e).__name__}). Retrieved {len(state.get('retrieved_chunks') or [])} "
            f"chunks and {len(state.get('tool_results') or [])} tool results."
        )
        return {
            "answer_draft": draft,
            "messages": msgs + [{"role": "assistant", "content": draft}],
        }

    text_blocks = [
        getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text"
    ]
    raw = "\n".join(t for t in text_blocks if t).strip()
    if not raw:
        raw = "(no answer produced)"

    cleaned, orphans = _enforce_citations(raw, state)
    if orphans:
        log.info("synthesize: %d unverified citation(s): %s", len(orphans), orphans)
    return {
        "answer_draft": cleaned,
        "messages": msgs + [{"role": "assistant", "content": cleaned}],
    }


__all__ = [
    "classify_query",
    "decompose",
    "retrieve_or_tool",
    "reflect",
    "synthesize",
]
