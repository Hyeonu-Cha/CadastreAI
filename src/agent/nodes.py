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
from datetime import datetime, timezone

from src.agent import llm
from src.agent.cost import log_cost
from src.agent.graph import AgentState, Classification
from src.agent.guardrails import log_refusal, screen_query
from src.agent.persona import (
    apply_publisher_boost,
    effective_persona,
    persona_disclaimer,
    persona_prompt_addendum,
    persona_router_hints,
)

log = logging.getLogger(__name__)


# --- Temporal context (Task 5.15) ----------------------------------
# The agent had no notion of "today", so it couldn't reason about recency
# or whether cited evidence is still current. Inject the date into each
# node's USER message rather than the system prompt: llm.py marks the
# system block with cache_control=ephemeral, and a volatile date there
# would bust that prompt cache. Australia/Sydney is the domain-correct
# clock; fall back to UTC where the tz database isn't present (e.g. a
# bare Windows runner without `tzdata`).
try:
    from zoneinfo import ZoneInfo

    _AGENT_TZ = ZoneInfo("Australia/Sydney")
    _AGENT_TZ_LABEL = "Australia/Sydney"
except Exception:  # noqa: BLE001 — no tzdata → UTC is a fine fallback
    _AGENT_TZ = timezone.utc
    _AGENT_TZ_LABEL = "UTC"


def _today_context() -> str:
    """One-line current-date preamble prepended to a node's user prompt."""
    today = datetime.now(_AGENT_TZ).strftime("%Y-%m-%d")
    return (
        f"[Context] Today's date is {today} ({_AGENT_TZ_LABEL}). Treat this "
        "as the current date when judging recency or whether cited material "
        "may be out of date.\n\n"
    )


def _with_today(user: str) -> str:
    """Prepend the current-date context to a node's user prompt (Task 5.15)."""
    return _today_context() + user


# Per-node model defaults, resolved at call time via `_resolve_*_model()`
# so the active provider (`CADASTRE_LLM_PROVIDER`) can swap between the
# Anthropic and OpenAI columns without re-importing this module. The
# existing import-time constants (`CLASSIFIER_MODEL`, ...) are preserved
# as the Anthropic-side default for backward compat with anything that
# imports them directly.
CLASSIFIER_MODEL = os.environ.get("CADASTRE_CLASSIFIER_MODEL", "claude-haiku-4-5")
CLASSIFIER_MODEL_OPENAI_DEFAULT = "gpt-4o-mini"
CLASSIFIER_MAX_TOKENS = 256


def _resolve_classifier_model() -> str:
    if llm.get_provider() == "openai":
        return os.environ.get(
            "CADASTRE_OPENAI_CLASSIFIER_MODEL", CLASSIFIER_MODEL_OPENAI_DEFAULT
        )
    return CLASSIFIER_MODEL


def guardrail_screen(state: AgentState) -> dict:
    """First node in the graph: pre-classification financial-product
    recommendation guardrail (Task X.04).

    Pulls the latest user message, runs the deterministic regex
    library in `src.agent.guardrails`, and either:

      - returns ``{}`` (allow path — graph continues to classify_query),
        or
      - returns a populated answer + reflection so the graph routes
        directly to END with a polite refusal (refuse path).

    On the refuse path we also stamp `state["guardrail"]` so the
    Streamlit UI can label the response distinctly. Refusals are
    logged at WARNING level via `log_refusal` for telemetry / regex
    tuning.
    """
    msgs = state.get("messages", [])
    user_query = next(
        (m["content"] for m in msgs if m.get("role") == "user"),
        "",
    )
    decision = screen_query(user_query)
    if decision.action == "allow":
        return {}
    log_refusal(user_query, decision)
    refusal = decision.refusal_text or ""
    return {
        "answer_draft": refusal,
        "messages": msgs + [{"role": "assistant", "content": refusal}],
        # Mark the run as complete so `_route_after_reflect` exits
        # immediately when the graph hops here. The reflection field
        # is also surfaced in the UI's reasoning trace.
        "reflection": {
            "is_complete": True,
            "missing": [],
            "refined_query": None,
        },
        "guardrail": {
            "blocked": True,
            "category": decision.category,
            "reason": decision.reason,
        },
    }


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
    """Route the question via the active provider's small model with a
    forced tool call.

    Returns a partial state with both the full `classification` dict
    and the top-level `query_type` mirror so downstream nodes that
    only care about query_type don't have to reach into the dict.

    Requires the provider's API key (ANTHROPIC_API_KEY or OPENAI_API_KEY,
    depending on `CADASTRE_LLM_PROVIDER`). Raises `RuntimeError` if the
    key is missing or the model returns an unexpected payload — silent
    fallback to a stub here would mask routing bugs.
    """
    query = _user_query(state)
    if not query:
        raise RuntimeError("classify_query: no user message in state")

    model = _resolve_classifier_model()
    log.debug("classify_query → %s (%s)", model, llm.get_provider())
    cls_input, usage = llm.call_with_tool(
        system=_CLASSIFIER_SYSTEM,
        user=_with_today(query),
        tool_def=_CLASSIFY_TOOL,
        tool_name="submit_classification",
        max_tokens=CLASSIFIER_MAX_TOKENS,
        model=model,
    )
    log_cost(usage, model, "classify_query")

    if cls_input is None:
        raise RuntimeError(
            f"classify_query: model {model} returned no tool_use / tool_call"
        )
    cls: Classification = cls_input  # type: ignore[assignment]
    log.info("classify_query: %s", json.dumps(cls, ensure_ascii=False))
    return {"classification": cls, "query_type": cls["query_type"]}


DECOMPOSER_MODEL = os.environ.get("CADASTRE_DECOMPOSER_MODEL", "claude-haiku-4-5")
DECOMPOSER_MODEL_OPENAI_DEFAULT = "gpt-4o-mini"
DECOMPOSER_MAX_TOKENS = 512
DECOMPOSE_MIN = 2
DECOMPOSE_MAX = 4


def _resolve_decomposer_model() -> str:
    if llm.get_provider() == "openai":
        return os.environ.get(
            "CADASTRE_OPENAI_DECOMPOSER_MODEL", DECOMPOSER_MODEL_OPENAI_DEFAULT
        )
    return DECOMPOSER_MODEL

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

    query = _user_query(state)
    if not query:
        raise RuntimeError("decompose: no user message in state")

    model = _resolve_decomposer_model()
    log.debug("decompose → %s (%s)", model, llm.get_provider())
    tool_input, usage = llm.call_with_tool(
        system=_DECOMPOSER_SYSTEM,
        user=_with_today(query),
        tool_def=_DECOMPOSE_TOOL,
        tool_name="submit_subquestions",
        max_tokens=DECOMPOSER_MAX_TOKENS,
        model=model,
    )
    log_cost(usage, model, "decompose")

    if tool_input is None:
        raise RuntimeError(
            f"decompose: model {model} returned no tool_use / tool_call"
        )
    raw = tool_input.get("sub_questions") or []
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
ROUTER_MODEL_OPENAI_DEFAULT = "gpt-4o-mini"
ROUTER_MAX_TOKENS = 1024
DOCS_TOP_K = 5
# Per-query cap on retrieved_chunks after dedupe + publisher_boost (Task
# 3.29). v3 averaged 8.93 chunks/query because each sub-question + each
# reflect-loop appended its own top-5; the synth's citation discipline
# slipped under that load. Cap at the v2 average to keep the synth
# context tight.
MAX_TOTAL_CHUNKS = 8
MAX_TOOL_CALLS_PER_QUESTION = 4

# Doc retriever for the agent. Default is hybrid (BGE dense + BM25 RRF)
# per Task 3.25 — `results/hybrid_comparison.md` showed hybrid wins
# every metric on the honest synthetic split, while v1/v2 of the agent
# shipped on dense-only. Override with `CADASTRE_AGENT_RETRIEVER=
# dense|bm25|hybrid` so eval runs can A/B against the v2 baseline
# without touching code.
DOCS_RETRIEVER_DEFAULT = "hybrid"
_AGENT_RETRIEVER: object | None = None


def _resolve_router_model() -> str:
    if llm.get_provider() == "openai":
        return os.environ.get(
            "CADASTRE_OPENAI_ROUTER_MODEL", ROUTER_MODEL_OPENAI_DEFAULT
        )
    return ROUTER_MODEL

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
    "\n\n"
    "COMPUTE TOOLS REQUIRE USER-PROVIDED NUMBERS. Route to a `compute_*` "
    "tool ONLY when every required number is stated in the question:\n"
    "  - compute_rental_yield: needs both `annual_rent_aud` AND "
    "`property_value_aud` as explicit numbers. Do NOT call it for "
    "'typical/median rental yield in {city}' — that is a market lookup, "
    "not a computation. Route those to sqm_rental_vacancy + "
    "abs_property_price_index (and/or use_docs).\n"
    "  - compute_mortgage_repayment: needs `principal_aud`, "
    "`annual_rate_pct`, AND `term_years`. Without those, route to "
    "rba_mortgage_rates (current rate) and let the synth answer "
    "qualitatively.\n"
    "  - compute_stamp_duty_nsw: needs `purchase_price_aud`. NSW only.\n\n"
    "WORKED EXAMPLES:\n"
    "  Q: 'What's the median rental yield in Perth?'\n"
    "  → tool_calls: [sqm_rental_vacancy(postcode_or_city='Perth'), "
    "abs_property_price_index(capital_city='Perth')]; use_docs=True.\n"
    "  Q: 'I get $32k/yr rent on a $650k unit — what's my yield?'\n"
    "  → tool_calls: [compute_rental_yield(annual_rent_aud=32000, "
    "property_value_aud=650000)]; use_docs=False.\n"
)


def _plan_subquestion(query: str, persona: str | None = None) -> dict:
    """Ask the active provider's small model for a routing plan for one
    sub-question.

    Factored out so tests can monkey-patch this without touching the
    SDK. Raises RuntimeError on missing API key or malformed response —
    silent fallbacks would hide planning bugs.

    `persona` (optional) appends a one-line tool-selection hint to the
    router system prompt — biases ties without restricting the toolset.
    """
    system_prompt = _ROUTER_SYSTEM
    hint = persona_router_hints(persona) if persona else ""
    if hint:
        system_prompt = f"{_ROUTER_SYSTEM}\n\n{hint}"
    model = _resolve_router_model()
    log.debug("router → %s (%s, persona=%s)", model, llm.get_provider(), persona)
    plan, usage = llm.call_with_tool(
        system=system_prompt,
        user=_with_today(query),
        tool_def=_ROUTER_TOOL,
        tool_name="submit_routing_plan",
        max_tokens=ROUTER_MAX_TOKENS,
        model=model,
    )
    log_cost(usage, model, "retrieve_or_tool")
    if plan is None:
        raise RuntimeError(
            f"router: model {model} returned no tool_use / tool_call"
        )
    log.info("router plan for %r: %s", query, json.dumps(plan, ensure_ascii=False))
    return plan


_TOOL_CACHE = None


def _get_tool_cache():
    """Lazy module-level ToolCache singleton.

    Tests can force a clean cache via `_get_tool_cache().clear()` or
    by monkey-patching the module-level `_TOOL_CACHE` attribute. We
    don't construct it at import time so unrelated tests that never
    touch tools pay zero cost.
    """
    global _TOOL_CACHE
    if _TOOL_CACHE is None:
        from src.agent.tool_cache import ToolCache

        _TOOL_CACHE = ToolCache()
    return _TOOL_CACHE


def _execute_tool(name: str, args: dict) -> dict:
    """Dispatch one tool call; filter unknown args, enforce required keys.

    Wraps unexpected exceptions in the result dict so a single bad arg
    doesn't sink the whole router pass. Tool-internal failures still
    surface — we just attribute them rather than propagating.

    Successful results go through a TTL cache (`tool_cache.ToolCache`)
    so back-to-back calls with the same args hit memory instead of the
    upstream API. Required-arg validation still fires before the cache
    so misuse is loud rather than memoised.
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

    cache = _get_tool_cache()
    return cache.wrap(
        name,
        filtered,
        lambda: fn(**filtered),
        skip_on=lambda r: isinstance(r, dict) and "error" in r,
    )


def _agent_retriever() -> object:
    """Process-wide cached doc retriever for the agent (Task 3.25).

    Default is hybrid (BGE dense + BM25 RRF). Honest-split eval
    (`results/hybrid_comparison.md`) shows hybrid wins every metric;
    the dense-only build that shipped in v1/v2 was the weakest of the
    three options measured in `results/ablation.md`. Override with
    `CADASTRE_AGENT_RETRIEVER=dense|bm25|hybrid`.

    Lazy import on first call so test/CLI users that never hit doc
    retrieval don't pay the BM25 pickle (~212 MB) or sentence-
    transformers load.
    """
    global _AGENT_RETRIEVER
    if _AGENT_RETRIEVER is not None:
        return _AGENT_RETRIEVER
    kind = os.environ.get("CADASTRE_AGENT_RETRIEVER", DOCS_RETRIEVER_DEFAULT).lower()
    if kind == "dense":
        from src.retrieval.retriever import Retriever

        _AGENT_RETRIEVER = Retriever()
    elif kind == "bm25":
        from src.index.bm25 import DEFAULT_INDEX as _BM25_INDEX
        from src.index.bm25 import BM25Index

        _AGENT_RETRIEVER = BM25Index.load(_BM25_INDEX)
    elif kind == "hybrid":
        # Pre-import torch on small-pagefile Windows hosts: cuBLAS DLLs
        # must allocate workspace BEFORE the 212 MB BM25 pickle becomes
        # resident, otherwise the eventual sentence-transformers load
        # OOMs on cuBLAS init. Importing torch reserves the DLL space;
        # the model loads later when the dense leg first runs.
        import torch  # noqa: F401
        from src.index.hybrid import HybridRetriever

        _AGENT_RETRIEVER = HybridRetriever()
    else:
        raise ValueError(
            f"unknown CADASTRE_AGENT_RETRIEVER={kind!r} (expected dense|bm25|hybrid)"
        )
    return _AGENT_RETRIEVER


def _dedupe_and_cap_chunks(chunks: list[dict], cap: int) -> list[dict]:
    """Dedupe by chunk_id (keep the first occurrence — already ordered
    by boosted_score desc from apply_publisher_boost) and truncate to
    `cap`. v3 piled chunks across sub-questions and reflect cycles to
    +19% over v2; capping keeps the synth context tight (Task 3.29).
    """
    seen: set[str] = set()
    out: list[dict] = []
    for c in chunks:
        cid = (c.get("chunk_id") or "").strip()
        if cid and cid in seen:
            continue
        if cid:
            seen.add(cid)
        out.append(c)
        if len(out) >= cap:
            break
    return out


def _retrieve_docs(query: str, k: int = DOCS_TOP_K) -> list[dict]:
    """Pull top-k chunks via the configured retriever (Task 3.25).

    Default retriever is hybrid (BGE dense + BM25 RRF) — see
    `_agent_retriever` for the env-driven override. Wrapped so tests
    can monkey-patch this without spinning up Qdrant or BM25 pickles.
    Returns the agent-state shape (`{chunk_id, score, payload}`).
    """
    hits = _agent_retriever().retrieve(query, k=k)
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
    chunks = _dedupe_and_cap_chunks(chunks, MAX_TOTAL_CHUNKS)

    return {
        "retrieved_chunks": chunks,
        "tool_results": tool_results,
        "iteration_count": iter_count + 1,
    }


REFLECTOR_MODEL = os.environ.get("CADASTRE_REFLECTOR_MODEL", "claude-haiku-4-5")
REFLECTOR_MODEL_OPENAI_DEFAULT = "gpt-4o-mini"
REFLECTOR_MAX_TOKENS = 768


def _resolve_reflector_model() -> str:
    if llm.get_provider() == "openai":
        return os.environ.get(
            "CADASTRE_OPENAI_REFLECTOR_MODEL", REFLECTOR_MODEL_OPENAI_DEFAULT
        )
    return REFLECTOR_MODEL
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
                    "True when the draft answers the user's question "
                    "using the available tool results and retrieved "
                    "chunks. Default to True unless there is a SPECIFIC, "
                    "concrete gap you can name and a refined query that "
                    "would plausibly fill it. Do not loop just because "
                    "more evidence might be nice — the agent has a "
                    "2-iteration budget."
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
    "ship, or whether the agent should run ONE more retrieval/tool pass. "
    "Default is to ship: mark is_complete=True unless there is a concrete, "
    "answerable gap. Specifically, mark complete when EITHER (a) every "
    "sub-question has at least one tool result OR retrieved chunk that "
    "addresses it, OR (b) the draft already states an answer and cites "
    "evidence. Only mark incomplete when you can name a specific missing "
    "piece (e.g. 'Melbourne leg of the comparison absent', 'no NSW stamp "
    "duty number for the $700k case') AND propose a refined query that "
    "would plausibly retrieve it. Do not loop on stylistic concerns, "
    "phrasing, or generic 'more context would help'. The agent has only "
    "2 iterations total — looping costs the user latency for marginal "
    "gain. Submit via the submit_reflection tool."
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
        date = payload.get("date")
        text = (payload.get("text") or "")[:REFLECT_EVIDENCE_PREVIEW_CHARS]
        lines.append(
            f"[{i}] {payload.get('publisher', '?')} ({date}) | {title}"
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
    prompt = _build_reflect_prompt(state)
    provider = llm.get_provider()
    # Pre-check the active provider's API key so missing-key configuration
    # raises loudly (caller bug) rather than silently flipping the agent
    # into the fallback "is_complete=True" branch below.
    if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set; reflect needs Claude."
        )
    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set; reflect needs GPT "
            "(CADASTRE_LLM_PROVIDER=openai)."
        )
    model = _resolve_reflector_model()
    log.debug("reflect → %s (%s)", model, provider)
    try:
        raw, usage = llm.call_with_tool(
            system=_REFLECTOR_SYSTEM,
            user=_with_today(prompt),
            tool_def=_REFLECT_TOOL,
            tool_name="submit_reflection",
            max_tokens=REFLECTOR_MAX_TOKENS,
            model=model,
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
    log_cost(usage, model, "reflect")

    if raw is None:
        log.warning("reflect: no tool_use / tool_call; marking complete")
        return {
            "reflection": {
                "is_complete": True,
                "missing": [],
                "refined_query": None,
            }
        }

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
SYNTHESIZER_MODEL_OPENAI_DEFAULT = "gpt-4o"
SYNTHESIZER_MAX_TOKENS = 1536


def _resolve_synthesizer_model() -> str:
    if llm.get_provider() == "openai":
        return os.environ.get(
            "CADASTRE_OPENAI_SYNTHESIZER_MODEL", SYNTHESIZER_MODEL_OPENAI_DEFAULT
        )
    return SYNTHESIZER_MODEL

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
    "knowledge.\n\n"
    "PLACEMENT IS LOAD-BEARING. Put each citation IMMEDIATELY after the "
    "value or fact it supports — same sentence, before the next punctuation "
    "mark or conjunction. Do NOT defer citations to a trailing parenthesis "
    "or to the Sources line.\n"
    "  GOOD: The median Sydney house price is $1,515,000 "
    "[tool:abs_property_price_index, retrieved:2026-05-02], up 4.2% YoY "
    "[tool:abs_property_price_index, retrieved:2026-05-02].\n"
    "  BAD : The median Sydney house price is $1,515,000, up 4.2% YoY. "
    "[tool:abs_property_price_index, retrieved:2026-05-02]\n"
    "If a sentence has two numeric values, EACH gets its own inline "
    "citation token — do not share one citation across multiple values.\n\n"
    "End with a 'Sources' line that lists each cited publisher and tool "
    "exactly once."
)

# Universal compliance baseline (Task X.03). Applied to every persona
# so that even an unrecognised persona key — or a future persona we
# haven't tuned — still gets the floor-level "not financial advice"
# behaviour. Per-persona language layered on top via persona_disclaimer().
_DISCLAIMER_BASELINE = (
    "COMPLIANCE: You are a research assistant, not a licensed "
    "financial, legal, tax, or buyer's-agent professional. Never "
    "produce personal financial, legal, or tax recommendations, and "
    "never recommend specific financial products (mortgages, "
    "insurance, super funds). Stick to summarising what the evidence "
    "says. CURRENCY: The document corpus was assembled before the "
    "Treasury Laws Amendment (Tax Reform No. 1) Act 2026 (enacted "
    "26 Jun 2026), which changed negative gearing and the CGT discount. "
    "Whenever the answer touches negative gearing, capital gains tax, or "
    "other tax treatment, state plainly that the retrieved evidence "
    "predates that reform and may be out of date, and direct the user to "
    "the ATO or a registered tax agent for current settings — do not "
    "present pre-reform tax rules as current. The persona-specific "
    "disclaimer policy below tells you exactly what closing-line "
    "disclaimer to append to the answer."
)

_SYNTHESIZER_SYSTEM = (
    "You are a careful Australian housing-market research assistant. "
    "Answer the user's question concisely and ONLY from the supplied "
    "evidence. Numbers must come from tool results; explanatory framing "
    "must come from retrieved chunks.\n\n"
    "USE THE EVIDENCE. When EVIDENCE — TOOL RESULTS contains data that "
    "addresses the question, you MUST quote that data (with the "
    "[tool:..., retrieved:...] citation). Never fall back to 'I cannot "
    "provide' or 'consult a real estate website' when the answer is in "
    "the tool results above. The tool's data fields are the authoritative "
    "current value as of `retrieved_at` — treat them as fresh.\n\n"
    + _DISCLAIMER_BASELINE
    + "\n\n"
    + _CITATION_RULES
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
        date = payload.get("date")
        text = (payload.get("text") or "")[:SYNTH_CHUNK_TEXT_LIMIT]
        header = f"[{i}] publisher={publisher!r} date={date!r} page={page!r} title={title!r}"
        if section:
            header += f" section={section!r}"
        # Chunk-level regime tag lands with Task 5.09; surface it when present
        # so the synthesizer can flag pre-reform evidence explicitly (F-5).
        regime = payload.get("regime")
        if regime:
            header += f" regime={regime!r}"
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
    msgs = state.get("messages", [])
    prompt = _build_synth_prompt(state)
    persona = effective_persona(state)
    system_prompt = (
        f"{_SYNTHESIZER_SYSTEM}\n\n"
        f"{persona_prompt_addendum(persona)}\n\n"
        f"{persona_disclaimer(persona)}"
    )

    # Pre-check the active provider's API key so misconfiguration raises
    # loudly (caller bug). Transient API errors below still fall through
    # to the stub draft so the graph can terminate.
    provider = llm.get_provider()
    if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set; synthesize needs Claude Sonnet."
        )
    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set; synthesize needs GPT "
            "(CADASTRE_LLM_PROVIDER=openai)."
        )

    model = _resolve_synthesizer_model()
    log.debug("synthesize → %s (%s, persona=%s)", model, provider, persona)
    try:
        raw, usage = llm.call_text(
            system=system_prompt,
            user=_with_today(prompt),
            max_tokens=SYNTHESIZER_MAX_TOKENS,
            model=model,
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
    log_cost(usage, model, "synthesize")

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
