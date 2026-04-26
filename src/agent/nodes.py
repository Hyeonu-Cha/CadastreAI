"""LangGraph node implementations for the CadastreAI agent.

Each function takes the full `AgentState` and returns a partial state
update that LangGraph merges in.

Implementation status
---------------------
- `classify_query`   → Task 3.13 (Claude Haiku, structured output)
- `decompose`        → stub; Task 3.14 will replace
- `retrieve_or_tool` → stub; Task 3.15 will replace
- `reflect`          → stub; Task 3.17/3.18 will replace
- `synthesize`       → stub; Task 3.17 prompt + post-processor will
                       harden citation discipline (Task 3.19)
"""
from __future__ import annotations

import json
import logging
import os

from src.agent.graph import AgentState, Classification

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


def decompose(state: AgentState) -> dict:
    """Split a complex query into 2–4 atomic sub-questions.

    Real impl (Task 3.14): only fires when classify_query flags
    `needs_decomposition=True`. Uses Claude Haiku again with a
    structured prompt. The stub returns an empty list so the
    caller treats the original question atomically.
    """
    log.debug("decompose stub: no sub-questions")
    return {"sub_questions": []}


def retrieve_or_tool(state: AgentState) -> dict:
    """Per sub-question, decide between retrieval, tool calls, or both.

    Real impl (Task 3.15): routes to `Retriever.search` for doc-shaped
    questions, to one of the structured tools (rba/abs/sqm/compute)
    for numeric questions, or both for comparative ones. The stub is
    a no-op but bumps `iteration_count` so the safety cap in Task
    3.11 is observable from tests.
    """
    new_count = state.get("iteration_count", 0) + 1
    log.debug("retrieve_or_tool stub: iteration_count -> %d", new_count)
    return {
        "retrieved_chunks": list(state.get("retrieved_chunks", [])),
        "tool_results": list(state.get("tool_results", [])),
        "iteration_count": new_count,
    }


def reflect(state: AgentState) -> dict:
    """Check coverage of sub-questions and citation discipline.

    Real impl (Task 3.17/3.18): asks Claude to inspect the draft
    against the original question + sub-questions + retrieved
    evidence. Returns `{is_complete, missing, refined_query}` —
    the graph loops back through retrieve_or_tool when not complete
    (respecting the iteration cap). The stub immediately marks the
    answer complete so the graph terminates after one pass.
    """
    log.debug("reflect stub: is_complete=True")
    return {
        "reflection": {
            "is_complete": True,
            "missing": [],
            "refined_query": None,
        }
    }


def synthesize(state: AgentState) -> dict:
    """Produce the user-facing answer with inline citations.

    Real impl: stuffs retrieved chunks and tool results into a Claude
    Sonnet prompt and asks for a sourced answer. The stub echoes the
    user's question back as a placeholder so the graph end-to-end
    test in Task 3.12 can assert that the synthesizer ran.
    """
    msgs = state.get("messages", [])
    user_query = next(
        (m["content"] for m in msgs if m.get("role") == "user"),
        "",
    )
    draft = f"[stub answer to: {user_query}]"
    log.debug("synthesize stub: draft=%r", draft)
    return {
        "answer_draft": draft,
        "messages": msgs + [{"role": "assistant", "content": draft}],
    }


__all__ = [
    "classify_query",
    "decompose",
    "retrieve_or_tool",
    "reflect",
    "synthesize",
]
