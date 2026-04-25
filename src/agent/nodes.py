"""LangGraph node stubs for the CadastreAI agent (Task 3.10).

Each function takes the full `AgentState` and returns a partial state
update that LangGraph merges in. The bodies here are deliberately
minimal — they exist so the graph wiring in Task 3.11 has runnable
nodes to connect, while Day 17 (3.13–3.15) and Day 18 (3.17/3.18)
will replace them with the real LLM-backed implementations.

Stub behaviours
---------------
- `classify_query`  → default to `"factual"`.
- `decompose`       → no decomposition; sub_questions stays empty.
- `retrieve_or_tool` → no-op; only bumps iteration_count so the loop
                       cap in 3.11 is observable in tests.
- `reflect`         → marks the answer as complete; loops won't fire.
- `synthesize`      → echoes the user query into a placeholder draft.

Each stub's docstring documents what the *real* implementation will
look like so the contract is locked in before LLM glue lands.
"""
from __future__ import annotations

import logging

from src.agent.graph import AgentState

log = logging.getLogger(__name__)


def classify_query(state: AgentState) -> dict:
    """Decide what kind of question we're answering.

    Real impl (Task 3.13): calls Claude Haiku with a structured-output
    prompt and returns one of {factual, comparative, computational,
    exploratory} plus needs_docs / needs_data / needs_decomposition
    flags. The stub just labels everything as "factual" so downstream
    routing has something to read.
    """
    log.debug("classify_query stub: query_type='factual'")
    return {"query_type": "factual"}


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
