"""LangGraph agent state definition (Task 3.09).

The CadastreAI agent is a small graph of nodes that route between doc
retrieval, tool calls, drafting an answer, and self-reflection. Every
node receives the full `AgentState` and returns a partial dict that
LangGraph merges in. Keeping state in one TypedDict (rather than spread
across closures) makes the graph easier to trace and test.

Field guide
-----------
- `messages`: chronological log of `{role, content}` dicts. The user
  question lands here as `{"role": "user", "content": ...}`; agent
  drafts and tool/result events append on the way through the graph.
- `query_type`: classifier output — one of "factual" / "comparative" /
  "computational" / "exploratory". Drives downstream routing.
- `sub_questions`: 0–N atomic questions produced by the decomposer.
  An empty list means the original question is treated atomically.
- `retrieved_chunks`: doc chunks pulled by the BGE retriever for any
  sub-question. Each entry keeps the chunk id, score, and payload so
  the synthesizer can build inline citations.
- `tool_results`: outputs from the structured tools (RBA / ABS / SQM /
  compute / chart). Each entry is the standard tool envelope
  `{data, source, retrieved_at, citation, ...}`.
- `answer_draft`: the synthesizer's current best answer text.
- `reflection`: dict produced by the reflect node — `{is_complete,
  missing, refined_query}`. Drives whether we loop or finish.
- `iteration_count`: monotonically increasing — Task 3.11 caps it at 4
  to keep the agent from spinning on a hopeless query.

Tasks 3.10/3.11 implement the actual nodes and edge wiring. This module
is intentionally narrow for now — pure data, no behaviour.

    >>> from src.agent.graph import AgentState, initial_state
    >>> state = initial_state("What's the cash rate today?")
    >>> state["iteration_count"]
    0
    >>> state["messages"][0]["role"]
    'user'
"""
from __future__ import annotations

from typing import Any, Literal, TypedDict

QueryType = Literal["factual", "comparative", "computational", "exploratory"]


class Message(TypedDict):
    role: Literal["user", "assistant", "tool"]
    content: str


class RetrievedChunk(TypedDict):
    chunk_id: str
    score: float
    payload: dict[str, Any]


class Reflection(TypedDict, total=False):
    is_complete: bool
    missing: list[str]
    refined_query: str | None


class AgentState(TypedDict, total=False):
    """Full state object passed between graph nodes.

    `total=False` so node return values can be partial dicts — LangGraph
    will merge them onto the running state. Initial state is built via
    `initial_state(query)` to make sure required fields exist.
    """

    messages: list[Message]
    query_type: QueryType | None
    sub_questions: list[str]
    retrieved_chunks: list[RetrievedChunk]
    tool_results: list[dict[str, Any]]
    answer_draft: str | None
    reflection: Reflection
    iteration_count: int


def initial_state(query: str) -> AgentState:
    """Seed state for a fresh run — only the user query is set."""
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    return {
        "messages": [{"role": "user", "content": query.strip()}],
        "query_type": None,
        "sub_questions": [],
        "retrieved_chunks": [],
        "tool_results": [],
        "answer_draft": None,
        "reflection": {},
        "iteration_count": 0,
    }


__all__ = [
    "AgentState",
    "Message",
    "QueryType",
    "Reflection",
    "RetrievedChunk",
    "initial_state",
]
