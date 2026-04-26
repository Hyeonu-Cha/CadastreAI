"""LangGraph agent state + graph wiring (Tasks 3.09 / 3.11).

The CadastreAI agent is a small graph of nodes that route between doc
retrieval, tool calls, drafting an answer, and self-reflection. Every
node receives the full `AgentState` and returns a partial dict that
LangGraph merges in. Keeping state in one TypedDict (rather than spread
across closures) makes the graph easier to trace and test.

Edge layout (Task 3.11):

    START → classify_query → decompose → retrieve_or_tool → synthesize
            → reflect → END  (when is_complete or iter ≥ MAX_ITERATIONS)
                      → retrieve_or_tool  (when reflection flags gaps)

The reflection-driven loop-back is what makes this an *agent* rather
than a pipeline — the model can decide a draft is incomplete and ask
for another retrieval/tool round. `MAX_ITERATIONS = 4` guards against
runaway loops; we exit unconditionally once that cap is hit, even if
reflection still says incomplete.

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
Persona = Literal[
    "first_home_buyer",
    "investor",
    "policy_researcher",
    "journalist",
    "general",
]


class Message(TypedDict):
    role: Literal["user", "assistant", "tool"]
    content: str


class Classification(TypedDict):
    """Structured output of `classify_query` (Task 3.13).

    `query_type` mirrors the top-level state field for convenience.
    The four boolean flags drive routing in Tasks 3.14 / 3.15:

    - `needs_decomposition` → run `decompose` (Task 3.14)
    - `needs_docs`          → retrieve from the BGE index
    - `needs_data`          → call one or more structured tools
    """

    persona: Persona
    query_type: QueryType
    needs_docs: bool
    needs_data: bool
    needs_decomposition: bool


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
    classification: Classification | None
    user_persona: Persona | None
    sub_questions: list[str]
    retrieved_chunks: list[RetrievedChunk]
    tool_results: list[dict[str, Any]]
    answer_draft: str | None
    reflection: Reflection
    iteration_count: int


def initial_state(query: str, *, user_persona: str | None = None) -> AgentState:
    """Seed state for a fresh run.

    `user_persona` (optional) is the caller-supplied persona that
    overrides the classifier's guess downstream. Pass through verbatim;
    the persona helpers handle validation + fallback to "general".
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string")
    state: AgentState = {
        "messages": [{"role": "user", "content": query.strip()}],
        "query_type": None,
        "classification": None,
        "user_persona": None,
        "sub_questions": [],
        "retrieved_chunks": [],
        "tool_results": [],
        "answer_draft": None,
        "reflection": {},
        "iteration_count": 0,
    }
    if user_persona:
        state["user_persona"] = user_persona  # type: ignore[typeddict-item]
    return state


MAX_ITERATIONS = 4


def _route_after_reflect(state: AgentState) -> str:
    """Conditional edge — loop or finish based on reflection + cap.

    Routing rules (Task 3.18):
      1. Iteration cap reached → end unconditionally. The cap exists to
         keep a stuck agent from spinning forever.
      2. Reflection says complete → end.
      3. Reflection says incomplete but offers no `refined_query` →
         end. There's nothing concrete to feed the next pass, so
         looping would just bump `iteration_count` until the cap.
      4. Otherwise → loop back through `retrieve_or_tool`, which will
         pick up `reflection.refined_query` as its single sub-question.
    """
    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        return "end"
    refl = state.get("reflection") or {}
    if refl.get("is_complete", False):
        return "end"
    refined = (refl.get("refined_query") or "").strip()
    if not refined:
        return "end"
    return "loop"


def build_graph():
    """Compile the agent's LangGraph state graph.

    Lazily imported so test/CLI users who don't actually run the graph
    aren't paying the langgraph import cost up front.
    """
    from langgraph.graph import END, START, StateGraph

    from src.agent.nodes import (
        classify_query,
        decompose,
        reflect,
        retrieve_or_tool,
        synthesize,
    )

    g: StateGraph = StateGraph(AgentState)
    g.add_node("classify_query", classify_query)
    g.add_node("decompose", decompose)
    g.add_node("retrieve_or_tool", retrieve_or_tool)
    g.add_node("synthesize", synthesize)
    g.add_node("reflect", reflect)

    g.add_edge(START, "classify_query")
    g.add_edge("classify_query", "decompose")
    g.add_edge("decompose", "retrieve_or_tool")
    g.add_edge("retrieve_or_tool", "synthesize")
    g.add_edge("synthesize", "reflect")
    g.add_conditional_edges(
        "reflect",
        _route_after_reflect,
        {"loop": "retrieve_or_tool", "end": END},
    )
    return g.compile()


__all__ = [
    "AgentState",
    "Classification",
    "MAX_ITERATIONS",
    "Message",
    "Persona",
    "QueryType",
    "Reflection",
    "RetrievedChunk",
    "build_graph",
    "initial_state",
]
