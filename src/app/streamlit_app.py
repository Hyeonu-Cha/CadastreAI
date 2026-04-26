"""CadastreAI Streamlit app (Task 4.01).

Run with:

    streamlit run src/app/streamlit_app.py

Layout
------
- Sidebar: persona selector (Homebuyer / Investor / Researcher / Just
  exploring), example-query list scoped to that persona, and a
  "Reset conversation" button.
- Main pane: conversation history (each turn shows the user's query
  and the agent's draft) plus a chat input at the bottom.

The agent itself is `src.agent.graph.build_graph()` — same code that
runs in the eval harness. The graph is built lazily and cached so we
only pay the wiring cost once per Streamlit process. Persona handling
beyond the dropdown lands in Task 4.02; this scaffold just records the
selection so 4.02 can read it off `session_state["persona"]`.
"""
from __future__ import annotations

import logging
import time

import streamlit as st

from src.app.citations import (
    Citation,
    citation_excerpt,
    match_doc_evidence,
    match_tool_evidence,
    parse_citations,
    renumber_answer,
)
from src.app.state import (
    UI_PERSONAS,
    TurnRecord,
    append_turn,
    example_queries_for,
    init_session_defaults,
    reset_history,
    ui_to_agent_persona,
)

log = logging.getLogger(__name__)

PAGE_TITLE = "CadastreAI — Australian housing-market research agent"


@st.cache_resource(show_spinner="Loading agent graph...")
def _load_graph():
    """Build the LangGraph agent once per Streamlit process.

    Imported lazily so a missing `agent` extra (anthropic / langgraph)
    surfaces here as a friendly error rather than at module import.
    """
    from src.agent.graph import build_graph

    return build_graph()


def _run_agent(query: str, persona: str) -> TurnRecord:
    """Drive one turn through the graph and return a `TurnRecord`."""
    from src.agent.graph import initial_state

    turn = TurnRecord(query=query, persona=persona)
    t0 = time.perf_counter()
    try:
        graph = _load_graph()
        final = graph.invoke(
            initial_state(query, user_persona=ui_to_agent_persona(persona))
        )
        turn.answer = final.get("answer_draft") or ""
        turn.classification = final.get("classification")
        turn.sub_questions = list(final.get("sub_questions") or [])
        turn.tool_results = list(final.get("tool_results") or [])
        turn.retrieved_chunks = list(final.get("retrieved_chunks") or [])
        turn.iteration_count = int(final.get("iteration_count") or 0)
    except Exception as exc:  # noqa: BLE001 — surface to the user
        log.exception("agent invocation failed")
        turn.error = f"{type(exc).__name__}: {exc}"
    finally:
        turn.elapsed_seconds = round(time.perf_counter() - t0, 2)
    return turn


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("CadastreAI")
        st.caption("Australian housing-market research agent.")

        persona = st.radio(
            "I'm a...",
            options=UI_PERSONAS,
            index=UI_PERSONAS.index(st.session_state["persona"]),
            key="persona",
            help="Used to tune retrieval weighting and prompt tone (Task 4.02).",
        )

        st.divider()
        st.subheader("Try an example")
        for q in example_queries_for(persona):
            if st.button(q, use_container_width=True, key=f"ex::{q}"):
                st.session_state["pending_query"] = q
                st.rerun()

        st.divider()
        st.toggle("Show reasoning trace", key="show_trace")
        if st.button("Reset conversation", type="secondary", use_container_width=True):
            reset_history(st.session_state)
            st.rerun()


def _render_citation_card(idx: int, citation: Citation, turn: TurnRecord) -> None:
    """Render one numbered source card under the answer.

    Doc citations show publisher / page / title / excerpt. Tool citations
    show the tool name, retrieval timestamp, and the result envelope as
    JSON. If no matching evidence is found we still render the chip so
    the user knows the model claimed it; the body just notes the gap.
    """
    if citation.kind == "doc":
        chunk = match_doc_evidence(citation, turn.retrieved_chunks)
        header = f"**[{idx}]** {citation.publisher}"
        if citation.page:
            header += f" · page {citation.page}"
        st.markdown(header)
        if chunk is None:
            st.caption("No matching retrieved chunk — model may have overreached.")
            return
        payload = chunk.get("payload") or {}
        title = (payload.get("title") or "").strip()
        section = (payload.get("section_heading") or "").strip()
        sub_bits = [b for b in (title, section) if b]
        if sub_bits:
            st.caption(" · ".join(sub_bits))
        st.markdown(f"> {citation_excerpt(chunk)}")
        return

    env = match_tool_evidence(citation, turn.tool_results)
    header = f"**[{idx}]** `{citation.tool}` · retrieved {citation.retrieved_at}"
    st.markdown(header)
    if env is None:
        st.caption("No matching tool result — model may have overreached.")
        return
    result = env.get("result") or {}
    if "data" in result:
        st.json(result["data"])
    if result.get("source"):
        st.caption(f"source: {result['source']}")


def _render_turn(turn: TurnRecord) -> None:
    with st.chat_message("user"):
        st.markdown(turn.query)
    with st.chat_message("assistant"):
        if turn.error:
            st.error(turn.error)
        else:
            citations = parse_citations(turn.answer or "")
            display_text = renumber_answer(turn.answer or "", citations)
            st.markdown(display_text or "_(no draft answer returned)_")
            if citations:
                with st.expander(
                    f"Cited sources ({len(citations)})", expanded=False
                ):
                    for i, c in enumerate(citations, start=1):
                        _render_citation_card(i, c, turn)
        meta_bits: list[str] = [f"persona: {turn.persona}"]
        if turn.iteration_count:
            meta_bits.append(f"iterations: {turn.iteration_count}")
        if turn.elapsed_seconds is not None:
            meta_bits.append(f"{turn.elapsed_seconds:.1f}s")
        st.caption(" · ".join(meta_bits))

        if st.session_state.get("show_trace") and not turn.error:
            with st.expander("Reasoning trace", expanded=False):
                if turn.classification:
                    st.markdown("**Classification**")
                    st.json(turn.classification)
                if turn.sub_questions:
                    st.markdown("**Sub-questions**")
                    for q in turn.sub_questions:
                        st.markdown(f"- {q}")
                if turn.tool_results:
                    st.markdown(f"**Tool calls** ({len(turn.tool_results)})")
                    for t in turn.tool_results:
                        st.markdown(f"- `{t.get('tool')}`({t.get('args')})")
                if turn.retrieved_chunks:
                    st.markdown(
                        f"**Retrieved chunks** ({len(turn.retrieved_chunks)})"
                    )


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    init_session_defaults(st.session_state)

    _render_sidebar()

    st.title("CadastreAI")
    st.write(
        "Ask anything about Australian housing — interest rates, prices, lending, "
        "rental markets, or policy. Answers cite their sources."
    )

    for turn in st.session_state["history"]:
        _render_turn(turn)

    user_query = st.chat_input("Ask about Australian housing...")
    pending = st.session_state.pop("pending_query", None)
    query = user_query or pending

    if query:
        with st.spinner("Thinking..."):
            turn = _run_agent(query, st.session_state["persona"])
        append_turn(st.session_state, turn)
        st.rerun()


if __name__ == "__main__":
    main()
