"""Unit tests for the structured agent-turn trace (Task 4.05).

Coverage:
  - 5 steps in the right order: classify, decompose, retrieve_or_tool,
    synthesize, reflect.
  - classify step surfaces persona, query_type, and any non-empty flag.
  - decompose step distinguishes atomic vs multi-step.
  - retrieve_or_tool step counts chunks + tools, lists top 5 chunks,
    flags errored tool calls (kind == 'warn').
  - reflect step explains iteration count: 1 = ok, 2-3 = looped (info),
    >= 4 = cap hit (warn).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.app.trace import build_trace_steps


@dataclass
class FakeTurn:
    """Minimal duck-typed turn object — no Streamlit / state.py dep."""

    classification: dict | None = None
    sub_questions: list = field(default_factory=list)
    tool_results: list = field(default_factory=list)
    retrieved_chunks: list = field(default_factory=list)
    iteration_count: int = 0
    answer: str = ""


def _full_classification(**overrides) -> dict:
    base = {
        "persona": "investor",
        "query_type": "computational",
        "needs_docs": True,
        "needs_data": True,
        "needs_decomposition": True,
    }
    base.update(overrides)
    return base


# ---------- shape ------------------------------------------------


def test_returns_five_steps_in_canonical_order():
    steps = build_trace_steps(FakeTurn(classification=_full_classification()))
    names = [s.name for s in steps]
    assert names == [
        "Classify query",
        "Decompose",
        "Retrieve / tools",
        "Synthesize",
        "Reflect",
    ]


# ---------- classify step ---------------------------------------


def test_classify_step_summarises_persona_and_type():
    steps = build_trace_steps(
        FakeTurn(
            classification=_full_classification(
                persona="first_home_buyer", query_type="factual"
            )
        )
    )
    s = steps[0]
    assert "first_home_buyer" in s.summary
    assert "factual" in s.summary


def test_classify_step_lists_only_active_flags():
    steps = build_trace_steps(
        FakeTurn(
            classification=_full_classification(
                needs_docs=True,
                needs_data=False,
                needs_decomposition=False,
            )
        )
    )
    flags_line = next(d for d in steps[0].details if d.startswith("flags:"))
    assert "docs" in flags_line
    assert "data" not in flags_line
    assert "decompose" not in flags_line


def test_classify_step_with_no_flags_says_none():
    steps = build_trace_steps(
        FakeTurn(
            classification=_full_classification(
                needs_docs=False,
                needs_data=False,
                needs_decomposition=False,
            )
        )
    )
    flags_line = next(d for d in steps[0].details if d.startswith("flags:"))
    assert "none" in flags_line.lower()


def test_classify_step_handles_missing_classification():
    steps = build_trace_steps(FakeTurn(classification=None))
    s = steps[0]
    assert "general" in s.summary  # default fallback
    assert "?" in s.summary  # query_type unknown


# ---------- decompose step --------------------------------------


def test_decompose_step_atomic_when_no_subquestions():
    steps = build_trace_steps(FakeTurn(sub_questions=[]))
    s = steps[1]
    assert "atomic" in s.summary.lower()
    assert s.details == []


def test_decompose_step_lists_subquestions_with_numbers():
    steps = build_trace_steps(
        FakeTurn(sub_questions=["What is X?", "What is Y?", "What is Z?"])
    )
    s = steps[1]
    assert "3" in s.summary
    assert s.details[0].startswith("1.")
    assert s.details[1].startswith("2.")
    assert "Z?" in s.details[2]


# ---------- retrieve_or_tool step -------------------------------


def test_retrieve_step_counts_chunks_and_tools():
    chunks = [
        {"score": 0.9, "payload": {"publisher": "RBA", "title": "Cash rate"}},
        {"score": 0.8, "payload": {"publisher": "ABS", "title": "Lending"}},
    ]
    tools = [
        {"tool": "rba_cash_rate", "result": {"source": "RBA F1.1"}},
    ]
    steps = build_trace_steps(
        FakeTurn(retrieved_chunks=chunks, tool_results=tools)
    )
    s = steps[2]
    assert "2 chunk" in s.summary
    assert "1 tool" in s.summary


def test_retrieve_step_lists_top_5_chunks_only():
    chunks = [
        {"score": 0.9 - i * 0.01, "payload": {"publisher": f"Pub{i}"}}
        for i in range(8)
    ]
    steps = build_trace_steps(FakeTurn(retrieved_chunks=chunks))
    s = steps[2]
    chunk_lines = [d for d in s.details if d.startswith("chunk ")]
    assert len(chunk_lines) == 5
    assert any("3 more" in d for d in s.details)


def test_retrieve_step_uses_boosted_score_when_present():
    chunks = [
        {"score": 0.5, "boosted_score": 0.9, "payload": {"publisher": "X"}},
    ]
    steps = build_trace_steps(FakeTurn(retrieved_chunks=chunks))
    s = steps[2]
    chunk_line = next(d for d in s.details if d.startswith("chunk 1"))
    assert "0.900" in chunk_line


def test_retrieve_step_flags_tool_errors_as_warn():
    tools = [
        {"tool": "rba_cash_rate", "result": {"source": "RBA"}},
        {"tool": "compute_stamp_duty_nsw", "error": "missing arg X"},
    ]
    steps = build_trace_steps(FakeTurn(tool_results=tools))
    s = steps[2]
    assert s.kind == "warn"
    err_line = next(d for d in s.details if "ERROR" in d)
    assert "missing arg X" in err_line


def test_retrieve_step_kind_ok_when_no_errors():
    tools = [{"tool": "rba_cash_rate", "result": {"source": "RBA"}}]
    steps = build_trace_steps(FakeTurn(tool_results=tools))
    s = steps[2]
    assert s.kind == "ok"


# ---------- synthesize step -------------------------------------


def test_synthesize_step_reports_iteration_and_length():
    steps = build_trace_steps(
        FakeTurn(iteration_count=2, answer="some answer text here")
    )
    s = steps[3]
    assert "2 iteration" in s.summary
    assert str(len("some answer text here")) in s.summary


# ---------- reflect step ----------------------------------------


def test_reflect_step_one_pass_is_ok():
    steps = build_trace_steps(FakeTurn(iteration_count=1))
    s = steps[4]
    assert s.kind == "ok"
    assert "one pass" in s.summary.lower()


def test_reflect_step_loop_is_info():
    steps = build_trace_steps(FakeTurn(iteration_count=2))
    s = steps[4]
    assert s.kind == "info"
    assert "loop" in s.summary.lower()


def test_reflect_step_cap_hit_is_warn():
    steps = build_trace_steps(FakeTurn(iteration_count=4))
    s = steps[4]
    assert s.kind == "warn"
    assert "cap" in s.summary.lower()
