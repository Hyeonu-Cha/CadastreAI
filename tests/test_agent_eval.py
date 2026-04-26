"""Unit tests for the agent eval harness (Task 3.21).

Pure offline. The metric primitives (tool_metrics, publisher_recall,
trajectory_efficiency, groundedness, jaccard, aggregate) get exhaustive
edge-case coverage; `run_agent_eval` is exercised by stubbing the graph
so we never need an API key for these tests.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval import agent_eval as ae

# ---------- jaccard / tool metrics ---------------------------------


def test_jaccard_empty_sets_score_one():
    assert ae.jaccard(set(), set()) == 1.0


def test_jaccard_full_overlap():
    assert ae.jaccard({"a", "b"}, {"a", "b"}) == 1.0


def test_jaccard_partial():
    assert ae.jaccard({"a", "b"}, {"a", "c"}) == pytest.approx(1 / 3)


def test_tool_metrics_perfect():
    m = ae.tool_metrics(["rba_cash_rate"], ["rba_cash_rate"])
    assert m["tool_call_accuracy"] == 1.0
    assert m["tool_call_recall"] == 1.0
    assert m["tool_call_precision"] == 1.0
    assert m["extra_tools"] == [] and m["missing_tools"] == []


def test_tool_metrics_extras_and_missings():
    m = ae.tool_metrics(
        ["rba_cash_rate", "abs_lending_indicators"],  # actual
        ["rba_cash_rate", "compute_rental_yield"],  # expected
    )
    assert m["extra_tools"] == ["abs_lending_indicators"]
    assert m["missing_tools"] == ["compute_rental_yield"]
    assert m["tool_call_accuracy"] == pytest.approx(1 / 3)
    assert m["tool_call_recall"] == 0.5
    assert m["tool_call_precision"] == 0.5


def test_tool_metrics_both_empty_is_perfect():
    """Doc-only query — no tools expected, none called."""
    m = ae.tool_metrics([], [])
    assert m["tool_call_accuracy"] == 1.0
    assert m["tool_call_recall"] == 1.0
    assert m["tool_call_precision"] == 1.0


def test_tool_metrics_unexpected_call_when_none_expected():
    """Doc-only query but the agent called a tool anyway."""
    m = ae.tool_metrics(["rba_cash_rate"], [])
    assert m["tool_call_recall"] == 0.0
    assert m["tool_call_precision"] == 0.0


# ---------- publisher recall ----------------------------------------


def test_publisher_recall_full_match():
    chunks = [{"payload": {"publisher": "RBA"}}, {"payload": {"publisher": "AHURI"}}]
    out = ae.publisher_recall(chunks, ["RBA", "AHURI"])
    assert out["publisher_recall"] == 1.0
    assert out["missing_publishers"] == []


def test_publisher_recall_partial():
    chunks = [{"payload": {"publisher": "RBA"}}]
    out = ae.publisher_recall(chunks, ["RBA", "AHURI"])
    assert out["publisher_recall"] == 0.5
    assert out["missing_publishers"] == ["AHURI"]


def test_publisher_recall_no_expected_is_one():
    out = ae.publisher_recall([{"payload": {"publisher": "X"}}], [])
    assert out["publisher_recall"] == 1.0


# ---------- trajectory efficiency ----------------------------------


def test_trajectory_efficiency_perfect():
    assert ae.trajectory_efficiency(["a"], ["a"], iteration_count=1) == 1.0


def test_trajectory_efficiency_extra_iterations():
    # 2 extra iterations, 0 extra tools → 1 / (1 + 2) = 1/3
    assert ae.trajectory_efficiency(["a"], ["a"], iteration_count=3) == pytest.approx(
        1 / 3
    )


def test_trajectory_efficiency_extra_tools():
    # 0 extra iterations, 1 extra tool → 1/2
    assert ae.trajectory_efficiency(
        ["a", "b"], ["a"], iteration_count=1
    ) == pytest.approx(1 / 2)


def test_trajectory_efficiency_missing_tools_not_penalised():
    """Missing tools belong to recall, not efficiency."""
    assert ae.trajectory_efficiency(
        ["a"], ["a", "b", "c"], iteration_count=1
    ) == 1.0


# ---------- groundedness -------------------------------------------


def test_groundedness_no_numbers_is_one():
    out = ae.groundedness("This is a qualitative explanation only.")
    assert out["groundedness"] == 1.0
    assert out["n_numeric_claims"] == 0


def test_groundedness_all_numbers_cited():
    text = (
        "The cash rate is 4.10% [tool:rba_cash_rate, retrieved:2026-04-26]. "
        "Median price is $1,400,000 [source:ABS, page:2]."
    )
    out = ae.groundedness(text)
    assert out["n_numeric_claims"] >= 2
    assert out["groundedness"] == 1.0


def test_groundedness_uncited_numbers_drag_score():
    text = (
        "The cash rate is 4.10% [tool:rba_cash_rate, retrieved:2026-04-26]. "
        "Inflation is 3.2% (no source). House prices fell 5%."
    )
    out = ae.groundedness(text)
    # 4.10% is grounded; 3.2% and 5% aren't.
    assert out["n_numeric_claims"] == 3
    assert out["n_grounded"] == 1
    assert out["groundedness"] == pytest.approx(1 / 3)


def test_groundedness_window_is_local():
    """A citation 200 chars away shouldn't ground a number."""
    text = "The figure was 17.4% in regional NSW. " + ("filler. " * 30) + "[source:RBA, page:3]"
    out = ae.groundedness(text)
    assert out["n_grounded"] == 0


# ---------- evaluate_one + aggregate -------------------------------


def _record(**overrides):
    base = {
        "id": "agent-001",
        "query": "What is the current RBA cash rate?",
        "persona": "general",
        "query_type": "factual",
        "needs_decomposition": False,
        "expected_tools": ["rba_cash_rate"],
        "expected_doc_publishers": [],
        "expected_n_subquestions": 0,
        "rationale": "x",
    }
    base.update(overrides)
    return base


def _state(**overrides):
    base = {
        "retrieved_chunks": [],
        "tool_results": [
            {
                "tool": "rba_cash_rate",
                "args": {"period": "latest"},
                "result": {
                    "data": {"rate_pct": 4.10},
                    "source": "RBA F1.1",
                    "retrieved_at": "2026-04-26T00:00:00+00:00",
                },
            }
        ],
        "iteration_count": 1,
        "answer_draft": "The cash rate is 4.10% [tool:rba_cash_rate, retrieved:2026-04-26].",
    }
    base.update(overrides)
    return base


def test_evaluate_one_perfect_factual():
    out = ae.evaluate_one(_record(), _state())
    assert out["tool_call_accuracy"] == 1.0
    assert out["trajectory_efficiency"] == 1.0
    assert out["groundedness"] == 1.0
    assert out["publisher_recall"] == 1.0
    # No --with-judge → no faithfulness key.
    assert "faithfulness" not in out


def test_evaluate_one_penalises_extras():
    rec = _record(expected_tools=["rba_cash_rate"])
    state = _state(
        tool_results=[
            *_state()["tool_results"],
            {
                "tool": "abs_lending_indicators",
                "args": {},
                "result": {
                    "data": {},
                    "source": "ABS 5601",
                    "retrieved_at": "2026-04-26T00:00:00+00:00",
                },
            },
        ],
        iteration_count=2,
    )
    out = ae.evaluate_one(rec, state)
    assert "abs_lending_indicators" in out["extra_tools"]
    # 1 extra iteration + 1 extra tool → 1/3
    assert out["trajectory_efficiency"] == pytest.approx(1 / 3)


def test_aggregate_macro_average():
    rows = [
        {"tool_call_accuracy": 1.0, "tool_call_recall": 1.0, "tool_call_precision": 1.0,
         "publisher_recall": 1.0, "trajectory_efficiency": 1.0, "groundedness": 1.0},
        {"tool_call_accuracy": 0.5, "tool_call_recall": 0.5, "tool_call_precision": 0.5,
         "publisher_recall": 0.0, "trajectory_efficiency": 0.5, "groundedness": 0.0},
    ]
    agg = ae.aggregate(rows)
    assert agg["n"] == 2
    assert agg["mean_tool_call_accuracy"] == 0.75
    assert agg["mean_groundedness"] == 0.5
    # No faithfulness in the rows → don't surface it.
    assert "mean_faithfulness" not in agg


def test_aggregate_empty():
    assert ae.aggregate([]) == {"n": 0}


# ---------- run_agent_eval (graph stubbed) -------------------------


def test_run_agent_eval_stubbed_graph(monkeypatch, tmp_path: Path):
    """Stub build_graph so we can exercise run_agent_eval offline."""
    queries = tmp_path / "q.jsonl"
    queries.write_text(
        "\n".join(
            json.dumps(_record(id=f"agent-{i:03d}")) for i in range(1, 4)
        ),
        encoding="utf-8",
    )

    class _StubGraph:
        def invoke(self, state):
            # Return a minimal but plausible final state.
            return {
                "retrieved_chunks": [],
                "tool_results": [
                    {
                        "tool": "rba_cash_rate",
                        "args": {"period": "latest"},
                        "result": {
                            "data": {"rate_pct": 4.10},
                            "source": "RBA F1.1",
                            "retrieved_at": "2026-04-26T00:00:00+00:00",
                        },
                    }
                ],
                "iteration_count": 1,
                "answer_draft": (
                    "Cash rate is 4.10% [tool:rba_cash_rate, retrieved:2026-04-26]."
                ),
            }

    from src.agent import graph as graph_mod

    monkeypatch.setattr(graph_mod, "build_graph", lambda: _StubGraph())
    report = ae.run_agent_eval(queries)
    assert report["aggregate"]["n"] == 3
    assert report["aggregate"]["mean_tool_call_accuracy"] == 1.0
    assert report["aggregate"]["mean_groundedness"] == 1.0
    assert report["failures"] == []


def test_run_agent_eval_records_failures(monkeypatch, tmp_path: Path):
    queries = tmp_path / "q.jsonl"
    queries.write_text(json.dumps(_record()), encoding="utf-8")

    class _ExplodingGraph:
        def invoke(self, state):
            raise RuntimeError("graph blew up")

    from src.agent import graph as graph_mod

    monkeypatch.setattr(graph_mod, "build_graph", lambda: _ExplodingGraph())
    report = ae.run_agent_eval(queries)
    assert report["aggregate"]["n"] == 0
    assert len(report["failures"]) == 1
    assert "graph blew up" in report["failures"][0]["error"]


def test_run_agent_eval_respects_limit(monkeypatch, tmp_path: Path):
    queries = tmp_path / "q.jsonl"
    queries.write_text(
        "\n".join(
            json.dumps(_record(id=f"agent-{i:03d}")) for i in range(1, 11)
        ),
        encoding="utf-8",
    )

    class _Stub:
        def invoke(self, state):
            return _state()

    from src.agent import graph as graph_mod

    monkeypatch.setattr(graph_mod, "build_graph", lambda: _Stub())
    report = ae.run_agent_eval(queries, limit=4)
    assert report["aggregate"]["n"] == 4
