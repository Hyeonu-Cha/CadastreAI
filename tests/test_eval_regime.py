"""Regime-quarantine tests for the eval harnesses (Task 5.01/5.02).

The 2026 tax reform (see docs/regime_change_gap_analysis.md, F-3) means
tax-dependent gold labels now reward repealed law. Queries tagged
`regime: pre_2026_reform` must be excluded from the *headline* metrics
while still being scored and reported separately. These tests pin:

  - the regime primitives (default = neutral; only pre-reform is off-headline);
  - retrieval_eval.run() splits headline vs legacy (stub retriever);
  - agent_eval.run_agent_eval() splits headline vs legacy (stub graph);
  - evaluate_one() stamps each row with its regime.

All offline — the retriever and graph are stubbed.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.eval import agent_eval as ae
from src.eval import retrieval_eval
from src.eval.regime import NEUTRAL_REGIME, is_headline, regime_of

# ---------- primitives ---------------------------------------------


def test_regime_of_absent_field_is_neutral():
    assert regime_of({"query": "q"}) == NEUTRAL_REGIME
    assert regime_of({"query": "q", "regime": ""}) == NEUTRAL_REGIME


def test_regime_of_reads_field():
    assert regime_of({"regime": "pre_2026_reform"}) == "pre_2026_reform"


def test_is_headline_excludes_only_pre_reform():
    assert is_headline("regime_neutral") is True
    assert is_headline("post_2026_reform") is True  # current law counts
    assert is_headline("pre_2026_reform") is False


# ---------- retrieval_eval split -----------------------------------


class _StubRetriever:
    """Returns one fixed gold hit for every query."""

    def retrieve(self, query, k):  # noqa: ANN001
        return [({"chunk_id": "g1"}, 1.0)]


def test_retrieval_run_quarantines_legacy(tmp_path: Path):
    q = tmp_path / "q.jsonl"
    q.write_text(
        "\n".join(
            [
                json.dumps({"query": "q1", "persona": "investor", "gold_chunk_ids": ["g1"]}),
                json.dumps(
                    {
                        "query": "q2",
                        "persona": "investor",
                        "regime": "pre_2026_reform",
                        "gold_chunk_ids": ["g1"],
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    out = retrieval_eval.run(q, tmp_path / "out.json", top_k=10, retriever=_StubRetriever())

    assert out["overall"]["n"] == 1, "headline excludes the pre-reform query"
    assert out["legacy_regime"]["n"] == 1
    assert out["overall_including_legacy"]["n"] == 2
    assert set(out["by_regime"]) == {"regime_neutral", "pre_2026_reform"}
    # Headline recall is a clean 1.0 (only the neutral query, which hits).
    assert out["overall"]["recall@5"] == 1.0
    assert all("regime" in pq for pq in out["per_query"])


# ---------- agent_eval split ---------------------------------------


def _agent_record(**overrides):
    base = {
        "id": "a1",
        "query": "q",
        "persona": "general",
        "expected_tools": [],
        "expected_doc_publishers": [],
    }
    base.update(overrides)
    return base


def test_evaluate_one_stamps_regime():
    out = ae.evaluate_one(
        _agent_record(regime="pre_2026_reform"),
        {"retrieved_chunks": [], "tool_results": [], "iteration_count": 1, "answer_draft": "x"},
    )
    assert out["regime"] == "pre_2026_reform"


def test_agent_run_quarantines_legacy(monkeypatch, tmp_path: Path):
    q = tmp_path / "q.jsonl"
    q.write_text(
        "\n".join(
            [
                json.dumps(_agent_record(id="a1")),
                json.dumps(_agent_record(id="a2", regime="pre_2026_reform", persona="investor")),
            ]
        ),
        encoding="utf-8",
    )

    class _StubGraph:
        def invoke(self, state):  # noqa: ANN001
            return {
                "retrieved_chunks": [],
                "tool_results": [],
                "iteration_count": 1,
                "answer_draft": "A qualitative answer with no numbers.",
            }

    from src.agent import graph as graph_mod

    monkeypatch.setattr(graph_mod, "build_graph", lambda: _StubGraph())
    report = ae.run_agent_eval(q)

    assert report["aggregate"]["n"] == 1, "headline excludes the pre-reform query"
    assert report["legacy_regime"]["n"] == 1
    assert report["aggregate_including_legacy"]["n"] == 2
    assert set(report["by_regime"]) == {"regime_neutral", "pre_2026_reform"}
