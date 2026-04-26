"""Static validation of `data/eval/agent_queries.jsonl` (Task 3.20).

The eval harness in Task 3.21 will treat this file as ground truth, so
the schema must be airtight before then. Each test pins one invariant
(parses, count, schema, persona/type vocab, tool names, decomp ↔
sub-question consistency) — keeping them separate makes a regression
trivial to localise.
"""
from __future__ import annotations

import json
from pathlib import Path

EVAL_PATH = Path("data/eval/agent_queries.jsonl")

REQUIRED_FIELDS = {
    "id",
    "query",
    "persona",
    "query_type",
    "needs_decomposition",
    "expected_tools",
    "expected_doc_publishers",
    "expected_n_subquestions",
    "rationale",
}

VALID_PERSONAS = {
    "first_home_buyer",
    "investor",
    "policy_researcher",
    "journalist",
    "general",
}

VALID_QUERY_TYPES = {"factual", "comparative", "computational", "exploratory"}

VALID_TOOLS = {
    "rba_cash_rate",
    "rba_mortgage_rates",
    "abs_property_price_index",
    "abs_building_approvals",
    "abs_lending_indicators",
    "sqm_rental_vacancy",
    "compute_rental_yield",
    "compute_mortgage_repayment",
    "compute_stamp_duty_nsw",
}

# Loose allowlist matching the publisher distribution in the indexed
# corpus (top-N from src.index). Anything outside this set in the
# expected_doc_publishers list is almost certainly a typo or a
# publisher we don't actually index.
VALID_PUBLISHERS = {
    "AHURI",
    "RBA",
    "Productivity Commission",
    "PropTrack",
    "Housing Australia (NHFIC)",
    "SQM Research",
    "APRA",
    "CoreLogic / Cotality",
    "Australian Treasury",
    "Grattan Institute",
}


def _records():
    return [json.loads(line) for line in EVAL_PATH.read_text(encoding="utf-8").splitlines()]


def test_file_exists_and_parses():
    assert EVAL_PATH.exists(), f"missing {EVAL_PATH}"
    rs = _records()
    assert rs, "agent_queries.jsonl is empty"


def test_has_30_queries():
    assert len(_records()) == 30


def test_ids_are_unique_and_well_formed():
    ids = [r["id"] for r in _records()]
    assert len(set(ids)) == len(ids), "duplicate ids"
    for i, rid in enumerate(ids, start=1):
        assert rid == f"agent-{i:03d}", f"id sequence broken at {rid}"


def test_required_fields_present():
    for r in _records():
        missing = REQUIRED_FIELDS - set(r)
        assert not missing, f"{r['id']}: missing fields {missing}"


def test_persona_and_query_type_vocab():
    for r in _records():
        assert r["persona"] in VALID_PERSONAS, f"{r['id']}: bad persona {r['persona']}"
        assert r["query_type"] in VALID_QUERY_TYPES, (
            f"{r['id']}: bad query_type {r['query_type']}"
        )


def test_expected_tools_are_real():
    for r in _records():
        for t in r["expected_tools"]:
            assert t in VALID_TOOLS, f"{r['id']}: unknown tool {t!r}"


def test_expected_publishers_are_in_corpus():
    for r in _records():
        for p in r["expected_doc_publishers"]:
            assert p in VALID_PUBLISHERS, f"{r['id']}: publisher not in corpus: {p!r}"


def test_decomposition_consistency():
    """needs_decomposition=True ↔ expected_n_subquestions in [2, 4]."""
    for r in _records():
        n = r["expected_n_subquestions"]
        if r["needs_decomposition"]:
            assert 2 <= n <= 4, f"{r['id']}: decomp=True needs 2–4 sub-qs, got {n}"
        else:
            assert n == 0, f"{r['id']}: decomp=False needs 0 sub-qs, got {n}"


def test_at_least_one_evidence_source_per_query():
    """Every query must call at least one tool OR retrieve at least one doc."""
    for r in _records():
        has_evidence = bool(r["expected_tools"]) or bool(r["expected_doc_publishers"])
        assert has_evidence, f"{r['id']}: no expected tools or publishers"


def test_persona_diversity():
    personas = {r["persona"] for r in _records()}
    # All five personas should appear at least once.
    assert personas == VALID_PERSONAS, f"missing personas: {VALID_PERSONAS - personas}"


def test_tool_coverage():
    """All 9 tools should appear at least once across the eval set."""
    seen = {t for r in _records() for t in r["expected_tools"]}
    missing = VALID_TOOLS - seen
    assert not missing, f"tools never exercised in eval set: {missing}"
