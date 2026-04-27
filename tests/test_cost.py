"""Unit tests for the per-query cost accountant (Task 4.12).

Coverage:
  - Pricing maths for input/output/cache_read/cache_write across models.
  - Unknown model returns a $0 breakdown instead of crashing.
  - Usage object can be a dict (tests) or an object with attributes (SDK).
  - Missing cache fields default to 0 (older response shapes).
  - CostTracker accumulates totals and produces a stable summary dict.
  - log_cost emits a structured log line we can match on.
"""
from __future__ import annotations

import logging

from src.agent.cost import (
    CACHE_READ_MULTIPLIER,
    CACHE_WRITE_MULTIPLIER,
    CostTracker,
    calculate_cost,
    log_cost,
)


class FakeUsage:
    """Mimics the Anthropic SDK's Usage object — attribute access only."""

    def __init__(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_input_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
    ):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read_input_tokens
        self.cache_creation_input_tokens = cache_creation_input_tokens


# ---------- pricing maths ---------------------------------------


def test_haiku_pure_input_output_no_cache():
    """1M in + 1M out @ $1.00/$5.00 = $6.00 exactly."""
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
    b = calculate_cost(usage, "claude-haiku-4-5")
    assert b.input_usd == 1.0
    assert b.output_usd == 5.0
    assert b.cache_read_usd == 0.0
    assert b.cache_write_usd == 0.0
    assert b.total_usd == 6.0


def test_sonnet_pricing():
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
    b = calculate_cost(usage, "claude-sonnet-4-6")
    assert b.input_usd == 3.0
    assert b.output_usd == 15.0


def test_opus_pricing():
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
    b = calculate_cost(usage, "claude-opus-4-7")
    assert b.input_usd == 5.0
    assert b.output_usd == 25.0


def test_cache_read_uses_0_1x_multiplier():
    """1M cache-read tokens on Haiku = 1M / 1M * $1 * 0.1 = $0.10."""
    usage = {"cache_read_input_tokens": 1_000_000}
    b = calculate_cost(usage, "claude-haiku-4-5")
    assert b.cache_read_usd == 0.10
    assert b.input_usd == 0.0
    assert CACHE_READ_MULTIPLIER == 0.10


def test_cache_write_uses_1_25x_multiplier():
    """1M cache-write tokens on Haiku = 1M / 1M * $1 * 1.25 = $1.25."""
    usage = {"cache_creation_input_tokens": 1_000_000}
    b = calculate_cost(usage, "claude-haiku-4-5")
    assert b.cache_write_usd == 1.25
    assert CACHE_WRITE_MULTIPLIER == 1.25


def test_total_sums_all_four_meters():
    usage = {
        "input_tokens": 1000,
        "output_tokens": 500,
        "cache_read_input_tokens": 2000,
        "cache_creation_input_tokens": 800,
    }
    b = calculate_cost(usage, "claude-haiku-4-5")
    expected = (
        (1000 / 1e6) * 1.0
        + (500 / 1e6) * 5.0
        + (2000 / 1e6) * 1.0 * 0.1
        + (800 / 1e6) * 1.0 * 1.25
    )
    assert abs(b.total_usd - expected) < 1e-12


# ---------- unknown / odd shapes -------------------------------


def test_unknown_model_returns_zero_cost_not_crash(caplog):
    with caplog.at_level(logging.WARNING):
        b = calculate_cost({"input_tokens": 100, "output_tokens": 50}, "made-up-model")
    assert b.total_usd == 0.0
    assert b.input_tokens == 100  # tokens still recorded
    assert any("no pricing" in r.message for r in caplog.records)


def test_usage_can_be_object_or_dict():
    obj = FakeUsage(input_tokens=100, output_tokens=200)
    d = {"input_tokens": 100, "output_tokens": 200}
    b1 = calculate_cost(obj, "claude-haiku-4-5")
    b2 = calculate_cost(d, "claude-haiku-4-5")
    assert b1.total_usd == b2.total_usd


def test_missing_cache_fields_default_to_zero():
    usage = {"input_tokens": 100, "output_tokens": 50}  # no cache fields
    b = calculate_cost(usage, "claude-haiku-4-5")
    assert b.cache_read_tokens == 0
    assert b.cache_write_tokens == 0


def test_none_usage_yields_zero_breakdown():
    b = calculate_cost(None, "claude-haiku-4-5")
    assert b.input_tokens == 0
    assert b.total_usd == 0.0


# ---------- CostTracker ---------------------------------------


def test_tracker_records_each_call():
    t = CostTracker()
    t.record("classify", FakeUsage(input_tokens=100, output_tokens=50), "claude-haiku-4-5")
    t.record("synth", FakeUsage(input_tokens=200, output_tokens=150), "claude-sonnet-4-6")
    s = t.summary()
    assert s["n_calls"] == 2
    assert s["input_tokens"] == 300
    assert s["output_tokens"] == 200
    assert len(s["by_node"]) == 2
    assert s["by_node"][0]["node"] == "classify"
    assert s["by_node"][1]["node"] == "synth"


def test_tracker_accumulates_total_usd():
    t = CostTracker()
    t.record("a", FakeUsage(input_tokens=1_000_000), "claude-haiku-4-5")  # $1.00
    t.record("b", FakeUsage(output_tokens=1_000_000), "claude-haiku-4-5")  # $5.00
    s = t.summary()
    assert s["total_usd"] == 6.0


def test_tracker_summary_round_trips_cache_tokens():
    t = CostTracker()
    t.record(
        "classify",
        FakeUsage(cache_read_input_tokens=500, cache_creation_input_tokens=2000),
        "claude-haiku-4-5",
    )
    s = t.summary()
    assert s["cache_read_tokens"] == 500
    assert s["cache_write_tokens"] == 2000


# ---------- log_cost -----------------------------------------


def test_log_cost_emits_structured_line(caplog):
    with caplog.at_level(logging.INFO, logger="src.agent.cost"):
        log_cost(
            FakeUsage(input_tokens=100, output_tokens=50),
            "claude-haiku-4-5",
            "classify",
        )
    matches = [r for r in caplog.records if "node=classify" in r.message]
    assert matches, "expected a structured cost log line"
    msg = matches[0].message
    assert "model=claude-haiku-4-5" in msg
    assert "in=100" in msg
    assert "out=50" in msg


def test_breakdown_as_dict_rounds_for_serialization():
    """Floats are rounded to 6dp so JSON output is compact and stable."""
    b = calculate_cost(
        {"input_tokens": 1, "output_tokens": 1},
        "claude-haiku-4-5",
    )
    d = b.as_dict()
    assert isinstance(d["total_usd"], float)
    # 1 token / 1M * $1 = 0.000001 — exactly representable at 6dp.
    assert d["input_usd"] == 0.000001
