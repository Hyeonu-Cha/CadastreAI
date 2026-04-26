"""Unit tests for the in-memory tool-result cache (Task 4.10).

Coverage:
  - get/set round-trips a value with its TTL.
  - get returns None on miss.
  - wrap() = cache miss → call fn → cache → return; second call hits.
  - different args produce distinct cache entries (key normalises but
    doesn't collapse).
  - sorted-key normalisation: {"a":1,"b":2} == {"b":2,"a":1}.
  - expiry: an entry past its TTL is evicted on access.
  - skip_on: a result matching the predicate is NOT cached (errors).
  - disabled cache always misses and never stores.
  - per-tool TTL table — known tools use their bespoke TTL, unknown
    falls back to DEFAULT_TTL_SECONDS.
"""
from __future__ import annotations

from src.agent.tool_cache import (
    DEFAULT_TTL_SECONDS,
    ToolCache,
    ttl_for,
)


class FakeClock:
    """Mutable monotonic clock for deterministic TTL tests."""

    def __init__(self, t0: float = 1000.0):
        self.now = t0

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


# ---------- ttl_for ----------------------------------------------


def test_ttl_for_known_tool_returns_bespoke_value():
    assert ttl_for("rba_cash_rate") == 60 * 60 * 24


def test_ttl_for_unknown_tool_returns_default():
    assert ttl_for("nonexistent_tool") == DEFAULT_TTL_SECONDS


# ---------- get / set ------------------------------------------


def test_set_then_get_roundtrips():
    c = ToolCache()
    c.set("rba_cash_rate", {}, {"data": {"rate_pct": 4.10}}, ttl=60)
    assert c.get("rba_cash_rate", {}) == {"data": {"rate_pct": 4.10}}
    assert c.hits == 1


def test_get_miss_increments_misses_and_returns_none():
    c = ToolCache()
    assert c.get("anything", {}) is None
    assert c.misses == 1
    assert c.hits == 0


def test_keys_normalise_across_arg_order():
    c = ToolCache()
    c.set("compute_stamp_duty_nsw", {"a": 1, "b": 2}, "result")
    assert c.get("compute_stamp_duty_nsw", {"b": 2, "a": 1}) == "result"


def test_different_args_get_distinct_entries():
    c = ToolCache()
    c.set("compute_stamp_duty_nsw", {"price": 700_000}, "low")
    c.set("compute_stamp_duty_nsw", {"price": 900_000}, "high")
    assert c.get("compute_stamp_duty_nsw", {"price": 700_000}) == "low"
    assert c.get("compute_stamp_duty_nsw", {"price": 900_000}) == "high"


def test_same_args_different_tools_dont_collide():
    c = ToolCache()
    c.set("tool_a", {"x": 1}, "A")
    c.set("tool_b", {"x": 1}, "B")
    assert c.get("tool_a", {"x": 1}) == "A"
    assert c.get("tool_b", {"x": 1}) == "B"


# ---------- TTL / expiry ---------------------------------------


def test_entry_expires_after_ttl():
    clock = FakeClock(1000.0)
    c = ToolCache(clock=clock)
    c.set("rba_cash_rate", {}, "fresh", ttl=60)
    assert c.get("rba_cash_rate", {}) == "fresh"

    clock.advance(61)
    assert c.get("rba_cash_rate", {}) is None
    assert c.evictions == 1


def test_entry_at_expiry_boundary_is_treated_as_expired():
    clock = FakeClock(1000.0)
    c = ToolCache(clock=clock)
    c.set("x", {}, "v", ttl=10)
    clock.advance(10)  # at the boundary, expires_at == now
    assert c.get("x", {}) is None


def test_entry_just_before_expiry_still_hits():
    clock = FakeClock(1000.0)
    c = ToolCache(clock=clock)
    c.set("x", {}, "v", ttl=10)
    clock.advance(9.9)
    assert c.get("x", {}) == "v"


def test_default_ttl_is_used_when_not_specified():
    """If we don't pass ttl, use ttl_for(name) — RBA = 86400s."""
    clock = FakeClock(1000.0)
    c = ToolCache(clock=clock)
    c.set("rba_cash_rate", {}, "v")
    clock.advance(60 * 60)  # 1h — well within the 24h tool TTL
    assert c.get("rba_cash_rate", {}) == "v"


# ---------- wrap() ---------------------------------------------


def test_wrap_misses_then_hits():
    c = ToolCache()
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return "computed"

    assert c.wrap("t", {}, fn) == "computed"
    assert calls["n"] == 1
    assert c.misses == 1

    assert c.wrap("t", {}, fn) == "computed"
    assert calls["n"] == 1, "second wrap should be a cache hit"
    assert c.hits == 1


def test_wrap_does_not_cache_when_skip_on_returns_true():
    """Errored results shouldn't be remembered."""
    c = ToolCache()
    calls = {"n": 0}

    def failing_fn():
        calls["n"] += 1
        return {"error": "transient blip"}

    is_error = lambda r: isinstance(r, dict) and "error" in r  # noqa: E731

    c.wrap("rba_cash_rate", {}, failing_fn, skip_on=is_error)
    c.wrap("rba_cash_rate", {}, failing_fn, skip_on=is_error)
    assert calls["n"] == 2, "errored result must NOT be cached"
    assert c.hits == 0


def test_wrap_does_cache_when_skip_on_returns_false():
    c = ToolCache()
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return {"data": {"ok": True}}

    is_error = lambda r: isinstance(r, dict) and "error" in r  # noqa: E731
    c.wrap("rba_cash_rate", {}, fn, skip_on=is_error)
    c.wrap("rba_cash_rate", {}, fn, skip_on=is_error)
    assert calls["n"] == 1


def test_wrap_uses_per_tool_ttl_by_default():
    """A tool from the table should get its bespoke TTL when ttl is unset."""
    clock = FakeClock(1000.0)
    c = ToolCache(clock=clock)
    c.wrap("compute_stamp_duty_nsw", {}, lambda: "v")
    clock.advance(60 * 60 * 24 * 6)  # 6 days — under the 7-day TTL
    assert c.get("compute_stamp_duty_nsw", {}) == "v"
    clock.advance(60 * 60 * 24 * 2)  # past 7 days now
    assert c.get("compute_stamp_duty_nsw", {}) is None


# ---------- disabled mode ---------------------------------------


def test_disabled_cache_misses_and_never_stores():
    c = ToolCache(enabled=False)
    c.set("x", {}, "v")
    assert c.get("x", {}) is None
    assert len(c) == 0


def test_disabled_wrap_always_calls_fn():
    c = ToolCache(enabled=False)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return "v"

    c.wrap("x", {}, fn)
    c.wrap("x", {}, fn)
    assert calls["n"] == 2


# ---------- clear / len ---------------------------------------


def test_clear_resets_entries_and_counters():
    c = ToolCache()
    c.set("x", {}, "v")
    c.get("x", {})  # hit
    c.get("y", {})  # miss
    assert len(c) == 1
    assert c.hits == 1
    assert c.misses == 1

    c.clear()
    assert len(c) == 0
    assert c.hits == 0
    assert c.misses == 0


def test_non_serializable_args_dont_crash():
    """A weird arg type (e.g. a date) shouldn't crash — default=str
    saves us. Cache still works for the same-typed arg."""
    import datetime as dt

    c = ToolCache()
    d = dt.date(2026, 4, 26)
    c.set("x", {"day": d}, "v")
    assert c.get("x", {"day": d}) == "v"
