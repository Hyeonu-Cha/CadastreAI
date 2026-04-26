"""In-memory TTL cache for structured tool results (Task 4.10).

The agent re-calls tools with identical args several times across a
single session — the loop-back path in particular often re-pulls the
RBA cash rate or an ABS series after reflection. Each call is a real
HTTP fetch, costing ~200-800ms even when the data hasn't moved. This
module caches `_execute_tool` results in-process with a per-tool TTL,
so back-to-back calls hit memory.

Why in-memory, not disk
-----------------------
Tools have side-effect-free, idempotent semantics, so a per-process
cache is enough for the agent loop and the Streamlit session. A disk
cache would help eval reruns but introduces concurrency and freshness
concerns we don't need yet — this is opt-in for now via
`ToolCache.from_config()`. Tests construct caches directly so they
never accidentally share state.

Per-tool TTLs reflect how often the upstream data actually moves:

  * RBA cash rate / ABS releases — monthly cadence; 24h TTL is fine
  * SQM vacancy — weekly cadence; 24h TTL is fine
  * compute_* (pure math)        — deterministic, cache for a week

Tools not in the table fall back to `DEFAULT_TTL_SECONDS`.
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 60 * 60  # 1 hour

_TTL_BY_TOOL: dict[str, int] = {
    "rba_cash_rate": 60 * 60 * 24,        # daily refresh is plenty
    "abs_property_index": 60 * 60 * 24,
    "abs_lending_indicators": 60 * 60 * 24,
    "sqm_vacancy": 60 * 60 * 24,
    "compute_stamp_duty_nsw": 60 * 60 * 24 * 7,    # deterministic
    "compute_mortgage_repayment": 60 * 60 * 24 * 7,
    "compute_rental_yield": 60 * 60 * 24 * 7,
    "make_chart": 60 * 5,                  # short — chart artefact paths drift
}


def ttl_for(name: str) -> int:
    return _TTL_BY_TOOL.get(name, DEFAULT_TTL_SECONDS)


def _normalize_key(name: str, args: dict | None) -> str:
    """Stable JSON-serialised cache key.

    `sort_keys=True` makes `{"a":1,"b":2}` and `{"b":2,"a":1}` collapse
    to the same key. `default=str` is a safety net for the rare argument
    that isn't natively JSON-serialisable (e.g. a Decimal or date) — we
    don't want a planner-injected weirdness to crash cache lookups; a
    cache miss is always recoverable.
    """
    payload = json.dumps(args or {}, sort_keys=True, default=str)
    return f"{name}::{payload}"


class ToolCache:
    """Tiny TTL cache with hit/miss counters."""

    def __init__(self, *, enabled: bool = True, clock: Callable[[], float] = time.time) -> None:
        self.enabled = enabled
        self._clock = clock
        self._store: dict[str, tuple[float, Any]] = {}
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, name: str, args: dict | None) -> Any:
        """Return the cached result if fresh, otherwise None.

        Expired entries are evicted on access — we don't run a periodic
        sweeper because the agent loop is short and the cache rarely
        grows beyond a dozen entries per session."""
        if not self.enabled:
            return None
        key = _normalize_key(name, args)
        item = self._store.get(key)
        if item is None:
            self.misses += 1
            return None
        expires_at, value = item
        if self._clock() >= expires_at:
            del self._store[key]
            self.evictions += 1
            self.misses += 1
            return None
        self.hits += 1
        return value

    def set(self, name: str, args: dict | None, value: Any, *, ttl: int | None = None) -> None:
        if not self.enabled:
            return
        key = _normalize_key(name, args)
        ttl_eff = ttl if ttl is not None else ttl_for(name)
        self._store[key] = (self._clock() + ttl_eff, value)

    def wrap(
        self,
        name: str,
        args: dict | None,
        fn: Callable[[], Any],
        *,
        ttl: int | None = None,
        skip_on: Callable[[Any], bool] | None = None,
    ) -> Any:
        """Get-or-compute. `skip_on(result) -> True` means don't cache.

        We use `skip_on` to avoid caching tool failures: a transient
        upstream blip shouldn't be remembered for the next 24 hours."""
        cached = self.get(name, args)
        if cached is not None:
            return cached
        result = fn()
        if skip_on is None or not skip_on(result):
            self.set(name, args, result, ttl=ttl)
        return result

    def clear(self) -> None:
        self._store.clear()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def __len__(self) -> int:
        return len(self._store)


__all__ = [
    "DEFAULT_TTL_SECONDS",
    "ToolCache",
    "ttl_for",
]
