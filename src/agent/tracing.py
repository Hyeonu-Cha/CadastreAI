"""LangSmith tracing helpers for the CadastreAI agent (Task 3.16).

LangChain's tracer auto-attaches to LangGraph runs when the standard
env vars are set, so this module is intentionally thin: a single
`enable_tracing()` entry point that

  - validates `LANGSMITH_API_KEY` is present,
  - sets `LANGSMITH_TRACING=true` and a default project name if the
    caller didn't, and
  - returns whether tracing actually became active.

A `traced_run(name, **metadata)` context manager wraps an otherwise
opaque `graph.invoke(...)` call in a named LangSmith run so the trace
shows the question (rather than just "RunnableSequence"). When tracing
is off, it's a no-op — callers don't need to branch.

Env vars (all optional):
    LANGSMITH_API_KEY    — required for tracing to actually upload runs
    LANGSMITH_TRACING    — defaults to "true" once enable_tracing() runs
    LANGSMITH_PROJECT    — default "cadastreai"
    LANGSMITH_ENDPOINT   — default "https://api.smith.langchain.com"
"""
from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Iterator

log = logging.getLogger(__name__)

DEFAULT_PROJECT = "cadastreai"
DEFAULT_ENDPOINT = "https://api.smith.langchain.com"


def enable_tracing(
    *,
    project: str | None = None,
    endpoint: str | None = None,
) -> bool:
    """Turn LangSmith tracing on if the API key is available.

    Idempotent — safe to call multiple times. Returns True when tracing
    is now active, False otherwise (with a warning logged so callers
    don't silently miss traces).
    """
    if not os.environ.get("LANGSMITH_API_KEY"):
        log.warning(
            "LANGSMITH_API_KEY not set — tracing is disabled. "
            "Set the key (and optionally LANGSMITH_PROJECT) to enable."
        )
        return False

    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", project or DEFAULT_PROJECT)
    os.environ.setdefault("LANGSMITH_ENDPOINT", endpoint or DEFAULT_ENDPOINT)
    log.info(
        "LangSmith tracing enabled (project=%s)",
        os.environ["LANGSMITH_PROJECT"],
    )
    return True


def is_tracing_enabled() -> bool:
    """Quick check used by callers that want to branch on tracing state."""
    return bool(os.environ.get("LANGSMITH_API_KEY")) and os.environ.get(
        "LANGSMITH_TRACING", ""
    ).lower() in {"1", "true", "yes"}


@contextlib.contextmanager
def traced_run(name: str, **metadata) -> Iterator[None]:
    """Wrap a block in a named LangSmith run when tracing is active.

    No-op when tracing is off, so callers can write::

        with traced_run("agent.invoke", query=q, idx=i):
            out = graph.invoke(state)

    without env-checking themselves. Failures in the tracer never
    propagate — a broken trace shouldn't break agent runs.
    """
    if not is_tracing_enabled():
        yield
        return

    try:
        from langsmith import trace as ls_trace
    except ImportError:  # pragma: no cover — agent extra missing
        log.warning("langsmith not importable; skipping trace for %s", name)
        yield
        return

    try:
        with ls_trace(name=name, run_type="chain", inputs=metadata or None):
            yield
    except Exception as e:  # noqa: BLE001 — never let tracing break the run
        log.warning("trace %s failed (%s: %s)", name, type(e).__name__, e)
        yield


__all__ = [
    "DEFAULT_ENDPOINT",
    "DEFAULT_PROJECT",
    "enable_tracing",
    "is_tracing_enabled",
    "traced_run",
]
