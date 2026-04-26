"""Tests for the LangSmith tracing helpers (Task 3.16).

Pure offline — we exercise the env-var contract and the no-op behaviour
of `traced_run`, but never actually upload runs to LangSmith.
"""
from __future__ import annotations

from src.agent import tracing


def test_enable_tracing_returns_false_without_api_key(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    assert tracing.enable_tracing() is False
    # Should NOT have set LANGSMITH_TRACING — that would falsely advertise tracing.
    import os

    assert "LANGSMITH_TRACING" not in os.environ


def test_enable_tracing_sets_defaults_when_key_present(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "fake-key")
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)

    assert tracing.enable_tracing() is True
    import os

    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_PROJECT"] == tracing.DEFAULT_PROJECT
    assert os.environ["LANGSMITH_ENDPOINT"] == tracing.DEFAULT_ENDPOINT


def test_enable_tracing_respects_explicit_project(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "fake-key")
    monkeypatch.setenv("LANGSMITH_PROJECT", "custom-project")
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)

    assert tracing.enable_tracing(project="ignored") is True
    import os

    # Existing env var wins (setdefault), so users can override per-shell.
    assert os.environ["LANGSMITH_PROJECT"] == "custom-project"


def test_is_tracing_enabled_requires_both_key_and_flag(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    assert tracing.is_tracing_enabled() is False

    monkeypatch.setenv("LANGSMITH_API_KEY", "fake-key")
    assert tracing.is_tracing_enabled() is False  # no flag yet

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    assert tracing.is_tracing_enabled() is True

    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    assert tracing.is_tracing_enabled() is False


def test_traced_run_is_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    # Should yield without raising and without trying to import langsmith.trace.
    with tracing.traced_run("agent.test", query="anything"):
        result = 42
    assert result == 42


def test_traced_run_swallows_tracer_failures(monkeypatch):
    """If the LangSmith client errors, the wrapped block must still run."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "fake-key")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")

    # Force an import-time failure path: monkey-patch the langsmith.trace
    # symbol to a function that raises when called.
    import langsmith

    def boom(*_a, **_k):
        raise RuntimeError("tracer offline")

    monkeypatch.setattr(langsmith, "trace", boom)
    ran = []
    with tracing.traced_run("agent.test", query="anything"):
        ran.append(True)
    assert ran == [True]


def test_validation_query_set_size():
    """The spec calls for 10 diverse queries."""
    from src.agent.validate import VALIDATION_QUERIES

    assert len(VALIDATION_QUERIES) == 10
    # Diversity check — surface area of personas/topics, loosely.
    text = " ".join(VALIDATION_QUERIES).lower()
    for keyword in ("first home", "investor", "compare", "negative gearing"):
        assert keyword in text, f"missing diversity keyword: {keyword}"
