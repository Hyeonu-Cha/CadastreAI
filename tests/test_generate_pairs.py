"""Unit tests for `src.training.generate_pairs` (Task X.05).

Coverage:

1. **Pure response parsing** (`_parse_response`): the parser is shared
   across providers, so the same JSON shape must round-trip whether
   Claude or Gemini produced it. Includes the code-fenced variant
   (Claude habit) and the strict-JSON variant (Gemini's
   `response_mime_type='application/json'` output).
2. **Default-model resolution**: `--provider gemini` without `--model`
   should pick the Gemini default; same for Anthropic.
3. **Provider validation**: `_run_async` rejects unknown providers
   *before* attempting to import a vendor SDK, so misconfiguration
   surfaces with a clean error rather than an ImportError.
4. **Missing-key errors**: each provider raises a clear RuntimeError
   when its API key env var is unset (the script is run unattended in
   batch jobs, so the failure mode needs to be diagnosable).
"""
from __future__ import annotations

import asyncio

import pytest

from src.training import generate_pairs as gp

# ---------- _parse_response -------------------------------------------


def test_parse_response_strict_json():
    """Gemini with response_mime_type='application/json' returns clean JSON."""
    text = '{"queries": [{"query": "What is the cash rate?", "persona": "researcher"}]}'
    out = gp._parse_response(text)
    assert out == [{"query": "What is the cash rate?", "persona": "researcher"}]


def test_parse_response_with_code_fences():
    """Claude sometimes wraps JSON in ```json fences — parser tolerates that."""
    text = (
        "```json\n"
        '{"queries": [\n'
        '  {"query": "Best suburb for first home buyers in NSW?", "persona": "homebuyer"},\n'
        '  {"query": "Average gross yield in Brisbane?", "persona": "investor"}\n'
        "]}\n"
        "```"
    )
    out = gp._parse_response(text)
    assert len(out) == 2
    assert {q["persona"] for q in out} == {"homebuyer", "investor"}


def test_parse_response_drops_invalid_personas():
    """Bogus personas are silently dropped — keeps downstream filtering simple."""
    text = (
        '{"queries": ['
        '{"query": "Q1", "persona": "homebuyer"},'
        '{"query": "Q2", "persona": "policy_wonk"},'
        '{"query": "Q3", "persona": "investor"}'
        "]}"
    )
    out = gp._parse_response(text)
    assert [q["persona"] for q in out] == ["homebuyer", "investor"]


def test_parse_response_raises_on_no_json():
    with pytest.raises(ValueError):
        gp._parse_response("the model refused to respond")


def test_parse_response_raises_when_all_invalid():
    text = '{"queries": [{"query": "", "persona": "homebuyer"}]}'
    with pytest.raises(ValueError):
        gp._parse_response(text)


# ---------- default model resolution ----------------------------------


def test_default_model_for_anthropic():
    assert gp._default_model_for("anthropic") == gp.DEFAULT_MODEL_ANTHROPIC


def test_default_model_for_gemini():
    assert gp._default_model_for("gemini") == gp.DEFAULT_MODEL_GEMINI


def test_default_model_for_unknown_raises():
    with pytest.raises(ValueError):
        gp._default_model_for("openai")


def test_providers_constant_includes_both():
    assert "anthropic" in gp.PROVIDERS
    assert "gemini" in gp.PROVIDERS


# ---------- provider validation in _run_async -------------------------


def test_run_async_rejects_unknown_provider(tmp_path):
    out = tmp_path / "out.jsonl"
    with pytest.raises(ValueError, match="unknown provider"):
        asyncio.run(
            gp._run_async(
                [{"chunk_id": "c1", "text": "x"}],
                out,
                provider="openai",
                n=1,
                model="x",
                max_tokens=10,
                concurrency=1,
            )
        )


def test_run_async_anthropic_requires_key(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = tmp_path / "out.jsonl"
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        asyncio.run(
            gp._run_async(
                [{"chunk_id": "c1", "text": "x"}],
                out,
                provider="anthropic",
                n=1,
                model="claude-haiku-4-5",
                max_tokens=10,
                concurrency=1,
            )
        )


def test_run_async_gemini_requires_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    out = tmp_path / "out.jsonl"
    # Either RuntimeError (key missing — the case we want) or
    # ImportError (google-genai not installed in this env). Both are
    # acceptable signals that we never reached an actual API call;
    # we only assert the RuntimeError path when the SDK is present.
    try:
        from google import genai  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        pytest.skip("google-genai SDK not installed in this test env")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        asyncio.run(
            gp._run_async(
                [{"chunk_id": "c1", "text": "x"}],
                out,
                provider="gemini",
                n=1,
                model="gemini-2.5-flash",
                max_tokens=10,
                concurrency=1,
            )
        )


# ---------- back-compat: DEFAULT_MODEL still points at the Claude id --


def test_default_model_alias_unchanged():
    """External callers may import DEFAULT_MODEL — keep it pointing at Anthropic."""
    assert gp.DEFAULT_MODEL == gp.DEFAULT_MODEL_ANTHROPIC
