"""Unit tests for the agent's provider abstraction (`src.agent.llm`).

The Anthropic branch is already exercised end-to-end by the node tests
(test_classify_query etc.) which monkey-patch ``anthropic.Anthropic``.
This module covers:

  * ``get_provider`` env handling (default, casing, unknown)
  * Tool-schema translation between Anthropic and OpenAI shapes
  * Usage normalisation (both shapes → cost.py-compatible dict)
  * ``call_with_tool`` / ``call_text`` provider dispatch — the OpenAI
    branch is stubbed via ``sys.modules`` so no real ``openai`` package
    is required and no network calls happen

The provider env var is reset per test via the ``_reset_provider`` fixture
to avoid cross-test contamination."""
from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from src.agent import llm


@pytest.fixture(autouse=True)
def _reset_provider(monkeypatch):
    """Each test starts on the default provider unless it sets one."""
    monkeypatch.delenv(llm.PROVIDER_ENV, raising=False)
    yield


# ---------- get_provider --------------------------------------------


def test_get_provider_default():
    assert llm.get_provider() == "anthropic"


def test_get_provider_explicit_openai(monkeypatch):
    monkeypatch.setenv(llm.PROVIDER_ENV, "openai")
    assert llm.get_provider() == "openai"


def test_get_provider_lowercases_and_strips(monkeypatch):
    monkeypatch.setenv(llm.PROVIDER_ENV, "  OpenAI  ")
    assert llm.get_provider() == "openai"


def test_unknown_provider_raises_in_call(monkeypatch):
    monkeypatch.setenv(llm.PROVIDER_ENV, "gemini")
    with pytest.raises(ValueError, match="unknown CADASTRE_LLM_PROVIDER"):
        llm.call_text(system="s", user="u", max_tokens=10, model="x")


# ---------- tool payload translation --------------------------------


_TOOL_DEF = {
    "name": "submit_x",
    "description": "describe submit_x",
    "input_schema": {
        "type": "object",
        "properties": {"foo": {"type": "string"}},
        "required": ["foo"],
    },
}


def test_anthropic_tool_payload_passthrough():
    tools, choice = llm._anthropic_tool_payload(_TOOL_DEF, "submit_x")
    assert tools == [_TOOL_DEF]
    assert choice == {"type": "tool", "name": "submit_x"}


def test_openai_tool_payload_translation():
    tools, choice = llm._openai_tool_payload(_TOOL_DEF, "submit_x")
    assert tools == [
        {
            "type": "function",
            "function": {
                "name": "submit_x",
                "description": "describe submit_x",
                "parameters": _TOOL_DEF["input_schema"],
            },
        }
    ]
    assert choice == {"type": "function", "function": {"name": "submit_x"}}


# ---------- usage normalisation -------------------------------------


def test_normalize_anthropic_usage_full():
    usage = SimpleNamespace(
        input_tokens=100,
        output_tokens=50,
        cache_read_input_tokens=200,
        cache_creation_input_tokens=10,
    )
    norm = llm._normalize_anthropic_usage(usage)
    assert norm == {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_input_tokens": 200,
        "cache_creation_input_tokens": 10,
    }


def test_normalize_anthropic_usage_none():
    assert llm._normalize_anthropic_usage(None) == {}


def test_normalize_openai_usage_splits_cache():
    """`prompt_tokens=300, cached=200` → input=100, cache_read=200."""
    usage = SimpleNamespace(
        prompt_tokens=300,
        completion_tokens=50,
        prompt_tokens_details=SimpleNamespace(cached_tokens=200),
    )
    norm = llm._normalize_openai_usage(usage)
    assert norm == {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_input_tokens": 200,
        "cache_creation_input_tokens": 0,
    }


def test_normalize_openai_usage_no_cache_details():
    """No prompt_tokens_details → all prompt is uncached input."""
    usage = SimpleNamespace(prompt_tokens=120, completion_tokens=30)
    norm = llm._normalize_openai_usage(usage)
    assert norm["input_tokens"] == 120
    assert norm["cache_read_input_tokens"] == 0


def test_normalize_openai_usage_none():
    assert llm._normalize_openai_usage(None) == {}


# ---------- OpenAI branch (stubbed via sys.modules) -----------------


def _install_fake_openai(monkeypatch, response):
    """Inject a fake `openai` module into sys.modules.

    `response` is the object the fake `chat.completions.create()` call
    returns — usually a SimpleNamespace shaped like the OpenAI SDK
    response (choices[0].message + usage). The fake records its kwargs
    on the returned ``calls`` list so tests can assert on them."""
    calls: list[dict] = []

    class _Completions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return response

    class _Chat:
        completions = _Completions()

    class _OpenAI:
        def __init__(self, **_kw):
            self.chat = _Chat()

    fake = ModuleType("openai")
    fake.OpenAI = _OpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake)
    return calls


def test_call_with_tool_openai_branch(monkeypatch):
    monkeypatch.setenv(llm.PROVIDER_ENV, "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "fake")

    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                arguments=json.dumps({"foo": "bar"})
                            )
                        )
                    ]
                )
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=4,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        ),
    )
    calls = _install_fake_openai(monkeypatch, fake_response)

    result, usage = llm.call_with_tool(
        system="sys",
        user="u",
        tool_def=_TOOL_DEF,
        tool_name="submit_x",
        max_tokens=128,
        model="gpt-4o-mini",
    )
    assert result == {"foo": "bar"}
    assert usage["input_tokens"] == 10
    assert usage["output_tokens"] == 4
    assert calls and calls[0]["model"] == "gpt-4o-mini"
    # Tool format must be translated to the OpenAI shape.
    assert calls[0]["tools"][0]["type"] == "function"
    assert calls[0]["tool_choice"]["function"]["name"] == "submit_x"


def test_call_with_tool_openai_no_tool_call(monkeypatch):
    """Empty tool_calls → returns (None, usage) — caller decides what to do."""
    monkeypatch.setenv(llm.PROVIDER_ENV, "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "fake")

    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[]))],
        usage=None,
    )
    _install_fake_openai(monkeypatch, fake_response)

    result, usage = llm.call_with_tool(
        system="sys",
        user="u",
        tool_def=_TOOL_DEF,
        tool_name="submit_x",
        max_tokens=128,
        model="gpt-4o-mini",
    )
    assert result is None
    assert usage == {}


def test_call_with_tool_openai_invalid_json(monkeypatch):
    """Invalid JSON in arguments → returns (None, usage); doesn't crash."""
    monkeypatch.setenv(llm.PROVIDER_ENV, "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "fake")

    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(arguments="not json {")
                        )
                    ]
                )
            )
        ],
        usage=None,
    )
    _install_fake_openai(monkeypatch, fake_response)

    result, _ = llm.call_with_tool(
        system="sys",
        user="u",
        tool_def=_TOOL_DEF,
        tool_name="submit_x",
        max_tokens=128,
        model="gpt-4o-mini",
    )
    assert result is None


def test_call_text_openai_branch(monkeypatch):
    monkeypatch.setenv(llm.PROVIDER_ENV, "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "fake")

    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content="  hello world  "))
        ],
        usage=SimpleNamespace(
            prompt_tokens=20,
            completion_tokens=8,
            prompt_tokens_details=SimpleNamespace(cached_tokens=15),
        ),
    )
    _install_fake_openai(monkeypatch, fake_response)

    text, usage = llm.call_text(
        system="sys",
        user="u",
        max_tokens=64,
        model="gpt-4o",
    )
    assert text == "hello world"
    # 20 prompt - 15 cached = 5 uncached input
    assert usage["input_tokens"] == 5
    assert usage["cache_read_input_tokens"] == 15


def test_openai_branch_raises_without_api_key(monkeypatch):
    monkeypatch.setenv(llm.PROVIDER_ENV, "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _install_fake_openai(monkeypatch, SimpleNamespace())  # never reached
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        llm.call_text(system="s", user="u", max_tokens=10, model="gpt-4o")


# ---------- Anthropic branch dispatch -------------------------------


def test_call_with_tool_anthropic_branch(monkeypatch):
    """Default provider routes to anthropic and unwraps tool_use.input."""
    import anthropic

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")

    captured: list[dict] = []

    class _M:
        def create(self, **kw):
            captured.append(kw)
            return SimpleNamespace(
                content=[
                    SimpleNamespace(type="tool_use", input={"foo": "bar"})
                ],
                usage=SimpleNamespace(
                    input_tokens=5,
                    output_tokens=3,
                    cache_read_input_tokens=0,
                    cache_creation_input_tokens=0,
                ),
            )

    monkeypatch.setattr(
        anthropic, "Anthropic", lambda **_kw: SimpleNamespace(messages=_M())
    )

    result, usage = llm.call_with_tool(
        system="sys",
        user="u",
        tool_def=_TOOL_DEF,
        tool_name="submit_x",
        max_tokens=128,
        model="claude-haiku-4-5",
    )
    assert result == {"foo": "bar"}
    assert usage["input_tokens"] == 5
    # Anthropic-shaped system block + cache_control survives the wrapper.
    assert captured[0]["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert captured[0]["tool_choice"] == {"type": "tool", "name": "submit_x"}
