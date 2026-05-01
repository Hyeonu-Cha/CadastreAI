"""Provider abstraction for the agent's LLM calls (Task X.08).

The agent's five nodes (classify, decompose, route, reflect, synthesize)
all need a forced-tool-call (the first four) or a plain-text completion
(the fifth). Originally every node hard-coded `anthropic.Anthropic(...)`;
this module factors that out so the same node code can run on either
Anthropic Claude or OpenAI GPT, switched by the env var
``CADASTRE_LLM_PROVIDER``.

Design choices that fall out of the existing test suite
-------------------------------------------------------
- The Anthropic branch still does ``import anthropic;
  anthropic.Anthropic(api_key).messages.create(...)`` exactly as before,
  so the ~30 tests that monkey-patch ``anthropic.Anthropic`` keep
  working without changes.
- ``cache_control: ephemeral`` is preserved on the Anthropic branch.
  OpenAI has automatic prompt caching with no opt-in surface, so the
  OpenAI branch passes a plain string system prompt.
- Tool schemas are written once in the Anthropic shape (``input_schema``
  + ``tool_choice={"type": "tool", "name": ...}``) and translated to
  OpenAI's function-calling shape (``parameters`` +
  ``tool_choice={"type": "function", "function": {"name": ...}}``)
  inside this module. Nodes don't need to know about either.

Provider selection
------------------
- ``CADASTRE_LLM_PROVIDER=anthropic`` (default) — Claude, Anthropic SDK.
- ``CADASTRE_LLM_PROVIDER=openai`` — GPT, OpenAI SDK.

Per-node model env vars are interpreted in provider-aware ways: when the
provider is OpenAI, the OpenAI-specific ``CADASTRE_OPENAI_*_MODEL`` vars
take precedence; otherwise the existing ``CADASTRE_*_MODEL`` vars apply.
The model passed into ``call_with_tool`` / ``call_text`` is the resolved
value, so the cost-accounting log line records the actual model used.

Usage normalisation
-------------------
Both branches return a dict with the Anthropic-shaped keys
(``input_tokens`` / ``output_tokens`` / ``cache_read_input_tokens`` /
``cache_creation_input_tokens``) so ``src.agent.cost`` can stay shape-
agnostic. For OpenAI we map ``prompt_tokens - cached_tokens`` →
``input_tokens`` and ``cached_tokens`` → ``cache_read_input_tokens``;
write-cache is always 0 (OpenAI's automatic cache has no separate
write meter).

The cost multipliers in cost.py are still Anthropic-flavoured (cache
read at 10%, write at 125%). For OpenAI the cache discount is 50%
not 90%, so cost lines on the OpenAI branch slightly *over*-attribute
the cache-read savings — flagged in cost.py rather than complicating
the math here.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

log = logging.getLogger(__name__)

PROVIDER_ENV = "CADASTRE_LLM_PROVIDER"
DEFAULT_PROVIDER = "anthropic"


def get_provider() -> str:
    """Resolve the active provider from ``CADASTRE_LLM_PROVIDER``.

    Defaults to anthropic. Lower-cased + stripped so casing typos in
    the env file don't silently fall through. Unknown values raise at
    call-time rather than here so importing this module never fails.
    """
    return (os.environ.get(PROVIDER_ENV) or DEFAULT_PROVIDER).strip().lower()


def _anthropic_tool_payload(tool_def: dict, tool_name: str) -> tuple[list[dict], dict]:
    """Pass-through — `tool_def` is already in Anthropic shape."""
    return [tool_def], {"type": "tool", "name": tool_name}


def _openai_tool_payload(tool_def: dict, tool_name: str) -> tuple[list[dict], dict]:
    """Translate an Anthropic-shaped tool to OpenAI's function-calling shape.

    Anthropic uses ``input_schema`` directly under the tool dict; OpenAI
    wraps the same JSON Schema under ``function.parameters`` and
    requires ``type: "function"``. The schema body itself is identical
    (both speak JSON Schema) so we copy it through verbatim.
    """
    schema = tool_def.get("input_schema") or {"type": "object", "properties": {}}
    fn = {
        "name": tool_def["name"],
        "description": tool_def.get("description", ""),
        "parameters": schema,
    }
    return (
        [{"type": "function", "function": fn}],
        {"type": "function", "function": {"name": tool_name}},
    )


def _normalize_anthropic_usage(usage: Any) -> dict:
    """Anthropic Usage → cost.py-compatible dict.

    Returns an empty dict when usage is missing — cost.py then logs
    zero tokens rather than crashing. Otherwise reads each meter via
    ``getattr`` so a Pydantic model and a plain dict both work."""
    if usage is None:
        return {}
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "cache_read_input_tokens": int(
            getattr(usage, "cache_read_input_tokens", 0) or 0
        ),
        "cache_creation_input_tokens": int(
            getattr(usage, "cache_creation_input_tokens", 0) or 0
        ),
    }


def _normalize_openai_usage(usage: Any) -> dict:
    """OpenAI Usage → cost.py-compatible dict.

    OpenAI exposes ``prompt_tokens`` / ``completion_tokens`` and (since
    late 2024) a nested ``prompt_tokens_details.cached_tokens`` for
    automatic prompt caching. We split prompt tokens into "uncached
    input" and "cache-read" so the cost accountant sees the same two
    meters it already knows about. There's no equivalent of cache-
    creation on OpenAI — they don't bill separately for it — so that
    field is always 0."""
    if usage is None:
        return {}
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    cached = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached = int(getattr(details, "cached_tokens", 0) or 0)
    return {
        "input_tokens": max(0, prompt - cached),
        "output_tokens": completion,
        "cache_read_input_tokens": cached,
        "cache_creation_input_tokens": 0,
    }


# ---------------------------------------------------------------------------
# Anthropic branch
# ---------------------------------------------------------------------------


def _call_anthropic_tool(
    *,
    system: str,
    user: str,
    tool_def: dict,
    tool_name: str,
    max_tokens: int,
    model: str,
) -> tuple[dict | None, dict]:
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set; agent node needs Claude."
        )
    client = anthropic.Anthropic(api_key=api_key)
    tools, tool_choice = _anthropic_tool_payload(tool_def, tool_name)
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[{
            "type": "text",
            "text": system,
            "cache_control": {"type": "ephemeral"},
        }],
        tools=tools,
        tool_choice=tool_choice,
        messages=[{"role": "user", "content": user}],
    )
    usage = _normalize_anthropic_usage(getattr(resp, "usage", None))
    tool_use = next(
        (b for b in resp.content if getattr(b, "type", None) == "tool_use"),
        None,
    )
    return (tool_use.input if tool_use is not None else None), usage


def _call_anthropic_text(
    *,
    system: str,
    user: str,
    max_tokens: int,
    model: str,
) -> tuple[str, dict]:
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set; agent node needs Claude."
        )
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[{
            "type": "text",
            "text": system,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": user}],
    )
    usage = _normalize_anthropic_usage(getattr(resp, "usage", None))
    text_blocks = [
        getattr(b, "text", "")
        for b in resp.content
        if getattr(b, "type", None) == "text"
    ]
    return "\n".join(t for t in text_blocks if t).strip(), usage


# ---------------------------------------------------------------------------
# OpenAI branch
# ---------------------------------------------------------------------------


def _call_openai_tool(
    *,
    system: str,
    user: str,
    tool_def: dict,
    tool_name: str,
    max_tokens: int,
    model: str,
) -> tuple[dict | None, dict]:
    from openai import OpenAI  # type: ignore[import-not-found]

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set; agent node needs GPT (CADASTRE_LLM_PROVIDER=openai)."
        )
    client = OpenAI(api_key=api_key)
    tools, tool_choice = _openai_tool_payload(tool_def, tool_name)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        tools=tools,
        tool_choice=tool_choice,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    usage = _normalize_openai_usage(getattr(resp, "usage", None))
    msg = resp.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None) or []
    if not tool_calls:
        return None, usage
    # We forced a single tool — take the first call. ``arguments`` is a
    # JSON string in the OpenAI shape; parse to dict to match Anthropic's
    # already-decoded ``input``.
    args_str = tool_calls[0].function.arguments or "{}"
    try:
        return json.loads(args_str), usage
    except json.JSONDecodeError as e:
        log.warning("openai tool call returned invalid JSON: %s", e)
        return None, usage


def _call_openai_text(
    *,
    system: str,
    user: str,
    max_tokens: int,
    model: str,
) -> tuple[str, dict]:
    from openai import OpenAI  # type: ignore[import-not-found]

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set; agent node needs GPT (CADASTRE_LLM_PROVIDER=openai)."
        )
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    usage = _normalize_openai_usage(getattr(resp, "usage", None))
    return (resp.choices[0].message.content or "").strip(), usage


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def call_with_tool(
    *,
    system: str,
    user: str,
    tool_def: dict,
    tool_name: str,
    max_tokens: int,
    model: str,
) -> tuple[dict | None, dict]:
    """Forced-tool-call across providers.

    Returns ``(tool_input, usage)``. ``tool_input`` is the decoded
    arguments dict the model passed to the forced tool, or ``None`` when
    the model bypassed the tool (rare on Anthropic with explicit
    tool_choice; possible on OpenAI). ``usage`` is the normalised dict
    described at the top of this module.

    Provider exceptions propagate — nodes that want a fallback (reflect,
    synthesize) wrap the call in their own try/except. Provider-key
    misconfiguration raises ``RuntimeError`` so it surfaces at the first
    call instead of silently degrading.
    """
    provider = get_provider()
    if provider == "anthropic":
        return _call_anthropic_tool(
            system=system,
            user=user,
            tool_def=tool_def,
            tool_name=tool_name,
            max_tokens=max_tokens,
            model=model,
        )
    if provider == "openai":
        return _call_openai_tool(
            system=system,
            user=user,
            tool_def=tool_def,
            tool_name=tool_name,
            max_tokens=max_tokens,
            model=model,
        )
    raise ValueError(
        f"unknown {PROVIDER_ENV}={provider!r}; expected 'anthropic' or 'openai'"
    )


def call_text(
    *,
    system: str,
    user: str,
    max_tokens: int,
    model: str,
) -> tuple[str, dict]:
    """Plain-text completion across providers.

    Returns ``(text, usage)``. ``text`` is the concatenation of any
    text blocks in the response (Anthropic) or the single message
    content string (OpenAI). Empty string when the model produced no
    text — caller decides whether to substitute a placeholder."""
    provider = get_provider()
    if provider == "anthropic":
        return _call_anthropic_text(
            system=system,
            user=user,
            max_tokens=max_tokens,
            model=model,
        )
    if provider == "openai":
        return _call_openai_text(
            system=system,
            user=user,
            max_tokens=max_tokens,
            model=model,
        )
    raise ValueError(
        f"unknown {PROVIDER_ENV}={provider!r}; expected 'anthropic' or 'openai'"
    )


__all__ = [
    "DEFAULT_PROVIDER",
    "PROVIDER_ENV",
    "call_text",
    "call_with_tool",
    "get_provider",
]
