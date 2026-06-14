"""Suggested follow-up questions based on a finished turn (Task 4.08).

After the agent answers, we want to show the user 2-3 follow-ups they
can click to keep the conversation going. There are two layers:

1. **AI-generated** (`ai_followups`) — a single cheap Haiku call that
   reads the *actual answer text* and proposes follow-ups tailored to
   what was just said. This is the default in the deployed app because
   reading the answer produces far more relevant suggestions than rules
   can (e.g. it can pick up a specific suburb or figure the answer
   mentioned). Gated by `CADASTRE_AI_FOLLOWUPS` (default on) and
   degrades silently to layer 2 on any error / missing key.

2. **Deterministic** (`suggest_followups`) — derives suggestions with
   no network call from things we already know:

     * which tools the agent called (e.g. stamp-duty NSW → suggest VIC)
     * which persona the user is wearing (investor vs homebuyer)
     * which keywords appear in the query (cash rate, vacancy, etc.)

   The catalogue is intentionally narrow. We'd rather emit nothing than
   emit a stale or non-sequitur suggestion.

`followup_questions` is the public entry the UI calls: it tries the AI
layer, then tops up (and falls back) with the deterministic layer so
the chip row is always full and never breaks the turn. `MAX_SUGGESTIONS
= 3` keeps the UI uncluttered.
"""
from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

MAX_SUGGESTIONS = 3

# AI follow-ups are on unless explicitly disabled. Any value in
# {"0", "false", "no", "off"} (case-insensitive) turns them off — handy
# for offline demos or to shave the post-answer Haiku round-trip.
_AI_FOLLOWUPS_ENV = "CADASTRE_AI_FOLLOWUPS"
_AI_DISABLED_VALUES = {"0", "false", "no", "off"}

# Cheap auxiliary task — Haiku by default, mirroring the agent's other
# structured-output nodes. Provider-aware like nodes.py: the OpenAI
# branch needs an OpenAI model id.
FOLLOWUP_MODEL = os.environ.get("CADASTRE_FOLLOWUP_MODEL", "claude-haiku-4-5")
FOLLOWUP_MODEL_OPENAI_DEFAULT = "gpt-4o-mini"


def _ai_followups_enabled() -> bool:
    return (os.environ.get(_AI_FOLLOWUPS_ENV) or "").strip().lower() not in _AI_DISABLED_VALUES


def _tool_followups(tool_results: list[dict]) -> list[str]:
    """Tool-call → follow-up rules. One rule per tool, at most."""
    seen: set[str] = set()
    out: list[str] = []
    for env in tool_results or []:
        name = (env.get("tool") or "").strip()
        if not name or name in seen or "error" in env:
            continue
        seen.add(name)
        if name == "rba_cash_rate":
            out.append("How has the cash rate moved over the past 12 months?")
        elif name == "compute_stamp_duty_nsw":
            out.append(
                "How does NSW stamp duty compare with VIC and QLD on the same purchase price?"
            )
        elif name == "compute_mortgage_repayment":
            out.append("What would the repayment look like if rates rose by 1 percentage point?")
        elif name == "compute_rental_yield":
            out.append("What's a typical rental vacancy rate in that market?")
        elif name == "abs_property_index":
            out.append("How does that capital city compare with the national average?")
        elif name == "abs_lending_indicators":
            out.append("How are first-home-buyer lending volumes tracking versus investors?")
        elif name == "sqm_vacancy":
            out.append("How has rental vacancy moved over the past year in that city?")
    return out


def _persona_followups(persona: str | None, query: str) -> list[str]:
    """Persona-flavoured exploratory follow-ups, used when tool rules
    don't cover enough ground."""
    p = (persona or "").strip().lower()
    q_lower = (query or "").lower()
    out: list[str] = []
    if p == "first_home_buyer":
        out.extend(
            [
                "What grants or stamp-duty concessions might I be eligible for?",
                "What are the typical upfront costs beyond the deposit?",
            ]
        )
    elif p == "investor":
        out.extend(
            [
                "What's the gross rental yield on a property in that price range?",
                "How have rental vacancy rates trended recently?",
            ]
        )
    elif p == "policy_researcher":
        out.extend(
            [
                "What does the recent evidence say about long-run trends in this area?",
                "How do these numbers compare across the major capital cities?",
            ]
        )
    elif p == "journalist":
        out.append("What's the most recent data point and how does it compare to a year ago?")

    # Light keyword sweetener — only when the question actually mentioned it.
    if "cash rate" in q_lower or "interest rate" in q_lower:
        out.append("What did the latest RBA statement give as the reason for that decision?")
    if "vacancy" in q_lower:
        out.append("Which capital city has the tightest rental market right now?")
    return out


def _generic_followups() -> list[str]:
    """Last-resort suggestions so the chip row is rarely empty."""
    return [
        "How does this compare across the major capital cities?",
        "What's the most recent data point on this?",
        "What's driving the recent trend?",
    ]


def suggest_followups(
    query: str,
    persona: str | None,
    tool_results: list[dict] | None = None,
) -> list[str]:
    """Return up to `MAX_SUGGESTIONS` deduped follow-up questions.

    Tool-driven suggestions come first (most specific to what just
    happened), then persona-flavoured, then generic. We dedupe on the
    question text — if a tool rule already proposed something, the
    persona pass won't double it up.
    """
    seen: set[str] = set()
    out: list[str] = []
    for src in (
        _tool_followups(tool_results or []),
        _persona_followups(persona, query),
        _generic_followups(),
    ):
        for s in src:
            if s in seen:
                continue
            seen.add(s)
            out.append(s)
            if len(out) == MAX_SUGGESTIONS:
                return out
    return out


# ---------------------------------------------------------------------------
# AI-generated layer
# ---------------------------------------------------------------------------

_FOLLOWUP_TOOL = {
    "name": "propose_followups",
    "description": (
        "Return up to three short follow-up questions the user is most "
        "likely to want to ask next, given the answer they just received."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": MAX_SUGGESTIONS,
                "description": (
                    "Each item is one self-contained question phrased "
                    "from the user's point of view, max ~12 words."
                ),
            }
        },
        "required": ["questions"],
    },
}

_FOLLOWUP_SYSTEM = (
    "You suggest follow-up questions for an Australian housing-market "
    "research assistant. Given the user's question and the assistant's "
    "answer, propose up to three SHORT, distinct follow-ups the user "
    "would naturally click next.\n"
    "Rules:\n"
    "- Each follow-up must be answerable from Australian housing data, "
    "policy documents, or the assistant's tools (prices, yields, "
    "vacancy, cash rate, stamp duty, grants, lending, supply).\n"
    "- Build on specifics the answer mentioned (a suburb, a figure, a "
    "scheme) rather than generic filler.\n"
    "- Phrase each as a question from the user's point of view, ~12 "
    "words or fewer.\n"
    "- Never give or invite personal financial advice; keep them "
    "informational.\n"
    "- Do not repeat the user's original question.\n"
    "Always answer by calling the propose_followups tool."
)


def _resolve_followup_model() -> str:
    """Provider-aware model id, mirroring nodes.py resolvers."""
    from src.agent.llm import get_provider

    if get_provider() == "openai":
        return os.environ.get(
            "CADASTRE_OPENAI_FOLLOWUP_MODEL", FOLLOWUP_MODEL_OPENAI_DEFAULT
        )
    return FOLLOWUP_MODEL


def _clean_questions(raw: object) -> list[str]:
    """Coerce the tool's `questions` payload into a clean, deduped list."""
    if not isinstance(raw, list):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        q = str(item).strip()
        if not q:
            continue
        # The model occasionally returns a leading bullet/number; trim it.
        q = q.lstrip("-*0123456789. ").strip()
        if not q or q.lower() in seen:
            continue
        seen.add(q.lower())
        out.append(q)
        if len(out) == MAX_SUGGESTIONS:
            break
    return out


def ai_followups(
    query: str,
    persona: str | None,
    answer: str | None,
    tool_results: list[dict] | None = None,
) -> list[str]:
    """LLM-generated follow-ups grounded in the actual answer.

    Returns up to `MAX_SUGGESTIONS` questions, or `[]` on any failure
    (missing key, provider error, empty answer) so the caller can fall
    back to the deterministic layer. Never raises.
    """
    if not (answer or "").strip():
        return []
    try:
        from src.agent.llm import call_with_tool

        tool_names = sorted(
            {
                (env.get("tool") or "").strip()
                for env in (tool_results or [])
                if env.get("tool") and "error" not in env
            }
        )
        user = (
            f"Persona: {persona or 'general'}\n"
            f"User question: {query}\n"
            f"Tools the assistant used: {', '.join(tool_names) or 'none'}\n\n"
            f"Assistant answer:\n{(answer or '')[:2000]}"
        )
        tool_input, _usage = call_with_tool(
            system=_FOLLOWUP_SYSTEM,
            user=user,
            tool_def=_FOLLOWUP_TOOL,
            tool_name="propose_followups",
            max_tokens=256,
            model=_resolve_followup_model(),
        )
    except Exception as e:  # noqa: BLE001 — never break the turn over chips
        log.warning("ai_followups failed, falling back to rules: %s", e)
        return []
    if not tool_input:
        return []
    return _clean_questions(tool_input.get("questions"))


def followup_questions(
    query: str,
    persona: str | None,
    answer: str | None = None,
    tool_results: list[dict] | None = None,
) -> list[str]:
    """Public entry for the UI: AI follow-ups with a deterministic safety net.

    When AI follow-ups are enabled (default), generate them from the
    answer text, then top up any short-fall from the deterministic
    catalogue so the chip row is always full. When disabled or when the
    AI layer yields nothing, fall back entirely to the deterministic
    suggester — identical behaviour to before this feature landed.
    """
    ai: list[str] = []
    if _ai_followups_enabled():
        ai = ai_followups(query, persona, answer, tool_results)
    if len(ai) >= MAX_SUGGESTIONS:
        return ai[:MAX_SUGGESTIONS]

    # Top up (or fully populate) from the deterministic layer, deduped
    # against whatever the AI already proposed.
    seen = {q.lower() for q in ai}
    out = list(ai)
    for q in suggest_followups(query, persona, tool_results):
        if q.lower() in seen:
            continue
        seen.add(q.lower())
        out.append(q)
        if len(out) == MAX_SUGGESTIONS:
            break
    return out


__all__ = [
    "MAX_SUGGESTIONS",
    "ai_followups",
    "followup_questions",
    "suggest_followups",
]
