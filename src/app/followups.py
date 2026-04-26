"""Suggested follow-up questions based on a finished turn (Task 4.08).

After the agent answers, we want to show the user 2-3 follow-ups they
can click to keep the conversation going. We could ask another LLM for
suggestions, but that's an extra round-trip and an extra failure mode
for what should feel instant. Instead this module derives suggestions
deterministically from things we already know:

  * which tools the agent called (e.g. stamp-duty NSW → suggest VIC)
  * which persona the user is wearing (investor vs homebuyer)
  * which keywords appear in the query (cash rate, vacancy, etc.)

The catalogue is intentionally narrow. We'd rather emit nothing than
emit a stale or non-sequitur suggestion. `MAX_SUGGESTIONS = 3` keeps
the UI uncluttered.
"""
from __future__ import annotations

MAX_SUGGESTIONS = 3


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


__all__ = ["MAX_SUGGESTIONS", "suggest_followups"]
