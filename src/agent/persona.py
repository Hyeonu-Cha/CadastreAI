"""Persona wiring helpers (Task 4.02).

The classifier (Task 3.13) infers a persona from the user's query. The
Streamlit UI (Task 4.01) lets the user pick one explicitly. The user's
self-identification is more authoritative than a one-shot classifier
guess, so when both exist we prefer the user's choice.

Where persona shows up downstream
---------------------------------
- **Synthesize prompt tone.** Different personas need different
  registers — a homebuyer wants concrete numbers and timelines, a
  researcher wants qualified hedges and methodology pointers.
- **Retrieval weighting.** The same publishers are useful for every
  persona, but their *relative* priority shifts: AHURI / Productivity
  Commission policy text is gold for researchers, dry for homebuyers;
  consumer-facing PropTrack / Housing Australia material is the
  reverse.
- **Tool-selection hints.** Investors think in yields and vacancies;
  homebuyers in stamp duty and repayments. The router prompt nudges,
  not forces — the planner can still call any tool when the question
  demands it.

Everything here is pure: no I/O, no Anthropic calls. Helpers are
imported by `nodes.py` for prompt-building and chunk re-ranking, and
by `streamlit_app.py` for persona forwarding.
"""
from __future__ import annotations

from typing import Literal

# Mirrors src.agent.graph.Persona — kept as plain str literals here to
# avoid a circular import (graph.py imports persona helpers via nodes).
Persona = Literal[
    "first_home_buyer",
    "investor",
    "policy_researcher",
    "journalist",
    "general",
]

DEFAULT_PERSONA: Persona = "general"


def effective_persona(state: dict) -> Persona:
    """Return the persona that downstream nodes should honour.

    Precedence:
      1. `state["user_persona"]` — set when the Streamlit UI (or any
         caller) supplied an explicit persona.
      2. `state["classification"]["persona"]` — the classifier's
         best-effort label.
      3. `"general"` — fallback when nothing else resolved.
    """
    user = (state.get("user_persona") or "").strip()
    if user in _PERSONA_PROMPT_ADDENDUM:
        return user  # type: ignore[return-value]
    cls = state.get("classification") or {}
    inferred = (cls.get("persona") or "").strip()
    if inferred in _PERSONA_PROMPT_ADDENDUM:
        return inferred  # type: ignore[return-value]
    return DEFAULT_PERSONA


# Synthesizer system-prompt tone notes per persona. Kept short — the
# bulk of the system prompt (citation grammar, "answer only from
# evidence") stays universal.
_PERSONA_PROMPT_ADDENDUM: dict[Persona, str] = {
    "first_home_buyer": (
        "The user is a prospective first home buyer. Lead with concrete "
        "dollar figures (deposit, stamp duty, monthly repayment), keep "
        "jargon out of the headline answer, and surface any first-home-"
        "buyer concessions or schemes the evidence references."
    ),
    "investor": (
        "The user is a property investor. Lead with investment metrics "
        "(rental yield, vacancy, capital growth, cash-flow). Distinguish "
        "gross vs net where the evidence supports it; flag tax-treatment "
        "items only when explicitly covered in retrieved policy text."
    ),
    "policy_researcher": (
        "The user is a policy / academic researcher. Use a formal "
        "register, attribute claims to specific publishers, and "
        "explicitly note methodology, scope, or year of any cited "
        "study. Do not summarise findings beyond what the evidence "
        "states."
    ),
    "journalist": (
        "The user is a journalist. Be precise and quotable: short "
        "factual sentences, exact numbers with units and dates, and a "
        "clear citation per claim so the user can verify."
    ),
    "general": (
        "The user has not declared a specific role. Answer plainly, "
        "explain a single piece of jargon when it first appears, and "
        "cite every numeric claim."
    ),
}


# Publisher priors per persona. These are MULTIPLIERS applied to the
# retriever's similarity scores — never zeroing a publisher (we still
# want every chunk to be considered), just nudging the order.
#
# Calibration intent (not a strict rule):
#   - 1.20 → "this persona's primary source"
#   - 1.10 → "directly useful"
#   - 1.00 → neutral
#   - 0.90 → "less directly useful for this persona"
#
# 10 publishers exist in the corpus; only ones we want to nudge appear
# here. Anything missing defaults to 1.0 (neutral) via .get().
_PUBLISHER_BOOST: dict[Persona, dict[str, float]] = {
    "first_home_buyer": {
        "Housing Australia (NHFIC)": 1.20,
        "PropTrack": 1.15,
        "CoreLogic / Cotality": 1.10,
        "RBA": 1.05,
        "AHURI": 0.90,
        "Productivity Commission": 0.90,
        "Grattan Institute": 0.95,
    },
    "investor": {
        "SQM Research": 1.20,
        "PropTrack": 1.15,
        "CoreLogic / Cotality": 1.15,
        "APRA": 1.05,
        "RBA": 1.05,
        "AHURI": 0.95,
        "Productivity Commission": 0.95,
    },
    "policy_researcher": {
        "AHURI": 1.20,
        "Productivity Commission": 1.20,
        "Grattan Institute": 1.15,
        "Australian Treasury": 1.15,
        "RBA": 1.10,
        "PropTrack": 0.90,
        "SQM Research": 0.95,
    },
    "journalist": {
        "RBA": 1.10,
        "Australian Treasury": 1.10,
        "AHURI": 1.05,
        "Productivity Commission": 1.05,
        "Grattan Institute": 1.05,
    },
    "general": {},  # no nudges — neutral ranking
}


# Router prompt hints per persona. Appended to `_ROUTER_SYSTEM` when
# the planner has a persona signal. Not strict — the planner can pick
# any tool the question warrants; the hint just biases ties.
_ROUTER_HINTS: dict[Persona, str] = {
    "first_home_buyer": (
        "The user is a first home buyer — for cost questions, prefer "
        "compute_stamp_duty_nsw and compute_mortgage_repayment with "
        "is_first_home_buyer=true where applicable."
    ),
    "investor": (
        "The user is a property investor — for income/return questions, "
        "prefer compute_rental_yield and sqm_rental_vacancy alongside "
        "the relevant ABS / RBA series."
    ),
    "policy_researcher": (
        "The user is a researcher — prefer document retrieval over "
        "tool calls when the question is conceptual, methodological, "
        "or historical. Tools are for current numeric snapshots."
    ),
    "journalist": (
        "The user is a journalist — pair every numeric tool result with "
        "a doc retrieval for context and attribution."
    ),
    "general": "",
}


# Per-persona "not financial advice" disclaimer (Task X.03).
#
# Proportional by persona: stakes are highest when the user is making a
# personal financial decision (first_home_buyer, investor), so they get
# the most explicit "not personal advice; consult a licensed
# professional" language. Researchers and journalists aren't making
# personal investment decisions on the back of an answer, so theirs is
# lighter and aimed at the right audience-appropriate caveats. Every
# persona still gets *some* disclaimer — the universal baseline below
# is enforced by the synthesizer system prompt regardless.
#
# These strings are appended to the system prompt verbatim, so they
# instruct the model on what closing line to include in the answer.
_PERSONA_DISCLAIMER: dict[Persona, str] = {
    "first_home_buyer": (
        "DISCLAIMER POLICY: Property purchase is one of the largest "
        "financial decisions a household makes. This answer is general "
        "information ONLY and is NOT financial, legal, tax, or "
        "buyer-agent advice. End every answer with a one-line "
        "disclaimer that recommends the user consult a licensed "
        "mortgage broker, buyer's agent, conveyancer/solicitor, and "
        "(if relevant) a financial adviser before acting."
    ),
    "investor": (
        "DISCLAIMER POLICY: Investment decisions carry capital and "
        "income risk. This answer is general information ONLY and is "
        "NOT financial, tax, or investment advice. End every answer "
        "with a one-line disclaimer that past performance does not "
        "guarantee future returns and that the user should consult a "
        "licensed financial adviser and tax professional before acting."
    ),
    "policy_researcher": (
        "DISCLAIMER POLICY: Researchers cite primary sources directly. "
        "End every answer with a one-line note that this is a "
        "research-assistant summary, not a peer-reviewed source, and "
        "that primary publications should be cited in any downstream "
        "work."
    ),
    "journalist": (
        "DISCLAIMER POLICY: Journalists verify before publication. End "
        "every answer with a one-line note that figures should be "
        "re-verified against the cited primary source before being "
        "quoted in published copy, and that this output is not "
        "financial advice for readers."
    ),
    "general": (
        "DISCLAIMER POLICY: End every answer with a one-line note that "
        "this is general information, not personal financial, legal, "
        "or tax advice, and that property decisions should involve "
        "appropriately licensed professionals."
    ),
}


def persona_prompt_addendum(persona: Persona) -> str:
    """One-paragraph tone instruction to append to the synth system prompt."""
    return _PERSONA_PROMPT_ADDENDUM.get(persona, _PERSONA_PROMPT_ADDENDUM[DEFAULT_PERSONA])


def persona_disclaimer(persona: Persona) -> str:
    """Per-persona disclaimer policy block for the synth system prompt.

    Proportional by persona: homebuyers and investors get the most
    explicit "not personal advice; consult a licensed professional"
    language; researchers and journalists get lighter, audience-
    appropriate caveats. Always returns a non-empty string — the
    fallback is the `general` persona's policy.
    """
    return _PERSONA_DISCLAIMER.get(persona, _PERSONA_DISCLAIMER[DEFAULT_PERSONA])


def persona_router_hints(persona: Persona) -> str:
    """Hint string to append to the router system prompt; '' when neutral."""
    return _ROUTER_HINTS.get(persona, "")


def persona_publisher_boost(persona: Persona) -> dict[str, float]:
    """Per-publisher score multipliers for retrieval re-ranking."""
    return dict(_PUBLISHER_BOOST.get(persona, {}))


def apply_publisher_boost(
    chunks: list[dict], persona: Persona, *, score_key: str = "score"
) -> list[dict]:
    """Re-rank chunks by `score * publisher_boost` (stable sort).

    Mutates each chunk's `boosted_score` (so the original `score` is
    preserved for inspection) and returns a new list ordered by the
    boosted score descending. Chunks whose payload publisher isn't in
    the boost table get a multiplier of 1.0 (neutral).

    No-op when the persona has no nudges (e.g. `general`).
    """
    boost_table = _PUBLISHER_BOOST.get(persona, {})
    if not boost_table:
        # Neutral — leave order alone but copy so callers can mutate freely.
        return list(chunks)
    out: list[dict] = []
    for c in chunks:
        publisher = ((c.get("payload") or {}).get("publisher") or "").strip()
        boost = boost_table.get(publisher, 1.0)
        new = dict(c)
        new["boosted_score"] = float(c.get(score_key, 0.0)) * boost
        new["boost"] = boost
        out.append(new)
    out.sort(key=lambda c: c.get("boosted_score", 0.0), reverse=True)
    return out


__all__ = [
    "DEFAULT_PERSONA",
    "Persona",
    "apply_publisher_boost",
    "effective_persona",
    "persona_disclaimer",
    "persona_prompt_addendum",
    "persona_publisher_boost",
    "persona_router_hints",
]
