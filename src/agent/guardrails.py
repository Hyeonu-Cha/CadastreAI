"""Pre-query guardrails that refuse financial-product recommendation
requests (Task X.04).

Scope is narrow on purpose. The agent's job is to summarise the
Australian housing-market evidence base — that includes locality
questions ("is Parramatta a good bet"), policy questions ("how does the
First Home Guarantee work"), and computational questions ("stamp duty
on $1.2M in NSW"). The X.03 disclaimer policy in the system prompt
already handles the "this isn't personal advice" framing for those.

What this module *does* refuse is requests for recommendations of
specific **financial products** — picking a mortgage / lender,
recommending insurance, choosing a super fund. We are not licensed to
do that and the model should never imply that we are. The check is
deliberately conservative: when a phrase clearly asks "which product
should I pick", refuse; everything else passes through to the agent.

Implementation is pure regex / keyword matching. We do *not* call an
LLM here — guardrails on the hot path must be fast and deterministic.
False negatives (slipping through to the agent) are recoverable
because the synthesizer also has the disclaimer policy. False
positives (refusing a legitimate question) are user-visible damage,
so the patterns target unambiguous recommendation phrasings.

The graph wires `guardrail_screen` as the first node — refused queries
short-circuit straight to END with a polite refusal answer; allowed
queries continue to `classify_query` as normal.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal

log = logging.getLogger(__name__)

GuardrailAction = Literal["allow", "refuse"]
GuardrailCategory = Literal[
    "mortgage_product",
    "insurance_product",
    "super_or_managed_fund",
    "specific_security_pick",
]


@dataclass(frozen=True)
class GuardrailDecision:
    """Outcome of a guardrail screening.

    On `allow`, all other fields are None and the agent runs normally.
    On `refuse`, `category`, `reason`, and `refusal_text` are populated
    and the graph short-circuits with `refusal_text` as the answer.
    """

    action: GuardrailAction
    category: GuardrailCategory | None = None
    reason: str | None = None
    refusal_text: str | None = None


# ---------------------------------------------------------------------
# Pattern library
# ---------------------------------------------------------------------
#
# Each pattern is `(category, reason, regex)`. The regex is compiled
# case-insensitive at module load. Order doesn't matter for correctness
# (the first match wins for which-category-was-blocked logging), but
# we put the strongest signals first to keep the log explanation
# specific.
#
# Three buckets — each is opt-in (you have to clearly ask for a product
# pick to trigger), not "any mention of mortgage". Compare:
#   - "what's the average mortgage rate in NSW"  → ALLOW (informational)
#   - "which mortgage should I get"              → REFUSE (product pick)
#   - "best home loan for me"                    → REFUSE (product pick)
#
_PATTERNS: list[tuple[GuardrailCategory, str, re.Pattern[str]]] = [
    # --- Mortgage / home-loan product picking ------------------------
    (
        "mortgage_product",
        "user is asking which specific mortgage / lender to choose",
        re.compile(
            r"\b("
            # "which mortgage should I", "what mortgage do you recommend"
            r"which (mortgage|home loan|lender|broker).{0,30}(should|do you)"
            r"|(what|which) (mortgage|home loan|lender).{0,30}(best|recommend)"
            # "best mortgage for me", "best home loan", "cheapest mortgage"
            r"|(best|cheapest|top) (mortgage|home loan|lender|broker)"
            # "recommend a mortgage / loan / broker / lender"
            r"|recommend.{0,15}(mortgage|home loan|broker|lender)"
            # "should I fix or float", "fixed or variable mortgage"
            r"|(fix(ed)? or (float|variable)|fixed-rate or variable)"
            # "should I refinance" (this is a product decision)
            r"|should i refinance"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    # --- Insurance product picking -----------------------------------
    (
        "insurance_product",
        "user is asking which insurance product to buy",
        re.compile(
            r"\b("
            # "which insurance ... should I", "which landlord insurance should I buy"
            r"which (([a-z]+ )?(insurance|insurer|policy)).{0,30}(should|do you|recommend|buy)"
            r"|(best|cheapest|top) (home insurance|landlord insurance|"
            r"life insurance|income protection|building insurance|contents insurance)"
            r"|recommend.{0,30}(insurance|insurer|policy)"
            r"|(should|do) i (need|get|buy) (life insurance|landlord insurance|"
            r"income protection|home insurance|building insurance)"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    # --- Super / managed fund picking --------------------------------
    (
        "super_or_managed_fund",
        "user is asking which super fund / managed fund to use",
        re.compile(
            r"\b("
            r"which (super|super fund|smsf|managed fund|etf).{0,30}(should|do you|recommend|best)"
            r"|(best|top) (super fund|smsf|managed fund|etf)"
            r"|recommend.{0,15}(super fund|smsf|managed fund|etf)"
            r"|should i (start|set up) (an? )?smsf"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    # --- Specific security picks (stocks, REITs) ---------------------
    (
        "specific_security_pick",
        "user is asking us to pick a specific listed security",
        re.compile(
            r"\b("
            r"(should|do) i (buy|sell|short) (shares?|stock|reit|etf|asx)"
            r"|(best|top) (reit|property etf|asx code)"
            r"|recommend (an? |me )?(stock|share|reit|etf|asx code)"
            r")\b",
            re.IGNORECASE,
        ),
    ),
]


# ---------------------------------------------------------------------
# Refusal text — fixed strings, no model call. Each one names *what*
# we won't do and offers a redirection to a licensed professional plus
# the kind of question we *can* answer. This is the literal text the
# agent returns to the user when guardrails fire.
# ---------------------------------------------------------------------
_REFUSAL_TEMPLATES: dict[GuardrailCategory, str] = {
    "mortgage_product": (
        "I can't recommend specific mortgage products, lenders, or brokers — "
        "that's a personal financial-product decision that needs a licensed "
        "mortgage broker or financial adviser who knows your full circumstances. "
        "I CAN help with the surrounding context: average mortgage rates from RBA "
        "data, stamp-duty calculations, the First Home Guarantee scheme rules, "
        "or how serviceability buffers work. Rephrase the question that way and "
        "I'll dig in."
    ),
    "insurance_product": (
        "I can't recommend specific insurance products or insurers — that's a "
        "licensed-adviser decision. I CAN explain how landlord, building, or "
        "income-protection insurance is treated in published research, or what "
        "the typical cost ranges look like in cited reports. Rephrase the "
        "question informationally and I'll help."
    ),
    "super_or_managed_fund": (
        "I can't recommend specific super funds, SMSFs, managed funds, or ETFs — "
        "that's a personal financial-advice question that needs a licensed "
        "adviser. I CAN summarise published research on how super interacts with "
        "housing policy (e.g. First Home Super Saver Scheme), or what the "
        "evidence says about property-vs-super returns. Reframe the question "
        "that way and I'll help."
    ),
    "specific_security_pick": (
        "I can't tell you whether to buy or sell a specific stock, REIT, or ETF "
        "— that's investment advice and needs a licensed financial adviser. I "
        "CAN summarise what the cited research says about the listed property "
        "sector, REIT structures, or housing-market exposure more broadly. "
        "Rephrase informationally and I'll help."
    ),
}


def screen_query(query: str) -> GuardrailDecision:
    """Run the pattern library against `query` and return a decision.

    Pure function — no I/O, no logging side-effects (callers log when
    they act on the decision). Returns `allow` for empty / whitespace
    queries; the graph's `initial_state` already validates non-empty
    inputs, so an empty here means an unusual caller path and we'd
    rather let downstream surfaces handle it than synthesize a
    refusal.
    """
    if not query or not query.strip():
        return GuardrailDecision(action="allow")
    for category, reason, pattern in _PATTERNS:
        if pattern.search(query):
            return GuardrailDecision(
                action="refuse",
                category=category,
                reason=reason,
                refusal_text=_REFUSAL_TEMPLATES[category],
            )
    return GuardrailDecision(action="allow")


def log_refusal(query: str, decision: GuardrailDecision) -> None:
    """Emit a structured WARNING log line for a refused query.

    Kept separate from `screen_query` so the pure function stays pure
    and the log message format is testable in isolation. We log the
    full query (not redacted) because guardrail false positives are
    the main thing this telemetry is for — knowing exactly what we
    refused is what makes the regex tunable.
    """
    if decision.action != "refuse":
        return
    log.warning(
        "guardrail refusal category=%s reason=%s query=%r",
        decision.category,
        decision.reason,
        query,
    )


__all__ = [
    "GuardrailAction",
    "GuardrailCategory",
    "GuardrailDecision",
    "log_refusal",
    "screen_query",
]
