"""Regime tagging for the eval sets (Task 5.01/5.02).

The 2026 housing tax reform (*Treasury Laws Amendment (Tax Reform No. 1)
Act 2026*, enacted 26 Jun 2026) changed negative gearing and the CGT
discount. The corpus predates it, so any eval query whose gold answer
depends on those settings now rewards the agent for reproducing
**repealed** law — see `docs/regime_change_gap_analysis.md`, finding F-3.
Left uncorrected, that makes the headline retrieval/agent metrics
measure fidelity to a superseded statute rather than answer quality.

Rather than physically split the query files (which duplicates data and
invites drift), we tag the affected rows in place with a `regime` field
and teach the harnesses to compute the **headline** metrics over the
regime-independent set, while still scoring and reporting the legacy set
separately. Quarantining is then one flag to flip once the corpus is
refreshed (Week 5 ingestion).

    regime ∈ {pre_2026_reform, post_2026_reform, regime_neutral}

An absent field means `regime_neutral` — the default, counted in the
headline. Only `pre_2026_reform` is excluded from the headline;
`post_2026_reform` is current and counts.
"""
from __future__ import annotations

LEGACY_REGIME = "pre_2026_reform"
NEUTRAL_REGIME = "regime_neutral"
VALID_REGIMES = frozenset({LEGACY_REGIME, "post_2026_reform", NEUTRAL_REGIME})


def regime_of(record: dict) -> str:
    """Regime tag for a query record; an absent/empty field → neutral."""
    return record.get("regime") or NEUTRAL_REGIME


def is_headline(regime: str) -> bool:
    """True when a regime counts toward the headline metrics.

    Everything except the quarantined pre-reform set counts — including
    `post_2026_reform`, which reflects current law.
    """
    return regime != LEGACY_REGIME


__all__ = [
    "LEGACY_REGIME",
    "NEUTRAL_REGIME",
    "VALID_REGIMES",
    "is_headline",
    "regime_of",
]
