"""Structural-break annotation for live-data tools (Task 5.24).

The 2026 negative-gearing / CGT reform was announced at 7:30pm AEST on
12 May 2026 (the grandfathering line). Investor-driven series — new
housing lending, dwelling prices (established vs new-build), and rental
vacancy / yields — can behave differently either side of that date, so a
figure drawn from after it shouldn't be extrapolated back across the
break (finding F-9, docs/regime_change_gap_analysis.md).

The live tools are point-in-time: each resolves a query to a single
`as_of` date, so there is no range to test for a crossing. Instead we
flag envelopes whose `as_of` falls on or after the announcement and
surface the caveat in the `citation` (the source-card surface) rather
than the answer body — the synthesizer already carries the tax-currency
caveat (#149) and the guardrail prepends one for tax *queries* (Task
5.18); this fills the gap for investor-*data* queries that never mention
negative gearing or CGT by name. Scoped to the three investor-driven
series; the cash rate and building approvals are not annotated.
"""
from __future__ import annotations

from datetime import date

# 7:30pm AEST 12 May 2026 — the reform announcement / grandfathering date.
REFORM_ANNOUNCEMENT_DATE = date(2026, 5, 12)

_STRUCTURAL_BREAK_NOTE = (
    "structural break: this figure is dated on or after the 12 May 2026 "
    "negative-gearing / CGT reform announcement, so investor-driven lending, "
    "price, and rental series may not be comparable across that date — do not "
    "extrapolate a trend through it"
)


def _as_date(value: object) -> date | None:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except (ValueError, TypeError):
        return None


def annotate_structural_break(envelope: dict) -> dict:
    """Flag an investor-series envelope whose `as_of` is on/after the reform
    announcement (Task 5.24).

    Mutates and returns the envelope: appends the note to `citation` (an
    existing field, so it rides the strict `extra_forbidden` tool schemas
    untouched — we deliberately do NOT add a key to `data`). No-op when
    `as_of` is missing / unparseable or pre-announcement, and idempotent
    (safe to call twice).
    """
    as_of = _as_date(envelope.get("as_of"))
    if as_of is None or as_of < REFORM_ANNOUNCEMENT_DATE:
        return envelope
    citation = envelope.get("citation")
    if isinstance(citation, str) and "structural break" not in citation:
        envelope["citation"] = f"{citation} [{_STRUCTURAL_BREAK_NOTE}]"
    return envelope


__all__ = ["REFORM_ANNOUNCEMENT_DATE", "annotate_structural_break"]
