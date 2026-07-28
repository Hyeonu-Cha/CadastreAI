"""CI guard: the 2026-reform eval quarantine can't silently regress (Task 5.27).

The quarantine (Task 5.01/5.02) only holds if BOTH invariants stay true:

  1. the regime tags stay on the affected queries in the eval data, and
  2. the harness rule keeps `pre_2026_reform` out of the headline.

If either drifts — someone strips the tags, or edits `is_headline` to
count legacy queries — the headline retrieval/agent metrics quietly
start crediting repealed-law answers again (finding F-3). This test
fails loudly if that happens, and runs as a named CI gate in
`.github/workflows/tests.yml`.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.eval.regime import LEGACY_REGIME, is_headline, regime_of

_EVAL_DIR = Path(__file__).resolve().parents[1] / "data" / "eval"
_HEADLINE_FILES = ("queries_all.jsonl", "agent_queries.jsonl")


def _load(name: str) -> list[dict]:
    text = (_EVAL_DIR / name).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_headline_files_still_carry_the_quarantine():
    """Each headline eval file must still tag its pre-reform queries, and
    every tagged query must fall outside the headline."""
    for name in _HEADLINE_FILES:
        rows = _load(name)
        tagged = [q for q in rows if regime_of(q) == LEGACY_REGIME]
        assert tagged, (
            f"{name}: no pre_2026_reform-tagged queries — the quarantine has "
            "been lost, so headline metrics would credit repealed-law answers (F-3)"
        )
        for q in tagged:
            assert not is_headline(regime_of(q)), (
                f"{name}: a pre_2026_reform query is counted in the headline: "
                f"{q.get('query') or q.get('id')!r}"
            )


def test_is_headline_rule_excludes_legacy_regime():
    """Guards the harness rule itself — pre_2026_reform is never headline,
    while neutral / post-reform queries are."""
    assert is_headline(LEGACY_REGIME) is False
    assert is_headline("regime_neutral") is True
    assert is_headline("post_2026_reform") is True
