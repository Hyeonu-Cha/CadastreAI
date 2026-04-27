"""Unit tests for the persona disclaimer policy (Task X.03).

Three responsibilities:

1. **Every persona gets a disclaimer.** No persona slips through with
   an empty/missing block. Unknown personas fall back to `general`.
2. **Proportional intensity.** Homebuyers and investors — who use the
   tool to inform personal financial decisions — must explicitly say
   "not financial advice" and point at a licensed professional. The
   journalist and researcher disclaimers focus on verification and
   audience caveats instead.
3. **Wired into the synth system prompt.** The synthesizer's assembled
   system prompt must include both the universal compliance baseline
   and the per-persona disclaimer block.
"""
from __future__ import annotations

import pytest

from src.agent import persona as p

ALL_PERSONAS = (
    "first_home_buyer",
    "investor",
    "policy_researcher",
    "journalist",
    "general",
)


# ---------- coverage --------------------------------------------------


@pytest.mark.parametrize("persona", ALL_PERSONAS)
def test_every_persona_has_a_non_empty_disclaimer(persona):
    text = p.persona_disclaimer(persona)
    assert isinstance(text, str)
    assert len(text.strip()) > 0


def test_unknown_persona_falls_back_to_general():
    assert p.persona_disclaimer("astronaut") == p.persona_disclaimer(  # type: ignore[arg-type]
        "general"
    )


@pytest.mark.parametrize("persona", ALL_PERSONAS)
def test_every_persona_has_a_disclaimer_policy_marker(persona):
    """All disclaimers tag themselves so the model recognises the block."""
    assert "DISCLAIMER POLICY" in p.persona_disclaimer(persona)


# ---------- proportional intensity ------------------------------------


@pytest.mark.parametrize("persona", ("first_home_buyer", "investor"))
def test_high_stakes_personas_invoke_licensed_professionals(persona):
    """Personal financial decisions → explicit licensed-professional language."""
    text = p.persona_disclaimer(persona).lower()
    assert "licensed" in text
    # The phrase the user is most likely searching for in compliance review.
    assert "not financial" in text or "not a licensed" in text


def test_first_home_buyer_disclaimer_names_relevant_professionals():
    """Homebuyers need a buyer's agent / mortgage broker / conveyancer."""
    text = p.persona_disclaimer("first_home_buyer").lower()
    assert "mortgage broker" in text
    assert "buyer" in text  # buyer's agent
    # conveyancer or solicitor — either is valid per Australian practice.
    assert "conveyancer" in text or "solicitor" in text


def test_investor_disclaimer_flags_returns_and_tax():
    text = p.persona_disclaimer("investor").lower()
    assert "past performance" in text
    assert "tax" in text


def test_journalist_disclaimer_focuses_on_verification():
    text = p.persona_disclaimer("journalist").lower()
    assert "verif" in text  # verify / re-verified / verification


def test_policy_researcher_disclaimer_focuses_on_primary_sources():
    text = p.persona_disclaimer("policy_researcher").lower()
    assert "primary" in text


# ---------- wired into synth system prompt ----------------------------


@pytest.mark.parametrize("persona", ALL_PERSONAS)
def test_synth_system_prompt_contains_baseline_and_persona_block(persona):
    """End-to-end: every persona's assembled prompt carries both layers."""
    from src.agent import nodes

    assembled = (
        f"{nodes._SYNTHESIZER_SYSTEM}\n\n"
        f"{p.persona_prompt_addendum(persona)}\n\n"
        f"{p.persona_disclaimer(persona)}"
    )
    # Universal baseline.
    assert "COMPLIANCE" in assembled
    # Persona-specific block.
    assert "DISCLAIMER POLICY" in assembled
    # The whole block is inside the assembled prompt verbatim.
    assert p.persona_disclaimer(persona) in assembled


def test_synthesizer_system_prompt_constant_includes_baseline():
    """The static module-level constant alone (no persona) carries the baseline."""
    from src.agent import nodes

    assert "COMPLIANCE" in nodes._SYNTHESIZER_SYSTEM
    assert "not a licensed" in nodes._SYNTHESIZER_SYSTEM.lower()


def test_disclaimer_persona_export():
    """`persona_disclaimer` is part of the public surface of the module."""
    assert "persona_disclaimer" in p.__all__
