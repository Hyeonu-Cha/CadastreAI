"""Tests for the compute-tool disambiguation rules in `_ROUTER_SYSTEM`
(Task 3.30, agent v5).

v3/v4 had a known regression class on agent-019: the router picked
`compute_rental_yield` for "median rental yields in Perth" — a market
lookup, not a computation. compute_rental_yield needs both
`annual_rent_aud` and `property_value_aud` as explicit numbers; the
agent had neither, so it hallucinated args.

These tests pin the disambiguation contract in the routing prompt so
a future prompt edit can't silently drop it.
"""
from __future__ import annotations

from src.agent.nodes import _ROUTER_SYSTEM


def test_router_prompt_names_compute_rental_yield_args():
    assert "annual_rent_aud" in _ROUTER_SYSTEM
    assert "property_value_aud" in _ROUTER_SYSTEM


def test_router_prompt_names_compute_mortgage_args():
    assert "principal_aud" in _ROUTER_SYSTEM
    assert "annual_rate_pct" in _ROUTER_SYSTEM
    assert "term_years" in _ROUTER_SYSTEM


def test_router_prompt_redirects_market_yield_lookup():
    assert "sqm_rental_vacancy" in _ROUTER_SYSTEM
    assert "abs_property_price_index" in _ROUTER_SYSTEM
    # The redirect rule must explicitly mention the market-lookup phrase
    # the v4 agent kept misclassifying.
    assert "rental yield" in _ROUTER_SYSTEM


def test_router_prompt_has_worked_examples():
    # WORKED EXAMPLES block — both the compute-when-numbers-given case
    # and the lookup-when-numbers-missing case.
    assert "WORKED EXAMPLES" in _ROUTER_SYSTEM
    # The lookup example asks about Perth yield without numbers —
    # locks in the routing-to-sqm + price-index pattern.
    assert "Perth" in _ROUTER_SYSTEM


def test_router_prompt_states_compute_requires_numbers():
    assert "COMPUTE TOOLS REQUIRE USER-PROVIDED NUMBERS" in _ROUTER_SYSTEM
