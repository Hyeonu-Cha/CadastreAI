"""Each tool's actual output must validate against its schema.

These are integration-flavoured smoke tests — they call the real tool
functions (which hit cached HTTP responses on disk) and pass the
result through `validate_tool_response`. The point is to catch shape
drift between an implementation and its declared contract.

The compute and chart helpers are pure-Python and run instantly. The
network-backed tools rely on the 24h disk cache, so the first run
populates it (~5 s of HTTP) and subsequent runs are offline.
"""
from __future__ import annotations

import pytest

from src.tools.chart import render_chart
from src.tools.compute import (
    compute_mortgage_repayment,
    compute_rental_yield,
    compute_stamp_duty_nsw,
)
from src.tools.schemas import (
    ABSBuildingApprovalsResponse,
    ABSLendingIndicatorsResponse,
    ABSPropertyPriceResponse,
    MortgageRepaymentResponse,
    RBACashRateResponse,
    RBAMortgageRatesResponse,
    RenderChartResponse,
    RentalYieldResponse,
    SQMRentalVacancyResponse,
    StampDutyNSWResponse,
    validate_tool_response,
)


def test_compute_rental_yield_schema():
    out = compute_rental_yield(36400, 950000)
    validate_tool_response(out, RentalYieldResponse)


def test_compute_mortgage_repayment_schema():
    out = compute_mortgage_repayment(800000, 6.05, 30)
    validate_tool_response(out, MortgageRepaymentResponse)


def test_compute_stamp_duty_nsw_schema():
    out = compute_stamp_duty_nsw(1_500_000)
    validate_tool_response(out, StampDutyNSWResponse)


def test_compute_stamp_duty_nsw_fhb_taper_schema():
    out = compute_stamp_duty_nsw(900_000, is_first_home_buyer=True)
    validate_tool_response(out, StampDutyNSWResponse)


def test_render_chart_schema():
    out = render_chart(
        {"cash_rate_pct": {"2025-01": 4.10, "2025-06": 3.85, "2026-01": 3.96}},
        "Cash rate sample",
    )
    validate_tool_response(out, RenderChartResponse)


# Network-backed tools — skipped if the cache is missing AND we're offline.
# In normal dev this runs against the 24h disk cache populated by the
# tools' own smoke-tests.

@pytest.mark.network
def test_rba_cash_rate_schema():
    from src.tools.rba_stats import rba_cash_rate

    out = rba_cash_rate("latest")
    validate_tool_response(out, RBACashRateResponse)


@pytest.mark.network
def test_rba_mortgage_rates_schema():
    from src.tools.rba_stats import rba_mortgage_rates

    out = rba_mortgage_rates("latest")
    validate_tool_response(out, RBAMortgageRatesResponse)


@pytest.mark.network
def test_abs_property_price_index_schema():
    from src.tools.abs_stats import abs_property_price_index

    out = abs_property_price_index("Sydney", "latest")
    validate_tool_response(out, ABSPropertyPriceResponse)


@pytest.mark.network
def test_abs_building_approvals_schema():
    from src.tools.abs_stats import abs_building_approvals

    out = abs_building_approvals("NSW", "latest")
    validate_tool_response(out, ABSBuildingApprovalsResponse)


@pytest.mark.network
def test_abs_lending_indicators_schema():
    from src.tools.abs_stats import abs_lending_indicators

    out = abs_lending_indicators("latest")
    validate_tool_response(out, ABSLendingIndicatorsResponse)


@pytest.mark.network
def test_sqm_rental_vacancy_postcode_schema():
    from src.tools.sqm import sqm_rental_vacancy

    out = sqm_rental_vacancy("2000")
    validate_tool_response(out, SQMRentalVacancyResponse)


@pytest.mark.network
def test_sqm_rental_vacancy_city_schema():
    from src.tools.sqm import sqm_rental_vacancy

    out = sqm_rental_vacancy("Sydney")
    validate_tool_response(out, SQMRentalVacancyResponse)
