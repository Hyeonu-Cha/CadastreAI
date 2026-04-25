"""Pydantic response schemas for every CadastreAI tool.

Every tool function (`rba_cash_rate`, `abs_property_price_index`,
`sqm_rental_vacancy`, `render_chart`, `compute_*`, …) returns a dict
that matches one of the models defined here. The agent can rely on
these as a typed contract: data shapes don't drift, missing/None
values are explicit, and the envelope (`source`, `retrieved_at`,
`citation`) is uniform.

The tool implementations still return plain dicts — these models are a
*validation contract*, not a constructor. Use `validate_tool_response`
to pass a tool's output through Pydantic and surface any drift loudly.

    >>> from src.tools.rba_stats import rba_cash_rate
    >>> from src.tools.schemas import RBACashRateResponse, validate_tool_response
    >>> validate_tool_response(rba_cash_rate("latest"), RBACashRateResponse)
    RBACashRateResponse(data=..., source=..., ...)
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------- Common envelope ---------------------------------------------------

class _ToolEnvelope(BaseModel):
    """Fields every tool response carries.

    `as_of` / `queried_period` / `resolved_date` are only meaningful for
    time-series tools — chart and compute helpers omit them. Models
    that need them subclass this and add the right field types.
    """

    model_config = ConfigDict(extra="forbid")

    source: str
    source_url: str | None = None
    retrieved_at: str
    citation: str


class _TemporalEnvelope(_ToolEnvelope):
    as_of: str
    queried_period: str
    resolved_date: str


# ---------- RBA --------------------------------------------------------------

class _RBACashRateData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rate_pct: float | None
    frequency: Literal["monthly_average"]


class RBACashRateResponse(_TemporalEnvelope):
    data: _RBACashRateData


class RBAMortgageRatesResponse(_TemporalEnvelope):
    """Mortgage-rates payload is open-ended.

    Default callers get the four-tag set in `KEY_MORTGAGE_SERIES`, but
    callers can request any subset of F6 series, so we accept a flexible
    `dict[tag, float]`. The companion `series` field maps each tag to
    the underlying RBA series ID for traceability.
    """

    data: dict[str, float | None]
    series: dict[str, str]


# ---------- ABS --------------------------------------------------------------

class _ABSPropertyPriceData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    capital_city: str
    median_house_price_aud: int | None
    median_attached_price_aud: int | None
    frequency: Literal["quarterly"]


class ABSPropertyPriceResponse(_TemporalEnvelope):
    data: _ABSPropertyPriceData


class _ABSBuildingApprovalsData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: str
    dwelling_units_approved: int | None
    frequency: Literal["monthly"]


class ABSBuildingApprovalsResponse(_TemporalEnvelope):
    data: _ABSBuildingApprovalsData


class _ABSLendingIndicatorsData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    total_number: float | None
    owner_occupier_number: float | None
    investor_number: float | None
    non_first_home_buyer_number: float | None
    first_home_buyer_number: float | None
    total_value_aud_m: float | None
    owner_occupier_value_aud_m: float | None
    investor_value_aud_m: float | None
    non_first_home_buyer_value_aud_m: float | None
    first_home_buyer_value_aud_m: float | None
    frequency: Literal["quarterly"]


class ABSLendingIndicatorsResponse(_TemporalEnvelope):
    data: _ABSLendingIndicatorsData


# ---------- SQM --------------------------------------------------------------

class _SQMVacancyData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    location: str
    vacancy_rate_pct: float | None
    listings: int | None
    properties: int | None
    frequency: Literal["monthly"]


class SQMRentalVacancyResponse(_TemporalEnvelope):
    data: _SQMVacancyData


# ---------- Chart ------------------------------------------------------------

class _ChartData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    png_base64: str
    mime_type: Literal["image/png"]
    title: str
    series_names: list[str]
    n_points: int = Field(ge=0)


class RenderChartResponse(_ToolEnvelope):
    """Chart isn't time-series, so it uses the non-temporal envelope."""

    data: _ChartData


# ---------- Compute ----------------------------------------------------------

class _RentalYieldData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    annual_rent_aud: float
    property_value_aud: float
    gross_yield_pct: float


class RentalYieldResponse(_ToolEnvelope):
    data: _RentalYieldData


class _MortgageRepaymentData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    principal_aud: float
    annual_rate_pct: float
    term_years: float
    frequency: Literal["monthly", "fortnightly", "weekly"]
    n_periods: int
    periodic_repayment_aud: float
    total_repaid_aud: float
    total_interest_aud: float


class MortgageRepaymentResponse(_ToolEnvelope):
    data: _MortgageRepaymentData


class _StampDutyNSWData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    purchase_price_aud: float
    is_first_home_buyer: bool
    fhb_concession_applied: bool
    full_duty_aud: float
    duty_aud: float


class StampDutyNSWResponse(_ToolEnvelope):
    data: _StampDutyNSWData


# ---------- Helper -----------------------------------------------------------

def validate_tool_response(payload: dict, model: type[BaseModel]) -> BaseModel:
    """Validate `payload` against `model`, returning the parsed instance.

    Raises `pydantic.ValidationError` on drift — extra keys, missing
    fields, or type mismatches all loud-fail. Useful as a smoke check
    in tests and as a guardrail in the agent loop.
    """
    return model.model_validate(payload)


__all__ = [
    "RBACashRateResponse",
    "RBAMortgageRatesResponse",
    "ABSPropertyPriceResponse",
    "ABSBuildingApprovalsResponse",
    "ABSLendingIndicatorsResponse",
    "SQMRentalVacancyResponse",
    "RenderChartResponse",
    "RentalYieldResponse",
    "MortgageRepaymentResponse",
    "StampDutyNSWResponse",
    "validate_tool_response",
]
