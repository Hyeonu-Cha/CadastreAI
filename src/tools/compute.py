"""Numeric helpers the agent calls when an answer needs a calculation.

Three functions, each returning the standard tool envelope:

  - `compute_rental_yield(annual_rent_aud, property_value_aud)` —
    gross rental yield as a percentage.
  - `compute_mortgage_repayment(principal_aud, annual_rate_pct,
    term_years, frequency)` — standard amortising-loan repayment.
  - `compute_stamp_duty_nsw(purchase_price_aud, *, is_first_home_buyer)` —
    NSW residential transfer duty using the 2024-25 brackets (effective
    1 July 2024; Revenue NSW indexes thresholds annually, so these are
    dated — see `_NSW_BRACKETS`), with optional First Home Buyers
    Assistance Scheme concession.

These are deliberately simple closed-form formulas — no API calls, no
caching needed. They exist so the agent doesn't have to reason about
numbers symbolically: it calls the helper and quotes the result with
the citation supplied here.

    >>> from src.tools.compute import compute_rental_yield
    >>> compute_rental_yield(36400, 950000)["data"]["gross_yield_pct"]
    3.83
    >>> from src.tools.compute import compute_mortgage_repayment
    >>> compute_mortgage_repayment(800000, 6.05, 30)["data"]["periodic_repayment_aud"]
    4822.15
    >>> from src.tools.compute import compute_stamp_duty_nsw
    >>> compute_stamp_duty_nsw(1_500_000)["data"]["duty_aud"]
    64910.0
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import UTC, datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

log = logging.getLogger(__name__)


def _envelope(data: dict, source: str, citation: str) -> dict:
    return {
        "data": data,
        "source": source,
        "source_url": None,
        "retrieved_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "citation": citation,
    }


def compute_rental_yield(
    annual_rent_aud: float,
    property_value_aud: float,
) -> dict:
    """Return gross rental yield = annual rent / property value, as a %.

    Net yield (after costs) needs strata, council rates, insurance,
    management fees etc. — not in scope here; the agent can subtract
    those itself once it has the gross figure.
    """
    if annual_rent_aud < 0:
        raise ValueError(f"annual_rent_aud must be >= 0; got {annual_rent_aud}")
    if property_value_aud <= 0:
        raise ValueError(
            f"property_value_aud must be > 0; got {property_value_aud}"
        )
    yield_pct = round((annual_rent_aud / property_value_aud) * 100, 2)
    return _envelope(
        data={
            "annual_rent_aud": annual_rent_aud,
            "property_value_aud": property_value_aud,
            "gross_yield_pct": yield_pct,
        },
        source="Local calculation — gross rental yield",
        citation=(
            f"Gross rental yield = {annual_rent_aud:,.0f} / "
            f"{property_value_aud:,.0f} = {yield_pct}% (gross, before costs)"
        ),
    )


def compute_mortgage_repayment(
    principal_aud: float,
    annual_rate_pct: float,
    term_years: float,
    frequency: str = "monthly",
) -> dict:
    """Return the periodic repayment for a standard amortising loan.

    Uses the textbook annuity formula:
        P_t = principal * r / (1 - (1+r)^-n)
    where `r` is the per-period rate and `n` the number of periods.
    `frequency` is one of 'monthly' (12/yr), 'fortnightly' (26/yr),
    or 'weekly' (52/yr).
    """
    if principal_aud <= 0:
        raise ValueError(f"principal_aud must be > 0; got {principal_aud}")
    if term_years <= 0:
        raise ValueError(f"term_years must be > 0; got {term_years}")
    if annual_rate_pct < 0:
        raise ValueError(f"annual_rate_pct must be >= 0; got {annual_rate_pct}")

    periods_per_year = {"monthly": 12, "fortnightly": 26, "weekly": 52}.get(frequency)
    if periods_per_year is None:
        raise ValueError(
            f"frequency must be 'monthly', 'fortnightly', or 'weekly'; got {frequency!r}"
        )

    n = int(round(term_years * periods_per_year))
    r = (annual_rate_pct / 100) / periods_per_year
    if r == 0:
        periodic = principal_aud / n
    else:
        periodic = principal_aud * r / (1 - math.pow(1 + r, -n))
    periodic = round(periodic, 2)
    total = round(periodic * n, 2)
    interest = round(total - principal_aud, 2)

    return _envelope(
        data={
            "principal_aud": principal_aud,
            "annual_rate_pct": annual_rate_pct,
            "term_years": term_years,
            "frequency": frequency,
            "n_periods": n,
            "periodic_repayment_aud": periodic,
            "total_repaid_aud": total,
            "total_interest_aud": interest,
        },
        source="Local calculation — amortising-loan repayment",
        citation=(
            f"{frequency.capitalize()} repayment on AUD {principal_aud:,.0f} "
            f"@ {annual_rate_pct}% p.a. over {term_years} years = "
            f"AUD {periodic:,.2f}"
        ),
    )


# NSW Office of State Revenue, transfer duty rates effective 1 July 2024
# (indexed annually). Brackets are (upper_threshold, base_duty,
# marginal_rate_per_dollar_above_lower_threshold). The last bracket
# uses math.inf for upper.
_NSW_BRACKETS: list[tuple[float, float, float]] = [
    (   17_000,        0.00, 0.0125),
    (   36_000,      212.50, 0.0150),
    (   97_000,      497.50, 0.0175),
    (  364_000,    1_565.00, 0.0350),
    (1_212_000,   10_910.00, 0.0450),
    (3_636_000,   49_070.00, 0.0550),
    (math.inf,   182_390.00, 0.0700),  # premium duty above $3.636M
]
_NSW_BRACKET_LOWERS: list[float] = [0.0] + [b[0] for b in _NSW_BRACKETS[:-1]]


def _nsw_full_duty(price: float) -> float:
    """Apply the bracketed NSW transfer-duty formula."""
    for i, (upper, base, rate) in enumerate(_NSW_BRACKETS):
        if price <= upper:
            lower = _NSW_BRACKET_LOWERS[i]
            return round(base + (price - lower) * rate, 2)
    # Unreachable — last bracket has upper=inf.
    raise RuntimeError(f"NSW duty bracket lookup failed for {price}")


def compute_stamp_duty_nsw(
    purchase_price_aud: float,
    *,
    is_first_home_buyer: bool = False,
) -> dict:
    """Return NSW residential transfer (stamp) duty for a purchase price.

    Uses the 2024-25 indexed brackets published by Revenue NSW. When
    `is_first_home_buyer=True`, applies the First Home Buyers Assistance
    Scheme: full exemption ≤ $800,000, linear concession to $1,000,000,
    full duty above $1,000,000.

    Note: this calculator excludes the foreign-buyer surcharge,
    off-the-plan concession, principal-place-of-residence concessions,
    and any specific exemptions (e.g. spousal transfers). The agent
    should flag these caveats when quoting the figure.
    """
    if purchase_price_aud <= 0:
        raise ValueError(
            f"purchase_price_aud must be > 0; got {purchase_price_aud}"
        )
    full_duty = _nsw_full_duty(purchase_price_aud)
    duty = full_duty
    concession_applied = False
    if is_first_home_buyer:
        if purchase_price_aud <= 800_000:
            duty = 0.0
            concession_applied = True
        elif purchase_price_aud < 1_000_000:
            # Linear taper over the $200k window.
            taper = (purchase_price_aud - 800_000) / 200_000
            duty = round(full_duty * taper, 2)
            concession_applied = True

    return _envelope(
        data={
            "purchase_price_aud": purchase_price_aud,
            "is_first_home_buyer": is_first_home_buyer,
            "fhb_concession_applied": concession_applied,
            "full_duty_aud": full_duty,
            "duty_aud": duty,
        },
        source="Revenue NSW — Transfer duty (2024-25 brackets, FHBAS thresholds)",
        citation=(
            f"NSW transfer duty on AUD {purchase_price_aud:,.0f} = "
            f"AUD {duty:,.2f}"
            + (
                f" (FHBAS concession from full duty AUD {full_duty:,.2f})"
                if concession_applied
                else ""
            )
            + ". Excludes foreign-buyer surcharge and off-the-plan concession."
        ),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="metric", required=True)

    ry = sub.add_parser("rental-yield", help="Gross rental yield")
    ry.add_argument("annual_rent_aud", type=float)
    ry.add_argument("property_value_aud", type=float)

    mr = sub.add_parser("mortgage", help="Amortising-loan repayment")
    mr.add_argument("principal_aud", type=float)
    mr.add_argument("annual_rate_pct", type=float)
    mr.add_argument("term_years", type=float)
    mr.add_argument(
        "--frequency",
        choices=("monthly", "fortnightly", "weekly"),
        default="monthly",
    )

    sd = sub.add_parser("stamp-duty-nsw", help="NSW transfer duty")
    sd.add_argument("purchase_price_aud", type=float)
    sd.add_argument("--first-home-buyer", action="store_true")

    args = p.parse_args()
    if args.metric == "rental-yield":
        out = compute_rental_yield(args.annual_rent_aud, args.property_value_aud)
    elif args.metric == "mortgage":
        out = compute_mortgage_repayment(
            args.principal_aud,
            args.annual_rate_pct,
            args.term_years,
            frequency=args.frequency,
        )
    else:  # stamp-duty-nsw
        out = compute_stamp_duty_nsw(
            args.purchase_price_aud,
            is_first_home_buyer=args.first_home_buyer,
        )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
