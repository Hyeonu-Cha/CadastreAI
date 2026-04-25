"""ABS statistics tools — three series the agent reasons about most often.

Public functions return a dict shaped roughly like:

    {
        "data":         <answer-shaped payload>,
        "source":       "ABS 6432.0 — Total Value of Dwellings, Table 2",
        "retrieved_at": "2026-04-25T03:14:15Z",
        "citation":     "ABS Total Value of Dwellings (cat. 6432.0), ...",
        "as_of":        "<period end of the returned data point>",
    }

Currently surfaces:

  - `abs_property_price_index(capital_city, period)` — cat. 6432.0
    Table 2: median sale price of established houses and attached
    dwellings per capital city (quarterly).
  - `abs_building_approvals(state, period)` — cat. 8731.0 Table 7:
    total number of dwelling units approved per state (monthly).
  - `abs_lending_indicators(period)` — cat. 5601.0 Table 1: value
    and number of new housing loan commitments by purpose (quarterly).

Task 3.08 will wrap these in Pydantic models; for now plain dicts so
early agent wiring isn't blocked on a schema decision. ABS publishes
XLSX only — no CSV — so this module needs `openpyxl` (in the `tools`
extra: `pip install -e ".[tools]"`). All downloads use the
`latest-release` URL pattern which redirects to the current period,
so the URL stays stable across ABS releases.

    >>> from src.tools.abs_stats import abs_property_price_index
    >>> abs_property_price_index("Sydney", "latest")
    {"data": {"median_house_price_aud": ..., ...}, ...}
    >>> from src.tools.abs_stats import abs_building_approvals
    >>> abs_building_approvals("NSW", "latest")
    {"data": {"dwelling_units_approved": 2840, ...}, "as_of": "2026-02-28", ...}
    >>> from src.tools.abs_stats import abs_lending_indicators
    >>> abs_lending_indicators("latest")
    {"data": {"owner_occupier_value_aud_m": 64287.0, ...}, ...}
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

log = logging.getLogger(__name__)

ABS_6432_TABLE2_URL = (
    "https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/"
    "total-value-dwellings/latest-release/643202.xlsx"
)
ABS_6432_PAGE_URL = (
    "https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/"
    "total-value-dwellings/latest-release"
)
ABS_8731_TABLE7_URL = (
    "https://www.abs.gov.au/statistics/industry/building-and-construction/"
    "building-approvals-australia/latest-release/8731007.xlsx"
)
ABS_8731_PAGE_URL = (
    "https://www.abs.gov.au/statistics/industry/building-and-construction/"
    "building-approvals-australia/latest-release"
)
ABS_5601_TABLE1_URL = (
    "https://www.abs.gov.au/statistics/economy/finance/"
    "lending-indicators/latest-release/560101.xlsx"
)
ABS_5601_PAGE_URL = (
    "https://www.abs.gov.au/statistics/economy/finance/"
    "lending-indicators/latest-release"
)
DEFAULT_CACHE_DIR = Path("data/cache/abs")
DEFAULT_CACHE_TTL_SECONDS = 24 * 60 * 60
USER_AGENT = "CadastreAI-research/0.1 (+contact: github.com/Hyeonu-Cha/CadastreAI)"

STATES = (
    "New South Wales", "Victoria", "Queensland", "South Australia",
    "Western Australia", "Tasmania", "Northern Territory",
    "Australian Capital Territory",
)
_STATE_ABBR = {
    "nsw": "New South Wales",
    "vic": "Victoria",
    "qld": "Queensland",
    "sa":  "South Australia",
    "wa":  "Western Australia",
    "tas": "Tasmania",
    "nt":  "Northern Territory",
    "act": "Australian Capital Territory",
}
_STATE_LOOKUP = {s.lower(): s for s in STATES} | _STATE_ABBR

# 5601.0 Table 1 column → tag mapping. ABS layout repeats the same 10
# series three times (Original | Seasonally Adjusted | Trend); we expose
# the Original block (cols 1-10) since it's what the agent should cite.
_LENDING_COLS = {
    "total_number":            1,
    "owner_occupier_number":   2,
    "investor_number":         3,
    "non_first_home_buyer_number": 4,
    "first_home_buyer_number": 5,
    "total_value_aud_m":       6,
    "owner_occupier_value_aud_m":  7,
    "investor_value_aud_m":    8,
    "non_first_home_buyer_value_aud_m": 9,
    "first_home_buyer_value_aud_m": 10,
}

CAPITAL_CITIES = (
    "Sydney", "Melbourne", "Brisbane", "Adelaide",
    "Perth", "Hobart", "Darwin", "Canberra",
)
_CITY_LOOKUP = {c.lower(): c for c in CAPITAL_CITIES}

_HOUSE_PREFIX = "Median Price of Established House Transfers (Unstratified)"
_UNIT_PREFIX = "Median Price of Attached Dwelling Transfers (Unstratified)"


def _cache_path(cache_dir: Path, name: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / name


def _fetch_xlsx(
    url: str,
    cache_path: Path,
    ttl_seconds: int,
    *,
    force_refresh: bool = False,
) -> Path:
    if not force_refresh and cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < ttl_seconds:
            log.debug("Cache hit %s (age=%.0fs)", cache_path, age)
            return cache_path
    log.info("Fetching %s", url)
    with httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=60.0,
        follow_redirects=True,
    ) as client:
        resp = client.get(url)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)
    return cache_path


def _normalise_city(city: str) -> str:
    key = (city or "").strip().lower()
    if key not in _CITY_LOOKUP:
        raise ValueError(
            f"capital_city must be one of {CAPITAL_CITIES}; got {city!r}"
        )
    return _CITY_LOOKUP[key]


def _normalise_state(state: str) -> str:
    key = (state or "").strip().lower()
    if key not in _STATE_LOOKUP:
        raise ValueError(
            f"state must be one of {STATES} (or NSW/VIC/QLD/SA/WA/TAS/NT/ACT); "
            f"got {state!r}"
        )
    return _STATE_LOOKUP[key]


def _resolve_period(period: str) -> date:
    """Same semantics as `rba_stats._resolve_period`."""
    period = (period or "").strip()
    if not period or period.lower() == "latest":
        return date.today()
    try:
        return datetime.strptime(period, "%Y-%m-%d").date()
    except ValueError:
        pass
    try:
        ym = datetime.strptime(period, "%Y-%m").date()
    except ValueError:
        ym = None
    if ym is not None:
        nxt = ym.replace(day=28) + timedelta(days=4)
        return nxt.replace(day=1) - timedelta(days=1)
    try:
        y = int(period)
        return date(y, 12, 31)
    except ValueError as e:
        raise ValueError(
            f"period must be 'latest', 'YYYY', 'YYYY-MM', or 'YYYY-MM-DD'; got {period!r}"
        ) from e


def _parse_643202(xlsx_path: Path, city: str) -> list[dict]:
    """Return list of `{period_end: date, median_house_aud: int|None,
    median_unit_aud: int|None}` for the given capital city, sorted asc.

    643202.xlsx Sheet 'Data1' layout: row 0 has full series titles, rows
    1–9 metadata, row 10+ data with a quarter-start date in column A and
    medians in $'000 per series. We multiply by 1000 to expose plain AUD.
    """
    import openpyxl

    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    if "Data1" not in wb.sheetnames:
        raise RuntimeError(f"'Data1' sheet not in {xlsx_path}; got {wb.sheetnames}")
    ws = wb["Data1"]

    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return []
    header = rows[0]

    house_col: int | None = None
    unit_col: int | None = None
    for i, h in enumerate(header):
        if not isinstance(h, str):
            continue
        if h.startswith(_HOUSE_PREFIX) and f" {city} " in h:
            house_col = i
        elif h.startswith(_UNIT_PREFIX) and f" {city} " in h:
            unit_col = i
    if house_col is None and unit_col is None:
        raise RuntimeError(f"No median series for {city!r} in {xlsx_path}")

    out: list[dict] = []
    for r in rows[10:]:
        if not r or not isinstance(r[0], datetime):
            continue
        # ABS labels each quarterly observation by the start-of-quarter
        # date (e.g. 2025-12-01 for "Dec quarter 2025"). Snap to quarter
        # end so `_last_at_or_before` picks the right point and the
        # citation reads naturally.
        q_start = r[0].date()
        q_end = _quarter_end(q_start)
        house = _to_aud(r[house_col]) if house_col is not None else None
        unit = _to_aud(r[unit_col]) if unit_col is not None else None
        if house is None and unit is None:
            continue
        out.append(
            {"period_end": q_end, "median_house_aud": house, "median_unit_aud": unit}
        )
    out.sort(key=lambda x: x["period_end"])
    return out


def _quarter_end(d: date) -> date:
    """Map a quarter-start month (Mar/Jun/Sep/Dec) to its end-of-quarter date."""
    m = d.month
    if m in (1, 2, 3):
        return date(d.year, 3, 31)
    if m in (4, 5, 6):
        return date(d.year, 6, 30)
    if m in (7, 8, 9):
        return date(d.year, 9, 30)
    return date(d.year, 12, 31)


def _month_end(d: date) -> date:
    """Last calendar day of `d`'s month."""
    nxt = d.replace(day=28) + timedelta(days=4)
    return nxt.replace(day=1) - timedelta(days=1)


def _to_aud(v: object) -> int | None:
    """ABS values are in $'000; multiply to plain AUD and round to int."""
    if v is None:
        return None
    try:
        return int(round(float(v) * 1000))
    except (TypeError, ValueError):
        return None


def _last_at_or_before(rows: list[dict], target: date) -> dict | None:
    found: dict | None = None
    for r in rows:
        if r["period_end"] <= target:
            found = r
        else:
            break
    return found


def abs_property_price_index(
    capital_city: str,
    period: str = "latest",
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    force_refresh: bool = False,
) -> dict:
    """Return median residential dwelling prices for a capital city/quarter.

    Despite the legacy name (mirroring the discontinued 6416.0 RPPI), this
    actually returns the *median* sale price published in 6432.0 Table 2 —
    one for established houses and one for attached dwellings (units /
    townhouses). Both values are in plain AUD; either may be `None` if
    not yet published for that quarter.
    """
    city = _normalise_city(capital_city)
    target = _resolve_period(period)

    cache_path = _cache_path(cache_dir, "643202.xlsx")
    _fetch_xlsx(
        ABS_6432_TABLE2_URL,
        cache_path,
        cache_ttl_seconds,
        force_refresh=force_refresh,
    )
    rows = _parse_643202(cache_path, city)
    if not rows:
        raise RuntimeError(f"No usable rows parsed from {ABS_6432_TABLE2_URL}")

    hit = _last_at_or_before(rows, target)
    if hit is None:
        first = rows[0]
        raise ValueError(
            f"period {period!r} resolves to {target}, before earliest "
            f"6432.0 record for {city} ({first['period_end']})"
        )
    retrieved_at = datetime.now(tz=UTC).isoformat(timespec="seconds")
    return {
        "data": {
            "capital_city": city,
            "median_house_price_aud": hit["median_house_aud"],
            "median_attached_price_aud": hit["median_unit_aud"],
            "frequency": "quarterly",
        },
        "as_of": hit["period_end"].isoformat(),
        "queried_period": period,
        "resolved_date": target.isoformat(),
        "source": "ABS 6432.0 — Total Value of Dwellings (Table 2)",
        "source_url": ABS_6432_PAGE_URL,
        "retrieved_at": retrieved_at,
        "citation": (
            f"ABS Total Value of Dwellings (cat. 6432.0), Table 2 "
            f"— Median Price of Transfers, {city}, "
            f"{hit['period_end'].isoformat()} ({ABS_6432_PAGE_URL})"
        ),
    }


def _parse_8731007(xlsx_path: Path, state: str) -> list[dict]:
    """Return `[{period_end, dwelling_units}]` for a state, sorted asc.

    8731007.xlsx Sheet 'Data1' columns 1-8 are the Original series for
    "Total number of dwelling units" by state, in this order: NSW, VIC,
    QLD, SA, WA, TAS, NT, ACT. Each row's date in column A is the
    *first* of the reference month — we snap to month end.
    """
    import openpyxl

    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    if "Data1" not in wb.sheetnames:
        raise RuntimeError(f"'Data1' sheet not in {xlsx_path}; got {wb.sheetnames}")
    ws = wb["Data1"]
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return []
    header = rows[0]

    # Match the *first* column whose title contains the state — that's
    # the Original series. ABS uses double-space delimiters in this table.
    target_marker = f" {state} "
    state_col: int | None = None
    for i, h in enumerate(header[1:9], start=1):
        if isinstance(h, str) and target_marker in h:
            state_col = i
            break
    if state_col is None:
        raise RuntimeError(f"No Original series for {state!r} in {xlsx_path}")

    out: list[dict] = []
    for r in rows[10:]:
        if not r or not isinstance(r[0], datetime):
            continue
        m_end = _month_end(r[0].date())
        v = r[state_col]
        units: int | None
        try:
            units = int(round(float(v))) if v is not None else None
        except (TypeError, ValueError):
            units = None
        if units is None:
            continue
        out.append({"period_end": m_end, "dwelling_units": units})
    out.sort(key=lambda x: x["period_end"])
    return out


def _parse_560101(xlsx_path: Path) -> list[dict]:
    """Return `[{period_end, values: dict[tag, float|None]}]` sorted asc.

    560101.xlsx Sheet 'Data1' columns 1-10 are the Original block of new
    housing loan commitments — see `_LENDING_COLS` for the tag → column
    mapping. Cols 1-5 are Number (counts), cols 6-10 are $ Millions.
    Frequency is *quarterly* despite the column titles not saying so.
    """
    import openpyxl

    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    if "Data1" not in wb.sheetnames:
        raise RuntimeError(f"'Data1' sheet not in {xlsx_path}; got {wb.sheetnames}")
    ws = wb["Data1"]
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return []

    out: list[dict] = []
    for r in rows[10:]:
        if not r or not isinstance(r[0], datetime):
            continue
        q_end = _quarter_end(r[0].date())
        values: dict[str, float | None] = {}
        for tag, col in _LENDING_COLS.items():
            v = r[col] if col < len(r) else None
            try:
                values[tag] = float(v) if v is not None else None
            except (TypeError, ValueError):
                values[tag] = None
        if all(v is None for v in values.values()):
            continue
        out.append({"period_end": q_end, "values": values})
    out.sort(key=lambda x: x["period_end"])
    return out


def abs_building_approvals(
    state: str,
    period: str = "latest",
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    force_refresh: bool = False,
) -> dict:
    """Return total dwelling units approved for a state in a given month.

    Source is ABS 8731.0 Table 7 (Original series). Returns the closest
    observation at-or-before the resolved period.
    """
    st = _normalise_state(state)
    target = _resolve_period(period)

    cache_path = _cache_path(cache_dir, "8731007.xlsx")
    _fetch_xlsx(
        ABS_8731_TABLE7_URL,
        cache_path,
        cache_ttl_seconds,
        force_refresh=force_refresh,
    )
    rows = _parse_8731007(cache_path, st)
    if not rows:
        raise RuntimeError(f"No usable rows parsed from {ABS_8731_TABLE7_URL}")

    hit = _last_at_or_before(rows, target)
    if hit is None:
        first = rows[0]
        raise ValueError(
            f"period {period!r} resolves to {target}, before earliest "
            f"8731.0 record for {st} ({first['period_end']})"
        )
    retrieved_at = datetime.now(tz=UTC).isoformat(timespec="seconds")
    return {
        "data": {
            "state": st,
            "dwelling_units_approved": hit["dwelling_units"],
            "frequency": "monthly",
        },
        "as_of": hit["period_end"].isoformat(),
        "queried_period": period,
        "resolved_date": target.isoformat(),
        "source": "ABS 8731.0 — Building Approvals (Table 7)",
        "source_url": ABS_8731_PAGE_URL,
        "retrieved_at": retrieved_at,
        "citation": (
            f"ABS Building Approvals (cat. 8731.0), Table 7 "
            f"— Total dwelling units approved, {st}, "
            f"{hit['period_end'].isoformat()} ({ABS_8731_PAGE_URL})"
        ),
    }


def abs_lending_indicators(
    period: str = "latest",
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    force_refresh: bool = False,
) -> dict:
    """Return new housing loan commitments (Number + Value) for a quarter.

    Source is ABS 5601.0 Table 1 (Original series). The ten exposed tags
    cover Total / Owner-occupier / Investor / Non-FHB / FHB, each as
    both a count and an AUD-millions value. Returns the closest
    observation at-or-before the resolved period.
    """
    target = _resolve_period(period)

    cache_path = _cache_path(cache_dir, "560101.xlsx")
    _fetch_xlsx(
        ABS_5601_TABLE1_URL,
        cache_path,
        cache_ttl_seconds,
        force_refresh=force_refresh,
    )
    rows = _parse_560101(cache_path)
    if not rows:
        raise RuntimeError(f"No usable rows parsed from {ABS_5601_TABLE1_URL}")

    hit = _last_at_or_before(rows, target)
    if hit is None:
        first = rows[0]
        raise ValueError(
            f"period {period!r} resolves to {target}, before earliest "
            f"5601.0 record ({first['period_end']})"
        )
    retrieved_at = datetime.now(tz=UTC).isoformat(timespec="seconds")
    data = {**hit["values"], "frequency": "quarterly"}
    return {
        "data": data,
        "as_of": hit["period_end"].isoformat(),
        "queried_period": period,
        "resolved_date": target.isoformat(),
        "source": "ABS 5601.0 — Lending Indicators (Table 1)",
        "source_url": ABS_5601_PAGE_URL,
        "retrieved_at": retrieved_at,
        "citation": (
            f"ABS Lending Indicators (cat. 5601.0), Table 1 "
            f"— New housing loan commitments, "
            f"{hit['period_end'].isoformat()} ({ABS_5601_PAGE_URL})"
        ),
    }


def _add_common_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    sub.add_argument("--ttl", type=int, default=DEFAULT_CACHE_TTL_SECONDS)
    sub.add_argument("--force-refresh", action="store_true")
    sub.add_argument("-v", "--verbose", action="store_true")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="metric", required=True)

    pp = sub.add_parser("property-price", help="Median residential prices (cat. 6432.0)")
    pp.add_argument("capital_city", help="One of Sydney, Melbourne, Brisbane, ...")
    pp.add_argument("period", nargs="?", default="latest")
    _add_common_args(pp)

    ba = sub.add_parser("building-approvals", help="Dwelling units approved (cat. 8731.0)")
    ba.add_argument("state", help="State name or abbreviation (NSW, VIC, ...)")
    ba.add_argument("period", nargs="?", default="latest")
    _add_common_args(ba)

    li = sub.add_parser("lending-indicators", help="New housing loan commitments (cat. 5601.0)")
    li.add_argument("period", nargs="?", default="latest")
    _add_common_args(li)

    args = p.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    if args.metric == "property-price":
        out = abs_property_price_index(
            args.capital_city,
            args.period,
            cache_dir=args.cache_dir,
            cache_ttl_seconds=args.ttl,
            force_refresh=args.force_refresh,
        )
    elif args.metric == "building-approvals":
        out = abs_building_approvals(
            args.state,
            args.period,
            cache_dir=args.cache_dir,
            cache_ttl_seconds=args.ttl,
            force_refresh=args.force_refresh,
        )
    else:  # lending-indicators
        out = abs_lending_indicators(
            args.period,
            cache_dir=args.cache_dir,
            cache_ttl_seconds=args.ttl,
            force_refresh=args.force_refresh,
        )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
