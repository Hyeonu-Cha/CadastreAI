"""RBA statistics tools — Cash Rate Target (F1.1) and Housing Lending Rates (F6).

Public functions return a dict shaped roughly like:

    {
        "data":         <answer-shaped payload>,
        "source":       "RBA F1.1 (Cash Rate Target)",
        "retrieved_at": "2026-04-25T03:14:15Z",
        "citation":     "RBA Statistical Table F1.1 — Cash Rate Target,
                         https://www.rba.gov.au/statistics/tables/...",
        "as_of":        "<effective date of the returned data point>",
    }

Task 3.08 will wrap these in Pydantic models; for now we keep plain dicts
so the early agent wiring doesn't depend on a schema decision.

Caching: every fetch goes through `data/cache/rba/`. The CSVs are small
(F1.1 ~40 KB, F6 ~37 KB) and update monthly, so a 24-hour TTL on disk is
plenty for development. Pass `force_refresh=True` to bypass.

    >>> from src.tools.rba_stats import rba_cash_rate, rba_mortgage_rates
    >>> rba_cash_rate("latest")
    {"data": {"rate_pct": 3.96, ...}, "as_of": "2026-03-31", ...}
    >>> rba_mortgage_rates("latest")
    {"data": {"owner_occupied_outstanding": 6.05, ...}, "as_of": "2026-02-28", ...}
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys
import time
from datetime import UTC, date, datetime
from pathlib import Path

import httpx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

log = logging.getLogger(__name__)

F11_HIST_URL = "https://www.rba.gov.au/statistics/tables/csv/f1.1-data.csv"
F11_TABLE_URL = "https://www.rba.gov.au/statistics/cash-rate/"
F6_DATA_URL = "https://www.rba.gov.au/statistics/tables/csv/f6-data.csv"
F6_TABLE_URL = "https://www.rba.gov.au/statistics/tables/#interest-rates"
DEFAULT_CACHE_DIR = Path("data/cache/rba")
DEFAULT_CACHE_TTL_SECONDS = 24 * 60 * 60
USER_AGENT = "CadastreAI-research/0.1 (+contact: github.com/Hyeonu-Cha/CadastreAI)"

# F6 has 42 series. We surface a curated subset — the rates the agent
# actually needs to reason about housing affordability. Each tag maps to
# the RBA Series ID found in the F6 CSV "Series ID" header row.
KEY_MORTGAGE_SERIES = {
    "owner_occupied_outstanding": "FLRHOOTA",
    "owner_occupied_new":         "FLRHOFTA",
    "investor_outstanding":       "FLRHIOTA",
    "investor_new":               "FLRHIFTA",
}


def _cache_path(cache_dir: Path, name: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / name


def _fetch_csv(
    url: str,
    cache_path: Path,
    ttl_seconds: int,
    *,
    force_refresh: bool = False,
) -> str:
    if not force_refresh and cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < ttl_seconds:
            log.debug("Cache hit %s (age=%.0fs)", cache_path, age)
            return cache_path.read_text(encoding="utf-8")
    log.info("Fetching %s", url)
    with httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=30.0,
        follow_redirects=True,
    ) as client:
        resp = client.get(url)
        resp.raise_for_status()
        text = resp.text
    cache_path.write_text(text, encoding="utf-8")
    return text


def _parse_f11_data(text: str) -> list[dict]:
    """Return list of `{period_end: date, rate_pct: float}`, sorted ascending.

    The F1.1 CSV is a monthly-average series, so each row is the month-end
    date and the average Cash Rate Target for that month. The first ~10 rows
    are metadata (Title, Description, Frequency, Type, Units, Source,
    Publication date, Series ID); we skip anything where column 0 doesn't
    parse as a date. Cash Rate Target lives in column 1; rows where col[1]
    is empty (pre-1990, before the RBA targeted the cash rate) are dropped.

    Monthly granularity is sufficient for housing-research queries — when a
    rate change lands mid-month the returned value is a weighted average,
    not the snapshot value, but that's what the official table publishes.
    For exact rate-change effective dates, use Statistical Table A2 (XLSX
    only, not yet wired up).
    """
    reader = csv.reader(io.StringIO(text))
    rows: list[dict] = []
    for raw in reader:
        if not raw:
            continue
        eff = _parse_date(raw[0])
        if eff is None:
            continue
        rate = _parse_float(raw[1] if len(raw) > 1 else "")
        if rate is None:
            continue
        rows.append({"period_end": eff, "rate_pct": rate})
    rows.sort(key=lambda r: r["period_end"])
    return rows


def _parse_date(s: str) -> date | None:
    s = s.strip()
    if not s:
        return None
    for fmt in ("%d-%b-%y", "%d-%b-%Y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _parse_float(s: str) -> float | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_f6_data(text: str, series_ids: list[str]) -> list[dict]:
    """Return list of `{period_end: date, rates: {series_id: float}}`.

    F6 is a wide CSV: one column per Series ID. We locate the `Series ID`
    metadata row to map column index → series id, then parse data rows
    where column 0 is a date. A row contributes to the output only if at
    least one of the requested series has a parseable float in its
    column. Missing values in other requested series stay absent rather
    than zero, so the caller can tell "not yet published" from "0.0%".
    """
    reader = csv.reader(io.StringIO(text))
    raw_rows = list(reader)

    series_id_row: list[str] | None = None
    for r in raw_rows:
        if r and r[0].strip().lower() == "series id":
            series_id_row = r
            break
    if series_id_row is None:
        return []
    col_for_series = {
        sid: idx for idx, sid in enumerate(series_id_row) if sid in series_ids
    }

    out: list[dict] = []
    for r in raw_rows:
        if not r:
            continue
        eff = _parse_date(r[0])
        if eff is None:
            continue
        rates: dict[str, float] = {}
        for sid, idx in col_for_series.items():
            if idx >= len(r):
                continue
            v = _parse_float(r[idx])
            if v is not None:
                rates[sid] = v
        if rates:
            out.append({"period_end": eff, "rates": rates})
    out.sort(key=lambda x: x["period_end"])
    return out


def _resolve_period(period: str) -> date:
    """Translate `period` into a target date for point-in-time lookup.

    Supported forms:
      - "latest"        → today
      - "YYYY"          → 31-Dec of that year
      - "YYYY-MM"       → last day of that month (approx via 28th + 4 days)
      - "YYYY-MM-DD"    → that exact date
    """
    period = period.strip()
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
        # Last day of month: jump to 28th, add 4 days, snap back.
        nxt = ym.replace(day=28)
        from datetime import timedelta
        nxt = nxt + timedelta(days=4)
        return nxt.replace(day=1) - timedelta(days=1)
    try:
        y = int(period)
        return date(y, 12, 31)
    except ValueError as e:
        raise ValueError(
            f"period must be 'latest', 'YYYY', 'YYYY-MM', or 'YYYY-MM-DD'; got {period!r}"
        ) from e


def _last_at_or_before(rows: list[dict], target: date) -> dict | None:
    found: dict | None = None
    for r in rows:
        if r["period_end"] <= target:
            found = r
        else:
            break
    return found


def rba_cash_rate(
    period: str = "latest",
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    force_refresh: bool = False,
) -> dict:
    """Return the RBA Cash Rate Target (monthly average) for `period`.

    Pulls F1.1 CSV (cached locally), parses the (month-end, rate) series,
    and picks the last entry on or before the resolved date. Granularity
    is monthly — see `_parse_f11_data` for the trade-off note.
    """
    target = _resolve_period(period)
    cache_path = _cache_path(cache_dir, "f1.1-data.csv")
    text = _fetch_csv(F11_HIST_URL, cache_path, cache_ttl_seconds, force_refresh=force_refresh)
    rows = _parse_f11_data(text)
    if not rows:
        raise RuntimeError(f"No usable rows parsed from {F11_HIST_URL}")

    hit = _last_at_or_before(rows, target)
    if hit is None:
        first = rows[0]
        raise ValueError(
            f"period {period!r} resolves to {target}, before earliest "
            f"F1.1 record ({first['period_end']})"
        )
    retrieved_at = datetime.now(tz=UTC).isoformat(timespec="seconds")
    return {
        "data": {
            "rate_pct": hit["rate_pct"],
            "frequency": "monthly_average",
        },
        "as_of": hit["period_end"].isoformat(),
        "queried_period": period,
        "resolved_date": target.isoformat(),
        "source": "RBA F1.1 (Cash Rate Target, monthly average)",
        "source_url": F11_TABLE_URL,
        "retrieved_at": retrieved_at,
        "citation": (
            f"RBA Statistical Table F1.1 — Cash Rate Target (monthly "
            f"average), {hit['period_end'].isoformat()} ({F11_TABLE_URL})"
        ),
    }


def rba_mortgage_rates(
    period: str = "latest",
    *,
    series: dict[str, str] | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    force_refresh: bool = False,
) -> dict:
    """Return Australian housing lending rates from RBA Statistical Table F6.

    Default `series` covers the four series most useful for housing
    research — outstanding/new loans × owner-occupier/investor — keyed
    by short tags (see `KEY_MORTGAGE_SERIES`). Pass a custom `series`
    dict (`{tag: rba_series_id}`) to surface other F6 series instead.

    Picks the last entry on or before the resolved period. Per-series
    values are dropped from the response when missing in that month
    (not all F6 series go back to 2019); callers should treat absence
    as "not published" rather than zero.
    """
    target = _resolve_period(period)
    series_map = dict(series) if series is not None else dict(KEY_MORTGAGE_SERIES)
    if not series_map:
        raise ValueError("series must be a non-empty mapping {tag: series_id}")

    cache_path = _cache_path(cache_dir, "f6-data.csv")
    text = _fetch_csv(F6_DATA_URL, cache_path, cache_ttl_seconds, force_refresh=force_refresh)
    rows = _parse_f6_data(text, list(series_map.values()))
    if not rows:
        raise RuntimeError(f"No usable rows parsed from {F6_DATA_URL}")

    hit = _last_at_or_before(rows, target)
    if hit is None:
        first = rows[0]
        raise ValueError(
            f"period {period!r} resolves to {target}, before earliest "
            f"F6 record ({first['period_end']})"
        )

    rates_by_tag: dict[str, float] = {}
    for tag, sid in series_map.items():
        if sid in hit["rates"]:
            rates_by_tag[tag] = hit["rates"][sid]

    retrieved_at = datetime.now(tz=UTC).isoformat(timespec="seconds")
    return {
        "data": rates_by_tag,
        "series": series_map,
        "as_of": hit["period_end"].isoformat(),
        "queried_period": period,
        "resolved_date": target.isoformat(),
        "source": "RBA F6 (Housing Lending Rates, monthly)",
        "source_url": F6_TABLE_URL,
        "retrieved_at": retrieved_at,
        "citation": (
            f"RBA Statistical Table F6 — Housing Lending Rates, "
            f"{hit['period_end'].isoformat()} ({F6_TABLE_URL})"
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "metric",
        choices=("cash-rate", "mortgage-rates"),
        nargs="?",
        default="cash-rate",
    )
    p.add_argument("period", nargs="?", default="latest")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    p.add_argument("--ttl", type=int, default=DEFAULT_CACHE_TTL_SECONDS)
    p.add_argument("--force-refresh", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    fn = rba_cash_rate if args.metric == "cash-rate" else rba_mortgage_rates
    out = fn(
        args.period,
        cache_dir=args.cache_dir,
        cache_ttl_seconds=args.ttl,
        force_refresh=args.force_refresh,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
