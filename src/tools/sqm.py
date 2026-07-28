"""SQM Research rental-vacancy tool.

SQM publishes monthly residential vacancy rates per postcode and per
capital-city region at https://sqmresearch.com.au/property/vacancy-rates.
The page embeds the full historical series inline as a JS literal:

    var data = [{"year":2005,"month":1,"listings":211,
                 "properties":6674,"vr":"0.0316"}, ...];

We just fetch the page and pull that array out — no JSON API needed.
`vr` is a decimal fraction; we expose it as a percentage to match the
chart legend (e.g. "0.0316" → 3.16).

    >>> from src.tools.sqm import sqm_rental_vacancy
    >>> sqm_rental_vacancy("2000")
    {"data": {"vacancy_rate_pct": 3.11, ...}, "as_of": "2026-03-31", ...}
    >>> sqm_rental_vacancy("Sydney")
    {"data": {"vacancy_rate_pct": ..., ...}, ...}

Output envelope mirrors `rba_stats` / `abs_stats` so the agent can
reason uniformly. Task 3.08 will wrap this in a Pydantic schema.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx

from src.tools.regime_break import annotate_structural_break

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

log = logging.getLogger(__name__)

SQM_VACANCY_URL = "https://sqmresearch.com.au/property/vacancy-rates"
DEFAULT_CACHE_DIR = Path("data/cache/sqm")
DEFAULT_CACHE_TTL_SECONDS = 24 * 60 * 60
# SQM 403s an obviously-non-browser UA, but is fine with a plain Chrome string.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Maps a capital-city name to SQM's region slug (state-City).
_CITY_REGIONS = {
    "sydney":    "nsw-Sydney",
    "melbourne": "vic-Melbourne",
    "brisbane":  "qld-Brisbane",
    "adelaide":  "sa-Adelaide",
    "perth":     "wa-Perth",
    "hobart":    "tas-Hobart",
    "darwin":    "nt-Darwin",
    "canberra":  "act-Canberra",
}

_DATA_RE = re.compile(r"var\s+data\s*=\s*(\[\s*\{.*?\}\s*\])\s*;", re.DOTALL)


def _cache_path(cache_dir: Path, name: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / name


def _build_url(target: str) -> tuple[str, dict[str, str]]:
    """Return `(url, query_params)` for a postcode-or-city input.

    A 4-digit string is treated as an Australian postcode; otherwise we
    look it up in `_CITY_REGIONS`. Anything else raises ValueError.
    """
    s = (target or "").strip()
    if not s:
        raise ValueError("postcode_or_city must not be empty")
    if s.isdigit() and len(s) == 4:
        return SQM_VACANCY_URL, {"postcode": s, "t": "1"}
    region = _CITY_REGIONS.get(s.lower())
    if region is None:
        raise ValueError(
            f"postcode_or_city must be a 4-digit postcode or one of "
            f"{sorted(_CITY_REGIONS)}; got {target!r}"
        )
    return SQM_VACANCY_URL, {"region": region, "t": "1", "type": "c"}


def _fetch_html(
    url: str,
    params: dict[str, str],
    cache_path: Path,
    ttl_seconds: int,
    *,
    force_refresh: bool = False,
) -> str:
    if not force_refresh and cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < ttl_seconds:
            log.debug("Cache hit %s (age=%.0fs)", cache_path, age)
            return cache_path.read_text(encoding="utf-8", errors="replace")
    log.info("Fetching %s %s", url, params)
    with httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=30.0,
        follow_redirects=True,
    ) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        text = resp.text
    cache_path.write_text(text, encoding="utf-8")
    return text


def _parse_data(html: str) -> list[dict]:
    """Extract the inline `var data = [...]` time series from SQM HTML.

    Returns a list of `{period_end: date, vacancy_rate_pct: float,
    listings: int, properties: int}` sorted ascending.
    """
    m = _DATA_RE.search(html)
    if m is None:
        raise RuntimeError("Couldn't find `var data = [...]` block in SQM HTML")
    raw = json.loads(m.group(1))
    out: list[dict] = []
    for item in raw:
        try:
            y, mo = int(item["year"]), int(item["month"])
            vr = float(item["vr"])
            listings = int(item["listings"])
            properties = int(item["properties"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append(
            {
                "period_end": _month_end(date(y, mo, 1)),
                "vacancy_rate_pct": round(vr * 100, 2),
                "listings": listings,
                "properties": properties,
            }
        )
    out.sort(key=lambda x: x["period_end"])
    return out


def _month_end(d: date) -> date:
    nxt = d.replace(day=28) + timedelta(days=4)
    return nxt.replace(day=1) - timedelta(days=1)


def _resolve_period(period: str) -> date:
    """Same semantics as `rba_stats._resolve_period` / `abs_stats._resolve_period`."""
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
        return _month_end(ym)
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


def _cache_name(target: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", target.strip().lower())
    return f"vacancy_{safe}.html"


def sqm_rental_vacancy(
    postcode_or_city: str,
    period: str = "latest",
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    force_refresh: bool = False,
) -> dict:
    """Return the SQM monthly residential vacancy rate for a postcode or capital city.

    `postcode_or_city` accepts either a 4-digit Australian postcode (e.g.
    "2000") or a capital-city name (Sydney/Melbourne/Brisbane/Adelaide/
    Perth/Hobart/Darwin/Canberra, case-insensitive). `period` follows the
    same semantics as the RBA/ABS tools — 'latest' | 'YYYY' | 'YYYY-MM' |
    'YYYY-MM-DD'.

    The returned `vacancy_rate_pct` is already a percentage (0.0316 in
    the source becomes 3.16). `listings` and `properties` are the raw
    numerator and denominator SQM uses, exposed for traceability.
    """
    url, params = _build_url(postcode_or_city)
    target_date = _resolve_period(period)

    cache_path = _cache_path(cache_dir, _cache_name(postcode_or_city))
    html = _fetch_html(
        url,
        params,
        cache_path,
        cache_ttl_seconds,
        force_refresh=force_refresh,
    )
    rows = _parse_data(html)
    if not rows:
        raise RuntimeError(f"No vacancy rows parsed for {postcode_or_city!r}")

    hit = _last_at_or_before(rows, target_date)
    if hit is None:
        first = rows[0]
        raise ValueError(
            f"period {period!r} resolves to {target_date}, before earliest "
            f"SQM record for {postcode_or_city} ({first['period_end']})"
        )
    retrieved_at = datetime.now(tz=UTC).isoformat(timespec="seconds")
    full_url = f"{url}?{httpx.QueryParams(params)}"
    return annotate_structural_break({
        "data": {
            "location": postcode_or_city,
            "vacancy_rate_pct": hit["vacancy_rate_pct"],
            "listings": hit["listings"],
            "properties": hit["properties"],
            "frequency": "monthly",
        },
        "as_of": hit["period_end"].isoformat(),
        "queried_period": period,
        "resolved_date": target_date.isoformat(),
        "source": "SQM Research — Residential Vacancy Rates",
        "source_url": full_url,
        "retrieved_at": retrieved_at,
        "citation": (
            f"SQM Research, Residential Vacancy Rates — {postcode_or_city}, "
            f"{hit['period_end'].isoformat()} ({full_url})"
        ),
    })


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "postcode_or_city",
        help="A 4-digit Australian postcode (e.g. 2000) or capital-city name",
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
    out = sqm_rental_vacancy(
        args.postcode_or_city,
        args.period,
        cache_dir=args.cache_dir,
        cache_ttl_seconds=args.ttl,
        force_refresh=args.force_refresh,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
