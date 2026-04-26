"""Detect and resolve ambiguous Australian place names (Task 4.04).

Several Australian suburbs and capital-city names collide with each
other or with overseas places. "Richmond" alone is ambiguous between a
Melbourne inner suburb, an outer-Sydney town, and a Tasmanian village —
and the agent's retrieval will silently pick whichever happens to score
highest. Asking the user once up front avoids that whole class of
wrong-answer bugs.

The catalogue is deliberately small. We're guarding against the most
common collisions a property-research user would actually hit, not
trying to be a gazetteer.

Detection rules
---------------
A place is flagged ambiguous when:
  * its name appears as a whole word (case-insensitive) in the query, AND
  * no nearby qualifier (state code, state name, country) disambiguates
    it within `_QUALIFIER_WINDOW` characters either side.

Qualifier matching is also case-insensitive and uses word boundaries so
"NSW" doesn't match inside "answer". A query like
"Newcastle NSW median price" is therefore *not* ambiguous; "Newcastle"
on its own is.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_QUALIFIER_WINDOW = 40  # chars around the place name to scan for qualifiers


@dataclass(frozen=True)
class PlaceOption:
    """One disambiguation choice (the qualified location)."""

    label: str  # human-readable, e.g. "Richmond, VIC (Melbourne suburb)"
    qualifier: str  # appended to the query, e.g. "Richmond, VIC, Australia"


@dataclass(frozen=True)
class Ambiguity:
    """An ambiguous place name found in a user query."""

    place: str  # the surface form as it appears in the query
    options: tuple[PlaceOption, ...]


_PLACES: dict[str, tuple[PlaceOption, ...]] = {
    "richmond": (
        PlaceOption("Richmond, VIC (Melbourne inner suburb)", "Richmond VIC"),
        PlaceOption("Richmond, NSW (outer Sydney)", "Richmond NSW"),
        PlaceOption("Richmond, TAS (Tasmania)", "Richmond TAS"),
    ),
    "brighton": (
        PlaceOption("Brighton, VIC (Melbourne bayside)", "Brighton VIC"),
        PlaceOption("Brighton, SA (Adelaide)", "Brighton SA"),
        PlaceOption("Brighton, QLD (Brisbane)", "Brighton QLD"),
    ),
    "st kilda": (
        PlaceOption("St Kilda, VIC (Melbourne)", "St Kilda VIC"),
        PlaceOption("St Kilda, SA (Adelaide)", "St Kilda SA"),
    ),
    "newcastle": (
        PlaceOption("Newcastle, NSW (Australia)", "Newcastle NSW Australia"),
        PlaceOption("Newcastle upon Tyne (UK)", "Newcastle upon Tyne UK"),
    ),
    "perth": (
        PlaceOption("Perth, WA (Australian capital)", "Perth WA Australia"),
        PlaceOption("Perth, TAS (Tasmania)", "Perth TAS"),
        PlaceOption("Perth, Scotland (UK)", "Perth Scotland UK"),
    ),
    "sydney": (
        PlaceOption("Sydney, NSW (Australia)", "Sydney NSW Australia"),
        PlaceOption("Sydney, Nova Scotia (Canada)", "Sydney Nova Scotia Canada"),
    ),
}

# Qualifier tokens that, if present near the place name, cancel the
# ambiguity. Lower-cased and matched with word boundaries.
_QUALIFIERS: frozenset[str] = frozenset(
    {
        "nsw",
        "vic",
        "qld",
        "wa",
        "sa",
        "tas",
        "act",
        "nt",
        "australia",
        "australian",
        "victoria",
        "queensland",
        "tasmania",
        "tassie",
        "uk",
        "scotland",
        "england",
        "tyne",
        "canada",
        "nova",
        "scotia",
    }
)


def _build_place_regexes() -> dict[str, re.Pattern[str]]:
    return {
        name: re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)
        for name in _PLACES
    }


_PLACE_REGEXES = _build_place_regexes()
_QUALIFIER_RE = re.compile(
    r"\b(" + "|".join(re.escape(q) for q in _QUALIFIERS) + r")\b",
    re.IGNORECASE,
)


def _has_nearby_qualifier(query: str, span: tuple[int, int]) -> bool:
    """Is there a qualifier within `_QUALIFIER_WINDOW` chars of `span`?"""
    start = max(0, span[0] - _QUALIFIER_WINDOW)
    end = min(len(query), span[1] + _QUALIFIER_WINDOW)
    return _QUALIFIER_RE.search(query[start:end]) is not None


def detect_ambiguity(query: str) -> list[Ambiguity]:
    """Return the ambiguous place names in `query`, in order of appearance.

    Empty list when the query is unambiguous (or empty). Each ambiguity
    is reported once even if the same place is mentioned multiple times.
    """
    if not query:
        return []

    found: list[Ambiguity] = []
    seen_keys: set[str] = set()
    matches: list[tuple[int, str, str]] = []
    for key, pattern in _PLACE_REGEXES.items():
        for m in pattern.finditer(query):
            if _has_nearby_qualifier(query, m.span()):
                continue
            matches.append((m.start(), key, m.group(0)))

    matches.sort(key=lambda x: x[0])
    for _start, key, surface in matches:
        if key in seen_keys:
            continue
        seen_keys.add(key)
        found.append(Ambiguity(place=surface, options=_PLACES[key]))
    return found


def apply_disambiguation(query: str, choices: dict[str, str]) -> str:
    """Append each chosen qualifier to the query as a parenthetical.

    `choices` maps the surface place name (as returned by
    `detect_ambiguity`) to the qualifier string the user picked. We
    don't try to splice the qualifier in-place — appending parenthetical
    hints keeps the original phrasing intact and is robust to weird
    casing, possessives, or punctuation around the place name.
    """
    if not choices:
        return query
    hints = ", ".join(f"{place}: {qual}" for place, qual in choices.items())
    return f"{query.rstrip()} (clarification — {hints})"


__all__ = [
    "Ambiguity",
    "PlaceOption",
    "apply_disambiguation",
    "detect_ambiguity",
]
