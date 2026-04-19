"""Shared helpers for source scrapers."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

HOUSING_KEYWORDS: tuple[str, ...] = (
    "housing",
    "house",
    "houses",
    "dwelling",
    "dwellings",
    "property",
    "properties",
    "mortgage",
    "mortgages",
    "rent",
    "rental",
    "rents",
    "home",
    "homes",
    "real estate",
    "residential",
    "land",
)

USER_AGENT = (
    "CadastreAI-Scraper/0.1 (+https://github.com/Hyeonu-Cha/CadastreAI; "
    "research-only; contact via GitHub issues)"
)


@dataclass
class Source:
    """One entry in sources.jsonl."""

    title: str
    publisher: str
    url: str
    date: str | None = None
    category: str | None = None
    extra: dict = field(default_factory=dict)


def matches_housing(text: str) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in HOUSING_KEYWORDS)


def slugify(text: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^\w\s-]", "", text).strip().lower()
    slug = re.sub(r"[\s_-]+", "-", slug)
    return slug[:max_len].strip("-")


def write_jsonl(sources: list[Source], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for src in sources:
            f.write(json.dumps(asdict(src), ensure_ascii=False) + "\n")
