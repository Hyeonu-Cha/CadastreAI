"""Render a series-dict as a base64-encoded PNG line chart.

The agent calls this when its answer would be clearer with a picture
(e.g. "show me cash-rate vs Sydney house prices over the last decade").
The chart is returned inline as a base64 string so it can be embedded
in the agent's structured response without a side-channel file.

`series_dict` is a mapping of `series_name -> {x_label: y_value}`. Each
entry becomes one line on the chart, sharing the same X axis. X labels
are sorted lexicographically — for ISO date strings (`YYYY-MM-DD`,
`YYYY-MM`, `YYYY`) that's also the chronological order, which is what
nearly every CadastreAI series needs.

    >>> from src.tools.chart import render_chart
    >>> out = render_chart(
    ...     {"cash_rate_pct": {"2024-01": 4.35, "2024-06": 4.10, "2025-01": 3.85}},
    ...     "RBA cash rate target",
    ... )
    >>> out["data"]["mime_type"]
    'image/png'
    >>> out["data"]["png_base64"][:8]
    'iVBORw0K'

The envelope matches the rest of the tool layer (`data`, `source`,
`retrieved_at`, `citation`) so Task 3.08's Pydantic wrap is a no-op.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import sys
from datetime import UTC, datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

log = logging.getLogger(__name__)

DEFAULT_FIGSIZE = (8.0, 4.5)
DEFAULT_DPI = 110


def render_chart(
    series_dict: dict[str, dict[str, float]],
    title: str,
    *,
    xlabel: str = "Period",
    ylabel: str = "Value",
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    dpi: int = DEFAULT_DPI,
) -> dict:
    """Render a multi-series line chart as a base64-encoded PNG.

    `series_dict[name]` is a `{x_label: y_value}` dict. X labels across
    series are unioned; missing values render as gaps. We import
    matplotlib lazily so importing this module is cheap.
    """
    if not series_dict:
        raise ValueError("series_dict must not be empty")
    for name, points in series_dict.items():
        if not isinstance(points, dict) or not points:
            raise ValueError(f"series {name!r} must be a non-empty dict")

    # Use the non-interactive Agg backend — no GUI, safe in headless
    # contexts (CI, agent runtime). Imported here so the module loads
    # quickly when render_chart isn't actually called.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_x = sorted({x for pts in series_dict.values() for x in pts})

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for name, pts in series_dict.items():
        ys = [pts.get(x) for x in all_x]
        ax.plot(all_x, ys, marker="o", markersize=3, linewidth=1.5, label=name)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if len(series_dict) > 1:
        ax.legend(loc="best", fontsize=9)
    if len(all_x) > 8:
        # Avoid axis-label collision for long series.
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_horizontalalignment("right")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    retrieved_at = datetime.now(tz=UTC).isoformat(timespec="seconds")
    return {
        "data": {
            "png_base64": png_b64,
            "mime_type": "image/png",
            "title": title,
            "series_names": list(series_dict.keys()),
            "n_points": len(all_x),
        },
        "source": "Local matplotlib render",
        "source_url": None,
        "retrieved_at": retrieved_at,
        "citation": f"Chart rendered locally — {title} ({retrieved_at})",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a JSON file with shape {title, series, xlabel?, ylabel?}",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write the rendered PNG to this path (decoded). If omitted, "
        "prints the JSON envelope to stdout.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    with open(args.input, encoding="utf-8") as f:
        spec = json.load(f)
    out = render_chart(
        spec["series"],
        spec["title"],
        xlabel=spec.get("xlabel", "Period"),
        ylabel=spec.get("ylabel", "Value"),
    )
    if args.output:
        with open(args.output, "wb") as f:
            f.write(base64.b64decode(out["data"]["png_base64"]))
        log.info("Wrote PNG to %s", args.output)
    else:
        # Drop the actual b64 blob from stdout; it's noisy.
        n_chars = len(out["data"]["png_base64"])
        slim = {**out, "data": {**out["data"], "png_base64": f"<{n_chars} chars>"}}
        print(json.dumps(slim, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
