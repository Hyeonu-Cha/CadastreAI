"""End-to-end smoke harness for the LangGraph agent (Task 3.12).

Runs three representative queries through `build_graph().invoke(...)`
and prints the final state for each — proves the graph is wired
correctly and lets reviewers see iteration_count, reflection, and
answer_draft for each query in one go. Also exports the graph
diagram in Mermaid form (and best-effort PNG) under `docs/`.

The agent runs against the Task 3.10 stubs, so the answers here are
placeholders — what we're verifying is the *control flow*, not the
content. Tasks 3.13–3.18 will replace the stubs with the real LLM
calls; this same harness will then exercise those.

    python -m src.agent.smoke
    python -m src.agent.smoke --diagram-only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.agent.graph import build_graph, initial_state

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

log = logging.getLogger(__name__)

SAMPLE_QUERIES = [
    "What is the current RBA cash rate?",
    "Compare median Sydney house prices to Melbourne over the last 12 months.",
    "If I buy a $950,000 house in NSW with a 20% deposit at 6.05% over 30 years, "
    "what's my monthly repayment?",
]

DEFAULT_DIAGRAM_DIR = Path("docs")


def _summarise_run(query: str, final_state: dict) -> dict:
    return {
        "query": query,
        "query_type": final_state.get("query_type"),
        "sub_questions": final_state.get("sub_questions"),
        "iteration_count": final_state.get("iteration_count"),
        "is_complete": (final_state.get("reflection") or {}).get("is_complete"),
        "answer_draft": final_state.get("answer_draft"),
    }


def export_diagram(out_dir: Path = DEFAULT_DIAGRAM_DIR) -> dict:
    """Write Mermaid + best-effort PNG of the compiled graph.

    The PNG export requires either a local mermaid renderer or network
    access to mermaid.ink — it can fail silently in offline contexts.
    The Mermaid `.mmd` is the source of truth.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    g = build_graph().get_graph()
    mmd_path = out_dir / "agent_graph.mmd"
    mmd_path.write_text(g.draw_mermaid(), encoding="utf-8")
    log.info("Wrote %s", mmd_path)

    png_path = out_dir / "agent_graph.png"
    try:
        png_bytes = g.draw_mermaid_png()
        png_path.write_bytes(png_bytes)
        log.info("Wrote %s", png_path)
        return {"mmd": str(mmd_path), "png": str(png_path)}
    except Exception as e:  # noqa: BLE001 — best-effort, online-only
        log.warning("PNG export skipped (%s: %s)", type(e).__name__, e)
        return {"mmd": str(mmd_path), "png": None}


def run_samples(queries: list[str] | None = None) -> list[dict]:
    queries = queries or SAMPLE_QUERIES
    g = build_graph()
    return [_summarise_run(q, g.invoke(initial_state(q))) for q in queries]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--diagram-only",
        action="store_true",
        help="Skip the sample-run loop; just (re-)export the graph diagram.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_DIAGRAM_DIR,
        help="Where to write the diagram files (default: docs/).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    diagram = export_diagram(args.out_dir)
    if args.diagram_only:
        print(json.dumps(diagram, ensure_ascii=False, indent=2))
        return
    runs = run_samples()
    print(
        json.dumps(
            {"diagram": diagram, "runs": runs}, ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
