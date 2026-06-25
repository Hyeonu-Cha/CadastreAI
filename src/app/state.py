"""Streamlit app state helpers (Task 4.01).

Pure, side-effect-free helpers — kept separate from the Streamlit
entrypoint so they can be unit-tested without spinning up Streamlit.
The entrypoint (`src/app/streamlit_app.py`) imports from here for any
logic that touches data shapes; UI rendering stays there.

Persona vocabulary
------------------
The Streamlit persona selector exposes four user-facing labels:

    Homebuyer / Investor / Researcher / Just exploring

These map onto the agent classifier's internal personas
(`first_home_buyer`, `investor`, `policy_researcher`, `general`)
because the classifier's vocabulary was designed for the eval set, not
for an end-user dropdown. `journalist` is intentionally not surfaced —
it shares behaviour with `policy_researcher` for our prompts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

UiPersona = Literal["Homebuyer", "Investor", "Researcher", "Just exploring"]
AgentPersona = Literal[
    "first_home_buyer", "investor", "policy_researcher", "general"
]

UI_PERSONAS: tuple[UiPersona, ...] = (
    "Homebuyer",
    "Investor",
    "Researcher",
    "Just exploring",
)
DEFAULT_PERSONA: UiPersona = "Just exploring"

_UI_TO_AGENT: dict[UiPersona, AgentPersona] = {
    "Homebuyer": "first_home_buyer",
    "Investor": "investor",
    "Researcher": "policy_researcher",
    "Just exploring": "general",
}


def ui_to_agent_persona(ui: str) -> AgentPersona:
    """Translate a UI label to the agent classifier's persona vocab."""
    if ui not in _UI_TO_AGENT:
        return "general"
    return _UI_TO_AGENT[ui]  # type: ignore[index]


@dataclass
class TurnRecord:
    """One question/answer pair in the user's session history."""

    query: str
    persona: UiPersona
    answer: str = ""
    classification: dict | None = None
    sub_questions: list[str] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    retrieved_chunks: list[dict] = field(default_factory=list)
    iteration_count: int = 0
    elapsed_seconds: float | None = None
    error: str | None = None
    # Memoized follow-up chips for this turn. `None` = not computed yet;
    # a list (possibly empty) = computed. The render path fills this once
    # so the Haiku follow-up call doesn't re-fire on every Streamlit rerun.
    followups: list[str] | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )


def init_session_defaults(session: dict) -> None:
    """Idempotently seed Streamlit's session_state with our defaults.

    Streamlit re-runs the script on every interaction, so the same
    keys get checked many times — this MUST be a no-op when the keys
    are already present.
    """
    session.setdefault("persona", DEFAULT_PERSONA)
    session.setdefault("history", [])
    session.setdefault("input_buffer", "")
    session.setdefault("show_trace", False)


def append_turn(session: dict, turn: TurnRecord) -> None:
    """Push a turn onto the session history (in place)."""
    session.setdefault("history", []).append(turn)


def reset_history(session: dict) -> None:
    """Clear conversation history but keep the persona selection."""
    session["history"] = []


def example_queries_for(persona: UiPersona) -> list[str]:
    """Hand-picked starter queries the user can click in the sidebar.

    Kept short and concrete so the homepage is never empty for a new
    user. Each query maps to a real evidence path the agent can answer:
    Homebuyer/Investor → tool-heavy, Researcher → doc-heavy, Just
    exploring → mixed.
    """
    examples: dict[UiPersona, list[str]] = {
        "Homebuyer": [
            "What's the current RBA cash rate?",
            "How much stamp duty would I pay on a $850,000 home in NSW?",
            "What's a typical monthly repayment on a $700,000 mortgage at today's rates?",
        ],
        "Investor": [
            "What's the rental yield on a $750,000 property earning $620 per week?",
            "How have mortgage rates moved over the past year?",
            "What's the rental vacancy rate in Sydney right now?",
        ],
        "Researcher": [
            "How has the ABS Property Price Index moved in the major capitals over two years?",
            "What does AHURI say about institutional investment in build-to-rent?",
            "How have ABS lending indicators for first home buyers tracked the cash rate?",
        ],
        "Just exploring": [
            "What is the current cash rate and how is it set?",
            "Is it cheaper to rent or buy in Sydney right now?",
            "How are housing approvals trending nationally?",
        ],
    }
    return examples.get(persona, examples["Just exploring"])
