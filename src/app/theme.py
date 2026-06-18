"""CadastreAI visual identity for the Streamlit app.

The design system (`CadastreAI-Design-System`) is a React/CSS kit that
Streamlit can't host directly, so this module translates its *foundations*
into a single injected stylesheet plus a couple of helpers:

- three webfonts — **Source Serif 4** (display + answer prose),
  **Hanken Grotesk** (UI chrome), **JetBrains Mono** (numbers / citations);
- the two-colour brand logic — deep navy ink + amber "AI signal" on warm
  parchment, with the doc/tool split (navy = retrieved sources, amber =
  live tool calls);
- a dark-navy sidebar framing a constrained (760px) reading column;
- citation `[N]` markers rendered as monospace chips, coloured by kind.

Kept side-effect-free and Streamlit-import-light (only `inject_theme`
touches `st`) so the token logic stays unit-testable — same split as
`state.py` vs `streamlit_app.py`.
"""
from __future__ import annotations

import re

from src.app.citations import Citation

# --- Design tokens (mirrors tokens/colors.css in the design system) -------
NAVY_800 = "#1e3a5f"  # brand ink — authority, the land-register heritage
NAVY_600 = "#2d5a8f"
AMBER_600 = "#d97706"  # the "AI signal" — sparing: primary CTA, live data
AMBER_500 = "#f59e0b"
PAPER_50 = "#faf8f4"  # warm parchment surface
PAPER_0 = "#ffffff"  # cards

# Citation chip palette — the product-wide doc/tool colour split.
_CHIP_DOC_BG, _CHIP_DOC_FG = "#e8eef6", NAVY_800  # navy = retrieved documents
_CHIP_TOOL_BG, _CHIP_TOOL_FG = "#fdedd3", "#b45309"  # amber = live tool calls

_FONT_DISPLAY = "'Source Serif 4', Georgia, serif"
_FONT_UI = "'Hanken Grotesk', -apple-system, BlinkMacSystemFont, sans-serif"
_FONT_MONO = "'JetBrains Mono', ui-monospace, 'SF Mono', monospace"

_FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700"
    "&family=Hanken+Grotesk:wght@400;500;600;700"
    "&family=JetBrains+Mono:wght@400;500;600&display=swap');"
)

# Injected stylesheet. Selectors lean on Streamlit's stable data-testids
# (class hashes churn between versions); broad fallbacks keep it resilient.
THEME_CSS = f"""
<style>
{_FONT_IMPORT}

:root {{
  --navy-800: {NAVY_800}; --navy-600: {NAVY_600};
  --amber-600: {AMBER_600}; --amber-500: {AMBER_500};
  --paper-50: {PAPER_50}; --paper-0: {PAPER_0};
  --border-subtle: #ece6da;
  --shadow-sm: 0 1px 2px rgba(20,40,63,0.06), 0 2px 6px rgba(20,40,63,0.06);
  --shadow-focus: 0 0 0 3px rgba(245,158,11,0.45);
  --ease-out: cubic-bezier(0.22, 1, 0.36, 1);
}}

/* --- Base type: UI chrome in Hanken Grotesk -------------------------- */
html, body, .stApp, [class*="st-"], button, input, textarea, select {{
  font-family: {_FONT_UI};
}}
.stApp {{ background: var(--paper-50); }}

/* Display headings + answer prose in Source Serif 4 (research authority) */
h1, h2, h3,
[data-testid="stHeading"], [data-testid="stHeading"] * {{
  font-family: {_FONT_DISPLAY};
  letter-spacing: -0.02em;
  color: var(--navy-800);
}}

/* Constrained reading column for legibility (--content-max 760px). */
.block-container, [data-testid="stMainBlockContainer"] {{
  max-width: 820px;
}}

/* Chat answers read as a considered economist's brief: serif, relaxed. */
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {{
  font-family: {_FONT_DISPLAY};
  font-size: 1.02rem;
  line-height: 1.65;
}}

/* Numbers, code, citation envelopes anchored in mono. */
code, kbd, pre, [data-testid="stMetricValue"], .stDataFrame {{
  font-family: {_FONT_MONO};
}}

/* --- Cards / expanders: white, hairline warm border, navy-tinted shadow */
[data-testid="stChatMessage"],
[data-testid="stExpander"] details {{
  background: var(--paper-0);
  border: 1px solid var(--border-subtle);
  border-radius: 10px;
  box-shadow: var(--shadow-sm);
}}

/* --- Dark navy sidebar: calm frame around the bright reading column --- */
section[data-testid="stSidebar"] {{
  background: var(--navy-800);
}}
section[data-testid="stSidebar"] * {{ color: #e7eef6 !important; }}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{ color: #ffffff !important; }}
/* Example-query / reset buttons: translucent-white hover affordance. */
section[data-testid="stSidebar"] .stButton > button {{
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.12);
  color: #e7eef6;
  text-align: left;
  transition: background 160ms var(--ease-out), transform 120ms var(--ease-out);
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
  background: rgba(255,255,255,0.10);
  border-color: rgba(255,255,255,0.22);
}}

/* --- Buttons: soft 6px radius, measured motion (no bounce). --------- */
.stButton > button {{
  border-radius: 6px;
  font-weight: 600;
  transition: filter 160ms var(--ease-out), transform 120ms var(--ease-out);
}}
.stButton > button:hover {{ filter: brightness(1.06); }}
.stButton > button:active {{ transform: translateY(0.5px); }}
/* Primary CTA = the amber signal (Streamlit reads primaryColor too). */
.stButton > button[kind="primary"],
[data-testid="stBaseButton-primary"] {{
  background: var(--amber-600);
  border-color: var(--amber-600);
  color: #fff;
}}

/* Focus = amber ring everywhere. */
:focus-visible {{ box-shadow: var(--shadow-focus) !important; outline: none !important; }}

@media (prefers-reduced-motion: reduce) {{
  * {{ transition: none !important; animation: none !important; }}
}}

/* --- Citation chips: monospace [N], coloured by source kind. -------- */
.cad-cite {{
  font-family: {_FONT_MONO};
  font-size: 0.72em; font-weight: 600;
  padding: 1px 5px; border-radius: 3px;
  vertical-align: 0.08em; white-space: nowrap;
}}
.cad-cite-doc {{ background: {_CHIP_DOC_BG}; color: {_CHIP_DOC_FG}; }}
.cad-cite-tool {{ background: {_CHIP_TOOL_BG}; color: {_CHIP_TOOL_FG}; }}
</style>
"""

_CITE_RE = re.compile(r"\[(\d+)\]")


def style_citation_markers(text: str, citations: list[Citation]) -> str:
    """Wrap `[N]` markers in monospace chips, coloured by source kind.

    `renumber_answer` has already rewritten the prose so marker `[k]`
    lines up with `citations[k-1]`, mirroring the order the source cards
    render in. We honour the product-wide colour split — navy for
    retrieved documents, amber for live tool calls — and leave any marker
    without a backing citation as navy (the neutral default).

    Returns HTML, so the caller must pass `unsafe_allow_html=True`.
    """

    def repl(m: re.Match[str]) -> str:
        n = int(m.group(1))
        kind = citations[n - 1].kind if 1 <= n <= len(citations) else "doc"
        cls = "cad-cite-tool" if kind == "tool" else "cad-cite-doc"
        return f'<span class="cad-cite {cls}">[{n}]</span>'

    return _CITE_RE.sub(repl, text)


def inject_theme(st) -> None:  # noqa: ANN001 — st module, kept import-light
    """Inject the CadastreAI stylesheet. Call once, right after page config."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)
