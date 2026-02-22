"""
Design-system helpers for the BIST Quant Research Cockpit.

Provides:
    - inject_global_css()  — single CSS injection for the entire app
    - page_header()        — standard page header with optional right slot
    - section_card()       — content card with title + description
    - metric_row()         — row of styled metric tiles
    - empty_state()        — empty/placeholder state with CTA
    - badge()              — inline coloured badge
    - PLOTLY_TEMPLATE      — Plotly layout dict matching the design system
    - apply_chart_style()  — apply the template to any go.Figure
"""

from __future__ import annotations

from typing import Any, Callable

import plotly.graph_objects as go
import streamlit as st

# ═══════════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS
# ═══════════════════════════════════════════════════════════════════════════════

# Backgrounds
BG_BASE = "#FFFFFF"
BG_SURFACE = "#FFFFFF"
BG_ELEVATED = "#FFFFFF"

# Borders
BORDER_DEFAULT = "#E0E0E0"
BORDER_MUTED = "#E0E0E0"

# Text
TEXT_PRIMARY = "#111827"
TEXT_SECONDARY = "#6B7280"
TEXT_MUTED = "#9CA3AF"

# Accent
ACCENT = "#10B981"
ACCENT_HOVER = "#059669"

# Semantic
SUCCESS = "#10B981"
WARNING = "#F59E0B"
DANGER = "#EF4444"
PURPLE = "#8B5CF6"
INFO = "#3B82F6"

# Regime mapping (consistent everywhere)
REGIME_COLORS: dict[str, str] = {
    "bull": SUCCESS,
    "recovery": WARNING,
    "bear": DANGER,
    "stress": PURPLE,
    "unknown": TEXT_MUTED,
}

# Typography
FONT_SANS = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
FONT_MONO = "'Geist Mono', 'Roboto Flex', 'Roboto Mono', 'Fira Code', 'SF Mono', monospace"


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS INJECTION
# ═══════════════════════════════════════════════════════════════════════════════

_GLOBAL_CSS = f"""
<style>
/* ── Font imports ──────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Base typography ───────────────────────────────────────────────────────── */
html, body, [class*="css"], .stApp, .stApp > header {{
    font-family: {{FONT_SANS}} !important;
    background-color: {{BG_BASE}} !important;
    color: {{TEXT_PRIMARY}} !important;
    -webkit-font-smoothing: antialiased;
}}

/* ── Main container ────────────────────────────────────────────────────────── */
.block-container {{
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1360px !important;
}}

/* ── Headings ──────────────────────────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6 {{
    font-family: {{FONT_SANS}} !important;
    color: {{TEXT_PRIMARY}} !important;
}}

h1 {{
    font-size: 1.75rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
    margin-bottom: 0.25rem !important;
}}

h2 {{
    font-size: 1.25rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
    margin-top: 1.5rem !important;
}}

h3 {{
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.2px !important;
    color: {{TEXT_MUTED}} !important;
    margin-bottom: 0.75rem !important;
    margin-top: 1.25rem !important;
}}

/* ── Body text ─────────────────────────────────────────────────────────────── */
p, li, span {{
    color: {{TEXT_SECONDARY}} !important;
}}

/* ── Divider ───────────────────────────────────────────────────────────────── */
hr {{
    border-color: {{BORDER_DEFAULT}} !important;
    opacity: 1 !important;
    margin: 1rem 0 !important;
}}

/* ── Metric tiles ──────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {{
    background: {{BG_SURFACE}} !important;
    border: 1px solid {{BORDER_DEFAULT}} !important;
    border-radius: 4px !important;
    padding: 16px 20px !important;
}}

[data-testid="stMetricLabel"] p {{
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    color: {{TEXT_MUTED}} !important;
    font-weight: 600 !important;
}}

[data-testid="stMetricValue"] {{
    font-family: {{FONT_MONO}} !important;
    font-size: 1.35rem !important;
    font-weight: 600 !important;
    color: {{TEXT_PRIMARY}} !important;
    letter-spacing: -0.02em !important;
}}

[data-testid="stMetricDelta"] {{
    font-family: {{FONT_MONO}} !important;
    font-size: 0.75rem !important;
}}

/* ── Monospace elements ────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] td,
.dataframe td, code, pre {{
    font-family: {{FONT_MONO}} !important;
}}

/* ── Sidebar ───────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {{
    background-color: {{BG_BASE}} !important;
    border-right: 1px solid {{BORDER_DEFAULT}} !important;
}}

[data-testid="stSidebar"] * {{
    color: {{TEXT_SECONDARY}} !important;
}}

[data-testid="stSidebarNav"] {{
    display: none !important;
}}

/* Sidebar page links */
[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"],
[data-testid="stSidebar"] .stPageLink a {{
    border-radius: 4px !important;
    padding: 6px 12px !important;
    margin: 2px 0 !important;
    transition: all 0.15s ease !important;
    font-size: 0.85rem !important;
    color: {{TEXT_SECONDARY}} !important;
}}

[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"]:hover,
[data-testid="stSidebar"] .stPageLink a:hover {{
    background: rgba(255, 255, 255, 0.05) !important;
    color: {{TEXT_PRIMARY}} !important;
}}

[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][aria-current="page"],
[data-testid="stSidebar"] .stPageLink a[aria-current="page"] {{
    background: #F4F7F9 !important;
    color: {{TEXT_PRIMARY}} !important;
    border-left: 3px solid {{ACCENT}} !important;
    border-radius: 0 4px 4px 0 !important;
}}

/* ── Cards / Containers with border ────────────────────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"] > div:first-child {{
    border-radius: 4px !important;
    border-color: {{BORDER_DEFAULT}} !important;
    background-color: {{BG_ELEVATED}} !important;
}}

div[data-testid="stVerticalBlock"] > div > div[data-testid="stVerticalBlock"] {{
    background-color: {{BG_ELEVATED}} !important;
    border: 1px solid {{BORDER_DEFAULT}} !important;
    border-radius: 4px !important;
    padding: 1rem !important;
}}

/* ── Buttons ───────────────────────────────────────────────────────────────── */
.stButton > button {{
    border-radius: 4px !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 8px 20px !important;
    transition: all 0.2s ease-in-out !important;
    border: 1px solid {{BORDER_MUTED}} !important;
    background: transparent !important;
    color: {{TEXT_SECONDARY}} !important;
    font-family: {{FONT_MONO}} !important;
}}

.stButton > button:hover {{
    background: rgba(0, 255, 0, 0.1) !important;
    color: {{ACCENT}} !important;
    border-color: {{ACCENT}} !important;
    box-shadow: 0 0 8px rgba(0, 255, 0, 0.4);
}}

/* Primary buttons */
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {{
    background: transparent !important;
    color: {{ACCENT}} !important;
    border: 1px solid {{ACCENT}} !important;
    font-weight: 600 !important;
}}

.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {{
    background: rgba(0, 255, 0, 0.2) !important;
    color: {{ACCENT}} !important;
}}

/* ── Download buttons ──────────────────────────────────────────────────────── */
.stDownloadButton > button {{
    border-radius: 4px !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    border: 1px solid {{BORDER_MUTED}} !important;
    background: transparent !important;
    color: {{TEXT_SECONDARY}} !important;
    font-family: {{FONT_MONO}} !important;
}}

.stDownloadButton > button:hover {{
    background: {{BG_ELEVATED}} !important;
    color: {{TEXT_PRIMARY}} !important;
}}

/* ── Inputs / Selects / Sliders ────────────────────────────────────────────── */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {{
    font-family: {{FONT_MONO}} !important;
    font-size: 0.85rem !important;
    background: {{BG_BASE}} !important;
    border: 1px solid {{BORDER_DEFAULT}} !important;
    border-radius: 4px !important;
    color: {{TEXT_PRIMARY}} !important;
}}

[data-testid="stNumberInput"] input:focus,
[data-testid="stTextInput"] input:focus {{
    border-color: {{ACCENT}} !important;
    box-shadow: 0 0 0 1px {{ACCENT}} !important;
}}

[data-baseweb="select"] {{
    font-family: {{FONT_MONO}} !important;
    font-size: 0.85rem !important;
}}

[data-baseweb="select"] > div {{
    background: {{BG_BASE}} !important;
    border: 1px solid {{BORDER_DEFAULT}} !important;
    border-radius: 4px !important;
}}

[data-baseweb="select"] > div:has(:focus) {{
    border-color: {{ACCENT}} !important;
}}

/* Slider track */
[data-testid="stSlider"] [role="slider"] {{
    background: {{ACCENT}} !important;
}}

/* ── Tabs ──────────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] {{
    border-bottom: 1px solid {{BORDER_DEFAULT}} !important;
}}

[data-testid="stTabs"] [role="tab"] {{
    font-size: 0.8rem !important;
    font-family: {{FONT_MONO}} !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px !important;
    color: {{TEXT_MUTED}} !important;
    padding: 10px 16px !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.15s ease !important;
    text-transform: uppercase !important;
}}

[data-testid="stTabs"] [role="tab"]:hover {{
    color: {{TEXT_SECONDARY}} !important;
}}

[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
    color: {{ACCENT}} !important;
    border-bottom: 2px solid {{ACCENT}} !important;
    font-weight: 600 !important;
}}

/* Remove default Streamlit tab line */
[data-testid="stTabs"] [role="tablist"] {{
    gap: 0 !important;
}}

/* ── Expanders ─────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {{
    border: 1px solid {{BORDER_DEFAULT}} !important;
    background-color: {{BG_ELEVATED}} !important;
    border-radius: 4px !important;
    overflow: hidden;
}}

[data-testid="stExpander"] summary {{
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: {{TEXT_SECONDARY}} !important;
    padding: 12px 16px !important;
    background: {{BG_ELEVATED}} !important;
}}

[data-testid="stExpander"] summary:hover {{
    color: {{TEXT_PRIMARY}} !important;
}}

/* ── Modals / Glassmorphism ────────────────────────────────────────────────── */
div[data-testid="stDialog"] > div {{
    background: rgba(0, 255, 0, 0.1) !important;
    backdrop-filter: blur(4px) !important;
    border: 1px solid {{BORDER_DEFAULT}} !important;
    border-radius: 4px !important;
}}

/* ── DataFrames ────────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {{
    font-family: {{FONT_MONO}} !important;
    font-size: 11px !important;
    border-radius: 4px !important;
    overflow: hidden;
}}

[data-testid="stDataFrame"] table {{
    font-family: {{FONT_MONO}} !important;
    font-size: 11px !important;
}}

[data-testid="stDataFrame"] th {{
    background-color: {{BG_BASE}} !important;
    color: {{TEXT_MUTED}} !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid {{BORDER_DEFAULT}} !important;
}}

[data-testid="stDataFrame"] td {{
    border-bottom: 1px solid rgba(38, 38, 38, 0.5) !important;
}}

/* ── Multiselect ───────────────────────────────────────────────────────────── */
[data-testid="stMultiSelect"] [data-baseweb="tag"] {{
    background: {{BG_ELEVATED}} !important;
    border-radius: 4px !important;
    font-size: 0.8rem !important;
}}

/* ── Alerts ────────────────────────────────────────────────────────────────── */
[data-testid="stAlert"] {{
    border-radius: 4px !important;
    font-size: 0.85rem !important;
    font-family: {{FONT_MONO}} !important;
}}

/* ── Status widget ─────────────────────────────────────────────────────────── */
[data-testid="stStatusWidget"] {{
    border-radius: 4px !important;
}}

/* ── Custom utilities ──────────────────────────────────────────────────────── */
.bq-page-header {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    border-bottom: 1px solid {{BORDER_DEFAULT}};
    padding-bottom: 16px;
    margin-bottom: 24px;
}}

.bq-page-header .bq-title {{
    font-size: 1.75rem;
    font-weight: 600;
    color: {{TEXT_PRIMARY}};
    letter-spacing: -0.02em;
    margin: 0;
}}

.bq-page-header .bq-subtitle {{
    font-size: 0.8rem;
    color: {{TEXT_MUTED}};
    margin-top: 4px;
}}

.bq-page-header .bq-right {{
    text-align: right;
}}

.bq-section-label {{
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: {{TEXT_MUTED}};
    margin-bottom: 12px;
    margin-top: 0;
}}

.bq-card {{
    background: {{BG_ELEVATED}};
    border: 1px solid {{BORDER_DEFAULT}};
    border-radius: 4px;
    padding: 20px;
    margin-bottom: 16px;
}}

.bq-card .bq-card-title {{
    font-size: 0.875rem;
    font-weight: 600;
    color: {{TEXT_PRIMARY}};
    margin: 0 0 4px 0;
}}

.bq-card .bq-card-desc {{
    font-size: 0.8rem;
    color: {{TEXT_MUTED}};
    margin: 0 0 16px 0;
}}

.bq-metric-strip {{
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
}}

.bq-metric-tile {{
    background: {{BG_ELEVATED}};
    border: 1px solid {{BORDER_DEFAULT}};
    border-radius: 4px;
    padding: 16px 20px;
    flex: 1;
    min-width: 140px;
}}

.bq-metric-tile.bq-accent-left {{
    border-left: 3px solid var(--accent-color, {{ACCENT}});
}}

.bq-metric-label {{
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: {{TEXT_MUTED}};
    margin-bottom: 6px;
}}

.bq-metric-value {{
    font-family: {{FONT_MONO}};
    font-size: 1.5rem;
    font-weight: 600;
    color: {{TEXT_PRIMARY}};
    letter-spacing: -0.02em;
    line-height: 1.2;
}}

.bq-metric-delta {{
    font-family: {{FONT_MONO}};
    font-size: 0.75rem;
    font-weight: 500;
    margin-top: 4px;
}}

.bq-metric-delta.positive {{ color: {{SUCCESS}}; }}
.bq-metric-delta.negative {{ color: {{DANGER}}; }}

.bq-empty-state {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 48px 24px;
    text-align: center;
    border: 1px dashed {{BORDER_MUTED}};
    border-radius: 4px;
    background: {{BG_BASE}};
    margin-bottom: 16px;
}}

.bq-empty-icon {{
    font-size: 2rem;
    margin-bottom: 12px;
    color: {{TEXT_MUTED}};
    opacity: 0.8;
}}

.bq-empty-title {{
    font-size: 1rem;
    font-weight: 600;
    color: {{TEXT_SECONDARY}};
    margin-bottom: 8px;
}}

.bq-empty-hint {{
    font-size: 0.85rem;
    color: {{TEXT_MUTED}};
    max-width: 360px;
    line-height: 1.5;
}}

.bq-badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-family: {{FONT_MONO}};
    font-weight: 600;
    line-height: 1.4;
    letter-spacing: 0.3px;
    text-transform: uppercase;
    border: 1px solid rgba(255,255,255,0.1);
}}

.bq-badge-success {{ background: rgba(0, 255, 0, 0.15); color: {{SUCCESS}}; border-color: rgba(0, 255, 0, 0.2); }}
.bq-badge-warning {{ background: rgba(245, 158, 11, 0.15); color: {{WARNING}}; border-color: rgba(245, 158, 11, 0.2); }}
.bq-badge-danger  {{ background: rgba(255, 49, 49, 0.15);  color: {{DANGER}}; border-color: rgba(255, 49, 49, 0.2); }}
.bq-badge-info    {{ background: rgba(59, 130, 246, 0.15);   color: {{INFO}}; border-color: rgba(59, 130, 246, 0.2); }}
.bq-badge-purple  {{ background: rgba(139, 92, 246, 0.15);  color: {{PURPLE}}; border-color: rgba(139, 92, 246, 0.2); }}
.bq-badge-neutral {{ background: rgba(100, 116, 139, 0.15); color: {{TEXT_MUTED}}; border-color: rgba(100, 116, 139, 0.2); }}

/* Regime badge (pill) */
.bq-regime-badge {{
    display: inline-block;
    padding: 3px 12px;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 700;
    font-family: {{FONT_MONO}};
    text-transform: uppercase;
}}

/* Nav card grid */
.bq-nav-card {{
    display: block;
    text-decoration: none !important;
    color: inherit !important;
}}

.bq-nav-card .bq-card {{
    transition: box-shadow 0.2s ease, border-color 0.2s ease, background 0.2s ease;
    cursor: pointer;
    background: {{BG_BASE}};
}}

.bq-nav-card:hover .bq-card {{
    border-color: {{ACCENT}};
    background: {{BG_ELEVATED}};
    box-shadow: 0 0 8px rgba(0, 255, 0, 0.2);
}}

.bq-nav-card .bq-card-icon {{
    font-size: 1.5rem;
    margin-bottom: 8px;
    color: {{ACCENT}};
}}

.bq-nav-card .bq-card-title {{
    font-size: 0.9rem;
    font-weight: 600;
    color: {{TEXT_PRIMARY}};
    margin: 4px 0;
}}

.bq-nav-card .bq-card-desc {{
    font-size: 0.78rem;
    color: {{TEXT_MUTED}};
    line-height: 1.45;
    margin: 0;
}}

/* Sidebar logo */
.bq-sidebar-logo {{
    text-align: center;
    padding: 8px 0 4px 0;
}}

.bq-sidebar-logo .bq-logo-icon {{
    font-size: 2rem;
    line-height: 1;
    color: {{ACCENT}};
}}

.bq-sidebar-logo .bq-logo-title {{
    font-size: 1.25rem;
    font-weight: 700;
    color: {{TEXT_PRIMARY}};
    letter-spacing: 0.3px;
    font-family: {{FONT_MONO}};
    text-transform: uppercase;
}}

.bq-sidebar-logo .bq-logo-sub {{
    font-size: 0.7rem;
    color: {{TEXT_MUTED}};
    letter-spacing: 1.5px;
    text-transform: uppercase;
}}

.bq-sidebar-footer {{
    text-align: center;
    color: {{TEXT_MUTED}};
    font-size: 0.72rem;
    margin-top: 1rem;
    padding-top: 0.5rem;
    border-top: 1px solid {{BORDER_DEFAULT}};
    font-family: {{FONT_MONO}};
}}

/* Config panel (for Backtest/Optimization) */
.bq-config-panel {{
    background: {{BG_ELEVATED}};
    border: 1px solid {{BORDER_DEFAULT}};
    border-radius: 4px;
    padding: 24px;
}}

.bq-config-panel h3 {{
    margin-top: 0 !important;
}}

/* Stat quick badge (Factor Lab) */
.bq-stat-row {{
    display: flex;
    gap: 12px;
    font-family: {{FONT_MONO}};
    font-size: 0.78rem;
    background: rgba(0, 255, 0, 0.05);
    border: 1px solid rgba(0, 255, 0, 0.15);
    border-radius: 4px;
    padding: 6px 10px;
    margin: 6px 0;
}}

.bq-stat-row span {{
    color: {{TEXT_SECONDARY}} !important;
}}

.bq-stat-row b {{
    color: {{TEXT_PRIMARY}} !important;
}}

/* Scrollbar */
::-webkit-scrollbar {{
    width: 4px;
    height: 4px;
}}
::-webkit-scrollbar-track {{
    background: transparent;
}}
::-webkit-scrollbar-thumb {{
    background: {{BORDER_DEFAULT}};
    border-radius: 4px;
}}
::-webkit-scrollbar-thumb:hover {{
    background: {{TEXT_MUTED}};
}}
</style>
"""


def inject_global_css() -> None:
    """Inject the global design-system CSS. Call once per page, typically inside render_sidebar()."""
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ═══════════════════════════════════════════════════════════════════════════════


def page_header(
    title: str,
    subtitle: str = "",
    right_slot: str = "",
) -> None:
    """
    Render a standardised page header.

    Args:
        title: Page title (may include leading emoji).
        subtitle: Gray subtitle text below the title.
        right_slot: Optional HTML to render in the right-aligned slot
                    (e.g. a date string, a badge, or a status indicator).
    """
    right_html = f'<div class="bq-right">{right_slot}</div>' if right_slot else ""
    st.markdown(
        f'<div class="bq-page-header">'
        f'<div><div class="bq-title">{title}</div><div class="bq-subtitle">{subtitle}</div></div>'
        f'{right_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION CARD
# ═══════════════════════════════════════════════════════════════════════════════


def section_card(
    title: str,
    description: str = "",
    *,
    content_fn: Callable[[], None] | None = None,
) -> None:
    """
    Render a titled card section. If *content_fn* is provided, it is called
    inside an ``st.container`` that visually sits within the card.

    For simple cases, use `with section_card_container(...)` instead.

    Args:
        title: Bold card title.
        description: Optional gray description below the title.
        content_fn: Callable that renders Streamlit content inside the card.
    """
    desc_html = f'<p class="bq-card-desc">{description}</p>' if description else ""
    st.markdown(
        f"""<div class="bq-card">
            <p class="bq-card-title">{title}</p>
            {desc_html}
        </div>""",
        unsafe_allow_html=True,
    )
    if content_fn is not None:
        content_fn()


def section_card_container(title: str, description: str = "") -> Any:
    """
    Return an ``st.container(border=True)`` with a card header already rendered.

    Usage::

        with section_card_container("My Section", "Some description"):
            st.write("Content goes here")
    """
    st.markdown(
        f'<div class="bq-section-label">{title}</div>'
        + (f'<div style="font-size:0.8rem;color:{TEXT_MUTED};margin:-8px 0 12px;">{description}</div>' if description else ""),
        unsafe_allow_html=True,
    )
    return st.container(border=True)


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC ROW
# ═══════════════════════════════════════════════════════════════════════════════


def metric_row(
    metrics: list[dict[str, str]],
) -> None:
    """
    Render a horizontal row of styled metric tiles using pure HTML.

    Each item in *metrics* should have keys:
        - ``label``: Uppercase label
        - ``value``: Large display value
        - ``delta`` (optional): Delta text like "+3.2%"
        - ``color`` (optional): Accent color for the left border

    Usage::

        metric_row([
            {"label": "CAGR", "value": "+18.4%", "delta": "+3.2%", "color": "#10B981"},
            {"label": "Sharpe", "value": "1.42"},
            {"label": "Max DD", "value": "-12.3%", "color": "#EF4444"},
        ])
    """
    tiles_html = ""
    for m in metrics:
        accent = m.get("color", "")
        accent_class = "bq-accent-left" if accent else ""
        accent_style = f'style="--accent-color: {accent};"' if accent else ""

        delta = m.get("delta", "")
        if delta:
            delta_class = "positive" if not delta.startswith("-") else "negative"
            delta_html = f'<div class="bq-metric-delta {delta_class}">{delta}</div>'
        else:
            delta_html = ""

        tiles_html += f"""<div class="bq-metric-tile {accent_class}" {accent_style}>
    <div class="bq-metric-label">{m["label"]}</div>
    <div class="bq-metric-value">{m["value"]}</div>
    {delta_html}
</div>"""

    # Try st.html if available (Streamlit >= 1.34), fallback to st.markdown
    html_payload = f'<div class="bq-metric-strip">{tiles_html}</div>'
    if hasattr(st, "html"):
        st.html(html_payload)
    else:
        st.markdown(html_payload, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# EMPTY STATE
# ═══════════════════════════════════════════════════════════════════════════════


def empty_state(
    icon: str = "📊",
    title: str = "No data yet",
    hint: str = "",
    action_label: str | None = None,
    action_key: str = "empty_action",
) -> bool:
    """
    Render a centered empty-state placeholder with optional CTA button.

    Returns True if the action button was clicked (or False if no button).
    """
    hint_html = f'<div class="bq-empty-hint">{hint}</div>' if hint else ""
    st.markdown(
        f"""<div class="bq-empty-state">
    <div class="bq-empty-icon">{icon}</div>
    <div class="bq-empty-title">{title}</div>
    {hint_html}
</div>""",
        unsafe_allow_html=True,
    )
    if action_label:
        _, center, _ = st.columns([1, 1, 1])
        with center:
            return st.button(action_label, key=action_key, type="primary", use_container_width=True)
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# BADGE
# ═══════════════════════════════════════════════════════════════════════════════


def badge(
    text: str,
    variant: str = "neutral",
) -> str:
    """
    Return an HTML string for an inline badge.

    Variants: ``success``, ``warning``, ``danger``, ``info``, ``purple``, ``neutral``.
    """
    return f'<span class="bq-badge bq-badge-{variant}">{text}</span>'


def regime_badge(label: str) -> str:
    """Return an HTML string for a regime pill badge."""
    key = label.lower()
    color = REGIME_COLORS.get(key, TEXT_MUTED)
    # Determine foreground
    fg_map = {
        "bull": "#002a18",
        "recovery": "#2a1e00",
        "bear": "#2a0008",
        "stress": "#160033",
    }
    fg = fg_map.get(key, BG_BASE)
    return (
        f'<span class="bq-regime-badge" '
        f'style="background-color:{color};color:{fg};">'
        f'{label}</span>'
    )


# ═══════════════════════════════════════════════════════════════════════════════
# NAV CARD (for home page)
# ═══════════════════════════════════════════════════════════════════════════════


def nav_card_html(icon: str, title: str, description: str, url: str) -> str:
    """Return HTML for a clickable navigation card."""
    return f"""<a class="bq-nav-card" href="{url}" target="_self">
    <div class="bq-card">
        <div class="bq-card-icon">{icon}</div>
        <div class="bq-card-title">{title}</div>
        <div class="bq-card-desc">{description}</div>
    </div>
</a>"""


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTLY CHART THEME
# ═══════════════════════════════════════════════════════════════════════════════

PLOTLY_TEMPLATE: dict[str, Any] = dict(
    template="plotly_white",
    margin=dict(l=16, r=16, t=40, b=16),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=FONT_SANS, color=TEXT_SECONDARY, size=12),
    title_font=dict(family=FONT_SANS, color=TEXT_PRIMARY, size=14, weight=600),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_SANS, size=11, color=TEXT_SECONDARY),
        borderwidth=0,
    ),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor=BG_SURFACE,
        bordercolor=BORDER_DEFAULT,
        font=dict(family=FONT_MONO, size=11, color=TEXT_PRIMARY),
    ),
    colorway=[
        ACCENT,       # Blue
        SUCCESS,      # Green
        WARNING,      # Amber
        PURPLE,       # Purple
        "#EC4899",    # Pink
        "#06B6D4",    # Cyan
        "#F97316",    # Orange
        "#84CC16",    # Lime
    ],
)

_AXIS_STYLE = dict(
    showgrid=True,
    gridcolor="#F4F7F9",
    zeroline=False,
    linecolor=BORDER_DEFAULT,
    tickfont=dict(family=FONT_MONO, size=10, color=TEXT_MUTED),
)


def apply_chart_style(fig: go.Figure, height: int | None = None) -> go.Figure:
    """
    Apply the design-system Plotly template to an existing figure.

    Args:
        fig: Plotly Figure to restyle.
        height: Optional fixed height in pixels.

    Returns:
        The same Figure with updated layout (mutated in place).
    """
    layout_update = dict(PLOTLY_TEMPLATE)
    if height is not None:
        layout_update["height"] = height
    fig.update_layout(**layout_update)
    fig.update_xaxes(**_AXIS_STYLE)
    fig.update_yaxes(**_AXIS_STYLE)
    return fig


def base_fig(title: str = "", height: int | None = None) -> go.Figure:
    """Create a new empty Figure pre-styled with the design-system theme."""
    fig = go.Figure()
    layout = dict(PLOTLY_TEMPLATE, title=dict(text=title, x=0))
    if height is not None:
        layout["height"] = height
    fig.update_layout(**layout)
    fig.update_xaxes(**_AXIS_STYLE)
    fig.update_yaxes(**_AXIS_STYLE)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def sidebar_logo(version: str = "v2.0.0") -> None:
    """Render the sidebar logo block."""
    st.markdown(
        f"""
        <div class="bq-sidebar-logo">
            <div class="bq-logo-icon">📈</div>
            <div class="bq-logo-title">BIST Quant</div>
            <div class="bq-logo-sub">Research Cockpit</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_footer(version: str = "v2.0.0") -> None:
    """Render the sidebar footer."""
    st.markdown(
        f'<div class="bq-sidebar-footer">{version} &nbsp;·&nbsp; Powered by bist_quant</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # CSS
    "inject_global_css",
    # Layout components
    "page_header",
    "section_card",
    "section_card_container",
    "metric_row",
    "empty_state",
    "badge",
    "regime_badge",
    "nav_card_html",
    # Sidebar
    "sidebar_logo",
    "sidebar_footer",
    # Chart
    "PLOTLY_TEMPLATE",
    "apply_chart_style",
    "base_fig",
    # Tokens
    "BG_BASE",
    "BG_SURFACE",
    "BG_ELEVATED",
    "BORDER_DEFAULT",
    "BORDER_MUTED",
    "TEXT_PRIMARY",
    "TEXT_SECONDARY",
    "TEXT_MUTED",
    "ACCENT",
    "ACCENT_HOVER",
    "SUCCESS",
    "WARNING",
    "DANGER",
    "PURPLE",
    "INFO",
    "REGIME_COLORS",
    "FONT_SANS",
    "FONT_MONO",
]
