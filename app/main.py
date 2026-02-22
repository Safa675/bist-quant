"""
BIST Quant Research Cockpit — entry point.

Launch with:
    streamlit run app/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── make the repo root importable ─────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── streamlit page config (must be first st call) ─────────────────────────────
import streamlit as st  # noqa: E402 — must come after sys.path patch

st.set_page_config(
    page_title="BIST Quant Research Cockpit",
    page_icon="🕯️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── bist_quant availability check ─────────────────────────────────────────────
try:
    import bist_quant  # noqa: F401

    BIST_QUANT_AVAILABLE = True
except ImportError:
    BIST_QUANT_AVAILABLE = False
    st.error(
        "**`bist_quant` not found.**  "
        "Install it from the repo root:\n\n"
        "```bash\npip install -e .[api,services]\n```"
    )

# ── shared helpers ─────────────────────────────────────────────────────────────
from app.layout import render_sidebar  # noqa: E402
from app.services import get_core_service, get_regime_classifier  # noqa: E402
from app.ui import (  # noqa: E402
    ACCENT,
    BG_SURFACE,
    BORDER_DEFAULT,
    PURPLE,
    REGIME_COLORS,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    metric_row,
    nav_card_html,
    page_header,
)
from app.utils import fmt_num, fmt_pct, load_csv_cached, regime_color, resolve_data_path  # noqa: E402
import pandas as pd  # noqa: E402

# ── sidebar ────────────────────────────────────────────────────────────────────
render_sidebar()

# ── hero section ───────────────────────────────────────────────────────────────
page_header(
    "📈 BIST Quant Research Cockpit",
    "Your BIST quantitative research workbench",
)

# ── summary row ───────────────────────────────────────────────────────────────
regime_info = get_regime_classifier()
label = regime_info.get("label", "Unknown")
color = regime_color(label)

signal_count = "—"
if BIST_QUANT_AVAILABLE:
    try:
        core = get_core_service()
        if core is not None:
            signals = core.list_available_signals()
            signal_count = fmt_num(len(signals) if signals else 0, decimals=0)
    except Exception:
        pass

last_date = "—"
try:
    df = load_csv_cached(str(resolve_data_path("xu100_prices.csv")))
    if not df.empty:
        date_col = next(
            (c for c in df.columns if "date" in c.lower()), df.columns[0]
        )
        series = pd.to_datetime(df[date_col], errors="coerce").dropna()
        if not series.empty:
            last_date = str(series.max().date())
except Exception:
    pass

metric_row([
    {"label": "Market Regime", "value": label, "color": color},
    {"label": "Available Signals", "value": signal_count, "color": ACCENT},
    {"label": "Last Data Update", "value": last_date, "color": PURPLE},
])

# ── quick-navigation card grid ─────────────────────────────────────────────────
st.markdown("### Navigate")

_NAV_CARDS = [
    ("📈", "Dashboard", "Market regime, XU100 overview, macro snapshot", "pages/1_Dashboard.py"),
    ("🔄", "Backtest", "Run & compare strategy backtests with full analytics", "pages/2_Backtest.py"),
    ("🧪", "Factor Lab", "Browse, combine, and analyse factor signals", "pages/3_Factor_Lab.py"),
    ("🔧", "Signal Construction", "Build custom signals with orthogonalization", "pages/4_Signal_Construction.py"),
    ("🔍", "Stock Screener", "Filter the BIST universe by fundamentals & signals", "pages/5_Screener.py"),
    ("📊", "Portfolio Analytics", "Deep-dive metrics, Monte Carlo, walk-forward", "pages/6_Analytics.py"),
    ("⚙️", "Optimization", "Parameter sweep heatmaps & strategy optimization", "pages/7_Optimization.py"),
]

import re as _re

rows = [_NAV_CARDS[:4], _NAV_CARDS[4:]]
for row_idx, row in enumerate(rows):
    cols = st.columns(len(row))
    for col, (icon, name, desc, page_path) in zip(cols, row):
        slug = _re.sub(r"^\d+_", "", page_path.split("/")[-1].replace(".py", ""))
        with col:
            st.markdown(
                nav_card_html(icon, name, desc, f"/{slug}"),
                unsafe_allow_html=True,
            )
