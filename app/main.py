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
from app.layout import page_header, render_sidebar  # noqa: E402
from app.services import get_core_service, get_regime_classifier  # noqa: E402
from app.utils import fmt_num, load_csv_cached, regime_color, resolve_data_path  # noqa: E402
import pandas as pd  # noqa: E402

# ── sidebar ────────────────────────────────────────────────────────────────────
render_sidebar()

# ── hero section ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="padding: 1rem 0 0.5rem 0;">
        <h1 style="font-size:2.4rem; font-weight:900; margin-bottom:0;">
            📈 BIST Quant Research Cockpit
        </h1>
        <p style="color:#aaa; font-size:1.1rem; margin-top:0.3rem;">
            Your BIST quantitative research workbench
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

# ── summary row ───────────────────────────────────────────────────────────────
col_regime, col_signals, col_date = st.columns(3)

# — Current regime —
with col_regime:
    regime_info = get_regime_classifier()
    label = regime_info.get("label", "Unknown")
    color = regime_color(label)
    st.markdown(
        f"""
        <div style="border-left: 2px solid {color}; padding-left:0.75rem;">
            <div style="font-size:0.62rem; color:#4a5a7a; text-transform:uppercase;
                        letter-spacing:1.4px; font-family:'JetBrains Mono',monospace;
                        font-weight:600;">Market Regime</div>
            <div style="font-size:1.6rem; font-weight:800; color:{color};
                        font-family:'JetBrains Mono',monospace; letter-spacing:-1px;">
                {label}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# — Available signals —
with col_signals:
    signal_count = "—"
    if BIST_QUANT_AVAILABLE:
        try:
            core = get_core_service()
            if core is not None:
                signals = core.list_available_signals()
                signal_count = fmt_num(len(signals) if signals else 0, decimals=0)
        except Exception:
            pass
    st.markdown(
        f"""
        <div style="border-left: 2px solid #3a9df8; padding-left:0.75rem;">
            <div style="font-size:0.62rem; color:#4a5a7a; text-transform:uppercase;
                        letter-spacing:1.4px; font-family:'JetBrains Mono',monospace;
                        font-weight:600;">Available Signals</div>
            <div style="font-size:1.6rem; font-weight:800; color:#3a9df8;
                        font-family:'JetBrains Mono',monospace; letter-spacing:-1px;">
                {signal_count}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# — Last data update —
with col_date:
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
    st.markdown(
        f"""
        <div style="border-left: 2px solid #b06aff; padding-left:0.75rem;">
            <div style="font-size:0.62rem; color:#4a5a7a; text-transform:uppercase;
                        letter-spacing:1.4px; font-family:'JetBrains Mono',monospace;
                        font-weight:600;">Last Data Update</div>
            <div style="font-size:1.6rem; font-weight:800; color:#b06aff;
                        font-family:'JetBrains Mono',monospace; letter-spacing:-1px;">
                {last_date}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

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

_card_css = """
<style>
a.nav-card-link {
    text-decoration: none !important;
    color: inherit !important;
    display: block;
}
.nav-card {
    border: 1px solid #1c2a3a;
    border-radius: 2px;
    padding: 0.85rem 1rem;
    background: #0d1220;
    transition: border-color 0.15s, background 0.15s;
    cursor: pointer;
}
a.nav-card-link:hover .nav-card {
    border-color: #3a9df8;
    background: #0f172a;
}
.nav-card-icon { font-size: 1.3rem; line-height: 1; margin-bottom: 0.4rem; }
.nav-card-title {
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    font-size: 0.88rem;
    color: #c9d1e0;
    margin: 0.2rem 0 0.15rem 0;
    letter-spacing: -0.1px;
}
.nav-card-desc {
    font-size: 0.73rem;
    color: #4a5a7a;
    line-height: 1.35;
}
</style>
"""
st.markdown(_card_css, unsafe_allow_html=True)

import re as _re

rows = [_NAV_CARDS[:4], _NAV_CARDS[4:]]
for row_idx, row in enumerate(rows):
    if row_idx > 0:
        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
    cols = st.columns(len(row))
    for col, (icon, name, desc, page_path) in zip(cols, row):
        # Derive URL slug: "pages/3_Factor_Lab.py" → "/Factor_Lab"
        slug = _re.sub(r"^\d+_", "", page_path.split("/")[-1].replace(".py", ""))
        with col:
            st.markdown(
                f"""
                <a class="nav-card-link" href="/{slug}" target="_self">
                    <div class="nav-card">
                        <div class="nav-card-icon">{icon}</div>
                        <div class="nav-card-title">{name}</div>
                        <div class="nav-card-desc">{desc}</div>
                    </div>
                </a>
                """,
                unsafe_allow_html=True,
            )
