"""
Shared sidebar and page-level layout helpers for the BIST Quant Research Cockpit.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.services import get_regime_classifier, is_realtime_connected
from app.ui import (
    inject_global_css,
    regime_badge,
    sidebar_footer,
    sidebar_logo,
    TEXT_MUTED,
    BORDER_DEFAULT,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)
from app.utils import load_csv_cached, resolve_data_path

_VERSION = "v2.0.0"

# ── navigation pages ──────────────────────────────────────────────────────────
_PAGES = [
    ("📈", "Dashboard", "pages/1_Dashboard.py"),
    ("🔄", "Backtest", "pages/2_Backtest.py"),
    ("🧪", "Factor Lab", "pages/3_Factor_Lab.py"),
    ("🔧", "Signal Construction", "pages/4_Signal_Construction.py"),
    ("🔍", "Stock Screener", "pages/5_Screener.py"),
    ("📊", "Portfolio Analytics", "pages/6_Analytics.py"),
    ("⚙️", "Optimization", "pages/7_Optimization.py"),
    ("🏦", "Professional", "pages/8_Professional.py"),
    ("⚖️", "Compliance", "pages/9_Compliance.py"),
    ("🤖", "Agents", "pages/10_Agents.py"),
]


def _last_xu100_date() -> str:
    """Return the last available date in xu100_prices.csv, or 'N/A'."""
    try:
        path = resolve_data_path("xu100_prices.csv")
        df = load_csv_cached(str(path))
        if df.empty:
            return "N/A"
        date_col = next(
            (c for c in df.columns if "date" in c.lower() or "Date" in c),
            df.columns[0],
        )
        series = pd.to_datetime(df[date_col], errors="coerce").dropna()
        if series.empty:
            return "N/A"
        return str(series.max().date())
    except Exception:
        return "N/A"


# ── public API ────────────────────────────────────────────────────────────────


def render_sidebar() -> None:
    """Render the shared sidebar: logo, regime badge, navigation, data status."""
    # Inject global design-system CSS (idempotent)
    inject_global_css()

    with st.sidebar:
        # ── logo ──────────────────────────────────────────────────────────────
        sidebar_logo(_VERSION)

        st.divider()

        # ── regime badge ──────────────────────────────────────────────────────
        regime_info = get_regime_classifier()
        label = regime_info.get("label", "Unknown")
        st.markdown(
            '<div class="bq-section-label">Market Regime</div>',
            unsafe_allow_html=True,
        )
        st.markdown(regime_badge(label), unsafe_allow_html=True)

        st.divider()

        # ── navigation ────────────────────────────────────────────────────────
        st.markdown(
            '<div class="bq-section-label">Navigation</div>',
            unsafe_allow_html=True,
        )
        for icon, name, page_path in _PAGES:
            st.page_link(page_path, label=f"{icon} {name}")

        st.divider()

        # ── data status expander ──────────────────────────────────────────────
        with st.expander("Data Status", expanded=False):
            last_date = _last_xu100_date()
            st.markdown(f"📅 **Last XU100 date:** `{last_date}`")

            connected = is_realtime_connected()
            dot = "🟢" if connected else "⚫"
            status = "Connected" if connected else "Offline"
            st.markdown(f"{dot} **Real-time:** {status}")

        # ── footer ────────────────────────────────────────────────────────────
        sidebar_footer(_VERSION)


def page_header(title: str, subtitle: str = "") -> None:
    """Compact page header with divider — delegates to ui.page_header."""
    from app.ui import page_header as _ui_page_header
    _ui_page_header(title, subtitle)


__all__ = ["render_sidebar", "page_header"]
