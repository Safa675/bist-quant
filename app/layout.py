"""
Shared sidebar and page-level layout helpers for the BIST Quant Research Cockpit.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.services import get_regime_classifier, is_realtime_connected
from app.utils import load_csv_cached, resolve_data_path

_VERSION = "v2.0.0"

# ── regime badge colours ──────────────────────────────────────────────────────
_BADGE_STYLES: dict[str, tuple[str, str]] = {
    # regime_lower: (bg_hex, text_hex)
    "bull": ("#2ecc71", "#0a3d1f"),
    "recovery": ("#f39c12", "#3d2600"),
    "bear": ("#e74c3c", "#3d0800"),
    "stress": ("#8e44ad", "#1e0033"),
    "unknown": ("#95a5a6", "#1a1a2e"),
}

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


def _regime_badge_html(label: str) -> str:
    key = label.lower()
    bg, fg = _BADGE_STYLES.get(key, _BADGE_STYLES["unknown"])
    return (
        f'<span style="'
        f"background-color:{bg};"
        f"color:{fg};"
        f"padding:2px 10px;"
        f"border-radius:12px;"
        f"font-weight:700;"
        f"font-size:0.85rem;"
        f'">{label}</span>'
    )


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
    # Hide the auto-generated Streamlit multipage nav (the symbol-less duplicate)
    st.markdown(
        """
        <style>
        [data-testid="stSidebarNav"] { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.sidebar:
        # ── logo ──────────────────────────────────────────────────────────────
        st.markdown(
            """
            <div style="text-align:center; padding: 0.5rem 0 0.25rem 0;">
                <span style="font-size:2.2rem;">📈</span><br>
                <span style="font-size:1.4rem; font-weight:800; letter-spacing:0.5px;">
                    BIST Quant
                </span><br>
                <span style="font-size:0.78rem; color:#aaa; letter-spacing:1px;">
                    RESEARCH COCKPIT
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # ── regime badge ──────────────────────────────────────────────────────
        regime_info = get_regime_classifier()
        label = regime_info.get("label", "Unknown")
        st.markdown("**Market Regime**")
        st.markdown(_regime_badge_html(label), unsafe_allow_html=True)

        st.divider()

        # ── navigation ────────────────────────────────────────────────────────
        st.markdown("**Navigation**")
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
        st.markdown(
            f"""
            <div style="text-align:center; color:#666; font-size:0.72rem;
                        margin-top:1rem; padding-top:0.5rem;
                        border-top:1px solid #333;">
                {_VERSION} &nbsp;·&nbsp; Powered by bist_quant
            </div>
            """,
            unsafe_allow_html=True,
        )


def page_header(title: str, subtitle: str = "") -> None:
    """Render a consistent H1 + optional subtitle + divider at the top of a page."""
    st.title(title)
    if subtitle:
        st.markdown(
            f'<p style="color:#aaa; margin-top:-0.8rem; font-size:1rem;">'
            f"{subtitle}</p>",
            unsafe_allow_html=True,
        )
    st.divider()


__all__ = ["render_sidebar", "page_header"]
