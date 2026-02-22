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
    "bull":     ("#00c97a", "#002a18"),
    "recovery": ("#f0c040", "#2a1e00"),
    "bear":     ("#ff3b5c", "#2a0008"),
    "stress":   ("#b06aff", "#160033"),
    "unknown":  ("#4a5a7a", "#0a0e1a"),
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
    # ── global terminal CSS ───────────────────────────────────────────────────
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Inter:wght@300;400;500;600;700&display=swap');

        /* Base font */
        html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

        /* Monospace for numbers / data */
        [data-testid="stMetricValue"],
        [data-testid="stDataFrame"] td,
        .dataframe td, code, pre {
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* Compress main container */
        .block-container {
            padding-top: 1.2rem !important;
            padding-bottom: 1rem !important;
            max-width: 1440px !important;
        }

        /* Metric tiles */
        [data-testid="stMetric"] {
            background: #111827 !important;
            border: 1px solid #1c2a3a !important;
            border-radius: 2px !important;
            padding: 0.55rem 0.8rem !important;
        }
        [data-testid="stMetricLabel"] p {
            font-size: 0.65rem !important;
            text-transform: uppercase !important;
            letter-spacing: 1.4px !important;
            color: #4a5a7a !important;
            font-weight: 600 !important;
        }
        [data-testid="stMetricValue"] {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 1.35rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.5px !important;
        }
        [data-testid="stMetricDelta"] {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.7rem !important;
        }

        /* Headings */
        h1 { font-size: 1.25rem !important; font-weight: 700 !important; letter-spacing: -0.2px !important; }
        h2 { font-size: 1.05rem !important; font-weight: 600 !important; }
        h3 {
            font-size: 0.68rem !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 1.8px !important;
            color: #4a5a7a !important;
            border-bottom: 1px solid #1c2a3a !important;
            padding-bottom: 0.35rem !important;
            margin-bottom: 0.6rem !important;
        }

        /* Divider */
        hr { border-color: #1c2a3a !important; opacity: 1 !important; margin: 0.7rem 0 !important; }

        /* Tabs */
        [data-testid="stTabs"] [role="tab"] {
            font-size: 0.72rem !important;
            text-transform: uppercase !important;
            letter-spacing: 1.1px !important;
        }

        /* Inputs / selects */
        [data-testid="stNumberInput"] input,
        [data-testid="stTextInput"] input,
        [data-baseweb="select"] * {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.8rem !important;
        }

        /* Expander header */
        [data-testid="stExpander"] summary {
            font-size: 0.72rem !important;
            text-transform: uppercase !important;
            letter-spacing: 1.1px !important;
            color: #4a5a7a !important;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: #07090e !important;
            border-right: 1px solid #1c2a3a !important;
        }
        [data-testid="stSidebarNav"] { display: none !important; }

        /* st.container borders — tighter corners */
        [data-testid="stVerticalBlockBorderWrapper"] > div:first-child {
            border-radius: 3px !important;
            border-color: #1c2a3a !important;
        }

        /* Dataframe */
        [data-testid="stDataFrame"] {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.78rem !important;
        }
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
    """Compact terminal-style page header — single-line topbar with divider."""
    # Strip leading emoji for the label, keep for accent
    import re
    clean = title.strip()
    st.markdown(
        f"""
        <div style="
            display:flex; align-items:baseline; justify-content:space-between;
            border-bottom: 1px solid #1c2a3a;
            padding-bottom: 0.55rem; margin-bottom: 0.9rem;">
            <div>
                <span style="font-size:1.15rem; font-weight:700;
                             letter-spacing:-0.3px; color:#c9d1e0;">{clean}</span>
            </div>
            <div style="text-align:right;">
                <span style="font-size:0.68rem; color:#4a5a7a;
                             letter-spacing:0.8px; text-transform:uppercase;
                             font-family:'JetBrains Mono',monospace;">{subtitle}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


__all__ = ["render_sidebar", "page_header"]
