"""Dashboard — market regime, XU100 overview, macro snapshot."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Dashboard · BIST Quant", page_icon="📈", layout="wide")

from app.charts import equity_curve, regime_timeline  # noqa: E402
from app.layout import page_header, render_sidebar  # noqa: E402
from app.services import get_regime_data  # noqa: E402
from app.utils import fmt_num, fmt_pct, load_csv_cached, regime_color, resolve_data_path  # noqa: E402

render_sidebar()
page_header("📈 Dashboard", "Market regime · XU100 overview · Macro snapshot")


# ── helpers ───────────────────────────────────────────────────────────────────

def _series_to_bands(series: dict[str, str]) -> list[dict]:
    """Convert {date_str: regime_str} dict into contiguous band segments."""
    if not series:
        return []
    dates = sorted(series.keys())
    bands: list[dict] = []
    current_regime = series[dates[0]]
    start = dates[0]
    for d in dates[1:]:
        if series[d] != current_regime:
            bands.append({"start": start, "end": d, "regime": current_regime})
            current_regime = series[d]
            start = d
    bands.append({"start": start, "end": dates[-1], "regime": current_regime})
    return bands


def _delta_pct(series: pd.Series, periods: int = 1) -> float | None:
    """Return percentage change over the last *periods* observations."""
    cleaned = series.dropna()
    if len(cleaned) < periods + 1:
        return None
    return (cleaned.iloc[-1] / cleaned.iloc[-(periods + 1)] - 1) * 100


def _xu100_chart(lookback_days: int = 504) -> go.Figure | None:
    """XU100 equity curve with regime band shading."""
    try:
        path = resolve_data_path("xu100_prices.csv")
        df = load_csv_cached(str(path))
        if df.empty:
            return None
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")
        df = df.tail(lookback_days)

        regime_info = get_regime_data()
        bands = _series_to_bands(regime_info.get("series", {}))

        # Only keep bands that overlap with chart window
        chart_start = str(df["Date"].min().date())
        chart_bands = [b for b in bands if b["end"] >= chart_start]

        # Normalise to 100
        base = df["Close"].iloc[0]
        normalised = (df["Close"] / base * 100).tolist()

        return equity_curve(
            dates=df["Date"].dt.strftime("%Y-%m-%d").tolist(),
            values=normalised,
            title="XU100 (normalised, last 2 years)",
            regime_bands=chart_bands,
        )
    except Exception as exc:
        st.warning(f"XU100 chart error: {exc}")
        return None


def _regime_distribution_chart(distribution: dict) -> go.Figure:
    """Donut chart for historical regime distribution."""
    labels = list(distribution.keys())
    values = [distribution[r].get("Count", 0) for r in labels]
    colors = [regime_color(r) for r in labels]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.55,
            marker=dict(colors=colors),
            textinfo="label+percent",
            textfont=dict(size=13),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
        title=dict(text="Historical Regime Distribution", x=0),
        font=dict(color="#e0e0e0"),
    )
    return fig


# ── data loading ──────────────────────────────────────────────────────────────

regime_info = get_regime_data()
current = regime_info.get("current", {})
label = regime_info.get("label", "Unknown")
color = regime_color(label)

# Macro data
usdtry_df = load_csv_cached(str(resolve_data_path("usdtry_data.csv")))
xau_df = load_csv_cached(str(resolve_data_path("xau_try_2013_2026.csv")))


# ── TOP ROW: regime card + 3 macro metrics ────────────────────────────────────

col_regime, col_usdtry, col_gold, col_xu100_price = st.columns([1.6, 1, 1, 1])

# Regime card
with col_regime:
    above_ma = current.get("above_ma")
    alloc = current.get("allocation")
    vol_pct = current.get("vol_percentile")
    real_vol = current.get("realized_vol")
    last_date = current.get("date")
    last_date_str = str(last_date.date()) if hasattr(last_date, "date") else str(last_date or "")

    ma_icon = "📈" if above_ma else "📉"
    alloc_str = f"{alloc * 100:.0f}%" if alloc is not None else "—"

    st.markdown(
        f"""
        <div style="
            border: 1.5px solid {color};
            border-radius: 12px;
            padding: 1.1rem 1.3rem;
            background: rgba(255,255,255,0.03);
        ">
            <div style="font-size:0.75rem; color:#aaa; text-transform:uppercase;
                        letter-spacing:1px; margin-bottom:0.3rem;">Market Regime</div>
            <div style="font-size:2.4rem; font-weight:900; color:{color};
                        line-height:1.1;">{label}</div>
            <div style="font-size:0.82rem; color:#bbb; margin-top:0.6rem; line-height:1.8;">
                {ma_icon} {'Above' if above_ma else 'Below'} 50-day MA &nbsp;|&nbsp;
                📊 Equity allocation: <b style="color:{color};">{alloc_str}</b>
            </div>
            <div style="font-size:0.78rem; color:#888; margin-top:0.2rem; line-height:1.8;">
                ⚡ Realized vol: {f"{real_vol*100:.1f}%" if real_vol else "—"} &nbsp;|&nbsp;
                📐 Vol percentile: {f"{vol_pct:.0f}th" if vol_pct is not None else "—"}<br>
                🗓 As of: {last_date_str}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# USD/TRY
with col_usdtry:
    val, chg = "—", None
    if not usdtry_df.empty:
        s = usdtry_df["USDTRY"].dropna()
        val = fmt_num(s.iloc[-1], 4)
        chg = _delta_pct(s, 1)
    st.metric(
        label="💵 USD/TRY",
        value=val,
        delta=fmt_pct(chg) if chg is not None else None,
        delta_color="inverse",
    )

# Gold (TRY/oz)
with col_gold:
    val, chg = "—", None
    if not xau_df.empty:
        s = xau_df["XAU_TRY"].dropna()
        val = fmt_num(s.iloc[-1] / 1000, 1) + "K"
        chg = _delta_pct(s, 1)
    st.metric(
        label="🥇 Gold (TRY/oz)",
        value=val,
        delta=fmt_pct(chg) if chg is not None else None,
    )

# XU100 last close
with col_xu100_price:
    val, chg = "—", None
    xu100_df = load_csv_cached(str(resolve_data_path("xu100_prices.csv")))
    if not xu100_df.empty:
        s = xu100_df["Close"].dropna()
        val = fmt_num(s.iloc[-1], 0)
        chg = _delta_pct(s, 1)
    st.metric(
        label="📈 XU100",
        value=val,
        delta=fmt_pct(chg) if chg is not None else None,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── MAIN CHART ROW ────────────────────────────────────────────────────────────

chart_col, dist_col = st.columns([3, 1])

with chart_col:
    lookback = st.select_slider(
        "Chart window",
        options=[63, 126, 252, 504, 756, 1260],
        value=504,
        format_func=lambda x: {63: "3M", 126: "6M", 252: "1Y",
                                504: "2Y", 756: "3Y", 1260: "5Y"}[x],
        label_visibility="collapsed",
    )
    fig = _xu100_chart(lookback)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("XU100 price data not available.")

with dist_col:
    dist = regime_info.get("distribution", {})
    if dist:
        st.plotly_chart(_regime_distribution_chart(dist), use_container_width=True)
    else:
        st.info("Regime distribution not available.")

# ── REGIME TIMELINE ───────────────────────────────────────────────────────────

series = regime_info.get("series", {})
if series:
    st.markdown("#### Regime History")
    dates_sorted = sorted(series.keys())
    # Show last 2 years of regime timeline
    dates_sorted = dates_sorted[-504:]
    regimes_sorted = [series[d] for d in dates_sorted]

    try:
        fig_timeline = regime_timeline(dates_sorted, regimes_sorted)
        fig_timeline.update_layout(
            height=180, margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
    except Exception as exc:
        st.warning(f"Timeline chart error: {exc}")

# ── MACRO DETAIL EXPANDER ─────────────────────────────────────────────────────

with st.expander("📊 Macro Detail", expanded=False):
    mc1, mc2 = st.columns(2)

    with mc1:
        st.markdown("**USD/TRY — Last 90 days**")
        if not usdtry_df.empty:
            usdtry_df["Date"] = pd.to_datetime(usdtry_df["Date"], errors="coerce")
            tail90 = usdtry_df.sort_values("Date").tail(90)
            fig_fx = go.Figure()
            fig_fx.add_trace(go.Scatter(
                x=tail90["Date"], y=tail90["USDTRY"],
                mode="lines", line=dict(color="#3498db", width=2),
                name="USD/TRY",
            ))
            fig_fx.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10), height=200,
                font=dict(color="#e0e0e0"),
            )
            st.plotly_chart(fig_fx, use_container_width=True)

    with mc2:
        st.markdown("**Gold (TRY/oz) — Last 90 days**")
        if not xau_df.empty:
            xau_df["Date"] = pd.to_datetime(xau_df["Date"], errors="coerce")
            tail90 = xau_df.sort_values("Date").tail(90)
            fig_gold = go.Figure()
            fig_gold.add_trace(go.Scatter(
                x=tail90["Date"], y=tail90["XAU_TRY"],
                mode="lines", line=dict(color="#f39c12", width=2),
                name="Gold (TRY)",
            ))
            fig_gold.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10), height=200,
                font=dict(color="#e0e0e0"),
            )
            st.plotly_chart(fig_gold, use_container_width=True)

    # Week-over-week changes table
    st.markdown("**Weekly Changes**")
    rows = []
    if not usdtry_df.empty:
        s = usdtry_df["USDTRY"].dropna()
        rows.append({"Asset": "USD/TRY", "Current": fmt_num(s.iloc[-1], 4),
                     "1D Δ": fmt_pct(_delta_pct(s, 1)), "1W Δ": fmt_pct(_delta_pct(s, 5)),
                     "1M Δ": fmt_pct(_delta_pct(s, 21))})
    if not xau_df.empty:
        s = xau_df["XAU_TRY"].dropna()
        rows.append({"Asset": "Gold (TRY/oz)", "Current": fmt_num(s.iloc[-1], 0),
                     "1D Δ": fmt_pct(_delta_pct(s, 1)), "1W Δ": fmt_pct(_delta_pct(s, 5)),
                     "1M Δ": fmt_pct(_delta_pct(s, 21))})
    if not xu100_df.empty:
        s = xu100_df["Close"].dropna()
        rows.append({"Asset": "XU100", "Current": fmt_num(s.iloc[-1], 0),
                     "1D Δ": fmt_pct(_delta_pct(s, 1)), "1W Δ": fmt_pct(_delta_pct(s, 5)),
                     "1M Δ": fmt_pct(_delta_pct(s, 21))})
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("Asset"), use_container_width=True)
