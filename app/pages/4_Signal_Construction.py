"""Signal Construction — build custom signals with orthogonalization."""

from __future__ import annotations

import sys
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Signal Construction · BIST Quant", page_icon="🔧", layout="wide"
)

from app.layout import page_header, render_sidebar  # noqa: E402
from app.utils import fmt_num, fmt_pct, run_in_thread  # noqa: E402

render_sidebar()
page_header("🔧 Signal Construction", "Build custom multi-indicator signals with orthogonalization")

# ── constants ─────────────────────────────────────────────────────────────────
INDICATORS_META: list[dict[str, Any]] = [
    {"key": "rsi",        "label": "RSI",                  "icon": "📊", "category": "Oscillator"},
    {"key": "macd",       "label": "MACD Histogram",        "icon": "📈", "category": "Trend"},
    {"key": "bollinger",  "label": "Bollinger %B",          "icon": "📉", "category": "Volatility"},
    {"key": "atr",        "label": "ATR (Cross-Sectional)", "icon": "📐", "category": "Volatility"},
    {"key": "stochastic", "label": "Stochastic %K",         "icon": "🎯", "category": "Oscillator"},
    {"key": "adx",        "label": "ADX Trend",             "icon": "↗️",  "category": "Trend"},
    {"key": "supertrend", "label": "Supertrend Direction",  "icon": "🌊", "category": "Trend"},
    {"key": "ta_consensus", "label": "TradingView Consensus", "icon": "🤝", "category": "Consensus"},
]

INDICATOR_DEFAULTS: dict[str, dict[str, Any]] = {
    "rsi":        {"period": 14, "oversold": 30.0, "overbought": 70.0},
    "macd":       {"fast": 12, "slow": 26, "signal": 9, "threshold": 0.0},
    "bollinger":  {"period": 20, "std_dev": 2.0, "lower": 0.2, "upper": 0.8},
    "atr":        {"period": 14, "lower_pct": 0.3, "upper_pct": 0.7},
    "stochastic": {"k_period": 14, "d_period": 3, "oversold": 20.0, "overbought": 80.0},
    "adx":        {"period": 14, "trend_threshold": 25.0},
    "supertrend": {"period": 10, "multiplier": 3.0},
    "ta_consensus": {"interval": "1d", "batch_size": 20},
}

# 13 five-factor axes (from five_factor_pipeline.py docs)
FIVE_FACTOR_AXES: list[dict[str, Any]] = [
    {"key": "size",             "label": "Size",                "icon": "📏", "desc": "Small cap vs Large cap premium"},
    {"key": "value",            "label": "Value",               "icon": "💎", "desc": "Value level vs Value growth (E/P, FCF/P, S/P)"},
    {"key": "profitability",    "label": "Profitability",       "icon": "💰", "desc": "High margin vs Future profitability"},
    {"key": "investment",       "label": "Investment",          "icon": "🏗",  "desc": "Conservative vs Reinvestment style"},
    {"key": "momentum",         "label": "Momentum",            "icon": "🚀", "desc": "12-1 month winner vs loser"},
    {"key": "risk",             "label": "Risk",                "icon": "⚠️",  "desc": "Low volatility vs High beta"},
    {"key": "quality",          "label": "Quality",             "icon": "🏆", "desc": "ROE, ROA, Piotroski F-score, accruals"},
    {"key": "liquidity",        "label": "Liquidity",           "icon": "💧", "desc": "Amihud illiquidity & real turnover"},
    {"key": "trading_intensity","label": "Trading Intensity",   "icon": "⚡", "desc": "Relative volume & activity"},
    {"key": "sentiment",        "label": "Sentiment",           "icon": "💬", "desc": "52w high proximity & price acceleration"},
    {"key": "fundmom",          "label": "Fundamental Momentum","icon": "📊", "desc": "Margin change & sales acceleration"},
    {"key": "carry",            "label": "Carry",               "icon": "🎁", "desc": "Dividend yield & shareholder yield"},
    {"key": "defensive",        "label": "Defensive",           "icon": "🛡",  "desc": "Earnings stability & low-beta profile"},
]

# category → color
AXIS_CATEGORY_COLORS: dict[str, str] = {
    "Original (FF5)": "#3498db",
    "Extended Axes":  "#2ecc71",
}

# ── session state ─────────────────────────────────────────────────────────────
for _k, _v in [
    ("sc_enabled_indicators", {"rsi": True, "macd": True, "supertrend": True}),
    ("sc_indicator_params",   {}),
    ("sc_universe",           "XU100"),
    ("sc_period",             "6mo"),
    ("sc_top_n",              20),
    ("sc_buy_threshold",      0.2),
    ("sc_sell_threshold",     -0.2),
    ("sc_orth_enabled",       False),
    ("sc_orth_axes",          ["momentum", "value", "quality", "size"]),
    ("sc_snap_future",        None),
    ("sc_snap_result",        None),
    ("sc_bt_future",          None),
    ("sc_bt_result",          None),
    ("sc_ff_axes_enabled",    {ax["key"]: True for ax in FIVE_FACTOR_AXES}),
    ("sc_ff_axes_weights",    {ax["key"]: 1.0 for ax in FIVE_FACTOR_AXES}),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─────────────────────────────────────────────────────────────────────────────
# TAB NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
tab_builder, tab_5f, tab_orth = st.tabs([
    "🔧 Indicator Builder",
    "🌐 Five-Factor Pipeline",
    "⊥ Orthogonalization",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Indicator Builder
# ═══════════════════════════════════════════════════════════════════════════════
with tab_builder:
    st.markdown("### Select & Configure Indicators")
    st.caption(
        "Toggle indicators on/off, tune their parameters, then run the signal "
        "snapshot to see cross-sectional scores or a full backtest."
    )

    # Indicator toggles + param editors ───────────────────────────────────────
    ind_cols = st.columns(2)
    for i, ind in enumerate(INDICATORS_META):
        key = ind["key"]
        with ind_cols[i % 2]:
            with st.container(border=True):
                hdr_c1, hdr_c2 = st.columns([4, 1])
                with hdr_c1:
                    enabled = st.toggle(
                        f"{ind['icon']} **{ind['label']}**",
                        value=st.session_state["sc_enabled_indicators"].get(key, False),
                        key=f"sc_toggle_{key}",
                    )
                    st.session_state["sc_enabled_indicators"][key] = enabled
                with hdr_c2:
                    cat_color = {"Oscillator": "#3498db", "Trend": "#2ecc71",
                                 "Volatility": "#f39c12", "Consensus": "#9b59b6"}.get(ind["category"], "#7f8c8d")
                    st.markdown(
                        f"<div style='text-align:right;'>"
                        f"<span style='background:{cat_color};color:#fff;border-radius:8px;"
                        f"padding:1px 6px;font-size:0.7rem;'>{ind['category']}</span></div>",
                        unsafe_allow_html=True,
                    )

                if enabled:
                    defaults = INDICATOR_DEFAULTS.get(key, {})
                    user_params = st.session_state["sc_indicator_params"].get(key, {})
                    merged = {**defaults, **user_params}

                    param_out: dict[str, Any] = {}
                    p_cols = st.columns(min(len(defaults), 3))

                    for pi, (pk, pv_default) in enumerate(defaults.items()):
                        pv_current = merged.get(pk, pv_default)
                        with p_cols[pi % len(p_cols)]:
                            if key == "ta_consensus" and pk == "interval":
                                param_out[pk] = st.selectbox(
                                    pk, ["1d", "1W", "1M"], index=["1d","1W","1M"].index(str(pv_current)) if str(pv_current) in ["1d","1W","1M"] else 0,
                                    key=f"sc_param_{key}_{pk}",
                                )
                            elif isinstance(pv_default, float):
                                lo, hi = (0.0, 1.0) if pv_default <= 1.0 else (0.0, float(pv_default) * 5)
                                param_out[pk] = st.number_input(
                                    pk,
                                    min_value=lo,
                                    max_value=hi if hi > 0 else 100.0,
                                    value=float(pv_current),
                                    step=0.5 if hi > 2 else 0.05,
                                    key=f"sc_param_{key}_{pk}",
                                )
                            else:
                                param_out[pk] = st.number_input(
                                    pk,
                                    min_value=1,
                                    max_value=500,
                                    value=int(pv_current),
                                    step=1,
                                    key=f"sc_param_{key}_{pk}",
                                )
                    st.session_state["sc_indicator_params"][key] = param_out

    enabled_indicators = [k for k, v in st.session_state["sc_enabled_indicators"].items() if v]

    st.markdown("---")
    st.markdown("### Universe & Execution Settings")
    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        universe = st.selectbox("Universe", ["XU100", "XU030", "XUTUM"], key="sc_universe_sel")
        st.session_state["sc_universe"] = universe
    with s2:
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2, key="sc_period_sel")
        st.session_state["sc_period"] = period
    with s3:
        top_n = st.number_input("Top-N stocks", min_value=5, max_value=100, value=st.session_state["sc_top_n"], key="sc_topn")
        st.session_state["sc_top_n"] = int(top_n)
    with s4:
        buy_thr = st.number_input("Buy threshold", min_value=-1.0, max_value=1.0, value=st.session_state["sc_buy_threshold"], step=0.05, key="sc_buy_thr")
        st.session_state["sc_buy_threshold"] = float(buy_thr)
    with s5:
        sell_thr = st.number_input("Sell threshold", min_value=-1.0, max_value=1.0, value=st.session_state["sc_sell_threshold"], step=0.05, key="sc_sell_thr")
        st.session_state["sc_sell_threshold"] = float(sell_thr)

    if not enabled_indicators:
        st.warning("⚠️ No indicators enabled. Toggle at least one indicator above.")

    st.markdown("---")
    rb1, rb2, rb3 = st.columns([1, 1, 3])
    with rb1:
        snap_btn = st.button(
            "📸 Run Snapshot",
            use_container_width=True,
            disabled=not enabled_indicators,
            help="Preview cross-sectional scores for the latest date",
            key="sc_snap_btn",
        )
    with rb2:
        bt_btn = st.button(
            "📈 Backtest",
            use_container_width=True,
            disabled=not enabled_indicators,
            help="Run a full backtest with equity curve",
            key="sc_bt_btn",
        )
    with rb3:
        if not enabled_indicators:
            st.caption("_Enable at least one indicator to run snapshot or backtest._")
        else:
            ind_summary = " · ".join(
                f"{INDICATORS_META[next(j for j, m in enumerate(INDICATORS_META) if m['key'] == k)]['icon']} {k}"
                for k in enabled_indicators
                if any(m["key"] == k for m in INDICATORS_META)
            )
            st.caption(f"Active indicators: **{ind_summary}**")

    # Build payload ────────────────────────────────────────────────────────────
    def _build_sc_payload() -> dict[str, Any]:
        indicators_payload: dict[str, Any] = {}
        for ind_key, is_enabled in st.session_state["sc_enabled_indicators"].items():
            indicators_payload[ind_key] = {
                "enabled": bool(is_enabled),
                "params": st.session_state["sc_indicator_params"].get(ind_key, {}),
            }
        return {
            "universe": st.session_state["sc_universe"],
            "period": st.session_state["sc_period"],
            "interval": "1d",
            "top_n": st.session_state["sc_top_n"],
            "max_symbols": 100,
            "buy_threshold": st.session_state["sc_buy_threshold"],
            "sell_threshold": st.session_state["sc_sell_threshold"],
            "indicators": indicators_payload,
        }

    # Trigger runs ────────────────────────────────────────────────────────────
    if snap_btn:
        try:
            from bist_quant.engines.signal_construction import run_signal_snapshot
            st.session_state["sc_snap_result"] = None
            st.session_state["sc_snap_future"] = run_in_thread(run_signal_snapshot, _build_sc_payload())
            st.rerun()
        except ImportError as exc:
            st.error(f"signal_construction engine unavailable: {exc}")

    if bt_btn:
        try:
            from bist_quant.engines.signal_construction import run_signal_backtest
            st.session_state["sc_bt_result"] = None
            st.session_state["sc_bt_future"] = run_in_thread(run_signal_backtest, _build_sc_payload())
            st.rerun()
        except ImportError as exc:
            st.error(f"signal_construction engine unavailable: {exc}")

    # Poll futures ─────────────────────────────────────────────────────────────
    _sf = st.session_state.get("sc_snap_future")
    if _sf is not None and not _sf.done():
        with st.spinner("Running signal snapshot… ⏳"):
            time.sleep(0.5)
            st.rerun()
    elif _sf is not None and _sf.done():
        try:
            st.session_state["sc_snap_result"] = _sf.result()
        except Exception as exc:
            st.session_state["sc_snap_result"] = {"error": str(exc)}
        st.session_state["sc_snap_future"] = None
        st.rerun()

    _btf = st.session_state.get("sc_bt_future")
    if _btf is not None and not _btf.done():
        with st.spinner("Running indicator backtest… ⏳"):
            time.sleep(0.5)
            st.rerun()
    elif _btf is not None and _btf.done():
        try:
            st.session_state["sc_bt_result"] = _btf.result()
        except Exception as exc:
            st.session_state["sc_bt_result"] = {"error": str(exc)}
        st.session_state["sc_bt_future"] = None
        st.rerun()

    # ─── Render Results ───────────────────────────────────────────────────────
    snap_res = st.session_state.get("sc_snap_result")
    bt_res = st.session_state.get("sc_bt_result")

    if snap_res is not None:
        st.markdown("---")
        if snap_res.get("error"):
            st.error(f"Snapshot error: {snap_res['error']}")
        else:
            _meta = snap_res.get("meta", {})
            _signals = snap_res.get("signals", [])
            _ind_sum = snap_res.get("indicator_summaries", [])

            st.markdown(
                f"### 📸 Signal Snapshot  "
                f"<span style='font-size:0.82rem;color:#aaa;'>"
                f"{_meta.get('universe','')} · {_meta.get('period','')} · "
                f"{_meta.get('symbols_used',0)} symbols · "
                f"as of {str(_meta.get('as_of',''))[:10]} · "
                f"{_meta.get('execution_ms',0)}ms"
                f"</span>",
                unsafe_allow_html=True,
            )

            if _signals:
                df_sig = pd.DataFrame(_signals)

                # Tabs: rank chart + table
                tab_rank, tab_tbl, tab_ind = st.tabs(
                    ["📊 Cross-Sectional Rank", "📋 Signal Table", "🔍 Indicator Breakdown"]
                )

                with tab_rank:
                    # Cross-sectional rank bar chart
                    df_sorted = df_sig.sort_values("combined_score", ascending=True).tail(40)
                    action_color = df_sorted["action"].map(
                        {"BUY": "#2ecc71", "SELL": "#e74c3c", "HOLD": "#7f8c8d"}
                    ).fillna("#7f8c8d")

                    fig_rank = go.Figure(go.Bar(
                        x=df_sorted["combined_score"],
                        y=df_sorted["symbol"],
                        orientation="h",
                        marker_color=action_color.tolist(),
                        text=df_sorted["action"],
                        textposition="inside",
                    ))
                    fig_rank.add_vline(
                        x=st.session_state["sc_buy_threshold"],
                        line_color="#2ecc71", line_dash="dash", line_width=1.5,
                        annotation_text="BUY", annotation_font_color="#2ecc71",
                    )
                    fig_rank.add_vline(
                        x=st.session_state["sc_sell_threshold"],
                        line_color="#e74c3c", line_dash="dash", line_width=1.5,
                        annotation_text="SELL", annotation_font_color="#e74c3c",
                    )
                    fig_rank.update_layout(
                        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=30, b=20),
                        font=dict(color="#e0e0e0"), height=max(350, len(df_sorted) * 22),
                        xaxis=dict(title="Combined Signal Score", range=[-1.05, 1.05]),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_rank, use_container_width=True)

                    # Action distribution donut
                    action_counts = df_sig["action"].value_counts()
                    total_sig = len(df_sig)
                    da1, da2, da3, da4 = st.columns(4)
                    da1.metric("📈 BUY", f"{action_counts.get('BUY', 0)}")
                    da2.metric("📉 SELL", f"{action_counts.get('SELL', 0)}")
                    da3.metric("⏸ HOLD", f"{action_counts.get('HOLD', 0)}")
                    da4.metric("📦 Universe", total_sig)

                with tab_tbl:
                    # Flat table showing symbol, action, score, votes
                    display_cols = [c for c in ["symbol", "action", "combined_score", "buy_votes", "sell_votes", "hold_votes"] if c in df_sig.columns]
                    def _color_action(val: str) -> str:
                        return {"BUY": "color: #2ecc71", "SELL": "color: #e74c3c"}.get(val, "color: #aaa")
                    st.dataframe(
                        df_sig[display_cols].style.applymap(_color_action, subset=["action"]),
                        use_container_width=True,
                        hide_index=True,
                        height=min(35 * len(df_sig) + 38, 500),
                    )

                with tab_ind:
                    if _ind_sum:
                        df_ind = pd.DataFrame(_ind_sum)
                        # Stacked bar per indicator: buy / hold / sell
                        fig_ind = go.Figure()
                        fig_ind.add_trace(go.Bar(
                            x=df_ind["name"], y=df_ind["buy_count"],
                            name="BUY", marker_color="#2ecc71",
                        ))
                        fig_ind.add_trace(go.Bar(
                            x=df_ind["name"], y=df_ind["hold_count"],
                            name="HOLD", marker_color="#7f8c8d",
                        ))
                        fig_ind.add_trace(go.Bar(
                            x=df_ind["name"], y=df_ind["sell_count"],
                            name="SELL", marker_color="#e74c3c",
                        ))
                        fig_ind.update_layout(
                            template="plotly_dark", barmode="stack",
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=20, r=20, t=20, b=20), height=300,
                            font=dict(color="#e0e0e0"),
                        )
                        st.plotly_chart(fig_ind, use_container_width=True)
            else:
                st.info("No signals returned. Try a different universe or period.")

    if bt_res is not None:
        st.markdown("---")
        if bt_res.get("error"):
            st.error(f"Backtest error: {bt_res['error']}")
        else:
            _bm = bt_res.get("metrics", {})
            _meta_bt = bt_res.get("meta", {})

            st.markdown(
                f"### 📈 Indicator Backtest Results  "
                f"<span style='font-size:0.82rem;color:#aaa;'>"
                f"{_meta_bt.get('universe','')} · {_meta_bt.get('period','')}"
                f"</span>",
                unsafe_allow_html=True,
            )

            bm1, bm2, bm3, bm4, bm5 = st.columns(5)
            bm1.metric("CAGR", fmt_pct(_bm.get("cagr", 0)))
            bm2.metric("Sharpe", fmt_num(_bm.get("sharpe", 0)))
            bm3.metric("Max DD", fmt_pct(_bm.get("max_dd", 0)))
            bm4.metric("Win Rate", fmt_pct(_bm.get("win_rate", 0)))
            bm5.metric("Volatility", fmt_pct(_bm.get("volatility", 0)))

            ec_list = bt_res.get("equity_curve", [])
            bench_list = bt_res.get("benchmark_curve", [])
            if ec_list:
                df_ec = pd.DataFrame(ec_list)
                df_ec["date"] = pd.to_datetime(df_ec["date"])
                fig_ec = go.Figure()
                if bench_list:
                    df_bench = pd.DataFrame(bench_list)
                    df_bench["date"] = pd.to_datetime(df_bench["date"])
                    fig_ec.add_trace(go.Scatter(
                        x=df_bench["date"], y=df_bench["value"] * 100,
                        name="Benchmark (equal-weight)", line=dict(color="#7f8c8d", width=1.5, dash="dot"),
                    ))
                fig_ec.add_trace(go.Scatter(
                    x=df_ec["date"], y=df_ec["value"] * 100,
                    name="Signal Portfolio", line=dict(color="#3498db", width=2.4),
                ))
                fig_ec.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=30, b=20),
                    font=dict(color="#e0e0e0"), height=380,
                    legend=dict(bgcolor="rgba(0,0,0,0)"),
                    yaxis=dict(ticksuffix="x"),
                )
                fig_ec.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
                fig_ec.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
                st.plotly_chart(fig_ec, use_container_width=True)

            # Current holdings
            holdings = bt_res.get("current_holdings", [])
            if holdings:
                st.markdown(f"**📋 Current Holdings ({len(holdings)}):** " + "  ".join(
                    f"`{h}`" for h in holdings[:20]
                ))


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Five-Factor Pipeline Preview
# ═══════════════════════════════════════════════════════════════════════════════
with tab_5f:
    st.markdown("### 🌐 Thirteen-Axis Multi-Factor Pipeline")
    st.caption(
        "The five-factor rotation signal (`five_factor_rotation`) is built from **13 factor axes** "
        "using quintile-based bucket selection, multi-lookback ensemble (21/63/126/252d), "
        "and exponentially-weighted factor selection. Toggle axes and tune their weights."
    )

    # Original vs Extended axes
    orig_axes = FIVE_FACTOR_AXES[:6]
    ext_axes = FIVE_FACTOR_AXES[6:]

    for group_name, axes, color in [
        ("Original 6 Axes (Fama-French inspired)", orig_axes, "#3498db"),
        ("7 Extended Axes (Quality, Liquidity, Sentiment…)", ext_axes, "#2ecc71"),
    ]:
        st.markdown(
            f"<div style='background:rgba(255,255,255,0.05);border-left:3px solid {color};"
            f"padding:6px 12px;border-radius:4px;margin-bottom:8px;'>"
            f"<b style='font-size:0.9rem;'>{group_name}</b></div>",
            unsafe_allow_html=True,
        )
        ax_cols = st.columns(3)
        for i, ax in enumerate(axes):
            with ax_cols[i % 3]:
                with st.container(border=True):
                    ec1, ec2 = st.columns([3, 1])
                    with ec1:
                        ax_on = st.toggle(
                            f"{ax['icon']} **{ax['label']}**",
                            value=st.session_state["sc_ff_axes_enabled"].get(ax["key"], True),
                            key=f"sc_ax_{ax['key']}",
                        )
                        st.session_state["sc_ff_axes_enabled"][ax["key"]] = ax_on
                    with ec2:
                        ax_w = st.number_input(
                            "Weight",
                            min_value=0.0,
                            max_value=5.0,
                            value=float(st.session_state["sc_ff_axes_weights"].get(ax["key"], 1.0)),
                            step=0.1,
                            key=f"sc_axw_{ax['key']}",
                            label_visibility="visible",
                        )
                        st.session_state["sc_ff_axes_weights"][ax["key"]] = ax_w if ax_on else 0.0
                    if ax_on:
                        st.caption(f"_{ax['desc']}_")

    # Enabled axes weight chart
    enabled_axes = [ax for ax in FIVE_FACTOR_AXES if st.session_state["sc_ff_axes_enabled"].get(ax["key"], True)]
    enabled_weights = [st.session_state["sc_ff_axes_weights"].get(ax["key"], 1.0) for ax in enabled_axes]
    total_aw = sum(enabled_weights)

    if enabled_axes and total_aw > 0:
        norm_aw = [w / total_aw for w in enabled_weights]
        st.markdown("---")
        st.markdown(f"**{len(enabled_axes)} active axes** · Portfolio weight allocation:")
        fig_aw = go.Figure(go.Bar(
            x=[ax["label"] for ax in enabled_axes],
            y=[w * 100 for w in norm_aw],
            text=[f"{w*100:.1f}%" for w in norm_aw],
            textposition="auto",
            marker_color=[
                AXIS_CATEGORY_COLORS["Original (FF5)"] if ax in orig_axes else AXIS_CATEGORY_COLORS["Extended Axes"]
                for ax in enabled_axes
            ],
        ))
        fig_aw.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=20, b=50),
            height=280, yaxis=dict(ticksuffix="%", title="Weight"), showlegend=False,
            xaxis=dict(tickangle=-30),
        )
        st.plotly_chart(fig_aw, use_container_width=True)

    # Pipeline architecture diagram
    st.markdown("---")
    st.markdown("#### 🔧 Pipeline Architecture")
    with st.expander("View pipeline flow", expanded=False):
        st.markdown("""
```
Input: BIST Price + Fundamental Data
          │
          ├─► [1] Size Axis         (small-cap premium)
          ├─► [2] Value Axis        (E/P, FCF/P, S/P composite)
          ├─► [3] Profitability     (margin level vs growth)
          ├─► [4] Investment        (conservative vs reinvestment)
          ├─► [5] Momentum          (12-1 month, skip-month corrected)
          ├─► [6] Risk              (realized vol + beta composite)
          ├─► [7] Quality           (ROE, ROA, Piotroski, accruals)
          ├─► [8] Liquidity         (Amihud, turnover/shares_outstanding)
          ├─► [9] Trading Intensity (relative vol, volume trend)
          ├─► [10] Sentiment        (52w high %, price acceleration)
          ├─► [11] Fundamental Mom  (margin change, sales accel)
          ├─► [12] Carry            (dividend + shareholder yield)
          └─► [13] Defensive        (earnings stability, low-beta)
                    │
                    ▼
          Quintile Scoring (0–100 per axis)
                    │
          Multi-Lookback Ensemble (21/63/126/252d)
                    │
          EW Factor Selection (rank² weights, 6m half-life)
                    │
          [Optional] Orthogonalization
                    │
          ► Final Combined Score (0–100) → portfolio
```
        """)

    # Ensemble lookback weights chart (static)
    st.markdown("#### 📅 Multi-Lookback Ensemble Weights")
    st.caption(
        "Each axis is computed at 4 lookback windows. "
        "Weights are determined by past return rank (rank² × exp-decay)."
    )
    lookback_labels = ["21d (1mo)", "63d (3mo)", "126d (6mo)", "252d (12mo)"]
    # Example relative weights (illustrative, uniform starting point)
    fig_lb = go.Figure(go.Bar(
        x=lookback_labels,
        y=[0.15, 0.25, 0.30, 0.30],
        text=["~15%", "~25%", "~30%", "~30%"],
        textposition="auto",
        marker_color=["#3498db", "#2ecc71", "#f39c12", "#e74c3c"],
    ))
    fig_lb.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=20, b=20),
        height=220, yaxis=dict(tickformat=".0%", title="Starting weight"),
        title="Illustrative starting weights (rank² × 6m decay updates these adaptively)",
        showlegend=False,
    )
    st.plotly_chart(fig_lb, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Orthogonalization
# ═══════════════════════════════════════════════════════════════════════════════
with tab_orth:
    st.markdown("### ⊥ Cross-Sectional Factor Orthogonalization")
    st.caption(
        "Orthogonalization projects each axis onto the residual of all preceding axes. "
        "This removes factor overlap, making each axis contribution independent (additive alpha)."
    )

    orth_c1, orth_c2 = st.columns([3, 2])

    with orth_c1:
        orth_enabled = st.toggle(
            "**Enable orthogonalization** in five-factor pipeline",
            value=st.session_state.get("sc_orth_enabled", False),
            key="sc_orth_toggle",
        )
        st.session_state["sc_orth_enabled"] = orth_enabled

        st.markdown("---")
        st.markdown("**Axis ordering** (orthogonalization is sequential — order matters)")
        st.caption(
            "Axes earlier in the list are projected first (treated as 'base' factors). "
            "Later axes project out all variance explained by preceding axes."
        )

        # Drag-sort substitute: reorder via multiselect order
        all_axis_keys = [ax["key"] for ax in FIVE_FACTOR_AXES]
        all_axis_labels = {ax["key"]: f"{ax['icon']} {ax['label']}" for ax in FIVE_FACTOR_AXES}

        orth_axes = st.multiselect(
            "Select axes to include in orthogonalization (in order)",
            options=all_axis_keys,
            default=st.session_state.get("sc_orth_axes", all_axis_keys[:4]),
            format_func=lambda k: all_axis_labels.get(k, k),
            key="sc_orth_axes_select",
        )
        st.session_state["sc_orth_axes"] = orth_axes

        min_overlap = st.number_input(
            "Min overlap (# stocks required for valid correlation)",
            min_value=5, max_value=100, value=20, step=5,
            key="sc_orth_overlap",
        )

    with orth_c2:
        st.markdown("**What orthogonalization does:**")
        st.markdown("""
- **Before:** Axes are correlated (e.g. Momentum × Quality ≈ 0.35)
- **After:** Each axis score is a purely incremental alpha contribution

**Mathematical process (date-by-date):**
1. Z-score each axis cross-sectionally
2. Project axis₂ out of axis₁ space → residual₂
3. Project axis₃ out of span(residual₁, residual₂) → residual₃
4. … repeat for all axes in order

**Result:** axis scores become orthogonal — pairwise correlation → ~0

**When to use:**
- When axes show high overlap (corr > 0.3)
- When you want to be sure each axis adds unique information
- Research mode for factor decomposition
        """)

    # Correlation preview ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🔗 Conceptual Correlation Structure")
    st.caption("Illustrative pairwise factor correlations (before orthogonalization). Actual values depend on the date and universe.")

    # Build a conceptual correlation heatmap
    axes_preview = ["momentum", "value", "quality", "size", "profitability", "risk"]
    corr_illustrative = np.array([
        [1.00,  0.08,  0.22, -0.15,  0.18, -0.28],
        [0.08,  1.00, -0.10, -0.12,  0.05,  0.06],
        [0.22, -0.10,  1.00, -0.08,  0.35,  0.12],
        [-0.15,-0.12, -0.08,  1.00, -0.05,  0.10],
        [0.18,  0.05,  0.35, -0.05,  1.00,  0.14],
        [-0.28, 0.06,  0.12,  0.10,  0.14,  1.00],
    ])
    fig_corr_before = px.imshow(
        corr_illustrative,
        x=axes_preview, y=axes_preview,
        color_continuous_scale="RdYlGn",
        zmin=-1, zmax=1, text_auto=".2f",
        title="Before orthogonalization (illustrative)",
    )
    fig_corr_before.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20), height=340,
    )

    # After orthogonalization: near-zero off-diagonal
    corr_after = np.eye(6)
    np.fill_diagonal(corr_after, 1.0)
    for i in range(6):
        for j in range(6):
            if i != j:
                corr_after[i, j] = np.random.uniform(-0.03, 0.03)

    fig_corr_after = px.imshow(
        corr_after,
        x=axes_preview, y=axes_preview,
        color_continuous_scale="RdYlGn",
        zmin=-1, zmax=1, text_auto=".2f",
        title="After orthogonalization (near-diagonal)",
    )
    fig_corr_after.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20), height=340,
    )

    hm_c1, hm_c2 = st.columns(2)
    with hm_c1:
        st.plotly_chart(fig_corr_before, use_container_width=True)
    with hm_c2:
        st.plotly_chart(fig_corr_after, use_container_width=True)

    # Configuration summary
    st.markdown("---")
    st.markdown("#### 📋 Current Configuration Summary")
    cfg_rows = []
    for ax_key in (orth_axes if orth_axes else orth_axes):
        ax_def = next((a for a in FIVE_FACTOR_AXES if a["key"] == ax_key), None)
        if ax_def:
            cfg_rows.append({
                "Order": len(cfg_rows) + 1,
                "Axis": f"{ax_def['icon']} {ax_def['label']}",
                "Key": ax_key,
                "Description": ax_def["desc"],
                "Enabled": "✅" if st.session_state["sc_ff_axes_enabled"].get(ax_key, True) else "❌",
                "Weight": f"{st.session_state['sc_ff_axes_weights'].get(ax_key, 1.0):.1f}",
            })
    if cfg_rows:
        st.dataframe(pd.DataFrame(cfg_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No axes selected for orthogonalization.")

    if orth_enabled and len(orth_axes) >= 2:
        st.success(
            f"✅ Orthogonalization enabled for **{len(orth_axes)} axes** "
            f"(min_overlap={min_overlap}). This will be applied when `five_factor_rotation` "
            f"is run via the `PortfolioEngine`. See `bist_quant/signals/orthogonalization.py`."
        )
    elif orth_enabled and len(orth_axes) < 2:
        st.warning("⚠️ Select at least 2 axes to enable orthogonalization.")
    else:
        st.info("ℹ️ Orthogonalization is currently **disabled**. Toggle above to enable.")
