"""Backtest Runner — run and compare strategy backtests with full analytics."""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import Future
from datetime import date, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="Backtest · BIST Quant", page_icon="🔄", layout="wide")

from app.charts import drawdown_chart, equity_curve, monthly_returns_heatmap  # noqa: E402
from app.layout import page_header, render_sidebar  # noqa: E402
from app.services import get_core_service  # noqa: E402
from app.utils import fmt_num, fmt_pct, run_in_thread  # noqa: E402

render_sidebar()
page_header("🔄 Backtest", "Configure a strategy, run it, explore every metric")

# ── session state init ────────────────────────────────────────────────────────
if "bt_future" not in st.session_state:
    st.session_state["bt_future"] = None       # Future object while running
if "bt_results" not in st.session_state:
    st.session_state["bt_results"] = []        # list of completed result dicts
if "bt_active_idx" not in st.session_state:
    st.session_state["bt_active_idx"] = None   # which result is displayed

# ── absorb optimization prefill (set by Phase-7 Optimization page) ───────────
_incoming_pf = st.session_state.pop("bt_prefill", None)
if _incoming_pf:
    st.session_state["_bt_pf"] = _incoming_pf
_bt_pf: dict[str, Any] = st.session_state.get("_bt_pf") or {}

# ── helpers ───────────────────────────────────────────────────────────────────

def _ec_to_df(equity_curve_list: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(equity_curve_list)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _monthly_to_heatmap_dict(monthly: list[dict]) -> dict[int, dict[int, float]]:
    """Convert [{month:'2020-01', strategy_return:0.03}] → {2020: {1: 3.0}}"""
    out: dict[int, dict[int, float]] = {}
    for row in monthly:
        yr_str, mo_str = row["month"].split("-")
        yr, mo = int(yr_str), int(mo_str)
        out.setdefault(yr, {})[mo] = row["strategy_return"] * 100
    return out


def _rolling_fig(rolling: list[dict]) -> go.Figure:
    df = pd.DataFrame(rolling)
    df["date"] = pd.to_datetime(df["date"])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Rolling Sharpe (63d)", "Rolling Volatility (63d)"),
                        vertical_spacing=0.12)

    sharpe_col = "rolling_sharpe_63d"
    vol_col = "rolling_volatility_63d"

    fig.add_trace(go.Scatter(
        x=df["date"], y=df[sharpe_col],
        mode="lines", name="Rolling Sharpe",
        line=dict(color="#3498db", width=1.8),
    ), row=1, col=1)
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1, row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["date"], y=df[vol_col].apply(lambda v: v * 100 if v else None),
        mode="lines", name="Realized Vol %",
        line=dict(color="#e67e22", width=1.8),
        fill="tozeroy", fillcolor="rgba(230,126,34,0.15)",
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(color="#e0e0e0"),
        showlegend=False,
        height=320,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
    return fig


def _combined_equity_fig(ec_list: list[dict]) -> go.Figure:
    """Equity curve + drawdown as a stacked subplot."""
    df = _ec_to_df(ec_list)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.06,
                        subplot_titles=("Equity Curve", "Drawdown"))

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["benchmark"] * 100,
        mode="lines", name="XU100",
        line=dict(color="#7f8c8d", width=1.5, dash="dot"),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["date"], y=df["value"] * 100,
        mode="lines", name="Strategy",
        line=dict(color="#3498db", width=2.2),
    ), row=1, col=1)

    dd_pct = (df["drawdown"] * 100).tolist()
    fig.add_trace(go.Scatter(
        x=df["date"], y=dd_pct,
        mode="lines", name="Drawdown",
        line=dict(color="#e74c3c", width=1.4),
        fill="tozeroy", fillcolor="rgba(231,76,60,0.2)",
        showlegend=False,
    ), row=2, col=1)
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.25)", line_width=1, row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(color="#e0e0e0"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=500,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)", row=1, col=1, ticksuffix="x")
    fig.update_yaxes(ticksuffix="%", row=2, col=1)
    return fig


def _render_results(result: dict[str, Any]) -> None:
    """Render the full results area for a completed backtest."""
    m = result.get("metrics", {})
    summary = result.get("summary", {})
    fname = result.get("factor_name", "strategy")
    start = result.get("start_date", "")
    end = result.get("end_date", "")

    st.markdown(
        f"<div style='color:#aaa; font-size:0.85rem; margin-bottom:0.5rem;'>"
        f"📋 <b style='color:#e0e0e0;'>{fname}</b> &nbsp;|&nbsp; "
        f"{start} → {end} &nbsp;|&nbsp; "
        f"{summary.get('trading_days','?')} trading days"
        f"</div>",
        unsafe_allow_html=True,
    )

    tab_curve, tab_monthly, tab_rolling, tab_risk, tab_holdings, tab_export = st.tabs([
        "📈 Equity Curve", "📅 Monthly Returns", "📊 Rolling Metrics",
        "⚠️ Risk", "🏆 Holdings", "💾 Export",
    ])

    # ── Equity Curve tab ──────────────────────────────────────────────────────
    with tab_curve:
        # Summary metric row
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("CAGR", fmt_pct(m.get("cagr", 0) * 100))
        c2.metric("Sharpe", fmt_num(m.get("sharpe", 0)))
        c3.metric("Sortino", fmt_num(m.get("sortino", 0)))
        c4.metric("Max DD", fmt_pct(m.get("max_drawdown", 0) * 100))
        c5.metric("Calmar", fmt_num(m.get("calmar", 0)))
        c6.metric("Ann. Vol", fmt_pct(m.get("annualized_volatility", 0) * 100))

        st.markdown("<br>", unsafe_allow_html=True)

        ec_list = result.get("equity_curve", [])
        if ec_list:
            st.plotly_chart(_combined_equity_fig(ec_list), use_container_width=True)

        # Secondary metrics
        st.markdown("---")
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total Return", fmt_pct(m.get("total_return", 0) * 100))
        d2.metric("Win Rate", fmt_pct(m.get("win_rate", 0) * 100))
        d3.metric("Beta", fmt_num(m.get("beta", 0)))
        d4.metric("Alpha (ann.)", fmt_pct(m.get("alpha_annual", 0) * 100))
        e1, e2 = st.columns(2)
        e1.metric("Tracking Error", fmt_pct(m.get("tracking_error", 0) * 100))
        e2.metric("Info Ratio", fmt_num(m.get("information_ratio", 0)))

    # ── Monthly Returns tab ───────────────────────────────────────────────────
    with tab_monthly:
        monthly = result.get("monthly_returns", [])
        if monthly:
            hm_dict = _monthly_to_heatmap_dict(monthly)
            fig_hm = monthly_returns_heatmap(hm_dict)
            fig_hm.update_layout(height=max(250, len(hm_dict) * 38 + 80))
            st.plotly_chart(fig_hm, use_container_width=True)

            # Also show as DataFrame
            with st.expander("Raw monthly data"):
                df_m = pd.DataFrame(monthly)
                df_m["strategy_return"] = (df_m["strategy_return"] * 100).round(2)
                df_m["benchmark_return"] = (df_m["benchmark_return"] * 100).round(2)
                df_m["excess_return"] = (df_m["excess_return"] * 100).round(2)
                df_m.columns = ["Month", "Strategy %", "Benchmark %", "Excess %"]
                st.dataframe(df_m, use_container_width=True, hide_index=True)
        else:
            st.info("No monthly returns data.")

    # ── Rolling Metrics tab ───────────────────────────────────────────────────
    with tab_rolling:
        rolling = result.get("rolling_metrics", [])
        if rolling:
            st.plotly_chart(_rolling_fig(rolling), use_container_width=True)
        else:
            st.info("No rolling metrics data.")

    # ── Risk tab ──────────────────────────────────────────────────────────────
    with tab_risk:
        risk = result.get("risk_metrics", {})
        scenario = result.get("scenario_analysis", {})

        r1, r2 = st.columns(2)
        with r1:
            st.markdown("**Tail Risk**")
            tail = risk.get("tail_risk", {})
            tr1, tr2 = st.columns(2)
            tr1.metric("VaR 95%", fmt_pct(tail.get("var_95", 0) * 100))
            tr2.metric("CVaR 95%", fmt_pct(tail.get("cvar_95", 0) * 100))
            tr3, tr4 = st.columns(2)
            tr3.metric("VaR 99%", fmt_pct(tail.get("var_99", 0) * 100))
            tr4.metric("CVaR 99%", fmt_pct(tail.get("cvar_99", 0) * 100))

        with r2:
            st.markdown("**Scenario Analysis**")
            sc1, sc2 = st.columns(2)
            sc1.metric("Best Day", fmt_pct(scenario.get("best_day", 0) * 100))
            sc2.metric("Worst Day", fmt_pct(scenario.get("worst_day", 0) * 100))
            sc3, sc4 = st.columns(2)
            sc3.metric("−2σ Day", fmt_pct(scenario.get("stress_1d_minus_2sigma", 0) * 100))
            sc4.metric("−3σ Day", fmt_pct(scenario.get("stress_1d_minus_3sigma", 0) * 100))

        mae_mfe = risk.get("mae_mfe", {})
        if mae_mfe:
            st.markdown("**MAE / MFE**")
            mf1, mf2, mf3, mf4 = st.columns(4)
            mf1.metric("MAE 1d", fmt_pct(mae_mfe.get("mae_1d", 0) * 100))
            mf2.metric("MFE 1d", fmt_pct(mae_mfe.get("mfe_1d", 0) * 100))
            mf3.metric("Worst 5d", fmt_pct(mae_mfe.get("worst_5d", 0) * 100))
            mf4.metric("Best 5d", fmt_pct(mae_mfe.get("best_5d", 0) * 100))

        sector = result.get("sector_exposure", {})
        if sector:
            st.markdown("**Sector Exposure (last rebalance)**")
            sec_df = pd.DataFrame(
                [{"Sector": k, "Weight %": round(v * 100, 1)} for k, v in sector.items()]
            ).sort_values("Weight %", ascending=False)
            st.dataframe(sec_df, use_container_width=True, hide_index=True)

    # ── Holdings tab ─────────────────────────────────────────────────────────
    with tab_holdings:
        holdings = result.get("top_holdings", [])
        if holdings:
            h_df = pd.DataFrame(holdings)
            h_df["weight"] = (h_df["weight"] * 100).round(1)
            h_df.columns = ["Ticker", "Weight %"]
            st.dataframe(h_df, use_container_width=True, hide_index=True)

            # Bar chart
            fig_h = go.Figure(go.Bar(
                x=h_df["Ticker"], y=h_df["Weight %"],
                marker_color="#3498db",
                text=h_df["Weight %"].apply(lambda v: f"{v}%"),
                textposition="outside",
            ))
            fig_h.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=20, b=10),
                yaxis_title="Weight %", height=300,
                font=dict(color="#e0e0e0"),
            )
            st.plotly_chart(fig_h, use_container_width=True)
        else:
            st.info("No holdings data.")

    # ── Export tab ────────────────────────────────────────────────────────────
    with tab_export:
        ec_list = result.get("equity_curve", [])
        if ec_list:
            ec_df = pd.DataFrame(ec_list)
            csv_bytes = ec_df.to_csv(index=False).encode()
            st.download_button(
                "⬇️ Equity Curve CSV",
                data=csv_bytes,
                file_name=f"{fname}_{start}_{end}_equity.csv",
                mime="text/csv",
            )

        full_json = json.dumps(
            {k: v for k, v in result.items() if k not in ("equity_curve", "drawdown_curve",
             "rolling_metrics", "daily_returns")},
            default=str, indent=2,
        )
        st.download_button(
            "⬇️ Full Result JSON",
            data=full_json.encode(),
            file_name=f"{fname}_{start}_{end}_result.json",
            mime="application/json",
        )


# ── LAYOUT: control panel (left) + results (right) ────────────────────────────
ctrl_col, spacer, result_col = st.columns([1, 0.05, 2.4])

with ctrl_col:
    st.markdown("### ⚙️ Strategy")

    core = get_core_service()
    signal_names: list[str] = []
    if core is not None:
        try:
            signal_names = core.list_available_signals()
        except Exception:
            pass

    if not signal_names:
        st.warning("Core service unavailable — no signals loaded.")
        st.stop()

    # ── prefill banner ────────────────────────────────────────────────────────
    if _bt_pf:
        _pf_sig = _bt_pf.get("signal", "?")
        _pf_top_n = _bt_pf.get("top_n")
        _banner_parts = [f"signal: **{_pf_sig}**"]
        if _pf_top_n is not None:
            _banner_parts.append(f"Top-N: **{_pf_top_n}**")
        st.info("⚙️ **Optimization prefill** — " + ", ".join(_banner_parts))
        if st.button("✖ Clear prefill", key="bt_clear_pf"):
            st.session_state.pop("_bt_pf", None)
            _bt_pf = {}
            st.rerun()

    # Determine default signal index (use prefill when available)
    _pf_signal = _bt_pf.get("signal")
    _sig_default_idx = (
        signal_names.index(_pf_signal)
        if _pf_signal and _pf_signal in signal_names
        else (signal_names.index("momentum") if "momentum" in signal_names else 0)
    )

    selected_signal = st.selectbox(
        "Signal / Factor",
        options=signal_names,
        index=_sig_default_idx,
    )

    # Load defaults from signal config
    signal_config: dict[str, Any] = {}
    if core is not None:
        try:
            signal_config = core.get_signal_details(selected_signal)
        except Exception:
            pass

    port_opts = signal_config.get("portfolio_options", {})
    timeline = signal_config.get("timeline", {})
    default_start = timeline.get("start_date", "2014-01-01")
    default_end = timeline.get("end_date", str(date.today()))

    st.markdown("### 📅 Date Range")
    col_s, col_e = st.columns(2)
    with col_s:
        start_date = st.date_input(
            "Start",
            value=datetime.strptime(default_start, "%Y-%m-%d").date(),
            min_value=date(2010, 1, 1),
            max_value=date.today(),
        )
    with col_e:
        end_date = st.date_input(
            "End",
            value=min(datetime.strptime(default_end[:10], "%Y-%m-%d").date(), date.today()),
            min_value=date(2010, 1, 1),
            max_value=date.today(),
        )

    st.markdown("### 🎛 Portfolio Options")

    rebalance_freq = st.selectbox(
        "Rebalance",
        ["monthly", "quarterly", "weekly"],
        index=["monthly", "quarterly", "weekly"].index(
            signal_config.get("rebalance_frequency", "monthly")
        ),
    )
    top_n = st.slider(
        "Top N stocks",
        min_value=5,
        max_value=50,
        value=int(_bt_pf.get("top_n") or port_opts.get("top_n", 20)),
        step=5,
    )
    max_weight = st.slider("Max position weight", 0.05, 0.50,
                           value=float(port_opts.get("max_position_weight", 0.25)),
                           step=0.05, format="%.0f%%",
                           help="Max single position weight (as decimal)")

    with st.expander("Risk Controls", expanded=False):
        use_regime = st.checkbox("Regime filter", value=bool(port_opts.get("use_regime_filter", True)))
        use_liq = st.checkbox("Liquidity filter", value=bool(port_opts.get("use_liquidity_filter", True)))
        use_slippage = st.checkbox("Slippage", value=bool(port_opts.get("use_slippage", True)))
        slippage_bps = st.slider("Slippage (bps)", 0, 50,
                                 value=int(port_opts.get("slippage_bps", 5)),
                                 disabled=not use_slippage)
        use_stop = st.checkbox("Stop loss", value=bool(port_opts.get("use_stop_loss", False)))
        stop_threshold = st.slider("Stop threshold", 0.05, 0.40,
                                   value=float(port_opts.get("stop_loss_threshold", 0.15)),
                                   step=0.01, format="%.0f%%",
                                   disabled=not use_stop)
        use_vol_tgt = st.checkbox("Volatility targeting",
                                  value=bool(port_opts.get("use_vol_targeting", False)))
        target_vol = st.slider("Target vol", 0.05, 0.60,
                               value=float(port_opts.get("target_downside_vol", 0.20)),
                               step=0.01, format="%.0f%%",
                               disabled=not use_vol_tgt)

    st.markdown("---")

    run_btn = st.button("▶ Run Backtest", type="primary", use_container_width=True)

    # ── Run logic ─────────────────────────────────────────────────────────────
    if run_btn and core is not None:
        if start_date >= end_date:
            st.error("Start date must be before end date.")
        else:
            fut: Future = run_in_thread(
                core.run_backtest,
                factor_name=selected_signal,
                start_date=str(start_date),
                end_date=str(end_date),
                rebalance_frequency=rebalance_freq,
                top_n=top_n,
                max_position_weight=max_weight,
                use_regime_filter=use_regime,
                use_liquidity_filter=use_liq,
                use_slippage=use_slippage,
                slippage_bps=float(slippage_bps),
                use_stop_loss=use_stop,
                stop_loss_threshold=stop_threshold,
                use_vol_targeting=use_vol_tgt,
                target_downside_vol=target_vol,
            )
            st.session_state["bt_future"] = fut

    # ── Run history sidebar ───────────────────────────────────────────────────
    if st.session_state["bt_results"]:
        st.markdown("### 📚 Run History")
        for i, r in enumerate(reversed(st.session_state["bt_results"])):
            idx = len(st.session_state["bt_results"]) - 1 - i
            m = r.get("metrics", {})
            label = (
                f"**{r.get('factor_name','?')}** "
                f"{r.get('start_date','')[:4]}–{r.get('end_date','')[:4]}  \n"
                f"CAGR {fmt_pct(m.get('cagr',0)*100)} | Sharpe {fmt_num(m.get('sharpe',0))}"
            )
            if st.button(label, key=f"hist_{idx}", use_container_width=True):
                st.session_state["bt_active_idx"] = idx

# ── Results panel ─────────────────────────────────────────────────────────────
with result_col:
    fut = st.session_state.get("bt_future")

    # Poll future
    if fut is not None:
        if not fut.done():
            with st.status("⏳ Running backtest…", expanded=True) as status:
                st.write(f"Factor: **{selected_signal}**  |  {start_date} → {end_date}")
                placeholder = st.empty()
                t0 = time.time()
                while not fut.done():
                    elapsed = time.time() - t0
                    placeholder.markdown(f"Elapsed: `{elapsed:.1f}s`")
                    time.sleep(0.4)
                    if elapsed > 300:
                        status.update(label="⌛ Timeout — backtest took too long.", state="error")
                        st.session_state["bt_future"] = None
                        break
                if fut.done():
                    status.update(label="✅ Backtest complete!", state="complete")
        # Future is done
        if fut.done():
            try:
                result = fut.result()
                # Store and display
                st.session_state["bt_results"].append(result)
                st.session_state["bt_active_idx"] = len(st.session_state["bt_results"]) - 1
                st.session_state["bt_future"] = None
            except Exception as exc:
                st.error(f"Backtest failed: {exc}")
                st.session_state["bt_future"] = None

    # Show active result
    active_idx = st.session_state.get("bt_active_idx")
    results = st.session_state.get("bt_results", [])

    if active_idx is not None and 0 <= active_idx < len(results):
        _render_results(results[active_idx])
    else:
        st.markdown(
            """
            <div style="
                display:flex; flex-direction:column; align-items:center;
                justify-content:center; height:60vh;
                color:#555; font-size:1.1rem;
            ">
                <div style="font-size:3rem; margin-bottom:1rem;">🔄</div>
                <div>Configure a strategy on the left and click <b>Run Backtest</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
