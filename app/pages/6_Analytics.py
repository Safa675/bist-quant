"""Portfolio Analytics -- deep-dive metrics, Monte Carlo, walk-forward analysis."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).parent.parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Analytics · BIST Quant", page_icon="📊", layout="wide"
)

from app.layout import page_header, render_sidebar  # noqa: E402

try:
    from bist_quant.analytics.advanced import (  # noqa: E402
        compute_performance_attribution_breakdown,
    )
    from bist_quant.analytics.core_metrics import (  # noqa: E402
        SeriesPoint,
        build_rolling_metrics,
        build_walk_forward_analysis,
        compute_performance_metrics,
        curve_to_returns,
        run_monte_carlo_bootstrap,
    )
    _ANALYTICS_OK = True
except ImportError as _e:
    _ANALYTICS_OK = False
    _ANALYTICS_ERR = str(_e)

render_sidebar()
page_header(
    "📊 Portfolio Analytics",
    "Deep-dive metrics, Monte Carlo & walk-forward analysis",
)

if not _ANALYTICS_OK:
    st.error(
        f"**bist_quant analytics unavailable**: {_ANALYTICS_ERR}\n\n"
        "Install the package from the repo root:\n```bash\npip install -e .[api,services]\n```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_XU100_CSV = _REPO_ROOT / "data" / "xu100_prices.csv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dicts_to_series(rows: list[dict[str, Any]]) -> list[SeriesPoint]:
    out: list[SeriesPoint] = []
    for r in rows:
        try:
            out.append(
                SeriesPoint(date=str(r["date"])[:10], value=float(r["value"]))
            )
        except (KeyError, TypeError, ValueError):
            pass
    return out


def _df_to_series(df: pd.DataFrame) -> list[SeriesPoint]:
    df = df.copy()
    date_col = df.columns[0]
    val_col = df.columns[1]
    out: list[SeriesPoint] = []
    for _, row in df.iterrows():
        try:
            out.append(
                SeriesPoint(
                    date=str(row[date_col])[:10], value=float(row[val_col])
                )
            )
        except (TypeError, ValueError):
            pass
    return out


@st.cache_data(show_spinner=False)
def _load_xu100_benchmark() -> list[SeriesPoint]:
    if not _XU100_CSV.exists():
        return []
    try:
        df = pd.read_csv(_XU100_CSV, parse_dates=True, index_col=0)
        close_col = next(
            (c for c in df.columns if c.lower() in ("close", "kapanish", "fiyat")),
            df.columns[0],
        )
        series = df[close_col].dropna().sort_index()
        curve = [
            SeriesPoint(date=str(d.date()), value=float(v))
            for d, v in series.items()
        ]
        return list(curve_to_returns(curve))
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Source selection
# ---------------------------------------------------------------------------

st.markdown("### 1 · Load Equity Curve")

src_col, opt_col = st.columns([2, 1])

with src_col:
    source = st.radio(
        "Source",
        ["Session history", "Upload CSV"],
        horizontal=True,
        label_visibility="collapsed",
    )

equity_curve_raw: list[SeriesPoint] = []
run_label = "Unnamed"

with src_col:
    if source == "Session history":
        bt_results: list[dict] = st.session_state.get("bt_results", [])
        if not bt_results:
            st.info(
                "No backtest results in this session. "
                "Run one on the Backtest page, or upload a CSV."
            )
        else:
            options = {
                (
                    f"Run {i + 1} — {r.get('strategy', 'Strategy')} "
                    f"({r.get('start_date', '')[:4]}–{r.get('end_date', '')[:4]})"
                ): i
                for i, r in enumerate(bt_results)
            }
            chosen_label = st.selectbox("Select run", list(options.keys()))
            chosen_idx = options[chosen_label]
            chosen = bt_results[chosen_idx]
            equity_curve_raw = _dicts_to_series(chosen.get("equity_curve", []))
            run_label = chosen.get("strategy", f"Run {chosen_idx + 1}")
    else:
        uploaded = st.file_uploader(
            "Upload equity curve CSV (columns: date, value)",
            type=["csv"],
            key="analytics_upload",
        )
        if uploaded:
            try:
                df_up = pd.read_csv(
                    io.StringIO(uploaded.read().decode("utf-8"))
                )
                if df_up.shape[1] < 2:
                    st.error("CSV must have at least two columns: date and value.")
                else:
                    equity_curve_raw = _df_to_series(df_up)
                    run_label = Path(uploaded.name).stem
                    st.success(f"Loaded {len(equity_curve_raw):,} data points.")
            except Exception as exc:
                st.error(f"Could not parse CSV: {exc}")

with opt_col:
    use_benchmark = st.checkbox(
        "Include XU100 benchmark (beta / alpha / attribution)", value=True
    )
    mc_iters = st.number_input(
        "Monte Carlo paths", min_value=100, max_value=5000, value=750, step=100
    )
    mc_horizon = st.number_input(
        "MC horizon (days)", min_value=21, max_value=1260, value=252, step=21
    )
    wf_splits = st.number_input(
        "Walk-forward splits", min_value=2, max_value=10, value=5, step=1
    )

if not equity_curve_raw:
    st.stop()

returns_raw = list(curve_to_returns(equity_curve_raw))
if len(returns_raw) < 30:
    st.error("Equity curve is too short — need at least 30 data points.")
    st.stop()

benchmark_returns: list[SeriesPoint] | None = (
    _load_xu100_benchmark() if use_benchmark else None
)
if benchmark_returns is not None and len(benchmark_returns) == 0:
    benchmark_returns = None

st.divider()

# ---------------------------------------------------------------------------
# Panel 1 -- Performance Metrics
# ---------------------------------------------------------------------------

st.markdown(f"### 2 · Performance Metrics — {run_label}")

metrics = compute_performance_metrics(
    returns_raw, equity_curve_raw, benchmark_returns
)

r1 = st.columns(4)
r2 = st.columns(4)
r3 = st.columns(4)

with r1[0]:
    st.metric("Total Return", f"{metrics.total_return:+.2f}%")
with r1[1]:
    st.metric("CAGR", f"{metrics.cagr:+.2f}%")
with r1[2]:
    st.metric("Ann. Volatility", f"{metrics.annualized_volatility:.2f}%")
with r1[3]:
    st.metric("Max Drawdown", f"{metrics.max_drawdown:.2f}%")

with r2[0]:
    st.metric("Sharpe", f"{metrics.sharpe:.3f}")
with r2[1]:
    st.metric("Sortino", f"{metrics.sortino:.3f}")
with r2[2]:
    st.metric("Calmar", f"{metrics.calmar:.3f}")
with r2[3]:
    st.metric("Win Rate", f"{metrics.win_rate:.1f}%")

with r3[0]:
    st.metric("VaR 95%", f"{metrics.var_95:.2f}%")
with r3[1]:
    st.metric("CVaR 95%", f"{metrics.cvar_95:.2f}%")
with r3[2]:
    beta_str = f"{metrics.beta:.3f}" if metrics.beta is not None else "—"
    st.metric("Beta (XU100)", beta_str)
with r3[3]:
    alpha_str = (
        f"{metrics.alpha_annual:+.2f}%"
        if metrics.alpha_annual is not None
        else "—"
    )
    st.metric("Alpha (annual)", alpha_str)

extra = st.columns(3)
with extra[0]:
    st.metric("Observations", f"{metrics.observations:,}")
with extra[1]:
    pf_str = (
        f"{metrics.profit_factor:.3f}"
        if metrics.profit_factor is not None
        else "—"
    )
    st.metric("Profit Factor", pf_str)
with extra[2]:
    st.metric("Downside Dev.", f"{metrics.downside_deviation:.2f}%")

st.divider()

# ---------------------------------------------------------------------------
# Panel 2 -- Rolling Metrics
# ---------------------------------------------------------------------------

st.markdown("### 3 · Rolling Metrics")

rolling = build_rolling_metrics(equity_curve_raw)

if not rolling:
    st.warning("Not enough data for rolling metrics (need at least 25 data points).")
else:
    r_df = pd.DataFrame(
        [
            {
                "date": p.date,
                "Rolling Sharpe": p.rolling_sharpe_63d,
                "Rolling Vol (%)": p.rolling_volatility_63d,
                "Rolling DD (%)": p.rolling_drawdown_126d,
            }
            for p in rolling
        ]
    )
    r_df["date"] = pd.to_datetime(r_df["date"])
    r_df = r_df.set_index("date").sort_index()

    rolling_fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "Rolling Sharpe (63d)",
            "Rolling Volatility % (63d)",
            "Rolling Max Drawdown % (126d)",
        ),
    )

    sharpe_s = r_df["Rolling Sharpe"].dropna()
    rolling_fig.add_trace(
        go.Scatter(
            x=sharpe_s.index,
            y=sharpe_s.values,
            name="Sharpe",
            line={"color": "#3b82f6", "width": 1.5},
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.12)",
        ),
        row=1, col=1,
    )
    rolling_fig.add_hline(
        y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)", row=1, col=1
    )

    vol_s = r_df["Rolling Vol (%)"].dropna()
    rolling_fig.add_trace(
        go.Scatter(
            x=vol_s.index,
            y=vol_s.values,
            name="Volatility",
            line={"color": "#f59e0b", "width": 1.5},
        ),
        row=2, col=1,
    )

    dd_s = r_df["Rolling DD (%)"].dropna()
    rolling_fig.add_trace(
        go.Scatter(
            x=dd_s.index,
            y=dd_s.values,
            name="Drawdown",
            line={"color": "#ef4444", "width": 1.5},
            fill="tozeroy",
            fillcolor="rgba(239,68,68,0.12)",
        ),
        row=3, col=1,
    )

    rolling_fig.update_layout(
        height=560,
        showlegend=False,
        margin={"l": 50, "r": 20, "t": 40, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
    )
    rolling_fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", showgrid=True)
    rolling_fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", showgrid=True)
    st.plotly_chart(rolling_fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Panel 3 -- Walk-Forward Analysis
# ---------------------------------------------------------------------------

st.markdown("### 4 · Walk-Forward Analysis")

wf_result = build_walk_forward_analysis(
    returns_raw,
    benchmark_returns,
    splits=int(wf_splits),
    train_ratio=0.7,
)

if not wf_result:
    st.warning(
        "Not enough data for walk-forward analysis "
        "(need at least 60 returns and at least 2 splits)."
    )
else:
    wf_df = pd.DataFrame(
        [
            {
                "Split": s.split,
                "Train Period": f"{s.train_start} -> {s.train_end}",
                "Test Period": f"{s.test_start} -> {s.test_end}",
                "IS CAGR %": s.train_cagr,
                "OOS CAGR %": s.test_cagr,
                "IS Sharpe": s.train_sharpe,
                "OOS Sharpe": s.test_sharpe,
                "p-value": (
                    f"{s.p_value:.3f}" if s.p_value is not None else "—"
                ),
            }
            for s in wf_result
        ]
    )

    split_labels = [f"Split {s.split}" for s in wf_result]

    cagr_fig = go.Figure(
        [
            go.Bar(
                name="In-Sample CAGR %",
                x=split_labels,
                y=[s.train_cagr for s in wf_result],
                marker_color="rgba(59,130,246,0.80)",
            ),
            go.Bar(
                name="Out-of-Sample CAGR %",
                x=split_labels,
                y=[s.test_cagr for s in wf_result],
                marker_color="rgba(16,185,129,0.80)",
            ),
        ]
    )
    cagr_fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
    cagr_fig.update_layout(
        barmode="group",
        title="In-Sample vs Out-of-Sample CAGR per Split",
        height=340,
        margin={"l": 50, "r": 20, "t": 50, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
        legend={"orientation": "h", "y": -0.18},
        yaxis={"title": "CAGR %", "gridcolor": "rgba(255,255,255,0.06)"},
        xaxis={"gridcolor": "rgba(255,255,255,0.06)"},
    )

    sharpe_fig = go.Figure(
        [
            go.Bar(
                name="In-Sample Sharpe",
                x=split_labels,
                y=[s.train_sharpe for s in wf_result],
                marker_color="rgba(139,92,246,0.80)",
            ),
            go.Bar(
                name="OOS Sharpe",
                x=split_labels,
                y=[s.test_sharpe for s in wf_result],
                marker_color="rgba(245,158,11,0.80)",
            ),
        ]
    )
    sharpe_fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
    sharpe_fig.update_layout(
        barmode="group",
        title="In-Sample vs Out-of-Sample Sharpe per Split",
        height=340,
        margin={"l": 50, "r": 20, "t": 50, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
        legend={"orientation": "h", "y": -0.18},
        yaxis={"title": "Sharpe", "gridcolor": "rgba(255,255,255,0.06)"},
        xaxis={"gridcolor": "rgba(255,255,255,0.06)"},
    )

    wf_left, wf_right = st.columns(2)
    with wf_left:
        st.plotly_chart(cagr_fig, use_container_width=True)
    with wf_right:
        st.plotly_chart(sharpe_fig, use_container_width=True)

    st.dataframe(
        wf_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "IS CAGR %": st.column_config.NumberColumn(format="%.2f"),
            "OOS CAGR %": st.column_config.NumberColumn(format="%.2f"),
            "IS Sharpe": st.column_config.NumberColumn(format="%.3f"),
            "OOS Sharpe": st.column_config.NumberColumn(format="%.3f"),
        },
    )

st.divider()

# ---------------------------------------------------------------------------
# Panel 4 -- Monte Carlo Simulation
# ---------------------------------------------------------------------------

st.markdown("### 5 · Monte Carlo Simulation")

with st.spinner(f"Running {int(mc_iters):,} bootstrap paths ..."):
    mc = run_monte_carlo_bootstrap(
        returns_raw,
        iterations=int(mc_iters),
        horizon_days=int(mc_horizon),
    )

mc_cols = st.columns(5)
with mc_cols[0]:
    st.metric("Expected Terminal", f"{mc.expected_terminal:.3f}x")
with mc_cols[1]:
    st.metric("Expected CAGR", f"{mc.expected_cagr:+.2f}%")
with mc_cols[2]:
    st.metric("Prob. of Loss", f"{mc.probability_of_loss:.1f}%")
with mc_cols[3]:
    st.metric("p05 Terminal", f"{mc.terminal_p05:.3f}x")
with mc_cols[4]:
    st.metric("p95 Terminal", f"{mc.terminal_p95:.3f}x")

fan_col, hist_col = st.columns([3, 2])

with fan_col:
    days = [p.day for p in mc.paths]
    fan_fig = go.Figure()
    fan_fig.add_trace(
        go.Scatter(
            x=days + days[::-1],
            y=[p.p95 for p in mc.paths] + [p.p05 for p in mc.paths][::-1],
            fill="toself",
            fillcolor="rgba(59,130,246,0.10)",
            line={"color": "rgba(0,0,0,0)"},
            name="p05-p95",
        )
    )
    fan_fig.add_trace(
        go.Scatter(
            x=days + days[::-1],
            y=[p.p75 for p in mc.paths] + [p.p25 for p in mc.paths][::-1],
            fill="toself",
            fillcolor="rgba(59,130,246,0.22)",
            line={"color": "rgba(0,0,0,0)"},
            name="p25-p75",
        )
    )
    fan_fig.add_trace(
        go.Scatter(
            x=days,
            y=[p.p50 for p in mc.paths],
            name="Median",
            line={"color": "#60a5fa", "width": 2},
        )
    )
    fan_fig.add_hline(y=1, line_dash="dot", line_color="rgba(255,255,255,0.3)")
    fan_fig.update_layout(
        title=f"Monte Carlo Fan — {mc.iterations:,} paths, {mc.horizon_days}d",
        yaxis_title="Portfolio Value (x initial)",
        xaxis_title="Trading Days",
        height=380,
        margin={"l": 55, "r": 20, "t": 50, "b": 45},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
        legend={"orientation": "h", "y": -0.22},
        yaxis={"gridcolor": "rgba(255,255,255,0.06)"},
        xaxis={"gridcolor": "rgba(255,255,255,0.06)"},
    )
    st.plotly_chart(fan_fig, use_container_width=True)

with hist_col:
    pct_vals = [
        mc.terminal_p05,
        mc.terminal_p25,
        mc.terminal_p50,
        mc.terminal_p75,
        mc.terminal_p95,
    ]
    hist_fig = go.Figure(
        go.Bar(
            x=["p05", "p25", "p50", "p75", "p95"],
            y=pct_vals,
            marker_color=["#ef4444", "#f59e0b", "#22c55e", "#3b82f6", "#8b5cf6"],
            text=[f"{v:.3f}x" for v in pct_vals],
            textposition="outside",
        )
    )
    hist_fig.add_hline(y=1, line_dash="dot", line_color="rgba(255,255,255,0.3)")
    hist_fig.update_layout(
        title="Terminal Value Distribution (percentiles)",
        yaxis_title="Value (x initial)",
        height=380,
        margin={"l": 55, "r": 20, "t": 50, "b": 45},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
        showlegend=False,
        yaxis={"gridcolor": "rgba(255,255,255,0.06)"},
    )
    st.plotly_chart(hist_fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Panel 5 -- Factor Attribution
# ---------------------------------------------------------------------------

st.markdown("### 6 · Factor Attribution")

if benchmark_returns is None or len(benchmark_returns) < 20:
    st.info(
        "Enable **Include XU100 benchmark** to compute factor attribution. "
        "At least 20 aligned data points are required."
    )
else:
    attr = compute_performance_attribution_breakdown(
        returns_raw, benchmark_returns
    )

    attr_factors = {
        "Asset Allocation": attr.asset_allocation_pct,
        "Stock Selection": attr.stock_selection_pct,
        "Sector Rotation": attr.sector_rotation_pct,
        "Currency Exposure": attr.currency_exposure_pct,
        "Benchmark Relative": attr.benchmark_relative_pct,
    }

    bar_col, summary_col = st.columns([3, 1])

    with bar_col:
        attr_fig = go.Figure(
            go.Bar(
                x=list(attr_factors.keys()),
                y=list(attr_factors.values()),
                marker_color=[
                    "#22c55e" if v >= 0 else "#ef4444"
                    for v in attr_factors.values()
                ],
                text=[f"{v:+.2f}%" for v in attr_factors.values()],
                textposition="outside",
            )
        )
        attr_fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
        attr_fig.update_layout(
            title="Performance Attribution vs XU100 Benchmark",
            yaxis_title="Contribution (%)",
            height=360,
            margin={"l": 55, "r": 20, "t": 50, "b": 45},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#e2e8f0"},
            showlegend=False,
            yaxis={"gridcolor": "rgba(255,255,255,0.06)"},
        )
        st.plotly_chart(attr_fig, use_container_width=True)

    with summary_col:
        st.markdown("**Attribution Summary**")
        st.metric("Benchmark Relative", f"{attr.benchmark_relative_pct:+.2f}%")
        st.metric("Asset Allocation", f"{attr.asset_allocation_pct:+.2f}%")
        st.metric("Stock Selection", f"{attr.stock_selection_pct:+.2f}%")
        st.metric("Sector Rotation", f"{attr.sector_rotation_pct:+.2f}%")
        st.metric("Currency Exposure", f"{attr.currency_exposure_pct:+.2f}%")
        st.metric("Style Drift Score", f"{attr.style_drift_score:.2f}")

st.divider()
st.caption(
    f"Analytics computed on {len(returns_raw):,} daily returns · {run_label}"
)
