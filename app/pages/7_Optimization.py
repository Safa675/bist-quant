"""Optimization — interactive parameter sweep with heatmap visualization."""

from __future__ import annotations

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

st.set_page_config(
    page_title="Optimization · BIST Quant", page_icon="⚙️", layout="wide"
)

from app.layout import page_header, render_sidebar  # noqa: E402
from app.services import get_core_service  # noqa: E402
from app.utils import fmt_num, fmt_pct, run_in_thread  # noqa: E402

render_sidebar()
page_header("⚙️ Optimization", "Parameter sweep heatmaps & strategy optimization")

# ── session state ─────────────────────────────────────────────────────────────
for _k, _v in [
    ("opt_future", None),
    ("opt_result", None),
    ("opt_signal", None),
    ("opt_active_param_specs", []),
    ("opt_base_request", {}),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── service ───────────────────────────────────────────────────────────────────
core = get_core_service()
if core is None:
    st.warning("Core service unavailable. Please check your bist_quant installation.")
    st.stop()

try:
    signal_names: list[str] = core.list_available_signals()
except Exception as _exc:
    st.error(f"Failed to load signals: {_exc}")
    st.stop()

if not signal_names:
    st.warning("No signals available.")
    st.stop()


# ── helpers ───────────────────────────────────────────────────────────────────

def _display_key(full_key: str) -> str:
    """Return the last path segment as a human-readable label."""
    return full_key.split(".")[-1]


def _infer_type(value: Any) -> str:
    """Guess 'int' or 'float' from a Python value."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    return "float"


def _trials_to_df(trials: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten trial list into a tidy DataFrame."""
    rows = []
    for t in trials:
        if not isinstance(t.get("params"), dict) or not isinstance(t.get("metrics"), dict):
            continue
        row: dict[str, Any] = {}
        for k, v in t["params"].items():
            row[_display_key(k)] = v
        m = t["metrics"]
        row["sharpe"] = m.get("sharpe")
        row["cagr"] = (m.get("cagr") or 0) * 100
        row["max_dd"] = (m.get("max_drawdown") or 0) * 100
        row["total_ret"] = (m.get("total_return") or 0) * 100
        row["feasible"] = t.get("feasible", False)
        row["score"] = t.get("score")
        row["trial_id"] = t.get("trial_id")
        rows.append(row)
    return pd.DataFrame(rows)


def _heatmap_fig(
    trials_df: pd.DataFrame,
    param_x: str,
    param_y: str,
    metric: str = "sharpe",
) -> go.Figure:
    """Build a 2-D heatmap of *metric* over param_x × param_y."""
    pivot = (
        trials_df.dropna(subset=[param_x, param_y, metric])
        .pivot_table(values=metric, index=param_y, columns=param_x, aggfunc="max")
    )
    if pivot.empty:
        return go.Figure()

    x_vals = [float(v) for v in pivot.columns.tolist()]
    y_vals = [float(v) for v in pivot.index.tolist()]
    z_vals = pivot.values.tolist()

    fig = go.Figure(
        go.Heatmap(
            x=x_vals,
            y=y_vals,
            z=z_vals,
            colorscale="RdYlGn",
            colorbar=dict(title=metric.replace("_", " ").title()),
            hoverongaps=False,
            hovertemplate=(
                f"{param_x}: %{{x}}<br>"
                f"{param_y}: %{{y}}<br>"
                f"{metric}: %{{z:.3f}}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        xaxis_title=param_x,
        yaxis_title=param_y,
        margin=dict(l=20, r=20, t=40, b=20),
        height=440,
    )
    return fig


def _scatter_fig(
    trials_df: pd.DataFrame,
    param_x: str,
    metric_y: str = "sharpe",
) -> go.Figure:
    """1-D line+scatter: metric vs single swept parameter."""
    df = trials_df.dropna(subset=[param_x, metric_y]).sort_values(param_x)
    fig = go.Figure(
        go.Scatter(
            x=df[param_x],
            y=df[metric_y],
            mode="lines+markers",
            marker=dict(
                color=df[metric_y],
                colorscale="RdYlGn",
                size=8,
                showscale=True,
                colorbar=dict(title=metric_y),
            ),
            line=dict(color="rgba(100,150,255,0.4)", width=1.5),
            hovertemplate=f"{param_x}: %{{x}}<br>{metric_y}: %{{y:.3f}}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e0e0e0"),
        xaxis_title=param_x,
        yaxis_title=metric_y,
        height=380,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.07)")
    return fig


def _render_best_trial_card(best: dict[str, Any], signal_name: str) -> None:
    """Render the best-configuration card with an auto-fill button."""
    m = best.get("metrics", {})
    params = best.get("params", {})

    st.markdown("#### 🏆 Best Configuration")
    bc1, bc2, bc3, bc4 = st.columns(4)
    bc1.metric("Sharpe", fmt_num(m.get("sharpe")))
    bc2.metric("CAGR", fmt_pct((m.get("cagr") or 0) * 100))
    bc3.metric("Max DD", fmt_pct((m.get("max_drawdown") or 0) * 100))
    bc4.metric("Win Rate", fmt_pct((m.get("win_rate") or 0) * 100))

    with st.container(border=True):
        st.markdown("**Optimal Parameters**")
        if params:
            n_cols = min(len(params), 4)
            p_cols = st.columns(n_cols)
            for i, (k, v) in enumerate(sorted(params.items())):
                p_cols[i % n_cols].metric(
                    _display_key(k),
                    fmt_num(float(v)) if isinstance(v, (int, float)) else str(v),
                )
        else:
            st.caption("No parameters recorded in best trial.")

        st.markdown("")
        if st.button(
            "↗️ Send Best Config to Backtest",
            type="secondary",
            use_container_width=True,
            key="opt_send_bt",
        ):
            best_sp: dict[str, Any] = {}
            best_top_n: int | None = None
            for k, v in params.items():
                dk = _display_key(k)
                if "signal_params" in k:
                    best_sp[dk] = v
                elif dk == "top_n":
                    best_top_n = int(v)

            st.session_state["bt_prefill"] = {
                "signal": signal_name,
                "signal_params": best_sp,
                "top_n": best_top_n,
            }
            st.success(
                "✅ Best config stored. Navigate to **🔄 Backtest** — "
                "the signal and top-N will be pre-populated."
            )


def _render_opt_results(result: dict[str, Any]) -> None:
    """Full results render: summary stats, heatmap, trial table, export."""
    signal_name = st.session_state.get("opt_signal", "?")

    s1, s2, s3 = st.columns(3)
    s1.metric("Total Trials", result.get("total_trials", 0))
    s2.metric("Feasible Trials", result.get("feasible_trials", 0))
    s3.metric("Method", str(result.get("method", "—")).upper())

    st.divider()

    best = result.get("best_trial") or {}
    _render_best_trial_card(best, signal_name)

    trials: list[dict[str, Any]] = result.get("trials", [])
    if not trials:
        st.info("No trial data returned from optimizer.")
        return

    trials_df = _trials_to_df(trials)
    if trials_df.empty:
        st.info("No numeric trial data to visualize.")
        return

    metric_cols = {"sharpe", "cagr", "max_dd", "total_ret", "feasible", "score", "trial_id"}
    param_cols = [c for c in trials_df.columns if c not in metric_cols]

    tab_hm, tab_table, tab_export = st.tabs(
        ["📊 Heatmap", "📋 All Trials", "💾 Export"]
    )

    # ── Heatmap tab ───────────────────────────────────────────────────────────
    with tab_hm:
        hm_metric = st.selectbox(
            "Metric to visualize",
            ["sharpe", "cagr", "max_dd", "total_ret"],
            format_func=lambda v: {
                "sharpe": "Sharpe Ratio",
                "cagr": "CAGR %",
                "max_dd": "Max Drawdown %",
                "total_ret": "Total Return %",
            }.get(v, v),
            key="opt_hm_metric",
        )

        if len(param_cols) >= 2:
            h1, h2 = st.columns(2)
            with h1:
                px = st.selectbox("X-axis param", param_cols, index=0, key="opt_hm_px")
            with h2:
                rest = [p for p in param_cols if p != px]
                py = st.selectbox("Y-axis param", rest, index=0, key="opt_hm_py")

            fig_hm = _heatmap_fig(trials_df, px, py, hm_metric)
            if fig_hm.data:
                st.plotly_chart(fig_hm, use_container_width=True)
                st.caption(
                    "Each cell shows the best observed "
                    f"**{hm_metric}** for that (X, Y) parameter pair."
                )
            else:
                st.info(
                    "Not enough data to build a heatmap. "
                    "Try increasing the number of trials or widening the sweep range."
                )

        elif len(param_cols) == 1:
            st.caption(f"Single parameter sweep over **{param_cols[0]}**")
            fig_sc = _scatter_fig(trials_df, param_cols[0], hm_metric)
            st.plotly_chart(fig_sc, use_container_width=True)

        else:
            st.info(
                "No swept parameter columns found. "
                "Ensure at least one param is enabled in the sweep config."
            )

    # ── All Trials tab ────────────────────────────────────────────────────────
    with tab_table:
        n_shown = st.slider("Max trials to display", 10, min(500, len(trials_df)), min(200, len(trials_df)), 10)
        show_df = trials_df.copy()
        for col in ["sharpe", "cagr", "max_dd", "total_ret", "score"]:
            if col in show_df.columns:
                show_df[col] = show_df[col].apply(
                    lambda v: f"{v:.4f}" if pd.notna(v) else "—"
                )
        sort_col = "trial_id" if "trial_id" in show_df.columns else show_df.columns[0]
        st.dataframe(
            show_df.sort_values(sort_col).head(n_shown),
            use_container_width=True,
            hide_index=True,
        )

    # ── Export tab ────────────────────────────────────────────────────────────
    with tab_export:
        csv_bytes = _trials_to_df(trials).to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download Full Sweep Results (CSV)",
            data=csv_bytes,
            file_name=f"{signal_name}_optimization_sweep.csv",
            mime="text/csv",
            use_container_width=True,
        )
        pareto = result.get("pareto_front", [])
        if pareto:
            pareto_csv = _trials_to_df(pareto).to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download Pareto Front (CSV)",
                data=pareto_csv,
                file_name=f"{signal_name}_pareto_front.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ── layout: config (left) | results (right) ──────────────────────────────────
cfg_col, _sp, result_col = st.columns([1, 0.04, 2.3])

# ═════════════════════════════════════════════════════════════════════════════
# LEFT — configuration panel
# ═════════════════════════════════════════════════════════════════════════════
with cfg_col:
    st.markdown("### 🎯 Signal")

    _default_sig_idx = signal_names.index("momentum") if "momentum" in signal_names else 0
    selected_signal = st.selectbox(
        "Signal / Factor",
        options=signal_names,
        index=_default_sig_idx,
        key="opt_sel_signal",
    )

    # Load signal details
    signal_details: dict[str, Any] = {}
    try:
        signal_details = core.get_signal_details(selected_signal)
    except Exception as _exc:
        st.warning(f"Could not load signal details: {_exc}")

    signal_params_defaults = signal_details.get("signal_params", {})
    portfolio_opts = signal_details.get("portfolio_options", {})
    timeline = signal_details.get("timeline", {})
    _default_start = timeline.get("start_date", "2014-01-01")
    _default_end = timeline.get("end_date", str(date.today()))

    # ── Date range ───────────────────────────────────────────────────────────
    st.markdown("### 📅 Date Range")
    _cs, _ce = st.columns(2)
    with _cs:
        start_date = st.date_input(
            "Start",
            value=datetime.strptime(_default_start[:10], "%Y-%m-%d").date(),
            min_value=date(2010, 1, 1),
            max_value=date.today(),
            key="opt_start",
        )
    with _ce:
        end_date = st.date_input(
            "End",
            value=min(datetime.strptime(_default_end[:10], "%Y-%m-%d").date(), date.today()),
            min_value=date(2010, 1, 1),
            max_value=date.today(),
            key="opt_end",
        )

    # ── Base portfolio options ───────────────────────────────────────────────
    st.markdown("### 🎛 Base Config")
    base_top_n = int(
        st.number_input(
            "Base Top N",
            min_value=5,
            max_value=100,
            value=int(portfolio_opts.get("top_n", 20)),
            step=5,
            key="opt_base_top_n",
        )
    )
    _rf_opts = ["monthly", "quarterly", "weekly"]
    _rf_default = signal_details.get("rebalance_frequency", "monthly")
    _rf_default = _rf_default if _rf_default in _rf_opts else "monthly"
    rebalance_freq = st.selectbox(
        "Rebalance",
        _rf_opts,
        index=_rf_opts.index(_rf_default),
        key="opt_rebalance_freq",
    )

    # ── Parameter sweep grid ─────────────────────────────────────────────────
    st.markdown("### 🔢 Parameter Sweep")

    all_sweep_specs: list[dict[str, Any]] = []

    # Signal-level params
    for _pk, _pv in sorted(signal_params_defaults.items()):
        if _pv is None or isinstance(_pv, bool):
            continue
        _pt = _infer_type(_pv)
        if _pt not in ("int", "float"):
            continue
        _fv = float(_pv)
        if _pt == "int":
            _iv = max(1, int(round(_fv)))
            all_sweep_specs.append({
                "label": _pk.replace("_", " ").title(),
                "key": f"factors.0.signal_params.{_pk}",
                "display_key": _pk,
                "type": "int",
                "default_val": _iv,
                "default_min": max(1, _iv // 2),
                "default_max": _iv * 3,
                "default_step": max(1, _iv // 5),
            })
        else:
            all_sweep_specs.append({
                "label": _pk.replace("_", " ").title(),
                "key": f"factors.0.signal_params.{_pk}",
                "display_key": _pk,
                "type": "float",
                "default_val": round(_fv, 4),
                "default_min": round(max(0.0001, _fv / 2), 4),
                "default_max": round(_fv * 2, 4),
                "default_step": round(max(0.001, _fv / 5), 4),
            })

    # Portfolio-level top_n sweep
    _top_n_def = int(portfolio_opts.get("top_n", 20))
    all_sweep_specs.append({
        "label": "Top N",
        "key": "top_n",
        "display_key": "top_n",
        "type": "int",
        "default_val": _top_n_def,
        "default_min": max(5, _top_n_def // 2),
        "default_max": min(80, _top_n_def * 3),
        "default_step": 5,
    })

    active_param_specs: list[dict[str, Any]] = []

    if not all_sweep_specs:
        st.info("No numeric tunable parameters found for this signal.")
    else:
        for _spec in all_sweep_specs:
            with st.expander(f"⚙️ {_spec['label']}", expanded=True):
                _enabled = st.checkbox(
                    "Sweep this parameter",
                    value=False,
                    key=f"opt_en_{_spec['key']}",
                )
                if _enabled:
                    _c1, _c2, _c3 = st.columns(3)
                    if _spec["type"] == "int":
                        with _c1:
                            _pmin = int(st.number_input(
                                "Min", value=int(_spec["default_min"]),
                                min_value=1, step=1,
                                key=f"opt_mn_{_spec['key']}",
                            ))
                        with _c2:
                            _pmax = int(st.number_input(
                                "Max", value=int(_spec["default_max"]),
                                min_value=2, step=1,
                                key=f"opt_mx_{_spec['key']}",
                            ))
                        with _c3:
                            _pstep = int(st.number_input(
                                "Step", value=int(_spec["default_step"]),
                                min_value=1, step=1,
                                key=f"opt_st_{_spec['key']}",
                            ))
                    else:
                        with _c1:
                            _pmin = float(st.number_input(
                                "Min", value=float(_spec["default_min"]),
                                format="%.4f",
                                key=f"opt_mn_{_spec['key']}",
                            ))
                        with _c2:
                            _pmax = float(st.number_input(
                                "Max", value=float(_spec["default_max"]),
                                format="%.4f",
                                key=f"opt_mx_{_spec['key']}",
                            ))
                        with _c3:
                            _pstep = float(st.number_input(
                                "Step", value=float(_spec["default_step"]),
                                min_value=0.0001, format="%.4f",
                                key=f"opt_st_{_spec['key']}",
                            ))

                    # estimated grid size
                    if _spec["type"] == "int" and _pstep > 0 and _pmax > _pmin:
                        _npts = max(1, (_pmax - _pmin) // _pstep + 1)
                        st.caption(f"~{_npts} grid point{'s' if _npts != 1 else ''}")

                    active_param_specs.append({
                        "key": _spec["key"],
                        "display_key": _spec["display_key"],
                        "label": _spec["label"],
                        "type": _spec["type"],
                        "min": _pmin,
                        "max": _pmax,
                        "step": _pstep,
                    })
                else:
                    st.caption(f"Default: `{_spec['default_val']}`")

    # ── Optimizer options ─────────────────────────────────────────────────────
    st.markdown("### 🔧 Optimizer")
    _co1, _co2 = st.columns(2)
    with _co1:
        opt_method = st.selectbox("Method", ["grid", "random"], key="opt_method")
    with _co2:
        max_trials = int(st.number_input(
            "Max Trials",
            min_value=4, max_value=500, value=50, step=10,
            key="opt_max_trials",
        ))
    train_ratio = st.slider(
        "Train / Val split",
        min_value=0.5, max_value=0.9, value=0.7, step=0.05,
        help="Fraction of the date range used for training; the rest scores each trial.",
        key="opt_train_ratio",
    )

    st.markdown("---")

    _n_active = len(active_param_specs)
    _is_running = st.session_state["opt_future"] is not None
    _btn_label = (
        f"▶ Run Optimization  ({_n_active} param{'s' if _n_active != 1 else ''})"
        if _n_active > 0
        else "▶ Run Optimization  (enable a param ↑)"
    )
    run_btn = st.button(
        _btn_label,
        type="primary",
        use_container_width=True,
        disabled=(_n_active == 0 or _is_running),
        key="opt_run_btn",
    )

    if run_btn and _n_active > 0 and not _is_running:
        if start_date >= end_date:
            st.error("Start date must be before end date.")
        else:
            _base_req: dict[str, Any] = {
                "factor_name": None,
                "factors": [{"name": selected_signal, "weight": 1.0}],
                "start_date": str(start_date),
                "end_date": str(end_date),
                "rebalance_frequency": rebalance_freq,
                "top_n": base_top_n,
            }
            _param_space = [
                {
                    "key": s["key"],
                    "type": s["type"],
                    "min": s["min"],
                    "max": s["max"],
                    "step": s["step"],
                }
                for s in active_param_specs
            ]
            _fut: Future = run_in_thread(
                core.optimize_strategy,
                base_request=_base_req,
                method=opt_method,
                parameter_space=_param_space,
                max_trials=max_trials,
                train_ratio=train_ratio,
            )
            st.session_state["opt_future"] = _fut
            st.session_state["opt_result"] = None
            st.session_state["opt_signal"] = selected_signal
            st.session_state["opt_active_param_specs"] = active_param_specs
            st.session_state["opt_base_request"] = _base_req
            st.rerun()

    if _is_running:
        st.info("⏳ Optimization in progress…")
    elif _n_active == 0:
        st.caption("⚠️ Enable at least one parameter above to begin.")


# ═════════════════════════════════════════════════════════════════════════════
# RIGHT — results panel
# ═════════════════════════════════════════════════════════════════════════════
with result_col:
    _fut = st.session_state.get("opt_future")

    # ── Poll running future ───────────────────────────────────────────────────
    if _fut is not None:
        if not _fut.done():
            with st.status(
                "⏳ Optimizing — running backtests for each trial…", expanded=True
            ) as _status:
                _ph = st.empty()
                _prog = st.progress(0.0)
                _t0 = time.time()
                while not _fut.done():
                    _el = time.time() - _t0
                    _prog.progress(min(0.95, _el / 300.0))
                    _ph.markdown(f"Elapsed: `{_el:.1f}s`  |  Trials running in background…")
                    time.sleep(0.5)
                    if _el > 900:
                        _status.update(
                            label="⌛ Timeout — optimization took too long.",
                            state="error",
                        )
                        st.session_state["opt_future"] = None
                        break
                if _fut.done():
                    _prog.progress(1.0)
                    _status.update(label="✅ Optimization complete!", state="complete")

        if _fut is not None and _fut.done():
            try:
                _res = _fut.result()
                st.session_state["opt_result"] = _res
                st.session_state["opt_future"] = None
                st.rerun()
            except Exception as _exc:
                st.error(f"Optimization failed: {_exc}")
                st.session_state["opt_future"] = None

    # ── Render completed results ──────────────────────────────────────────────
    _opt_result = st.session_state.get("opt_result")

    if _opt_result is None and st.session_state.get("opt_future") is None:
        st.markdown(
            """
            <div style="
                display:flex; flex-direction:column; align-items:center;
                justify-content:center; height:55vh;
                color:#555; font-size:1.1rem;
            ">
                <div style="font-size:3.5rem; margin-bottom:1rem;">⚙️</div>
                <div>Configure parameters on the left and click
                    <b>Run Optimization</b></div>
                <div style="margin-top:0.6rem; color:#444; font-size:0.9rem;">
                    Results include a Sharpe heatmap, full trial table,
                    and best-config auto-fill for the Backtest page.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif _opt_result is not None:
        _render_opt_results(_opt_result)
