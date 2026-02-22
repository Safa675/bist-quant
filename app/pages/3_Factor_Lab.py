"""Factor Lab — browse, combine, and analyse factor signals."""

from __future__ import annotations

import sys
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Factor Lab · BIST Quant", page_icon="🧪", layout="wide")

from app.layout import page_header, render_sidebar  # noqa: E402
from app.services import get_core_service  # noqa: E402
from app.ui import (  # noqa: E402
    ACCENT,
    BG_ELEVATED,
    BG_SURFACE,
    BORDER_DEFAULT,
    FONT_MONO,
    FONT_SANS,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    apply_chart_style,
    badge,
    metric_row,
)
from app.utils import fmt_num, fmt_pct, run_in_thread  # noqa: E402

render_sidebar()
page_header("🧪 Factor Lab", "Browse, combine, and back-test factor signals")

# ── Category taxonomy ─────────────────────────────────────────────────────────
CATEGORY_MAP: dict[str, list[str]] = {
    "Momentum": [
        "momentum", "consistent_momentum", "residual_momentum",
        "momentum_reversal_volatility", "sector_rotation",
        "short_term_reversal", "trend_following", "low_volatility",
    ],
    "Value": [
        "value", "asset_growth", "investment", "dividend_rotation",
        "macro_hedge", "small_cap",
    ],
    "Quality": [
        "profitability", "roa", "accrual", "earnings_quality", "fscore_reversal",
    ],
    "Technical": [
        "sma", "donchian", "xu100",
    ],
    "Composite": [
        "betting_against_beta", "breakout_value", "five_factor_rotation",
        "momentum_asset_growth", "pairs_trading", "quality_momentum",
        "quality_value", "size_rotation", "size_rotation_momentum",
        "size_rotation_quality", "small_cap_momentum", "trend_value",
    ],
}

CATEGORY_COLORS: dict[str, str] = {
    "Momentum": ACCENT,
    "Value":    "#10B981",
    "Quality":  "#8B5CF6",
    "Technical":"#F59E0B",
    "Composite":"#EC4899",
}

CATEGORY_ICONS: dict[str, str] = {
    "Momentum": "🚀",
    "Value":    "💎",
    "Quality":  "🏆",
    "Technical":"📡",
    "Composite":"🔀",
}

REBALANCE_ICON: dict[str, str] = {
    "monthly":   "🗓 Monthly",
    "weekly":    "📆 Weekly",
    "daily":     "📅 Daily",
    "quarterly": "🗓 Quarterly",
    "annual":    "📋 Annual",
}


def _signal_category(name: str) -> str:
    for cat, signals in CATEGORY_MAP.items():
        if name in signals:
            return cat
    return "Composite"


def _hex_to_rgb(hex_color: str) -> str:
    """Convert '#2563EB' → '37,99,235' for use in rgba()."""
    h = hex_color.lstrip("#")
    return f"{int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)}"


# ── session state ─────────────────────────────────────────────────────────────
for _key, _default in [
    ("fl_cat_filter",     "All"),
    ("fl_catalog",        None),
    ("fl_selected",       []),
    ("fl_weights",        {}),
    ("fl_bt_future",      None),
    ("fl_bt_results",     None),
    ("fl_ind_future",     None),
    ("fl_ind_results",    {}),
    ("fl_quick_bt",       {}),
    ("fl_quick_future",   None),
    ("fl_quick_signal",   None),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

# ── load signal catalog ───────────────────────────────────────────────────────
core = get_core_service()


@st.cache_data(ttl=600, show_spinner=False)
def _load_catalog() -> dict[str, dict[str, Any]]:
    if core is None:
        return {}
    try:
        configs = core.load_signal_configs()
        return {k: dict(v) for k, v in configs.items()}
    except Exception as exc:
        return {}


catalog: dict[str, dict[str, Any]] = _load_catalog()

if not catalog:
    st.error("No signals available. Ensure the bist_quant package and data are installed.")
    st.stop()

all_signal_names = sorted(catalog.keys())

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Signal Catalog
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## 📋 Signal Catalog")
st.caption(f"**{len(all_signal_names)} signals** registered across 5 categories")

# Category filter ──────────────────────────────────────────────────────────────
all_categories = ["All"] + list(CATEGORY_MAP.keys())
cat_cols = st.columns(len(all_categories))
for idx, cat in enumerate(all_categories):
    icon = "" if cat == "All" else CATEGORY_ICONS.get(cat, "")
    count = (
        len(all_signal_names)
        if cat == "All"
        else sum(1 for s in all_signal_names if _signal_category(s) == cat)
    )
    with cat_cols[idx]:
        if st.button(
            f"{icon} {cat}  ({count})",
            key=f"cat_btn_{cat}",
            use_container_width=True,
        ):
            st.session_state["fl_cat_filter"] = cat
            st.rerun()

active_cat: str = st.session_state.get("fl_cat_filter", "All")

# Visual indicator for active category
st.markdown(
    f"<div style='font-size:0.8rem;color:#aaa;margin:4px 0 12px;'>"
    f"Showing: <b style='color:#e0e0e0;'>{active_cat}</b></div>",
    unsafe_allow_html=True,
)

# Search box ──────────────────────────────────────────────────────────────────
search_col, _ = st.columns([2, 3])
with search_col:
    search_query = st.text_input(
        "🔍 Search signals",
        placeholder="e.g. momentum, value, rsi…",
        key="fl_search",
        label_visibility="collapsed",
    )


def _filter_signals() -> list[str]:
    names = list(all_signal_names)
    if active_cat != "All":
        names = [n for n in names if _signal_category(n) == active_cat]
    if search_query.strip():
        q = search_query.strip().lower()
        names = [
            n for n in names
            if q in n.lower()
            or q in (catalog.get(n, {}).get("description") or "").lower()
        ]
    return names


visible_signals = _filter_signals()

if not visible_signals:
    st.info("No signals match the current filter.")
else:
    num_cols = 3
    rows = [visible_signals[i:i + num_cols] for i in range(0, len(visible_signals), num_cols)]

    for row in rows:
        cols = st.columns(num_cols)
        for col_idx, signal_name in enumerate(row):
            cfg = catalog.get(signal_name, {})
            cat = _signal_category(signal_name)
            color = CATEGORY_COLORS.get(cat, "#7f8c8d")
            icon = CATEGORY_ICONS.get(cat, "📌")
            desc = cfg.get("description") or "—"
            rebal = REBALANCE_ICON.get(cfg.get("rebalance_frequency", ""), "")
            enabled = cfg.get("enabled", True)
            timeline = cfg.get("timeline") or {}
            start_yr = str(timeline.get("start_date", ""))[:4] if timeline.get("start_date") else "?"
            end_yr = str(timeline.get("end_date", ""))[:4] if timeline.get("end_date") else "?"
            quick_stats = st.session_state["fl_quick_bt"].get(signal_name, {})

            with cols[col_idx]:
                with st.container(border=True):
                    st.markdown(
                        f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px;'>"
                        f"<span style='font-size:1rem;font-weight:600;color:{TEXT_PRIMARY};'>{signal_name}</span>"
                        f"<span class='bq-badge' style='background:rgba({_hex_to_rgb(color)},0.15);"
                        f"color:{color};'>{icon} {cat}</span>"
                        f"{'<span class=\"bq-badge bq-badge-danger\">(disabled)</span>' if not enabled else ''}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    st.caption(f"_{desc[:100]}{'…' if len(desc) > 100 else ''}_")

                    meta_parts = []
                    if rebal:
                        meta_parts.append(rebal)
                    if start_yr != "?":
                        meta_parts.append(f"📅 {start_yr}–{end_yr}")
                    if meta_parts:
                        st.caption("  ·  ".join(meta_parts))

                    # Quick backtest stats badge
                    if quick_stats and not quick_stats.get("error"):
                        qm = quick_stats.get("metrics", {})
                        st.markdown(
                            f"<div class='bq-stat-row'>"
                            f"<span>CAGR <b>{fmt_pct(qm.get('cagr', 0) * 100)}</b></span>"
                            f"<span>Sharpe <b>{fmt_num(qm.get('sharpe', 0))}</b></span>"
                            f"<span>DD <b>{fmt_pct(qm.get('max_drawdown', 0) * 100)}</b></span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                    elif quick_stats and quick_stats.get("error"):
                        st.caption(f"⚠️ Backtest error: {quick_stats['error'][:60]}")

                    with st.expander("Details & Run Backtest"):
                        signal_params = cfg.get("signal_params") or cfg.get("parameters") or {}
                        portfolio_opts = cfg.get("portfolio_options") or {}

                        if signal_params:
                            st.markdown("**Parameters:**")
                            param_rows = [{"Parameter": k, "Default": v} for k, v in signal_params.items()]
                            st.dataframe(
                                pd.DataFrame(param_rows),
                                use_container_width=True,
                                hide_index=True,
                                height=min(35 * len(param_rows) + 38, 200),
                            )
                        else:
                            st.caption("_No configurable signal_params in config_")

                        # Param schemas from factor_lab PARAM_SCHEMAS
                        try:
                            from bist_quant.engines.factor_lab import PARAM_SCHEMAS
                            schema = PARAM_SCHEMAS.get(signal_name, [])
                            if schema:
                                st.markdown("**Tunable param ranges:**")
                                schema_rows = [
                                    {
                                        "Param": p["label"],
                                        "Default": p["default"],
                                        "Min": p.get("min", "—"),
                                        "Max": p.get("max", "—"),
                                        "Type": p.get("type", "—"),
                                    }
                                    for p in schema
                                ]
                                st.dataframe(
                                    pd.DataFrame(schema_rows),
                                    use_container_width=True,
                                    hide_index=True,
                                )
                        except Exception:
                            pass

                        if portfolio_opts:
                            regime = "✅" if portfolio_opts.get("use_regime_filter") else "❌"
                            vol = "✅" if portfolio_opts.get("use_vol_targeting") else "❌"
                            sl = "✅" if portfolio_opts.get("use_stop_loss") else "❌"
                            topn = portfolio_opts.get("top_n", "—")
                            st.markdown(
                                f"**Portfolio:** Regime {regime} · Vol target {vol} "
                                f"· Stop-loss {sl} · Top-N **{topn}**"
                            )

                        b_col1, b_col2 = st.columns(2)
                        with b_col1:
                            is_selected = signal_name in st.session_state["fl_selected"]
                            if is_selected:
                                if st.button("✅ In Portfolio", key=f"add_{signal_name}", use_container_width=True):
                                    st.session_state["fl_selected"].remove(signal_name)
                                    st.session_state["fl_weights"].pop(signal_name, None)
                                    st.rerun()
                            else:
                                if st.button("➕ Add to Portfolio", key=f"add_{signal_name}", use_container_width=True):
                                    st.session_state["fl_selected"].append(signal_name)
                                    st.rerun()

                        with b_col2:
                            is_qbt_loading = (
                                st.session_state.get("fl_quick_signal") == signal_name
                                and st.session_state.get("fl_quick_future") is not None
                                and not st.session_state["fl_quick_future"].done()
                            )
                            if is_qbt_loading:
                                st.button("⏳ Running…", key=f"qbt_{signal_name}", disabled=True, use_container_width=True)
                            elif st.button("⚡ Quick Backtest", key=f"qbt_{signal_name}", use_container_width=True):
                                if core is not None:
                                    st.session_state["fl_quick_signal"] = signal_name
                                    st.session_state["fl_quick_future"] = run_in_thread(
                                        core.run_backtest,
                                        factor_name=signal_name,
                                        start_date="2019-01-01",
                                        end_date="2024-12-31",
                                    )
                                    st.rerun()

# ── poll quick backtest future ────────────────────────────────────────────────
_qf = st.session_state.get("fl_quick_future")
if _qf is not None and _qf.done():
    _sig = st.session_state.get("fl_quick_signal", "")
    try:
        st.session_state["fl_quick_bt"][_sig] = _qf.result()
    except Exception as exc:
        st.session_state["fl_quick_bt"][_sig] = {"error": str(exc)}
    st.session_state["fl_quick_future"] = None
    st.session_state["fl_quick_signal"] = None
    st.rerun()
elif _qf is not None and not _qf.done():
    time.sleep(0.4)
    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Multi-Factor Combination
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## 🔀 Multi-Factor Portfolio Combination")
st.caption(
    "Select signals, assign weights (auto-normalized to sum 100%), then run the "
    "combined backtest to compare the blended equity curve vs individual factors."
)

sel_col, ctrl_col = st.columns([3, 1])
with sel_col:
    selected = st.multiselect(
        "Choose signals to combine",
        options=all_signal_names,
        default=st.session_state["fl_selected"],
        placeholder="Select 2+ signals…",
        key="fl_multi_select",
    )
    st.session_state["fl_selected"] = list(selected)

with ctrl_col:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Equal Weights", use_container_width=True) and selected:
        n = len(selected)
        eq = round(100.0 / n, 1)
        st.session_state["fl_weights"] = {s: eq for s in selected}
        st.rerun()

if selected:
    st.markdown("**Assign weights** (normalized to 100% before running)")
    current_weights: dict[str, float] = {}
    slider_cols = st.columns(min(len(selected), 4))
    for idx, sig in enumerate(selected):
        default_w = st.session_state["fl_weights"].get(sig, round(100.0 / len(selected), 1))
        with slider_cols[idx % min(len(selected), 4)]:
            cat = _signal_category(sig)
            color = CATEGORY_COLORS.get(cat, "#7f8c8d")
            st.markdown(
                f"<span style='font-size:0.75rem;background:{color};color:#fff;"
                f"border-radius:8px;padding:1px 6px;'>{CATEGORY_ICONS.get(cat,'')}{cat}</span>",
                unsafe_allow_html=True,
            )
            w = st.slider(sig, min_value=0.0, max_value=100.0, value=float(default_w), step=5.0, key=f"wslider_{sig}")
            current_weights[sig] = w
            st.session_state["fl_weights"][sig] = w

    total_w = sum(current_weights.values())
    if total_w > 0:
        norm_weights = {s: w / total_w for s, w in current_weights.items()}
        wt_cols = st.columns(len(selected))
        for idx, (sig, nw) in enumerate(norm_weights.items()):
            color = CATEGORY_COLORS.get(_signal_category(sig), "#7f8c8d")
            with wt_cols[idx]:
                st.markdown(
                    f"<div style='text-align:center;'>"
                    f"<div style='font-size:1.35rem;font-weight:700;color:{color};'>{nw*100:.1f}%</div>"
                    f"<div style='font-size:0.72rem;color:#aaa;'>{sig}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)

    cfg_cols = st.columns([1, 1, 1, 2])
    with cfg_cols[0]:
        comb_start = st.date_input("Start", value=pd.Timestamp("2016-01-01"), key="fl_comb_start")
    with cfg_cols[1]:
        comb_end = st.date_input("End", value=pd.Timestamp("2024-12-31"), key="fl_comb_end")
    with cfg_cols[2]:
        comb_scheme = st.selectbox(
            "Weighting scheme",
            ["custom", "equal", "risk_parity", "min_variance"],
            key="fl_scheme",
        )
    with cfg_cols[3]:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button(
            "🚀 Run Combined Backtest",
            use_container_width=True,
            disabled=(len(selected) < 2 or core is None or total_w == 0),
            key="fl_run_btn",
        )

    if run_btn and len(selected) >= 2 and core is not None and total_w > 0:
        factors_payload = [
            {"name": s, "weight": float(norm_weights[s])}
            for s in selected
            if norm_weights.get(s, 0) > 0
        ]
        st.session_state["fl_bt_results"] = None
        st.session_state["fl_ind_results"] = {}
        st.session_state["fl_bt_future"] = run_in_thread(
            core.combine_factors,
            factors=factors_payload,
            start_date=str(comb_start),
            end_date=str(comb_end),
            weighting_scheme=comb_scheme,
        )
        for sig in selected:
            st.session_state["fl_ind_results"][sig] = run_in_thread(
                core.run_backtest,
                factor_name=sig,
                start_date=str(comb_start),
                end_date=str(comb_end),
            )
        st.rerun()

    # Poll combined future ─────────────────────────────────────────────────────
    _bf = st.session_state.get("fl_bt_future")
    if _bf is not None and not _bf.done():
        with st.spinner("Running combined backtest… ⏳"):
            time.sleep(0.5)
            st.rerun()
    elif _bf is not None and _bf.done():
        try:
            st.session_state["fl_bt_results"] = _bf.result()
        except Exception as exc:
            st.session_state["fl_bt_results"] = {"error": str(exc)}
        st.session_state["fl_bt_future"] = None
        st.rerun()

    # Poll individual futures ──────────────────────────────────────────────────
    _ind = st.session_state.get("fl_ind_results", {})
    any_pending = any(isinstance(v, Future) and not v.done() for v in _ind.values())
    for sig, fut in list(_ind.items()):
        if isinstance(fut, Future) and fut.done():
            try:
                _ind[sig] = fut.result()
            except Exception as exc:
                _ind[sig] = {"error": str(exc)}
    if any_pending:
        time.sleep(0.4)
        st.rerun()


# ─── render combination results ───────────────────────────────────────────────
def _render_combination_results(
    combined: dict[str, Any],
    individual: dict[str, Any],
    sig_list: list[str],
) -> None:
    m = combined.get("metrics", {})
    st.markdown("### Combined Portfolio Results")

    metric_row([
        {"label": "CAGR", "value": fmt_pct(m.get("cagr", 0) * 100)},
        {"label": "Sharpe", "value": fmt_num(m.get("sharpe", 0))},
        {"label": "Sortino", "value": fmt_num(m.get("sortino", 0))},
        {"label": "Max DD", "value": fmt_pct(m.get("max_drawdown", 0) * 100)},
        {"label": "Ann. Vol", "value": fmt_pct(m.get("annualized_volatility", 0) * 100)},
    ])

    tab_curve, tab_breakdown, tab_corr = st.tabs(
        ["📈 Equity Curve vs Factors", "🗂 Factor Breakdown", "🔗 Correlations"]
    )

    with tab_curve:
        ec_list = combined.get("equity_curve", [])
        if not ec_list:
            st.info("No equity curve data.")
        else:
            df_ec = pd.DataFrame(ec_list)
            df_ec["date"] = pd.to_datetime(df_ec["date"])
            fig = go.Figure()

            if "benchmark" in df_ec.columns:
                bench = df_ec.dropna(subset=["benchmark"])
                if not bench.empty:
                    fig.add_trace(go.Scatter(
                        x=bench["date"], y=bench["benchmark"] * 100,
                        name="XU100", line=dict(color=TEXT_MUTED, width=1.5, dash="dot"), mode="lines",
                    ))

            colors_ind = ["#EF4444", "#10B981", "#8B5CF6", "#F59E0B", "#06B6D4", "#F97316"]
            for idx_s, sn in enumerate(sig_list):
                ind_r = individual.get(sn, {})
                if isinstance(ind_r, dict) and not ind_r.get("error"):
                    ind_ec = ind_r.get("equity_curve", [])
                    if ind_ec:
                        df_i = pd.DataFrame(ind_ec)
                        df_i["date"] = pd.to_datetime(df_i["date"])
                        fig.add_trace(go.Scatter(
                            x=df_i["date"], y=df_i["value"] * 100,
                            name=sn, line=dict(color=colors_ind[idx_s % len(colors_ind)], width=1.2, dash="dot"),
                            opacity=0.55, mode="lines",
                        ))

            fig.add_trace(go.Scatter(
                x=df_ec["date"], y=df_ec["value"] * 100,
                name="✦ Combined", line=dict(color=ACCENT, width=2.8), mode="lines",
            ))
            apply_chart_style(fig, height=430)
            fig.update_layout(
                legend=dict(orientation="h"),
                yaxis=dict(ticksuffix="x"),
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab_breakdown:
        breakdown = combined.get("factor_breakdown", {})
        rows_bd = []
        if breakdown:
            for sn, bd in breakdown.items():
                rows_bd.append({
                    "Signal": sn,
                    "Category": _signal_category(sn),
                    "Weight": f"{bd.get('weight', 0) * 100:.1f}%",
                    "CAGR": fmt_pct(bd.get("cagr", 0) * 100),
                    "Sharpe": fmt_num(bd.get("sharpe", 0)),
                    "Max DD": fmt_pct(bd.get("max_drawdown", 0) * 100),
                })
        else:
            for sn in sig_list:
                ind_r = individual.get(sn, {})
                if isinstance(ind_r, dict) and not ind_r.get("error"):
                    im = ind_r.get("metrics", {})
                    rows_bd.append({
                        "Signal": sn,
                        "Category": _signal_category(sn),
                        "Weight": "—",
                        "CAGR": fmt_pct(im.get("cagr", 0) * 100),
                        "Sharpe": fmt_num(im.get("sharpe", 0)),
                        "Max DD": fmt_pct(im.get("max_drawdown", 0) * 100),
                    })
        if rows_bd:
            st.dataframe(pd.DataFrame(rows_bd), use_container_width=True, hide_index=True)
            # CAGR bar chart
            bar_names = [r["Signal"] for r in rows_bd] + ["Combined"]
            try:
                bar_cagr = [
                    float(r["CAGR"].replace("%","").replace("+",""))
                    for r in rows_bd
                ] + [m.get("cagr", 0) * 100]
            except Exception:
                bar_cagr = [0] * len(rows_bd) + [m.get("cagr", 0) * 100]
            bar_colors = [CATEGORY_COLORS.get(_signal_category(r["Signal"]), TEXT_MUTED) for r in rows_bd] + [ACCENT]
            fig_bar = go.Figure(go.Bar(
                x=bar_names, y=bar_cagr, marker_color=bar_colors,
                text=[f"{v:.1f}%" for v in bar_cagr], textposition="auto",
            ))
            apply_chart_style(fig_bar, height=260)
            fig_bar.update_layout(
                yaxis=dict(ticksuffix="%", title="CAGR"), showlegend=False,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    with tab_corr:
        corr_data = combined.get("correlation_matrix", {})
        if corr_data:
            factors_ord = list(corr_data.keys())
            corr_vals = [[corr_data[r].get(c, 0) for c in factors_ord] for r in factors_ord]
            fig_corr = px.imshow(
                corr_vals, x=factors_ord, y=factors_ord,
                color_continuous_scale="RdYlGn", zmin=-1, zmax=1, text_auto=".2f",
            )
            apply_chart_style(fig_corr, height=320)
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Correlation matrix unavailable for this result.")


# Render results ───────────────────────────────────────────────────────────────
_combined_res = st.session_state.get("fl_bt_results")
_ind_res = st.session_state.get("fl_ind_results", {})
_ind_resolved = {s: v for s, v in _ind_res.items() if not isinstance(v, Future)}

if _combined_res is not None:
    if isinstance(_combined_res, dict) and _combined_res.get("error"):
        st.error(f"Combination error: {_combined_res['error']}")
    elif isinstance(_combined_res, dict):
        _render_combination_results(_combined_res, _ind_resolved, st.session_state["fl_selected"])
elif not selected:
    st.info("👆 Select 2+ signals above and click **Run Combined Backtest** to compare factor blends.")
elif len(selected) < 2:
    st.info("👆 Select at least **2 signals** to enable the combination backtest.")
else:
    st.info("👆 Configure weights and click **🚀 Run Combined Backtest**.")
