"""Stock Screener — filter the BIST universe by fundamentals & signals."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).parent.parent.parent
for _p in [str(_REPO_ROOT), str(_REPO_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Stock Screener · BIST Quant", page_icon="🔍", layout="wide"
)

from app.layout import page_header, render_sidebar  # noqa: E402
from app.utils import fmt_num, fmt_pct, run_in_thread  # noqa: E402

render_sidebar()
page_header("🔍 Stock Screener", "Filter the BIST universe by fundamentals & signals")

# ── static metadata ────────────────────────────────────────────────────────────
_SECTORS_CSV = _REPO_ROOT / "data" / "bist_sector_classification.csv"

@st.cache_data(ttl=3600, show_spinner=False)
def _load_sectors() -> list[str]:
    try:
        df = pd.read_csv(_SECTORS_CSV)
        return sorted(df["sector"].dropna().unique().tolist())
    except Exception:
        return []

SECTORS: list[str] = _load_sectors()

# sector → emoji mapping (Turkish sector names)
SECTOR_ICONS: dict[str, str] = {
    "İMALAT": "🏭",
    "MALİ KURULUŞLAR": "🏦",
    "TEKNOLOJİ": "💻",
    "BİLGİ VE İLETİŞİM": "📡",
    "ELEKTRİK GAZ VE SU": "⚡",
    "GAYRİMENKUL FAALİYETLERİ": "🏢",
    "TOPTAN VE PERAKENDE TİCARET": "🛒",
    "ULAŞTIRMA VE DEPOLAMA": "🚢",
    "İNŞAAT VE BAYINDIRLIK": "🔨",
    "MADENCİLİK VE TAŞ OCAKÇILIĞI": "⛏️",
    "OTELLER VE LOKANTALAR": "🍽️",
    "TARIM, ORMANCILIK VE BALIKÇILIK": "🌾",
    "EĞİTİM SAĞLIK SPOR VE EĞLENCE HİZMETLERİ": "🏥",
    "MESLEKİ, BİLİMSEL VE TEKNİK FAALİYETLER": "🔬",
    "İDARİ VE DESTEK HİZMET FAALİYETLERİ": "📋",
}

REC_COLORS = {"AL": "#2ecc71", "TUT": "#f39c12", "SAT": "#e74c3c"}
REC_ICONS  = {"AL": "📈", "TUT": "⏸", "SAT": "📉"}

TEMPLATE_LABELS: dict[str, str] = {
    "small_cap": "🔸 Small Cap",
    "mid_cap": "🔷 Mid Cap",
    "large_cap": "🔹 Large Cap",
    "high_dividend": "💰 High Dividend",
    "high_upside": "🚀 High Upside",
    "low_pe": "📉 Low P/E",
    "high_roe": "⭐ High ROE",
    "high_net_margin": "💎 High Margin",
    "buy_recommendation": "✅ BUY Rated",
    "sell_recommendation": "❌ SELL Rated",
    "high_return": "🏆 High Return",
    "high_foreign_ownership": "🌍 Foreign Owned",
}

INDEX_OPTIONS = ["XU100", "XU050", "XU030", "XUTUM"]
SORT_FIELD_OPTIONS: list[tuple[str, str]] = [
    ("upside_potential",        "Upside Potential"),
    ("market_cap_usd",          "Market Cap (USD)"),
    ("pe",                      "P/E"),
    ("pb",                      "P/B"),
    ("roe",                     "ROE"),
    ("net_margin",              "Net Margin"),
    ("dividend_yield",          "Dividend Yield"),
    ("return_1m",               "Return 1M"),
    ("return_1y",               "Return 1Y"),
    ("rsi_14",                  "RSI 14"),
    ("volume_3m",               "Volume 3M"),
]

# ── session state ──────────────────────────────────────────────────────────────
_SS_DEFAULTS: dict[str, Any] = {
    "sc5_run_future":   None,
    "sc5_raw_result":   None,      # full result from run_stock_filter
    "sc5_df":           None,      # post-processed DataFrame (all rows)
    "sc5_error":        None,
    "sc5_watchlist":    [],
    "sc5_sort_col":     "upside_potential",
    "sc5_sort_desc":    True,
    "sc5_index":        "XU100",
    "sc5_sectors":      [],
    "sc5_rec_filter":   [],
    "sc5_template":     "",
    "sc5_show_spark":   True,
    "sc5_spark_n":      20,
    # numeric filter state
    "sc5_pe_min":       0.0,
    "sc5_pe_max":       0.0,
    "sc5_roe_min":      0.0,
    "sc5_roe_max":      0.0,
    "sc5_nm_min":       0.0,
    "sc5_nm_max":       0.0,
    "sc5_mcap_min":     0.0,
    "sc5_mcap_max":     0.0,
    "sc5_div_min":      0.0,
    "sc5_rsi_min":      0.0,
    "sc5_rsi_max":      100.0,
    "sc5_ret1m_min":   -100.0,
    "sc5_ret1m_max":    100.0,
    "sc5_page":         1,
    "sc5_page_size":    50,
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── helpers ────────────────────────────────────────────────────────────────────
def _val(key: str) -> Any:
    return st.session_state.get(key, _SS_DEFAULTS.get(key))


def _build_sc5_payload() -> dict[str, Any]:
    """Build run_stock_filter payload from current session state."""
    filters: dict[str, Any] = {}
    def _add(key: str, min_val: float, max_val: float) -> None:
        mn = min_val if min_val != 0.0 else None
        mx = max_val if max_val != 0.0 else None
        if mn is not None or mx is not None:
            filters[key] = {k: v for k, v in {"min": mn, "max": mx}.items() if v is not None}

    _add("pe",            _val("sc5_pe_min"),    _val("sc5_pe_max"))
    _add("roe",           _val("sc5_roe_min"),   _val("sc5_roe_max"))
    _add("net_margin",    _val("sc5_nm_min"),    _val("sc5_nm_max"))
    _add("market_cap_usd",_val("sc5_mcap_min"),  _val("sc5_mcap_max"))
    _add("dividend_yield",_val("sc5_div_min"),   0.0)
    _add("rsi_14",        _val("sc5_rsi_min"),   _val("sc5_rsi_max"))
    _add("return_1m",     _val("sc5_ret1m_min"), _val("sc5_ret1m_max"))

    sel_index = _val("sc5_index") or "XU100"

    payload: dict[str, Any] = {
        "index":       sel_index,
        "data_source": "local",
        "sort_by":     _val("sc5_sort_col"),
        "sort_desc":   _val("sc5_sort_desc"),
        "limit":       2000,
        # request full column set so sector + all metrics are available client-side
        "fields": [
            "symbol", "name", "sector",
            "recommendation",
            "market_cap_usd", "market_cap",
            "pe", "forward_pe", "pb", "ev_ebitda", "ev_sales",
            "dividend_yield", "upside_potential", "analyst_target_price",
            "roe", "roa", "net_margin", "ebitda_margin",
            "foreign_ratio", "foreign_change_1w", "foreign_change_1m", "float_ratio",
            "volume_3m", "volume_12m",
            "return_1w", "return_1m", "return_ytd", "return_1y",
            "revenue_growth_yoy", "net_income_growth_yoy",
            "rsi_14", "macd_hist", "atr_14_pct",
        ],
    }
    if filters:
        payload["filters"] = filters
    if _val("sc5_template"):
        payload["template"] = _val("sc5_template")
    return payload


def _apply_client_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply sector + recommendation filters that aren't in the engine payload."""
    out = df.copy()
    sectors = _val("sc5_sectors")
    if sectors:
        sector_lower = [s.lower() for s in sectors]
        out = out[out["sector"].str.lower().isin(sector_lower)]
    recs = _val("sc5_rec_filter")
    if recs:
        out = out[out["recommendation"].isin(recs)]
    return out


def _load_sparklines(symbols: list[str], n_days: int = 30) -> dict[str, list[float]]:
    """Return last n_days close prices per symbol from the engine's in-memory cache."""
    try:
        from bist_quant.engines.stock_filter import _SCREEN_CACHE  # type: ignore
        close_df: pd.DataFrame | None = _SCREEN_CACHE.get("close_df")
        if close_df is None or close_df.empty:
            return {}
        result: dict[str, list[float]] = {}
        for sym in symbols:
            if sym in close_df.columns:
                series = pd.to_numeric(close_df[sym], errors="coerce").dropna().tail(n_days)
                if len(series) >= 5:
                    result[sym] = series.tolist()
        return result
    except Exception:
        return {}


def _build_spark_fig(
    symbols: list[str],
    spark_data: dict[str, list[float]],
    ret_1m_map: dict[str, float],
    n_cols: int = 4,
) -> go.Figure:
    """Render a grid of mini sparkline charts for the given symbols."""
    available = [s for s in symbols if s in spark_data]
    if not available:
        return go.Figure()

    n_rows = max(1, (len(available) + n_cols - 1) // n_cols)
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=available,
        vertical_spacing=0.08,
        horizontal_spacing=0.04,
    )
    for idx, sym in enumerate(available):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        prices = spark_data[sym]
        if not prices:
            continue
        ret = ret_1m_map.get(sym, 0.0)
        color = "#2ecc71" if ret >= 0 else "#e74c3c"
        norm = [p / prices[0] * 100 if prices[0] else p for p in prices]
        fig.add_trace(
            go.Scatter(
                x=list(range(len(norm))),
                y=norm,
                mode="lines",
                line=dict(color=color, width=1.8),
                showlegend=False,
                hovertemplate=f"<b>{sym}</b><br>Day %{{x}}<br>%{{y:.1f}}<extra></extra>",
            ),
            row=row,
            col=col,
        )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=5, r=5, t=30, b=5),
        height=max(160, n_rows * 130),
        showlegend=False,
        font=dict(color="#e0e0e0", size=9),
    )
    for ann in fig.layout.annotations:
        ann.update(font=dict(size=9, color="#aaa"))
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    return fig


def _style_results_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a display-ready copy with formatted columns."""
    display_cols = [
        "symbol", "sector", "recommendation",
        "market_cap_usd", "pe", "pb", "roe", "net_margin",
        "dividend_yield", "rsi_14", "return_1m", "return_1y",
        "upside_potential", "volume_3m",
    ]
    available = [c for c in display_cols if c in df.columns]
    disp = df[available].copy()
    # round numerics
    for col in ["pe", "pb", "roe", "net_margin", "dividend_yield",
                "return_1m", "return_1y", "upside_potential", "rsi_14", "volume_3m"]:
        if col in disp.columns:
            disp[col] = pd.to_numeric(disp[col], errors="coerce").round(1)
    if "market_cap_usd" in disp.columns:
        disp["market_cap_usd"] = pd.to_numeric(disp["market_cap_usd"], errors="coerce").round(0)
    return disp


def _signal_strength_label(row: pd.Series) -> str:
    """Map return + RSI to a signal strength bucket (for coloring)."""
    ret = float(row.get("return_1m", 0) or 0)
    rsi = float(row.get("rsi_14", 50) or 50)
    if ret > 5 and rsi < 70:
        return "Strong"
    if ret > 0 and rsi < 60:
        return "Moderate"
    if ret < -5 or rsi > 75:
        return "Weak"
    return "Neutral"


# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT: 2-column (filter panel | results)
# ═══════════════════════════════════════════════════════════════════════════════
filter_col, result_col = st.columns([1, 3], gap="large")

# ─────────────────────────────────────────────────────────────────────────────
# LEFT: FILTER PANEL
# ─────────────────────────────────────────────────────────────────────────────
with filter_col:
    st.markdown("### ⚙️ Screener Filters")

    # Universe
    sel_index = st.selectbox(
        "📦 Universe",
        options=INDEX_OPTIONS,
        index=INDEX_OPTIONS.index(_val("sc5_index")),
        key="sc5_index_sel",
    )
    st.session_state["sc5_index"] = sel_index

    # Sector multi-select
    sel_sectors = st.multiselect(
        "🏭 Sectors",
        options=SECTORS,
        default=_val("sc5_sectors"),
        placeholder="All sectors",
        format_func=lambda s: f"{SECTOR_ICONS.get(s, '•')} {s}",
        key="sc5_sec_ms",
    )
    st.session_state["sc5_sectors"] = sel_sectors

    # Template presets
    st.markdown("**⚡ Quick Presets**")
    preset_cols = st.columns(2)
    for i, (tkey, tlabel) in enumerate(TEMPLATE_LABELS.items()):
        with preset_cols[i % 2]:
            active = _val("sc5_template") == tkey
            btn_style = "primary" if active else "secondary"
            if st.button(tlabel, key=f"sc5_tpl_{tkey}", type=btn_style, use_container_width=True):
                st.session_state["sc5_template"] = "" if active else tkey
                st.rerun()

    st.markdown("---")
    st.markdown("**📊 Fundamental Filters**")

    with st.expander("📈 Valuation", expanded=True):
        c1, c2 = st.columns(2)
        pe_min = c1.number_input("P/E min", value=float(_val("sc5_pe_min")), min_value=0.0, max_value=500.0, step=1.0, key="sc5_pe_min_in", help="0 = no lower bound")
        pe_max = c2.number_input("P/E max", value=float(_val("sc5_pe_max")), min_value=0.0, max_value=5000.0, step=1.0, key="sc5_pe_max_in", help="0 = no upper bound")
        st.session_state["sc5_pe_min"] = pe_min
        st.session_state["sc5_pe_max"] = pe_max
        c3, c4 = st.columns(2)
        mcap_min = c3.number_input("MCap USD mn min", value=float(_val("sc5_mcap_min")), min_value=0.0, max_value=1e6, step=100.0, key="sc5_mcap_min_in")
        mcap_max = c4.number_input("MCap USD mn max", value=float(_val("sc5_mcap_max")), min_value=0.0, max_value=1e6, step=100.0, key="sc5_mcap_max_in")
        st.session_state["sc5_mcap_min"] = mcap_min
        st.session_state["sc5_mcap_max"] = mcap_max
        div_min = st.number_input("Dividend Yield min (%)", value=float(_val("sc5_div_min")), min_value=0.0, max_value=100.0, step=0.5, key="sc5_div_min_in")
        st.session_state["sc5_div_min"] = div_min

    with st.expander("⭐ Quality", expanded=True):
        c5, c6 = st.columns(2)
        roe_min = c5.number_input("ROE min (%)", value=float(_val("sc5_roe_min")), min_value=-200.0, max_value=500.0, step=1.0, key="sc5_roe_min_in")
        roe_max = c6.number_input("ROE max (%)", value=float(_val("sc5_roe_max")), min_value=0.0, max_value=1000.0, step=1.0, key="sc5_roe_max_in")
        st.session_state["sc5_roe_min"] = roe_min
        st.session_state["sc5_roe_max"] = roe_max
        c7, c8 = st.columns(2)
        nm_min = c7.number_input("Net Margin min (%)", value=float(_val("sc5_nm_min")), min_value=-500.0, max_value=200.0, step=1.0, key="sc5_nm_min_in")
        nm_max = c8.number_input("Net Margin max (%)", value=float(_val("sc5_nm_max")), min_value=0.0, max_value=1000.0, step=1.0, key="sc5_nm_max_in")
        st.session_state["sc5_nm_min"] = nm_min
        st.session_state["sc5_nm_max"] = nm_max

    with st.expander("📉 Technical  / Momentum", expanded=False):
        rsi_min = st.slider("RSI 14 range", 0.0, 100.0, (_val("sc5_rsi_min"), _val("sc5_rsi_max")), step=1.0, key="sc5_rsi_slider")
        st.session_state["sc5_rsi_min"] = float(rsi_min[0])
        st.session_state["sc5_rsi_max"] = float(rsi_min[1])
        r1m = st.slider("Return 1M range (%)", -50.0, 50.0, (_val("sc5_ret1m_min"), _val("sc5_ret1m_max")), step=0.5, key="sc5_r1m_slider")
        st.session_state["sc5_ret1m_min"] = float(r1m[0])
        st.session_state["sc5_ret1m_max"] = float(r1m[1])

    with st.expander("📋 Recommendation", expanded=False):
        rec_sel = st.multiselect(
            "Analyst rating",
            options=["AL", "TUT", "SAT"],
            default=_val("sc5_rec_filter"),
            format_func=lambda r: f"{REC_ICONS.get(r, '')} {r}",
            key="sc5_rec_ms",
        )
        st.session_state["sc5_rec_filter"] = rec_sel

    st.markdown("---")
    st.markdown("**🔢 Sort**")
    sort_labels = [lbl for _, lbl in SORT_FIELD_OPTIONS]
    sort_keys  = [key for key, _ in SORT_FIELD_OPTIONS]
    cur_sort_idx = sort_keys.index(_val("sc5_sort_col")) if _val("sc5_sort_col") in sort_keys else 0
    sel_sort = st.selectbox("Sort by", options=sort_labels, index=cur_sort_idx, key="sc5_sort_sel")
    st.session_state["sc5_sort_col"] = sort_keys[sort_labels.index(sel_sort)]
    sort_dir = st.radio("Order", ["Descending ↓", "Ascending ↑"], horizontal=True, key="sc5_sort_dir")
    st.session_state["sc5_sort_desc"] = sort_dir.startswith("Desc")

    st.markdown("---")
    run_btn = st.button("▶ Run Screener", type="primary", use_container_width=True, key="sc5_run")
    if st.button("🔄 Clear Filters", use_container_width=True, key="sc5_clear"):
        for k, v in _SS_DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Trigger run
# ─────────────────────────────────────────────────────────────────────────────
if run_btn:
    try:
        from bist_quant.engines.stock_filter import run_stock_filter
        st.session_state["sc5_raw_result"] = None
        st.session_state["sc5_df"] = None
        st.session_state["sc5_error"] = None
        st.session_state["sc5_page"] = 1
        payload = _build_sc5_payload()
        st.session_state["sc5_run_future"] = run_in_thread(run_stock_filter, payload)
        st.rerun()
    except ImportError as exc:
        st.session_state["sc5_error"] = f"Import error: {exc}"

# Poll future
_fut = st.session_state.get("sc5_run_future")
if _fut is not None and not _fut.done():
    with result_col:
        with st.spinner("Running screener… loading prices & computing metrics ⏳"):
            time.sleep(0.5)
            st.rerun()
elif _fut is not None and _fut.done():
    try:
        _res = _fut.result()
        st.session_state["sc5_raw_result"] = _res
        # Convert rows to DataFrame
        rows = _res.get("rows", [])
        if rows:
            df_all = pd.DataFrame(rows)
            st.session_state["sc5_df"] = df_all
        else:
            st.session_state["sc5_df"] = pd.DataFrame()
    except Exception as exc:
        st.session_state["sc5_error"] = str(exc)
    st.session_state["sc5_run_future"] = None
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# RIGHT: RESULTS PANEL
# ─────────────────────────────────────────────────────────────────────────────
with result_col:
    err = st.session_state.get("sc5_error")
    if err:
        st.error(f"Screener error: {err}")

    raw_result = st.session_state.get("sc5_raw_result")
    df_all: pd.DataFrame | None = st.session_state.get("sc5_df")

    # ── Welcome / placeholder ────────────────────────────────────────────────
    if raw_result is None and not err:
        st.markdown("### 📊 Configure filters and click **▶ Run Screener**")
        with st.container(border=True):
            ic1, ic2, ic3, ic4 = st.columns(4)
            ic1.metric("Universe", "XU100", help="604 stocks in BIST All-Share")
            ic2.metric("Sectors", str(len(SECTORS)), help="Available sector filters")
            ic3.metric("Filter Fields", "29", help="Fundamental + technical + flow metrics")
            ic4.metric("Presets", str(len(TEMPLATE_LABELS)), help="Quick-start templates")

        st.markdown("**Available columns:** ticker · sector · P/E · P/B · ROE · Net Margin · Market Cap · Dividend Yield · RSI · Return 1M/1Y · Analyst Rating · Sparkline")
        with st.expander("💡 Usage tips"):
            st.markdown("""
- **Run Screener** loads price data and computes fundamentals from local CSV/Parquet files. First run takes ~5–15 seconds; subsequent runs use the in-memory cache (valid for 10 min).
- **Presets** auto-populate filter bounds (e.g. "Low P/E" sets `pe.max = 12`). Combine presets with custom sliders.
- **Sector filter** is applied client-side after the engine run — instant, no re-fetch needed.
- **Watchlist**: tick the checkbox next to any row to add the ticker to `st.session_state["sc5_watchlist"]`.
- **Sparklines** show the 30-day normalised price trend for the top 20 results.
            """)

    # ── Results ──────────────────────────────────────────────────────────────
    if df_all is not None:
        meta = raw_result.get("meta", {}) if raw_result else {}
        applied_filters = raw_result.get("applied_filters", []) if raw_result else []

        # Apply client-side sector + recommendation filters
        df_view = _apply_client_filters(df_all)

        # Re-sort locally (engine already sorted, but client filters may change order)
        sort_col = _val("sc5_sort_col")
        if sort_col in df_view.columns:
            df_view = df_view.sort_values(
                sort_col,
                ascending=not _val("sc5_sort_desc"),
                na_position="last",
            )

        total_view = len(df_view)
        page_size = int(_val("sc5_page_size"))

        # ── summary bar ──────────────────────────────────────────────────────
        sm1, sm2, sm3, sm4, sm5 = st.columns(5)
        sm1.metric("Total engine matches", meta.get("total_matches", "–"))
        sm2.metric("After filters", total_view)
        sm3.metric("As-of", str(meta.get("as_of", "–"))[:10])
        sm4.metric("Data source", meta.get("data_source", "local").capitalize())
        sm5.metric("Time", f"{meta.get('execution_ms', '–')} ms")

        # applied filters chips
        if applied_filters:
            chips = " · ".join(
                f"**{f.get('label', f.get('key','?'))}** "
                + (f"≥ {f['min']:.1f}" if f.get("min") is not None else "")
                + (f" ≤ {f['max']:.1f}" if f.get("max") is not None else "")
                for f in applied_filters
            )
            st.caption(f"🔧 Filters applied: {chips}")

        # ── rec distribution badges ───────────────────────────────────────────
        if "recommendation" in df_view.columns:
            rec_counts = df_view["recommendation"].fillna("–").value_counts()
            bd1, bd2, bd3, bd4 = st.columns(4)
            bd1.markdown(
                f"<div style='background:#27ae60;border-radius:6px;padding:6px 12px;text-align:center;'>"
                f"📈 <b>AL</b> {rec_counts.get('AL', 0)}</div>", unsafe_allow_html=True
            )
            bd2.markdown(
                f"<div style='background:#e67e22;border-radius:6px;padding:6px 12px;text-align:center;'>"
                f"⏸ <b>TUT</b> {rec_counts.get('TUT', 0)}</div>", unsafe_allow_html=True
            )
            bd3.markdown(
                f"<div style='background:#c0392b;border-radius:6px;padding:6px 12px;text-align:center;'>"
                f"📉 <b>SAT</b> {rec_counts.get('SAT', 0)}</div>", unsafe_allow_html=True
            )
            bd4.markdown(
                f"<div style='background:#2c3e50;border-radius:6px;padding:6px 12px;text-align:center;'>"
                f"❓ <b>N/A</b> {rec_counts.get('–', 0) + rec_counts.get('nan', 0) + (total_view - rec_counts.sum())}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("")

        # ── pagination controls ─────────────────────────────────────────────
        total_pages = max(1, (total_view + page_size - 1) // page_size)
        cur_page = min(int(_val("sc5_page")), total_pages)
        offset = (cur_page - 1) * page_size
        df_page = df_view.iloc[offset : offset + page_size].copy()

        pc1, pc2, pc3 = st.columns([1, 2, 1])
        with pc1:
            if st.button("◀ Prev", disabled=cur_page <= 1, key="sc5_prev"):
                st.session_state["sc5_page"] = max(1, cur_page - 1)
                st.rerun()
        with pc2:
            st.caption(f"Page **{cur_page}** / {total_pages}  ·  {total_view} results  ·  rows {offset+1}–{min(offset+page_size, total_view)}")
        with pc3:
            if st.button("Next ▶", disabled=cur_page >= total_pages, key="sc5_next"):
                st.session_state["sc5_page"] = min(total_pages, cur_page + 1)
                st.rerun()

        # ── main results table ───────────────────────────────────────────────
        disp_df = _style_results_df(df_page)

        # Colour map: recommendation text colour
        def _color_rec(val: str) -> str:
            return f"color: {REC_COLORS.get(str(val).upper(), '#aaa')}; font-weight: bold"

        def _color_rsi(val: float) -> str:
            try:
                v = float(val)
                if v > 70:  return "color: #e74c3c"
                if v < 30:  return "color: #2ecc71"
            except Exception:
                pass
            return "color: #e0e0e0"

        def _color_ret(val: float) -> str:
            try:
                v = float(val)
                if v >  5:  return "color: #2ecc71"
                if v < -5:  return "color: #e74c3c"
            except Exception:
                pass
            return "color: #e0e0e0"

        styled = disp_df.style
        if "recommendation" in disp_df.columns:
            styled = styled.applymap(_color_rec, subset=["recommendation"])
        for _rc in ["return_1m", "return_1y", "upside_potential"]:
            if _rc in disp_df.columns:
                styled = styled.applymap(_color_ret, subset=[_rc])
        if "rsi_14" in disp_df.columns:
            styled = styled.applymap(_color_rsi, subset=["rsi_14"])

        st.dataframe(
            styled,
            use_container_width=True,
            hide_index=True,
            height=min(38 * len(df_page) + 38, 600),
            column_config={
                "symbol":          st.column_config.TextColumn("Ticker"),
                "sector":          st.column_config.TextColumn("Sector"),
                "recommendation":  st.column_config.TextColumn("Rating"),
                "market_cap_usd":  st.column_config.NumberColumn("MktCap USD mn", format="%.0f"),
                "pe":              st.column_config.NumberColumn("P/E", format="%.1f"),
                "pb":              st.column_config.NumberColumn("P/B", format="%.1f"),
                "roe":             st.column_config.NumberColumn("ROE %", format="%.1f"),
                "net_margin":      st.column_config.NumberColumn("Net Margin %", format="%.1f"),
                "dividend_yield":  st.column_config.NumberColumn("Div Yield %", format="%.1f"),
                "rsi_14":          st.column_config.NumberColumn("RSI 14", format="%.0f"),
                "return_1m":       st.column_config.NumberColumn("Ret 1M %", format="%.1f"),
                "return_1y":       st.column_config.NumberColumn("Ret 1Y %", format="%.1f"),
                "upside_potential":st.column_config.NumberColumn("Upside %", format="%.1f"),
                "volume_3m":       st.column_config.NumberColumn("Vol 3M mn", format="%.1f"),
            },
        )

        # ── watchlist buttons ────────────────────────────────────────────────
        st.markdown("---")
        wl_expander = st.expander(
            f"⭐ Watchlist ({len(_val('sc5_watchlist'))} stocks)", expanded=len(_val("sc5_watchlist")) > 0
        )
        with wl_expander:
            if "symbol" in df_page.columns:
                wl_cols = st.columns(min(len(df_page), 6))
                for ci, sym in enumerate(df_page["symbol"].tolist()[:6]):
                    with wl_cols[ci]:
                        in_wl = sym in _val("sc5_watchlist")
                        label = f"✓ {sym}" if in_wl else f"＋ {sym}"
                        if st.button(label, key=f"sc5_wl_{sym}_{ci}", use_container_width=True,
                                     type="primary" if in_wl else "secondary"):
                            wl = list(_val("sc5_watchlist"))
                            if in_wl:
                                wl.remove(sym)
                            else:
                                wl.append(sym)
                            st.session_state["sc5_watchlist"] = wl
                            st.rerun()

            current_wl = _val("sc5_watchlist")
            if current_wl:
                st.markdown("**Current watchlist:** " + "  ".join(f"`{s}`" for s in current_wl))
                if st.button("🗑️ Clear watchlist", key="sc5_clr_wl"):
                    st.session_state["sc5_watchlist"] = []
                    st.rerun()

        # ── sparklines ───────────────────────────────────────────────────────
        if "symbol" in df_page.columns:
            spark_n = min(int(_val("sc5_spark_n")), len(df_page))
            top_symbols = df_page["symbol"].tolist()[:spark_n]

            with st.expander(f"📈 Sparklines — 30-day price trend (top {spark_n} stocks)", expanded=True):
                n_spark = st.slider("Stocks to show", 4, min(len(df_page), 40), spark_n, 4, key="sc5_spk_n_sl")
                st.session_state["sc5_spark_n"] = n_spark
                top_symbols_vis = df_page["symbol"].tolist()[:n_spark]

                with st.spinner("Loading price data for sparklines…"):
                    spark_data = _load_sparklines(top_symbols_vis, n_days=30)

                if spark_data:
                    ret_1m_map: dict[str, float] = {}
                    if "return_1m" in df_page.columns:
                        ret_1m_map = {
                            str(r["symbol"]): float(r.get("return_1m") or 0)
                            for _, r in df_page[["symbol", "return_1m"]].iterrows()
                        }
                    n_cols_spark = 4 if n_spark > 8 else (3 if n_spark > 4 else 2)
                    fig_spark = _build_spark_fig(
                        top_symbols_vis, spark_data, ret_1m_map, n_cols=n_cols_spark
                    )
                    st.plotly_chart(fig_spark, use_container_width=True)
                    st.caption(
                        "Green line = positive 1M return. Red = negative. "
                        "Y-axis: normalised to 100 at start of window. "
                        "Sparklines sourced from local price cache."
                    )
                else:
                    st.info("Sparkline data not available (price cache not yet loaded — run the screener first).")

        # ── sector breakdown chart ────────────────────────────────────────────
        if "sector" in df_view.columns and total_view > 0:
            with st.expander("🥧 Sector Distribution", expanded=False):
                sector_counts = df_view["sector"].value_counts()
                fig_sec = go.Figure(go.Pie(
                    labels=[f"{SECTOR_ICONS.get(s, '•')} {s}" for s in sector_counts.index],
                    values=sector_counts.values.tolist(),
                    hole=0.4,
                    textinfo="percent+label",
                    marker=dict(colors=[
                        f"hsl({int(i * 360 / len(sector_counts))}, 60%, 50%)"
                        for i in range(len(sector_counts))
                    ]),
                ))
                fig_sec.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10, r=10, t=20, b=10), height=320,
                    showlegend=True, legend=dict(font=dict(size=9)),
                    font=dict(color="#e0e0e0", size=9),
                )
                st.plotly_chart(fig_sec, use_container_width=True)

        # ── top/bottom tables ─────────────────────────────────────────────────
        with st.expander("🏆 Top 10 by ROE vs Bottom 10 by P/E", expanded=False):
            t1, t2 = st.columns(2)
            with t1:
                st.markdown("**Top 10 ROE**")
                if "roe" in df_view.columns:
                    top_roe = df_view[["symbol", "sector", "roe", "net_margin"]].dropna(subset=["roe"]).nlargest(10, "roe")
                    st.dataframe(top_roe.round(1), use_container_width=True, hide_index=True)
            with t2:
                st.markdown("**Lowest P/E (positive only)**")
                if "pe" in df_view.columns:
                    low_pe = df_view[["symbol", "sector", "pe", "roe"]].dropna(subset=["pe"])
                    low_pe = low_pe[low_pe["pe"] > 0].nsmallest(10, "pe")
                    st.dataframe(low_pe.round(1), use_container_width=True, hide_index=True)

        # ── column group tabs (full field view) ───────────────────────────────
        with st.expander("📊 Full Column Groups", expanded=False):
            tab_val, tab_qual, tab_mom, tab_flow = st.tabs(
                ["💎 Valuation", "⭐ Quality", "🚀 Momentum", "🌍 Flow"]
            )
            val_cols  = ["symbol", "sector", "recommendation", "market_cap_usd", "pe", "forward_pe", "pb", "ev_ebitda", "ev_sales", "dividend_yield"]
            qual_cols = ["symbol", "sector", "roe", "roa", "net_margin", "ebitda_margin", "revenue_growth_yoy", "net_income_growth_yoy"]
            mom_cols  = ["symbol", "sector", "rsi_14", "macd_hist", "atr_14_pct", "return_1w", "return_1m", "return_ytd", "return_1y", "upside_potential"]
            flow_cols = ["symbol", "sector", "foreign_ratio", "foreign_change_1w", "foreign_change_1m", "float_ratio", "volume_3m", "volume_12m"]

            for tab_obj, cols in [(tab_val, val_cols), (tab_qual, qual_cols), (tab_mom, mom_cols), (tab_flow, flow_cols)]:
                with tab_obj:
                    avail = [c for c in cols if c in df_view.columns]
                    if avail:
                        st.dataframe(
                            df_view[avail].head(100).round(2),
                            use_container_width=True, hide_index=True, height=400,
                        )

