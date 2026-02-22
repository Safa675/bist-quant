"""Professional Tools — Options Greeks, Stress Tests & Crypto Trade Sizing."""

from __future__ import annotations

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

st.set_page_config(
    page_title="Professional · BIST Quant", page_icon="🏦", layout="wide"
)

from app.layout import page_header, render_sidebar  # noqa: E402
from app.utils import fmt_num  # noqa: E402

# ---------------------------------------------------------------------------
# Import analytics
# ---------------------------------------------------------------------------
try:
    from bist_quant.analytics.professional import (
        CryptoTradeInput,
        OptionGreeksInput,
        StressFactorShock,
        build_crypto_trade_plan,
        compute_option_greeks,
        run_portfolio_stress_test,
    )
    _PRO_OK = True
except ImportError as _e:
    _PRO_OK = False
    _PRO_ERR = str(_e)

render_sidebar()
page_header(
    "🏦 Professional Tools",
    "Options Greeks (Black-Scholes), portfolio stress tests & crypto trade sizing",
)

if not _PRO_OK:
    st.error(
        f"**bist_quant professional analytics unavailable**: {_PRO_ERR}\n\n"
        "Install from repo root:\n```bash\npip install -e .[api,services]\n```"
    )
    st.stop()

# ── tabs ───────────────────────────────────────────────────────────────────────
tab_greeks, tab_stress, tab_crypto = st.tabs(
    ["📐 Options Greeks", "💥 Stress Tests", "₿ Crypto Trade Sizing"]
)

# ===========================================================================
# TAB 1 — Options Greeks (Black-Scholes)
# ===========================================================================
with tab_greeks:
    st.subheader("Black-Scholes Options Greeks")
    st.markdown(
        "Calculate theoretical price and all five Greeks for a vanilla call or put option."
    )

    col_inp, col_out = st.columns([1, 1], gap="large")

    with col_inp:
        with st.form("greeks_form"):
            opt_type = st.selectbox("Option Type", ["call", "put"])
            spot = st.number_input(
                "Spot Price (S)", min_value=0.01, value=100.0, step=0.5
            )
            strike = st.number_input(
                "Strike Price (K)", min_value=0.01, value=100.0, step=0.5
            )
            expiry_days = st.number_input(
                "Days to Expiry", min_value=1, max_value=1825, value=30, step=1
            )
            vol_pct = st.slider(
                "Implied Volatility (%)", min_value=1, max_value=200, value=25
            )
            rfr_pct = st.slider(
                "Risk-Free Rate (%)", min_value=-5, max_value=30, value=5
            )
            submitted_greeks = st.form_submit_button(
                "Calculate Greeks", use_container_width=True, type="primary"
            )

    with col_out:
        if submitted_greeks:
            inp = OptionGreeksInput(
                option_type=opt_type,  # type: ignore[arg-type]
                spot=spot,
                strike=strike,
                time_years=expiry_days / 365,
                volatility=vol_pct / 100,
                risk_free_rate=rfr_pct / 100,
            )
            g = compute_option_greeks(inp)

            st.markdown("#### Results")

            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.metric("Theoretical Price", f"{g.theoretical_price:,.4f}")
                st.metric("Delta (Δ)", f"{g.delta:+.4f}")
                st.metric("Gamma (Γ)", f"{g.gamma:.6f}")
            with metrics_col2:
                st.metric("Theta / Day (Θ)", f"{g.theta_per_day:+.4f}")
                st.metric("Vega / 1% IV (ν)", f"{g.vega_per_1pct:.4f}")
                st.metric("Rho / 1% RFR (ρ)", f"{g.rho_per_1pct:+.4f}")

            # Greek sensitivity gauge chart
            st.markdown("#### Greek Sensitivity Overview")
            fig = go.Figure()
            labels = ["Delta", "Gamma (×100)", "Theta", "Vega", "Rho"]
            values = [
                g.delta,
                g.gamma * 100,
                g.theta_per_day,
                g.vega_per_1pct,
                g.rho_per_1pct,
            ]
            colors = [
                "#2ecc71" if v >= 0 else "#e74c3c" for v in values
            ]
            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=values,
                    marker_color=colors,
                    text=[f"{v:.4f}" for v in values],
                    textposition="outside",
                )
            )
            fig.update_layout(
                height=320,
                margin=dict(t=20, b=20, l=0, r=0),
                yaxis_title="Value",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#ccc",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Fill in the option parameters and click **Calculate Greeks**.")

    # ── volatility smile surface (illustrative) ────────────────────────────
    st.divider()
    st.subheader("Implied Volatility Smile (Illustrative)")
    st.caption(
        "Shows how theoretical fair-value shifts as vol changes for the current spot/strike. "
        "Connect a live options feed to replace with real market IV data."
    )

    strikes_range = [spot * k for k in [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]]
    vol_vals = [
        compute_option_greeks(
            OptionGreeksInput(
                option_type="call",
                spot=spot,
                strike=k,
                time_years=expiry_days / 365,
                volatility=vol_pct / 100,
                risk_free_rate=rfr_pct / 100,
            )
        ).theoretical_price
        for k in strikes_range
    ]
    fig_smile = go.Figure(
        go.Scatter(
            x=strikes_range,
            y=vol_vals,
            mode="lines+markers",
            line=dict(color="#3498db", width=2),
            marker=dict(size=7),
            name="Call Price",
        )
    )
    fig_smile.add_vline(x=spot, line_dash="dash", line_color="#f39c12", annotation_text="Spot")
    fig_smile.update_layout(
        height=280,
        margin=dict(t=20, b=20, l=0, r=0),
        xaxis_title="Strike",
        yaxis_title="Theoretical Call Price",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#ccc",
    )
    st.plotly_chart(fig_smile, use_container_width=True)


# ===========================================================================
# TAB 2 — Portfolio Stress Tests
# ===========================================================================
with tab_stress:
    st.subheader("Portfolio Stress Test")
    st.markdown(
        "Simulate multi-factor shocks and calculate expected portfolio loss. "
        "Each factor shock is scaled by its portfolio beta."
    )

    portfolio_value = st.number_input(
        "Portfolio Value (USD / TRY)",
        min_value=1_000.0,
        max_value=1_000_000_000.0,
        value=1_000_000.0,
        step=10_000.0,
        format="%.0f",
    )

    st.markdown("#### Add Factor Shocks")
    st.caption("Shock % = directional move of the factor (e.g. -20 = factor drops 20%). Beta = portfolio sensitivity.")

    # Dynamic shock rows stored in session state
    if "stress_shocks" not in st.session_state:
        st.session_state.stress_shocks: list[dict[str, Any]] = [
            {"factor": "Equity Market", "shock_pct": -20.0, "beta": 1.0},
            {"factor": "USD/TRY FX", "shock_pct": 30.0, "beta": -0.4},
            {"factor": "Interest Rates", "shock_pct": 200.0, "beta": -0.15},
        ]

    # Render editable shock table
    shock_df = pd.DataFrame(st.session_state.stress_shocks)
    edited = st.data_editor(
        shock_df,
        num_rows="dynamic",
        column_config={
            "factor": st.column_config.TextColumn("Factor", width="medium"),
            "shock_pct": st.column_config.NumberColumn("Shock (%)", format="%.2f", width="small"),
            "beta": st.column_config.NumberColumn("Portfolio Beta", format="%.3f", width="small"),
        },
        use_container_width=True,
        key="stress_editor",
    )

    run_stress = st.button("▶ Run Stress Test", type="primary")

    if run_stress and not edited.empty:
        shocks = [
            StressFactorShock(
                factor=str(row["factor"]),
                shock_pct=float(row["shock_pct"]),
                beta=float(row["beta"]),
            )
            for _, row in edited.iterrows()
        ]
        result = run_portfolio_stress_test(portfolio_value, shocks)

        st.divider()
        c1, c2 = st.columns(2)
        loss_color = "inverse" if result.scenario_loss_pct > 0 else "normal"
        c1.metric(
            "Scenario Loss (%)",
            f"{result.scenario_loss_pct:+.2f}%",
            delta_color=loss_color,
        )
        c2.metric(
            "Scenario Loss (Value)",
            f"${result.scenario_loss_value:,.2f}",
        )

        st.markdown("#### Factor Contribution to Loss")
        factor_df = pd.DataFrame(result.by_factor)
        factor_df["loss_contribution"] = factor_df["loss_pct"] / result.scenario_loss_pct * 100 if result.scenario_loss_pct != 0 else 0

        fig_stress = go.Figure(
            go.Bar(
                x=factor_df["factor"],
                y=factor_df["loss_pct"],
                marker_color=[
                    "#e74c3c" if v <= 0 else "#2ecc71"
                    for v in factor_df["loss_pct"]
                ],
                text=[f"{v:+.2f}%" for v in factor_df["loss_pct"]],
                textposition="outside",
            )
        )
        fig_stress.update_layout(
            height=350,
            margin=dict(t=20, b=20, l=0, r=0),
            yaxis_title="Loss Contribution (%)",
            xaxis_title="Factor",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#ccc",
        )
        st.plotly_chart(fig_stress, use_container_width=True)

        st.dataframe(
            factor_df.rename(
                columns={
                    "factor": "Factor",
                    "loss_pct": "Loss (%)",
                    "loss_contribution": "Share of Total Loss (%)",
                }
            ).style.format({"Loss (%)": "{:+.4f}", "Share of Total Loss (%)": "{:.2f}"}),
            use_container_width=True,
        )

    # ── pre-built scenario library ─────────────────────────────────────────
    st.divider()
    st.subheader("Pre-built Scenario Library")

    _SCENARIOS = {
        "2008 GFC": [
            {"factor": "Global Equities", "shock_pct": -55.0, "beta": 1.0},
            {"factor": "Credit Spreads", "shock_pct": 500.0, "beta": -0.3},
            {"factor": "USD FX", "shock_pct": 15.0, "beta": -0.2},
        ],
        "2020 COVID Crash": [
            {"factor": "Equities", "shock_pct": -34.0, "beta": 1.0},
            {"factor": "Oil", "shock_pct": -70.0, "beta": 0.1},
            {"factor": "Volatility (VIX)", "shock_pct": 400.0, "beta": -0.5},
        ],
        "Turkey 2021 Currency Crisis": [
            {"factor": "USD/TRY", "shock_pct": 60.0, "beta": -0.8},
            {"factor": "BIST100", "shock_pct": -30.0, "beta": 1.0},
            {"factor": "Turkish Bonds", "shock_pct": -25.0, "beta": 0.4},
        ],
        "Rate Shock +300bps": [
            {"factor": "Interest Rates", "shock_pct": 300.0, "beta": -0.25},
            {"factor": "Equities (re-pricing)", "shock_pct": -15.0, "beta": 1.0},
        ],
    }

    chosen_scenario = st.selectbox(
        "Load a pre-built scenario", ["— select —"] + list(_SCENARIOS.keys())
    )
    if chosen_scenario != "— select —":
        preset_shocks = [
            StressFactorShock(**s) for s in _SCENARIOS[chosen_scenario]
        ]
        preset_result = run_portfolio_stress_test(portfolio_value, preset_shocks)
        sc1, sc2 = st.columns(2)
        sc1.metric(
            f"{chosen_scenario} — Loss (%)",
            f"{preset_result.scenario_loss_pct:+.2f}%",
        )
        sc2.metric(
            "Loss (Value)",
            f"${preset_result.scenario_loss_value:,.2f}",
        )
        if st.button("Load into editor"):
            st.session_state.stress_shocks = list(_SCENARIOS[chosen_scenario])
            st.rerun()


# ===========================================================================
# TAB 3 — Crypto Trade Sizing
# ===========================================================================
with tab_crypto:
    st.subheader("Crypto Trade Sizing Calculator")
    st.markdown(
        "Risk-based position sizing with leverage, liquidation price estimation and fee impact."
    )

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        with st.form("crypto_form"):
            pair = st.text_input("Trading Pair", value="BTC/USDT").upper()
            side = st.selectbox("Side", ["long", "short"])
            entry = st.number_input(
                "Entry Price (USDT)", min_value=0.0001, value=65_000.0, step=100.0
            )
            equity = st.number_input(
                "Account Equity (USDT)", min_value=1.0, value=10_000.0, step=100.0
            )
            risk_pct = st.slider(
                "Risk per Trade (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1
            )
            leverage = st.slider(
                "Leverage (×)", min_value=1, max_value=125, value=10
            )
            stop_dist_pct = st.slider(
                "Stop Distance (%)", min_value=0.1, max_value=30.0, value=2.0, step=0.1
            )
            taker_fee = st.number_input(
                "Taker Fee (bps)", min_value=0, max_value=100, value=5
            )
            submitted_crypto = st.form_submit_button(
                "Calculate Position", use_container_width=True, type="primary"
            )

    with col_right:
        if submitted_crypto:
            plan = build_crypto_trade_plan(
                CryptoTradeInput(
                    pair=pair,
                    side=side,  # type: ignore[arg-type]
                    entry_price=entry,
                    equity=equity,
                    risk_pct=risk_pct,
                    leverage=leverage,
                    stop_distance_pct=stop_dist_pct,
                    taker_fee_bps=taker_fee,
                )
            )

            st.markdown("#### Trade Plan")
            st.markdown(
                f"""
| Parameter | Value |
|---|---|
| **Pair** | {plan.pair} |
| **Side** | {plan.side.upper()} |
| **Notional** | ${plan.notional:,.2f} |
| **Margin Required** | ${plan.margin_required:,.2f} |
| **Quantity** | {plan.quantity:.6f} |
| **Liquidation Price** | ${plan.liquidation_price:,.4f} |
| **Max Loss (risk cap)** | ${plan.max_loss:,.2f} |
| **Estimated Fees** | ${plan.estimated_fees:,.2f} |
"""
            )

            # Visualise P&L corridor
            import math
            price_range = [entry * (1 + pct / 100) for pct in range(-15, 16)]
            if side == "long":
                pnl = [(p - entry) * plan.quantity for p in price_range]
            else:
                pnl = [(entry - p) * plan.quantity for p in price_range]

            colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in pnl]
            fig_crypto = go.Figure(
                go.Bar(x=price_range, y=pnl, marker_color=colors)
            )
            fig_crypto.add_hline(y=0, line_dash="dash", line_color="#aaa")
            fig_crypto.add_vline(
                x=plan.liquidation_price,
                line_color="#e74c3c",
                line_dash="dot",
                annotation_text="Liq.",
                annotation_position="top left",
            )
            fig_crypto.update_layout(
                height=320,
                title="P&L Corridor (entry ± 15%)",
                margin=dict(t=40, b=20, l=0, r=0),
                xaxis_title="Price (USDT)",
                yaxis_title="Unrealised P&L (USDT)",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#ccc",
            )
            st.plotly_chart(fig_crypto, use_container_width=True)

            # Risk ratios
            rr = abs((entry * stop_dist_pct / 100)) / (risk_pct / 100 * equity) if equity > 0 else 0
            st.info(
                f"**R-multiple:** You risk ${plan.max_loss:,.2f} to trade ${plan.notional:,.2f} notional "
                f"({leverage}× leverage). Margin utilisation: **{plan.margin_required / equity * 100:.1f}%** of equity."
            )
        else:
            st.info("Fill in the trade parameters and click **Calculate Position**.")

    # ── forex pip helper ───────────────────────────────────────────────────
    st.divider()
    st.subheader("Forex Pip Value")
    try:
        from bist_quant.analytics.professional import compute_forex_pip_value

        fp_col1, fp_col2, fp_col3 = st.columns(3)
        fx_pair = fp_col1.text_input("FX Pair", value="EURUSD")
        lot_size = fp_col2.number_input("Lot Size", value=100_000.0, step=1000.0)
        conv_rate = fp_col3.number_input("Account Conversion Rate", value=1.0, step=0.01)

        if st.button("Compute Pip Value"):
            pip = compute_forex_pip_value(fx_pair, lot_size, conv_rate)
            c1, c2, c3 = st.columns(3)
            c1.metric("Pip Size", f"{pip.pip_size:.5f}")
            c2.metric("Pip Value (Quote)", f"{pip.pip_value_quote:.4f}")
            c3.metric("Pip Value (Account)", f"{pip.pip_value_account:.4f}")
    except ImportError:
        st.warning("compute_forex_pip_value not available.")
