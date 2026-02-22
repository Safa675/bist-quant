"""Compliance — Transaction rule-engine checks with pass/fail checklist output."""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).parent.parent.parent
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Compliance · BIST Quant", page_icon="⚖️", layout="wide"
)

from app.layout import page_header, render_sidebar  # noqa: E402

# ---------------------------------------------------------------------------
# Import analytics
# ---------------------------------------------------------------------------
try:
    from bist_quant.analytics.professional import (
        ComplianceHit,
        ComplianceRule,
        TransactionRecord,
        detect_user_activity_anomalies,
        monitor_position_limits,
        run_compliance_rule_engine,
    )
    _COMP_OK = True
except ImportError as _e:
    _COMP_OK = False
    _COMP_ERR = str(_e)

render_sidebar()
page_header(
    "⚖️ Compliance",
    "Transaction rule-engine checks, position limit monitoring & activity anomaly detection",
)

if not _COMP_OK:
    st.error(
        f"**bist_quant compliance unavailable**: {_COMP_ERR}\n\n"
        "Install from repo root:\n```bash\npip install -e .[api,services]\n```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Default rule library
# ---------------------------------------------------------------------------
_DEFAULT_RULES: list[dict[str, Any]] = [
    {
        "id": "MAX_QTY",
        "field": "quantity",
        "comparator": ">",
        "threshold": 10_000,
        "message": "Order quantity exceeds max single-order limit (10,000 units).",
        "severity": "critical",
    },
    {
        "id": "MAX_NOTIONAL",
        "field": "price",
        "comparator": ">",
        "threshold": 500,
        "message": "Execution price above $500 — validate against pre-trade fairness benchmark.",
        "severity": "warning",
    },
    {
        "id": "LARGE_SIDE_BUY",
        "field": "quantity",
        "comparator": ">",
        "threshold": 5_000,
        "message": "Buy order > 5,000 units triggers enhanced pre-trade review.",
        "severity": "warning",
    },
]

# ── session state ──────────────────────────────────────────────────────────
for _k, _v in [
    ("comp_rules", list(_DEFAULT_RULES)),
    ("comp_results", []),
    ("comp_history", []),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ===========================================================================
# Layout: two columns — form + results
# ===========================================================================
col_form, col_results = st.columns([1, 1], gap="large")

# ---------------------------------------------------------------------------
# Transaction Entry Form
# ---------------------------------------------------------------------------
with col_form:
    st.subheader("Transaction Details")

    with st.form("compliance_form"):
        tx_id = st.text_input("Transaction ID", value=f"TXN-{uuid.uuid4().hex[:8].upper()}")
        user_id = st.text_input("User ID", value="USR-001")
        order_id = st.text_input("Order ID", value="ORD-001")
        symbol = st.text_input("Symbol", value="THYAO")
        side = st.selectbox("Side", ["buy", "sell"])
        quantity = st.number_input("Quantity", min_value=0.0, value=1000.0, step=100.0)
        price = st.number_input("Execution Price", min_value=0.0, value=150.0, step=0.5)
        venue = st.selectbox(
            "Venue",
            ["BIST", "XCME", "XNAS", "XNYS", "OTC", "CRYPTO"],
        )
        strategy_id = st.text_input("Strategy ID", value="STRAT-MOMENTUM")
        timestamp = st.text_input(
            "Timestamp (ISO)",
            value=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        submitted = st.form_submit_button(
            "▶ Run Compliance Check", use_container_width=True, type="primary"
        )

    if submitted:
        record = TransactionRecord(
            id=tx_id,
            timestamp=timestamp,
            user_id=user_id,
            order_id=order_id,
            symbol=symbol,
            side=side,  # type: ignore[arg-type]
            quantity=quantity,
            price=price,
            venue=venue,
            strategy_id=strategy_id,
        )
        rules = [ComplianceRule(**r) for r in st.session_state.comp_rules]
        hits = run_compliance_rule_engine(record, rules)

        # Store results
        st.session_state.comp_results = hits
        st.session_state.comp_history.append(
            {
                "timestamp": timestamp,
                "tx_id": tx_id,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "hits": len(hits),
                "status": "FAIL" if hits else "PASS",
            }
        )

# ---------------------------------------------------------------------------
# Results — pass/fail checklist
# ---------------------------------------------------------------------------
with col_results:
    st.subheader("Compliance Check Results")

    results: list[ComplianceHit] = st.session_state.comp_results
    rules_checked = st.session_state.comp_rules

    if not results and not st.session_state.comp_history:
        st.info("Submit a transaction above to run the compliance rule engine.")
    else:
        if st.session_state.comp_history:
            last_run = st.session_state.comp_history[-1]
            overall_pass = last_run["status"] == "PASS"

            # ── overall badge ─────────────────────────────────────────────
            if overall_pass:
                st.success("### ✅ COMPLIANCE PASSED")
                st.caption(f"All {len(rules_checked)} rules satisfied for transaction `{last_run['tx_id']}`.")
            else:
                st.error(
                    f"### ❌ COMPLIANCE FAILED — "
                    f"{last_run['hits']} rule{'s' if last_run['hits'] != 1 else ''} triggered"
                )

            st.divider()

            # ── checklist ─────────────────────────────────────────────────
            st.markdown("#### Rule Checklist")

            hit_ids = {h.rule_id for h in results}

            for rule in rules_checked:
                rule_id = rule["id"]
                triggered = rule_id in hit_ids

                if triggered:
                    hit = next(h for h in results if h.rule_id == rule_id)
                    icon = "🔴" if hit.severity == "critical" else "🟡"
                    with st.container():
                        st.markdown(
                            f"{icon} **`{rule_id}`** — {hit.message}  \n"
                            f"<span style='color:#aaa;font-size:0.82rem;'>"
                            f"Severity: **{hit.severity.upper()}**</span>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(f"🟢 **`{rule_id}`** — {rule['message']}")

            # ── hit detail table ───────────────────────────────────────────
            if results:
                st.divider()
                st.markdown("#### Triggered Rules — Detail")
                df_hits = pd.DataFrame(
                    [
                        {
                            "Rule ID": h.rule_id,
                            "Severity": h.severity.upper(),
                            "Message": h.message,
                        }
                        for h in results
                    ]
                )

                def _color_severity(val: str) -> str:
                    if val == "CRITICAL":
                        return "color: #e74c3c; font-weight: 700"
                    if val == "WARNING":
                        return "color: #f39c12; font-weight: 600"
                    return ""

                st.dataframe(
                    df_hits.style.applymap(_color_severity, subset=["Severity"]),
                    use_container_width=True,
                    hide_index=True,
                )

# ===========================================================================
# Rule Editor
# ===========================================================================
st.divider()
st.subheader("Compliance Rule Editor")
st.caption(
    "Add, edit or remove rules applied during the check. Changes apply immediately to the next submission."
)

rule_df = pd.DataFrame(st.session_state.comp_rules)
edited_rules = st.data_editor(
    rule_df,
    num_rows="dynamic",
    column_config={
        "id": st.column_config.TextColumn("Rule ID", width="small"),
        "field": st.column_config.SelectboxColumn(
            "Transaction Field",
            options=["quantity", "price", "quantity"],
            width="small",
        ),
        "comparator": st.column_config.SelectboxColumn(
            "Comparator",
            options=[">", ">=", "<", "<=", "==", "!="],
            width="small",
        ),
        "threshold": st.column_config.NumberColumn("Threshold", format="%.4f"),
        "message": st.column_config.TextColumn("Message", width="large"),
        "severity": st.column_config.SelectboxColumn(
            "Severity", options=["warning", "critical"]
        ),
    },
    use_container_width=True,
    key="rule_editor",
)

col_save, col_reset = st.columns([1, 5])
if col_save.button("💾 Save Rules", type="primary"):
    st.session_state.comp_rules = edited_rules.to_dict("records")
    st.success("Rules saved.")
if col_reset.button("↩ Reset to Defaults"):
    st.session_state.comp_rules = list(_DEFAULT_RULES)
    st.rerun()

# ===========================================================================
# Position Limit Monitor
# ===========================================================================
st.divider()
st.subheader("Position Limit Monitor")
st.caption("Positions exceeding their notional limit are flagged automatically.")

if "pos_limits" not in st.session_state:
    st.session_state.pos_limits = [
        {"symbol": "THYAO", "value": 1_200_000, "limit": 1_000_000},
        {"symbol": "EREGL", "value": 450_000, "limit": 500_000},
        {"symbol": "ASELS", "value": 780_000, "limit": 750_000},
        {"symbol": "KCHOL", "value": 300_000, "limit": 600_000},
    ]

edited_pos = st.data_editor(
    pd.DataFrame(st.session_state.pos_limits),
    num_rows="dynamic",
    column_config={
        "symbol": st.column_config.TextColumn("Symbol"),
        "value": st.column_config.NumberColumn("Current Value", format="%.0f"),
        "limit": st.column_config.NumberColumn("Limit", format="%.0f"),
    },
    use_container_width=True,
    key="pos_editor",
)

if st.button("Check Position Limits"):
    breaches = monitor_position_limits(edited_pos.to_dict("records"))
    if breaches:
        st.error(f"**{len(breaches)} position limit breach(es) detected:**")
        df_b = pd.DataFrame(breaches)
        df_b["excess"] = df_b["value"] - df_b["limit"]
        df_b["excess_pct"] = (df_b["excess"] / df_b["limit"] * 100).round(2)
        st.dataframe(
            df_b.rename(
                columns={
                    "symbol": "Symbol",
                    "value": "Current Value",
                    "limit": "Limit",
                    "excess": "Excess",
                    "excess_pct": "Excess (%)",
                }
            ).style.format(
                {
                    "Current Value": "{:,.0f}",
                    "Limit": "{:,.0f}",
                    "Excess": "{:+,.0f}",
                    "Excess (%)": "{:+.2f}%",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("All positions within limits.")

# ===========================================================================
# Activity Anomaly Detection
# ===========================================================================
st.divider()
st.subheader("User Activity Anomaly Detection")
st.caption(
    "Paste a JSON-like activity log or enter user IDs to detect statistically unusual activity (z-score ≥ 2.2)."
)

if "activity_events" not in st.session_state:
    st.session_state.activity_events = [
        {"user_id": "USR-001"},
        {"user_id": "USR-001"},
        {"user_id": "USR-001"},
        {"user_id": "USR-002"},
        {"user_id": "USR-003"},
        {"user_id": "USR-003"},
        {"user_id": "USR-004"},
        {"user_id": "USR-004"},
        {"user_id": "USR-004"},
        {"user_id": "USR-005"},
        {"user_id": "USR-001"},
        {"user_id": "USR-001"},
        {"user_id": "USR-001"},
        {"user_id": "USR-001"},
        {"user_id": "USR-001"},
        {"user_id": "USR-001"},
        {"user_id": "USR-001"},
        {"user_id": "USR-001"},
        {"user_id": "USR-001"},
        {"user_id": "USR-001"},
    ]

events_df = st.data_editor(
    pd.DataFrame(st.session_state.activity_events),
    num_rows="dynamic",
    column_config={
        "user_id": st.column_config.TextColumn("User ID"),
    },
    use_container_width=True,
    key="activity_editor",
)

if st.button("Detect Anomalies"):
    anomalies = detect_user_activity_anomalies(events_df.to_dict("records"))
    if anomalies:
        st.warning(f"**{len(anomalies)} anomalous user(s) detected (z-score ≥ 2.2):**")
        st.dataframe(
            pd.DataFrame(anomalies)
            .rename(
                columns={
                    "user_id": "User ID",
                    "actions_per_hour": "Actions / Hour",
                    "z_score": "Z-Score",
                }
            )
            .style.format({"Z-Score": "{:+.4f}"}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("No anomalous activity detected.")

# ===========================================================================
# Audit history
# ===========================================================================
st.divider()
st.subheader("Compliance Run History")

history = st.session_state.comp_history
if history:
    df_hist = pd.DataFrame(history[::-1])  # newest first

    def _style_status(val: str) -> str:
        if val == "FAIL":
            return "color: #e74c3c; font-weight: 700"
        return "color: #2ecc71; font-weight: 700"

    st.dataframe(
        df_hist.rename(
            columns={
                "timestamp": "Timestamp",
                "tx_id": "Transaction ID",
                "symbol": "Symbol",
                "side": "Side",
                "quantity": "Qty",
                "price": "Price",
                "hits": "Rules Triggered",
                "status": "Status",
            }
        ).style.applymap(_style_status, subset=["Status"]),
        use_container_width=True,
        hide_index=True,
    )
    if st.button("🗑 Clear History"):
        st.session_state.comp_history = []
        st.session_state.comp_results = []
        st.rerun()
else:
    st.info("No compliance checks run yet.")
