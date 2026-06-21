"""Compliance and alert analytics: rule engine, position limits, anomaly detection, alerting.

Split out of the former professional.py monolith. Covers transaction compliance
checking, position-limit monitoring, user activity anomaly detection, metric
threshold alerting, alert grouping/deduplication, and escalation planning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ..core_metrics import mean, sample_std_dev
from .._shared import _compare, _parse_date, _to_fixed

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class TransactionRecord:
    id: str; timestamp: str; user_id: str; order_id: str; symbol: str
    side: Literal["buy","sell"]; quantity: float; price: float
    venue: str; strategy_id: str

@dataclass
class ComplianceRule:
    id: str; field: str; comparator: str; threshold: float
    message: str; severity: Literal["warning","critical"]

@dataclass
class ComplianceHit:
    rule_id: str; message: str; severity: Literal["warning","critical"]

@dataclass
class AlertCondition:
    id: str; name: str; metric: str; comparator: str; threshold: float
    severity: Literal["info","warning","critical"]
    channels: list[str]; group_key: str = ""; cooldown_sec: int = 0

@dataclass
class NotificationAlert:
    condition_id: str; name: str; severity: Literal["info","warning","critical"]
    channels: list[str]; message: str; metric_value: float
    triggered_at: str; group_key: str

# ---------------------------------------------------------------------------
# Compliance & Alerts
# ---------------------------------------------------------------------------

def run_compliance_rule_engine(record: TransactionRecord, rules: list[ComplianceRule]) -> list[ComplianceHit]:
    hits = []
    for rule in rules:
        raw = getattr(record, rule.field, None)
        obs = raw if isinstance(raw, (int, float)) else float(raw) if raw is not None else float('nan')
        if _compare(obs, rule.comparator, rule.threshold):
            hits.append(ComplianceHit(rule.id, rule.message, rule.severity))
    return hits

def monitor_position_limits(limits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted([l for l in limits if l["value"] > l["limit"]],
                  key=lambda l: l["value"] - l["limit"], reverse=True)

def detect_user_activity_anomalies(events: list[dict[str, str]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for e in events: counts[e["user_id"]] = counts.get(e["user_id"], 0) + 1
    vals = list(counts.values())
    avg = mean(vals); sd = sample_std_dev(vals)
    if sd <= 0: return []
    return sorted([{"user_id": u, "actions_per_hour": _to_fixed(c,4), "z_score": _to_fixed((c-avg)/sd,4)}
                   for u, c in counts.items() if (c-avg)/sd >= 2.2],
                  key=lambda x: x["z_score"], reverse=True)

def _get_metric(metrics: dict, path: str) -> float:
    cursor: Any = metrics
    for p in path.split("."):
        if not isinstance(cursor, dict): return float('nan')
        cursor = cursor.get(p)
    return cursor if isinstance(cursor, (int, float)) else float('nan')

def evaluate_alert_conditions(metrics: dict, conditions: list[AlertCondition],
                              state: dict[str, str | None], now_iso: str) -> dict[str, Any]:
    now = _parse_date(now_iso)
    next_state = dict(state)
    alerts: list[NotificationAlert] = []
    for cond in conditions:
        val = _get_metric(metrics, cond.metric)
        if not _compare(val, cond.comparator, cond.threshold): continue
        last = _parse_date(state.get(cond.id, "") or "1970-01-01")
        cd_ms = max(0, cond.cooldown_sec) * 1000
        if state.get(cond.id) and (now - last).total_seconds()*1000 < cd_ms: continue
        alerts.append(NotificationAlert(
            cond.id, cond.name, cond.severity, cond.channels,
            f"{cond.name}: {cond.metric}={_to_fixed(val,4)} breached {cond.comparator} {cond.threshold}",
            _to_fixed(val,6), now_iso, cond.group_key or cond.id))
        next_state[cond.id] = now_iso
    return {"alerts": alerts, "nextState": next_state}

def group_alerts(alerts: list[NotificationAlert], window_sec: int = 120) -> list[NotificationAlert]:
    if not alerts: return []
    srt = sorted(alerts, key=lambda a: a.triggered_at)
    grouped: list[NotificationAlert] = []
    last_by: dict[str, str] = {}
    for a in srt:
        last = last_by.get(a.group_key)
        if last:
            dt = (_parse_date(a.triggered_at) - _parse_date(last)).total_seconds()
            if dt <= window_sec: continue
        grouped.append(a); last_by[a.group_key] = a.triggered_at
    return grouped

def build_escalation_plan(alert: NotificationAlert) -> list[dict[str, Any]]:
    if alert.severity == "critical":
        return [{"after_minutes": 0, "channel": "push", "action": "Instant push notification"},
                {"after_minutes": 1, "channel": "sms", "action": "Escalate to on-call desk"},
                {"after_minutes": 3, "channel": "email", "action": "Send incident summary"},
                {"after_minutes": 5, "channel": "webhook", "action": "Open incident ticket"}]
    if alert.severity == "warning":
        return [{"after_minutes": 0, "channel": "push", "action": "Notify portfolio manager"},
                {"after_minutes": 5, "channel": "email", "action": "Send warning digest"}]
    return [{"after_minutes": 0, "channel": "push", "action": "Record informational update"}]


__all__ = [
    "TransactionRecord",
    "ComplianceRule",
    "ComplianceHit",
    "AlertCondition",
    "NotificationAlert",
    "run_compliance_rule_engine",
    "monitor_position_limits",
    "detect_user_activity_anomalies",
    "evaluate_alert_conditions",
    "group_alerts",
    "build_escalation_plan",
]
