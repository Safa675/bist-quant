"""Compliance endpoints — rule engine, position limits."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from bist_quant.api.schemas import ComplianceTransactionRequest, PositionLimitsRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/compliance", tags=["compliance"])

# Default compliance rules used when none are supplied in the request.
DEFAULT_COMPLIANCE_RULES: list[dict[str, Any]] = [
    {
        "id": "max_order_qty",
        "field": "quantity",
        "comparator": ">",
        "threshold": 100_000,
        "message": "Order quantity exceeds 100 000",
        "severity": "warning",
    },
    {
        "id": "max_notional",
        "field": "price",
        "comparator": ">",
        "threshold": 10_000_000,
        "message": "Notional value exceeds 10M TRY",
        "severity": "critical",
    },
]


@router.get("/rules")
def compliance_rules() -> dict[str, Any]:
    """Return the set of active compliance rules."""
    return {"rules": DEFAULT_COMPLIANCE_RULES}


@router.post("/check")
def compliance_check(payload: ComplianceTransactionRequest) -> dict[str, Any]:
    """Check a transaction against compliance rules."""
    try:
        from bist_quant.analytics.professional import (
            ComplianceRule,
            TransactionRecord,
            run_compliance_rule_engine,
        )

        # Build TransactionRecord from dict
        tx = payload.transaction
        record = TransactionRecord(
            id=str(tx.get("id", "")),
            timestamp=str(tx.get("timestamp", "")),
            user_id=str(tx.get("user_id", "")),
            order_id=str(tx.get("order_id", "")),
            symbol=str(tx.get("symbol", "")),
            side=tx.get("side", "buy"),
            quantity=float(tx.get("quantity", 0)),
            price=float(tx.get("price", 0)),
            venue=str(tx.get("venue", "")),
            strategy_id=str(tx.get("strategy_id", "")),
        )

        # Use provided rules or defaults
        raw_rules = payload.rules if payload.rules else DEFAULT_COMPLIANCE_RULES
        rules = [
            ComplianceRule(
                id=str(r.get("id", "")),
                field=str(r.get("field", "")),
                comparator=str(r.get("comparator", "")),
                threshold=float(r.get("threshold", 0)),
                message=str(r.get("message", "")),
                severity=r.get("severity", "warning"),
            )
            for r in raw_rules
        ]

        hits = run_compliance_rule_engine(record, rules)
        return {
            "transaction_id": record.id,
            "hits": [
                {
                    "rule_id": h.rule_id,
                    "message": h.message,
                    "severity": h.severity,
                }
                for h in hits
            ],
            "passed": len(hits) == 0,
        }
    except Exception as exc:
        logger.exception("compliance_check failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/position-limits")
def compliance_position_limits(payload: PositionLimitsRequest) -> dict[str, Any]:
    """Check for position limit breaches."""
    try:
        from bist_quant.analytics.professional import monitor_position_limits

        breaches = monitor_position_limits(payload.positions)
        return {
            "breaches": breaches,
            "total_checked": len(payload.positions),
            "breach_count": len(breaches),
        }
    except Exception as exc:
        logger.exception("compliance_position_limits failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/activity-anomalies")
def compliance_activity_anomalies(payload: dict[str, Any]) -> dict[str, Any]:
    """Detect anomalous user activity from event logs."""
    try:
        from bist_quant.analytics.professional import detect_user_activity_anomalies

        events = payload.get("events", [])
        if not isinstance(events, list):
            raise HTTPException(status_code=422, detail="`events` must be a list of objects")

        normalized: list[dict[str, str]] = []
        for row in events:
            if not isinstance(row, dict):
                continue
            user_id = str(row.get("user_id", "")).strip()
            if not user_id:
                continue
            normalized.append({"user_id": user_id})

        anomalies = detect_user_activity_anomalies(normalized)
        return {
            "events_count": len(normalized),
            "anomalies": anomalies,
            "anomaly_count": len(anomalies),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("compliance_activity_anomalies failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
