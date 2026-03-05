"""Signal Construction endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from bist_quant.api.schemas import (
    SignalConstructionBacktestRequest,
    SignalConstructionFiveFactorRequest,
    SignalConstructionOrthogonalizationRequest,
    SignalConstructionSnapshotRequest,
)
from bist_quant.services import CoreBackendService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/signal-construction", tags=["signal-construction"])


@router.post("/snapshot")
def signal_construction_snapshot(payload: SignalConstructionSnapshotRequest) -> dict[str, Any]:
    """Run multi-indicator signal snapshot."""
    try:
        from bist_quant.engines.signal_construction import run_signal_snapshot

        return run_signal_snapshot(payload.model_dump())
    except Exception as exc:
        logger.exception("signal_construction_snapshot failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/backtest")
def signal_construction_backtest(payload: SignalConstructionBacktestRequest) -> dict[str, Any]:
    """Run multi-indicator signal backtest."""
    try:
        from bist_quant.engines.signal_construction import run_signal_backtest

        return run_signal_backtest(payload.model_dump())
    except Exception as exc:
        logger.exception("signal_construction_backtest failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/five-factor")
def signal_construction_five_factor(payload: SignalConstructionFiveFactorRequest) -> dict[str, Any]:
    """Run five-factor rotation backtest via core service."""
    try:
        core = CoreBackendService(strict_paths=False)
        return core.run_backtest(
            factor_name=payload.factor_name,
            start_date=payload.start_date,
            end_date=payload.end_date,
            rebalance_frequency=payload.rebalance_frequency,
            top_n=payload.top_n,
            max_position_weight=payload.max_position_weight,
        )
    except Exception as exc:
        logger.exception("signal_construction_five_factor failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/orthogonalization")
def signal_construction_orthogonalization(
    payload: SignalConstructionOrthogonalizationRequest,
) -> dict[str, Any]:
    """Return orthogonalization configuration diagnostics for selected axes."""
    try:
        axes = [str(axis).strip().lower() for axis in payload.axes if str(axis).strip()]
        if not axes:
            axes = ["momentum", "value", "quality", "size", "profitability", "risk"]

        return {
            "enabled": payload.enabled,
            "axes": axes,
            "min_overlap": payload.min_overlap,
            "epsilon": payload.epsilon,
            "status": "configured",
            "note": (
                "This endpoint exposes orthogonalization configuration for the five-factor "
                "pipeline. Use /five-factor to evaluate runtime impact in backtest results."
            ),
        }
    except Exception as exc:
        logger.exception("signal_construction_orthogonalization failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

