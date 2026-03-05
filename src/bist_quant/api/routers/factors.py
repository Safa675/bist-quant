"""Factor Lab & Signal Construction endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from bist_quant.api.schemas import FactorCombineRequest, FactorSnapshotRequest
from bist_quant.services import CoreBackendService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/factors", tags=["factors"])


@router.get("/catalog")
def factors_catalog() -> dict[str, Any]:
    """List all available signals with categories and parameter schemas."""
    try:
        from bist_quant.engines.factor_lab import build_factor_catalog

        result = build_factor_catalog()
        return result
    except Exception as exc:
        logger.exception("factors_catalog failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{name}")
def factors_detail(name: str) -> dict[str, Any]:
    """Get detailed info for a single signal/factor."""
    try:
        core = CoreBackendService(strict_paths=False)
        detail = core.get_signal_details(name)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"Signal '{name}' not found")
        return detail
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("factors_detail failed for %s", name)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/snapshot")
def factors_snapshot(payload: FactorSnapshotRequest) -> dict[str, Any]:
    """Run cross-sectional signal scoring snapshot."""
    try:
        from bist_quant.engines.signal_construction import run_signal_snapshot

        result = run_signal_snapshot(payload.model_dump())
        return result
    except Exception as exc:
        logger.exception("factors_snapshot failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/combine")
def factors_combine(payload: FactorCombineRequest) -> dict[str, Any]:
    """Combine multiple factors with weighting and optional timing."""
    try:
        core = CoreBackendService(strict_paths=False)

        # Build factor specs from request
        factors = []
        for sig in payload.signals:
            factor_spec: dict[str, Any] = {"name": sig.get("name", ""), "weight": sig.get("weight", 1.0)}
            # Pass through any extra signal params
            for k, v in sig.items():
                if k not in ("name", "weight"):
                    factor_spec[k] = v
            factors.append(factor_spec)

        result = core.combine_factors(
            factors=factors,
            start_date=payload.start_date,
            end_date=payload.end_date,
            weighting_scheme=payload.method,
            timing_enabled=payload.timing_enabled,
            timing_lookback=payload.timing_lookback,
            timing_threshold=payload.timing_threshold,
            benchmark=payload.benchmark,
            rebalance_frequency=payload.rebalance_frequency,
            top_n=payload.top_n,
            max_position_weight=payload.max_position_weight,
        )
        return result
    except Exception as exc:
        logger.exception("factors_combine failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
