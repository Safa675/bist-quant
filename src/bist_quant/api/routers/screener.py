"""Stock screener endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
import pandas as pd

from bist_quant.api.schemas import ScreenerRunRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/screener", tags=["screener"])


@router.get("/metadata")
def screener_metadata() -> dict[str, Any]:
    """Return screener metadata: templates, filters, indexes, etc."""
    try:
        from bist_quant.engines.stock_filter import get_stock_filter_metadata

        return get_stock_filter_metadata()
    except Exception as exc:
        logger.exception("screener_metadata failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/run")
def screener_run(payload: ScreenerRunRequest) -> dict[str, Any]:
    """Run the stock screener with the given filters."""
    try:
        from bist_quant.engines.stock_filter import run_stock_filter

        result = run_stock_filter(payload.model_dump())
        return result
    except Exception as exc:
        logger.exception("screener_run failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/sparklines")
def screener_sparklines(payload: dict[str, Any]) -> dict[str, list[float]]:
    """Return short close-price sparkline series by symbol.

    Request:
    {
      "symbols": ["THYAO", "GARAN"],
      "points": 60
    }
    """
    try:
        symbols_raw = payload.get("symbols")
        if not isinstance(symbols_raw, list) or len(symbols_raw) == 0:
            raise HTTPException(
                status_code=422,
                detail="`symbols` must be a non-empty list of ticker strings.",
            )

        points_raw = payload.get("points", 60)
        points = int(points_raw) if points_raw is not None else 60
        points = max(20, min(points, 252))

        # Normalize + deduplicate symbol names.
        symbols: list[str] = []
        seen: set[str] = set()
        for item in symbols_raw:
            if not isinstance(item, str):
                continue
            symbol = item.strip().upper().split(".")[0]
            if symbol and symbol not in seen:
                symbols.append(symbol)
                seen.add(symbol)
        if not symbols:
            raise HTTPException(
                status_code=422,
                detail="No valid symbols provided after normalization.",
            )

        from bist_quant.engines.stock_filter import _SCREEN_CACHE, run_stock_filter

        close_cache = _SCREEN_CACHE.get("close_df")
        if not isinstance(close_cache, pd.DataFrame) or close_cache.empty:
            # Warm cache lazily so sparklines can be requested independently.
            run_stock_filter({"limit": 1, "chart_points": points})
            close_cache = _SCREEN_CACHE.get("close_df")

        if not isinstance(close_cache, pd.DataFrame) or close_cache.empty:
            return {}

        response: dict[str, list[float]] = {}
        for symbol in symbols:
            if symbol not in close_cache.columns:
                continue
            series = pd.to_numeric(close_cache[symbol], errors="coerce").dropna().tail(points)
            if len(series) < 5:
                continue
            response[symbol] = [round(float(value), 6) for value in series.tolist()]
        return response
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("screener_sparklines failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
