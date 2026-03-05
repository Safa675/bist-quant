"""Professional trading tools endpoints (Greeks, stress test, crypto sizing)."""

from __future__ import annotations

import dataclasses
import logging
import math
from typing import Any

from fastapi import APIRouter, HTTPException

from bist_quant.api.schemas import (
    CryptoSizingRequest,
    GreeksRequest,
    PipValueRequest,
    StressTestRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/professional", tags=["professional"])


def _safe_dict(obj: Any) -> Any:
    """Convert dataclass to JSON-safe dict."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _safe_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _safe_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_dict(v) for v in obj]
    return str(obj)


@router.post("/greeks")
def professional_greeks(payload: GreeksRequest) -> dict[str, Any]:
    """Compute Black-Scholes option Greeks."""
    try:
        from bist_quant.analytics.professional import (
            OptionGreeksInput,
            compute_option_greeks,
        )

        inp = OptionGreeksInput(
            option_type=payload.option_type,
            spot=payload.spot,
            strike=payload.strike,
            time_years=payload.time_years,
            volatility=payload.volatility,
            risk_free_rate=payload.risk_free_rate,
        )
        result = compute_option_greeks(inp)
        return _safe_dict(result)  # type: ignore[return-value]
    except Exception as exc:
        logger.exception("professional_greeks failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/stress")
def professional_stress(payload: StressTestRequest) -> dict[str, Any]:
    """Run a portfolio stress test with factor shocks."""
    try:
        from bist_quant.analytics.professional import (
            StressFactorShock,
            run_portfolio_stress_test,
        )

        shocks = [
            StressFactorShock(
                factor=s.get("factor", ""),
                shock_pct=float(s.get("shock_pct", 0)),
                beta=float(s.get("beta", 1.0)),
            )
            for s in payload.shocks
        ]
        result = run_portfolio_stress_test(
            portfolio_value=payload.portfolio_value,
            shocks=shocks,
        )
        return _safe_dict(result)  # type: ignore[return-value]
    except Exception as exc:
        logger.exception("professional_stress failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/crypto-sizing")
def professional_crypto_sizing(payload: CryptoSizingRequest) -> dict[str, Any]:
    """Compute crypto position sizing and trade plan."""
    try:
        from bist_quant.analytics.professional import (
            CryptoTradeInput,
            build_crypto_trade_plan,
        )

        inp = CryptoTradeInput(
            pair=payload.pair,
            side=payload.side,
            entry_price=payload.entry_price,
            equity=payload.equity,
            risk_pct=payload.risk_pct,
            leverage=payload.leverage,
            stop_distance_pct=payload.stop_distance_pct,
            taker_fee_bps=payload.taker_fee_bps,
        )
        result = build_crypto_trade_plan(inp)
        return _safe_dict(result)  # type: ignore[return-value]
    except Exception as exc:
        logger.exception("professional_crypto_sizing failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/pip-value")
def professional_pip_value(payload: PipValueRequest) -> dict[str, Any]:
    """Compute forex pip value for a pair and lot size."""
    try:
        from bist_quant.analytics.professional import compute_forex_pip_value

        result = compute_forex_pip_value(
            pair=payload.pair,
            lot_size=payload.lot_size,
            account_conversion_rate=payload.account_conversion_rate,
        )
        return _safe_dict(result)  # type: ignore[return-value]
    except Exception as exc:
        logger.exception("professional_pip_value failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
