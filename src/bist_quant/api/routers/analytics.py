"""Analytics endpoints — run performance/risk/MC/walk-forward/attribution analytics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException

from bist_quant.api.schemas import AnalyticsRunRequest
from bist_quant.common.data_paths import get_data_paths

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


def _to_series_points(raw: list[dict[str, Any]]) -> list[Any]:
    """Convert [{date, value}] dicts to SeriesPoint instances."""
    from bist_quant.analytics.core_metrics import SeriesPoint

    points: list[SeriesPoint] = []
    for item in raw:
        date_val = str(item.get("date", "")).strip()
        value_val = item.get("value")
        if not date_val or value_val is None:
            continue
        points.append(SeriesPoint(date=date_val, value=float(value_val)))
    return points


def _load_xu100_curve() -> list[dict[str, Any]]:
    """Load XU100 close curve as [{date, value}] from local data paths."""
    data_paths = get_data_paths()
    xu100_path: Path = data_paths.xu100_prices
    if not xu100_path.exists():
        return []

    try:
        if xu100_path.suffix == ".parquet":
            frame = pd.read_parquet(xu100_path)
        else:
            frame = pd.read_csv(xu100_path)
    except Exception:
        return []

    if frame.empty:
        return []

    date_col = next((c for c in frame.columns if "date" in c.lower()), frame.columns[0])
    close_col = next(
        (c for c in frame.columns if c.lower() in {"close", "kapanis", "kapanış", "fiyat"}),
        frame.columns[-1],
    )
    frame["date"] = pd.to_datetime(frame[date_col], errors="coerce")
    frame["value"] = pd.to_numeric(frame[close_col], errors="coerce")
    frame = frame.dropna(subset=["date", "value"]).sort_values("date")
    return [
        {"date": str(row.date.date()), "value": float(row.value)}
        for row in frame.itertuples(index=False)
    ]


@router.get("/benchmark/xu100")
def analytics_benchmark_xu100() -> dict[str, Any]:
    """Return XU100 benchmark curve as {date,value} points."""
    return {"symbol": "XU100", "curve": _load_xu100_curve()}


@router.post("/run")
def analytics_run(payload: AnalyticsRunRequest) -> dict[str, Any]:
    """Run selected analytics methods on an equity curve."""
    try:
        from bist_quant.analytics.core_metrics import (
            apply_transaction_costs,
            build_rolling_metrics,
            build_walk_forward_analysis,
            compute_performance_metrics,
            curve_to_returns,
            run_monte_carlo_bootstrap,
            run_stress_scenarios,
        )

        equity_points = _to_series_points(payload.equity_curve)
        if len(equity_points) < 2:
            raise HTTPException(
                status_code=422, detail="equity_curve must have at least 2 valid points"
            )

        returns = curve_to_returns(equity_points)

        benchmark_points = _to_series_points(payload.benchmark_curve)
        if payload.include_benchmark and len(benchmark_points) < 2:
            if payload.benchmark_symbol.upper() == "XU100":
                benchmark_points = _to_series_points(_load_xu100_curve())

        benchmark_returns = (
            curve_to_returns(benchmark_points) if len(benchmark_points) >= 2 else None
        )

        methods = {m.lower().strip() for m in payload.methods}
        result: dict[str, Any] = {
            "methods": sorted(methods),
            "benchmark": {
                "enabled": bool(payload.include_benchmark),
                "symbol": payload.benchmark_symbol.upper(),
                "points": len(benchmark_points),
            },
        }

        if "performance" in methods:
            perf = compute_performance_metrics(
                returns=returns,
                equity_curve=equity_points,
                benchmark_returns=benchmark_returns,
            )
            result["performance"] = _dataclass_to_dict(perf)

        if "rolling" in methods:
            rolling = build_rolling_metrics(equity_points)
            result["rolling"] = [_dataclass_to_dict(r) for r in rolling]

        if "walk_forward" in methods:
            wf = build_walk_forward_analysis(
                returns,
                benchmark_returns=benchmark_returns,
                splits=payload.walk_forward_splits,
                train_ratio=payload.train_ratio,
            )
            result["walk_forward"] = [_dataclass_to_dict(row) for row in wf]

        if "monte_carlo" in methods:
            mc = run_monte_carlo_bootstrap(
                returns=returns,
                iterations=payload.monte_carlo_iterations,
                horizon_days=payload.monte_carlo_horizon,
            )
            result["monte_carlo"] = _dataclass_to_dict(mc)

        if "stress" in methods:
            stress = run_stress_scenarios(returns)
            result["stress"] = [_dataclass_to_dict(s) for s in stress]

        if "cost" in methods or "transaction_costs" in methods:
            tca = apply_transaction_costs(
                returns=returns,
                slippage_bps=payload.slippage_bps,
                spread_bps=payload.spread_bps,
                market_impact_bps=payload.market_impact_bps,
                tax_rate_pct=payload.tax_rate_pct,
                rebalance_every_days=payload.rebalance_every_days,
            )
            result["transaction_costs"] = _dataclass_to_dict(tca)

        if "risk" in methods:
            try:
                from bist_quant.analytics.advanced import compute_advanced_risk_metrics

                risk = compute_advanced_risk_metrics(returns)
                result["risk"] = _dataclass_to_dict(risk)
            except Exception:
                result["risk"] = None

        if "attribution" in methods:
            if benchmark_returns is None:
                result["attribution"] = None
            else:
                try:
                    from bist_quant.analytics.advanced import (
                        compute_performance_attribution_breakdown,
                    )

                    attr = compute_performance_attribution_breakdown(
                        strategy_returns=returns,
                        benchmark_returns=benchmark_returns,
                    )
                    result["attribution"] = _dataclass_to_dict(attr)
                except Exception:
                    result["attribution"] = None

        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("analytics_run failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert dataclass instances to dicts, handling nested structures."""
    import dataclasses
    import math

    if obj is None:
        return None
    if isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _dataclass_to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(v) for v in obj]
    return str(obj)

