"""GARCH/EWMA volatility forecasting and proxy return series."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .core_metrics import (
    SeriesPoint,
    _sort_points,
    normal_cdf,
    quantile,
    sample_std_dev,
)
from ._shared import (
    PricePoint,
    VolatilityRegime,
    _clamp,
    _make_deterministic_noise,
    _rolling_std,
    _to_fixed,
)


@dataclass
class VolatilityForecastPoint:
    date: str
    forecast_vol_pct: float
    realized_vol_pct: float
    regime: VolatilityRegime


@dataclass
class VolatilityForecastResult:
    window: int
    latest_forecast_vol_pct: float
    latest_realized_vol_pct: float
    latest_regime: VolatilityRegime
    adaptive_size_pct: float
    regime_distribution_pct: dict[str, float]
    series: list[VolatilityForecastPoint]


def build_garch_volatility_forecast(
    returns: list[SeriesPoint],
    *,
    window: int = 50,
    target_vol_pct: float,
    base_allocation_pct: float,
    alpha: float = 0.08,
    beta: float = 0.9,
) -> VolatilityForecastResult:
    """EWMA/GARCH(1,1) volatility forecast with regime classification."""
    ordered = _sort_points(returns)
    values = [r.value for r in ordered]

    if not values:
        return VolatilityForecastResult(
            window=window,
            latest_forecast_vol_pct=0,
            latest_realized_vol_pct=0,
            latest_regime="normal",
            adaptive_size_pct=_clamp(base_allocation_pct, 0, 100),
            regime_distribution_pct={"low": 0, "normal": 100, "high": 0},
            series=[],
        )

    base_variance = max(sample_std_dev(values) ** 2, 1e-8)
    a = _clamp(alpha, 0.01, 0.2)
    b = _clamp(beta, 0.6, 0.98)
    omega = base_variance * max(1e-5, 1 - a - b)

    variance = base_variance
    forecast_vols: list[float] = []
    series: list[VolatilityForecastPoint] = []

    for i, pt in enumerate(ordered):
        ret = pt.value
        variance = omega + a * ret * ret + b * variance
        forecast = math.sqrt(max(variance, 0)) * math.sqrt(252) * 100
        realized = _rolling_std(values, i, window) * math.sqrt(252) * 100
        forecast_vols.append(forecast)
        series.append(VolatilityForecastPoint(
            date=pt.date,
            forecast_vol_pct=_to_fixed(forecast, 3),
            realized_vol_pct=_to_fixed(realized, 3),
            regime="normal",
        ))

    low_th = quantile(forecast_vols, 0.33)
    high_th = quantile(forecast_vols, 0.67)
    low_c = normal_c = high_c = 0

    for row in series:
        if row.forecast_vol_pct <= low_th:
            row.regime = "low"
            low_c += 1
        elif row.forecast_vol_pct >= high_th:
            row.regime = "high"
            high_c += 1
        else:
            normal_c += 1

    latest = series[-1] if series else VolatilityForecastPoint("", 0, 0, "normal")
    regime_mult = {"low": 1.1, "normal": 1.0, "high": 0.74}
    raw_adaptive = (
        (base_allocation_pct * target_vol_pct) / latest.forecast_vol_pct
        if latest.forecast_vol_pct > 0
        else base_allocation_pct
    )
    adaptive = raw_adaptive * regime_mult[latest.regime]
    total = max(1, len(series))

    return VolatilityForecastResult(
        window=window,
        latest_forecast_vol_pct=_to_fixed(latest.forecast_vol_pct, 2),
        latest_realized_vol_pct=_to_fixed(latest.realized_vol_pct, 2),
        latest_regime=latest.regime,
        adaptive_size_pct=_to_fixed(_clamp(adaptive, 0, 100), 2),
        regime_distribution_pct={
            "low": _to_fixed((low_c / total) * 100, 2),
            "normal": _to_fixed((normal_c / total) * 100, 2),
            "high": _to_fixed((high_c / total) * 100, 2),
        },
        series=series,
    )


def build_proxy_asset_return_series(
    base_returns: list[SeriesPoint],
    symbols: list[str],
) -> dict[str, list[SeriesPoint]]:
    """Create synthetic proxy return series for correlation analysis."""
    ordered = _sort_points(base_returns)
    if not ordered or not symbols:
        return {}
    safe_symbols = symbols[:12]
    output: dict[str, list[SeriesPoint]] = {}
    for sym_idx, symbol in enumerate(safe_symbols):
        seed = sum(ord(c) for c in symbol)
        lag = seed % 4
        scale = 0.76 + (seed % 35) / 100
        rows: list[SeriesPoint] = []
        for i, pt in enumerate(ordered):
            src_idx = max(0, i - lag)
            base = ordered[src_idx].value
            noise = _make_deterministic_noise(seed + sym_idx * 11, i)
            rows.append(SeriesPoint(
                date=pt.date,
                value=_clamp(base * scale + noise, -0.18, 0.18),
            ))
        output[symbol] = rows
    return output


__all__ = [
    "VolatilityForecastPoint",
    "VolatilityForecastResult",
    "build_garch_volatility_forecast",
    "build_proxy_asset_return_series",
]
