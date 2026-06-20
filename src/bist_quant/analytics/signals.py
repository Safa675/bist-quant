"""Indicator combination signal (RSI + MACD + SMA trend + volatility regime)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .core_metrics import sample_std_dev
from ._shared import (
    PricePoint,
    _ema,
    _price_to_returns,
    _to_fixed,
)
from .core_metrics import mean


@dataclass
class IndicatorCombinationSignal:
    trend_signal: Literal["bullish", "neutral", "bearish"]
    momentum_signal: Literal["overbought", "neutral", "oversold"]
    volatility_signal: Literal["compressed", "normal", "expanded"]
    rsi: float
    macd: float


def build_indicator_combination_signal(
    points: list[PricePoint],
) -> IndicatorCombinationSignal:
    """RSI, MACD, SMA trend, and volatility regime signal."""
    ordered = sorted(points, key=lambda p: p.date)
    closes = [p.close for p in ordered]
    if len(closes) < 30:
        return IndicatorCombinationSignal("neutral", "neutral", "normal", 50, 0)

    sma20 = mean(closes[-20:])
    sma50 = mean(closes[-50:])
    trend: Literal["bullish", "neutral", "bearish"] = (
        "bullish" if sma20 > sma50 * 1.005 else "bearish" if sma20 < sma50 * 0.995 else "neutral"
    )

    gains = losses = 0.0
    for i in range(len(closes) - 14, len(closes)):
        delta = closes[i] - closes[i - 1]
        if delta > 0:
            gains += delta
        else:
            losses += abs(delta)
    rs = gains / losses if losses > 0 else 100
    rsi = 100 - 100 / (1 + rs)
    momentum: Literal["overbought", "neutral", "oversold"] = (
        "overbought" if rsi >= 70 else "oversold" if rsi <= 30 else "neutral"
    )

    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    macd_line = ema12[-1] - ema26[-1]

    recent_rets = [r.value for r in _price_to_returns(ordered)]
    vol_now = sample_std_dev(recent_rets[-21:])
    vol_base = sample_std_dev(recent_rets[-84:-21]) if len(recent_rets) > 84 else vol_now
    vol_signal: Literal["compressed", "normal", "expanded"] = (
        "expanded" if vol_now > vol_base * 1.25 else "compressed" if vol_now < vol_base * 0.8 else "normal"
    )

    return IndicatorCombinationSignal(
        trend_signal=trend,
        momentum_signal=momentum,
        volatility_signal=vol_signal,
        rsi=_to_fixed(rsi, 2),
        macd=_to_fixed(macd_line, 4),
    )


__all__ = [
    "IndicatorCombinationSignal",
    "build_indicator_combination_signal",
]
