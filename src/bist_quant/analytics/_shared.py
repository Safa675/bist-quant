"""Shared types and private helpers for the analytics domain modules.

Split out of ``advanced.py`` so that volatility / position-sizing /
portfolio-construction / etc. sub-modules can share the same primitive
helpers (clamp, rounding, weight normalisation, deterministic noise,
price-to-returns, EMA, timeframe grouping) without a circular import on
``advanced`` itself.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass
from typing import Literal

from .core_metrics import (
    SeriesPoint,
    mean,
    sample_std_dev,
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

VolatilityRegime = Literal["low", "normal", "high"]
PortfolioConstructionMethod = Literal[
    "mpt", "risk_parity", "min_variance", "equal_risk_contribution", "factor_based"
]
MarketRegime = Literal["bull", "bear", "sideways"]
ChartTimeframe = Literal["1D", "1W", "1M"]


# ---------------------------------------------------------------------------
# Shared data classes
# ---------------------------------------------------------------------------


@dataclass
class PricePoint:
    date: str
    close: float
    volume: float | None = None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float, hi: float) -> float:
    if not math.isfinite(value):
        return lo
    return max(lo, min(hi, value))


def _to_fixed(value: float, digits: int = 4) -> float:
    if not math.isfinite(value):
        return 0.0
    return round(value, digits)


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    entries = {k: v for k, v in weights.items() if math.isfinite(v) and v > 0}
    if not entries:
        return {}
    total = sum(entries.values())
    if total <= 0:
        return {}
    return {k: _to_fixed(v / total, 6) for k, v in entries.items()}


def _rolling_std(values: list[float], end_index: int, window: int) -> float:
    start = max(0, end_index - window + 1)
    return sample_std_dev(values[start : end_index + 1])


def _make_deterministic_noise(seed: int, index: int) -> float:
    raw = ((seed * 31 + index * 17) % 97) - 48
    return raw / 12_000


def _compare(left: float, comp: str, right: float) -> bool:
    if not math.isfinite(left) or not math.isfinite(right):
        return False
    if comp == ">": return left > right
    if comp == ">=": return left >= right
    if comp == "<": return left < right
    if comp == "<=": return left <= right
    if comp == "==": return left == right
    return left != right


def _parse_date(s: str) -> datetime.datetime:
    try:
        return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return datetime.datetime(1970, 1, 1)


def _price_to_returns(points: list[PricePoint]) -> list[SeriesPoint]:
    ordered = sorted(points, key=lambda p: p.date)
    if len(ordered) < 2:
        return []
    returns: list[SeriesPoint] = []
    for i in range(1, len(ordered)):
        prev = ordered[i - 1].close
        cur = ordered[i].close
        if not math.isfinite(prev) or not math.isfinite(cur) or prev <= 0:
            continue
        returns.append(SeriesPoint(date=ordered[i].date, value=cur / prev - 1))
    return returns


def _build_ma_strategy_returns(
    points: list[PricePoint], fast: int, slow: int
) -> list[SeriesPoint]:
    ordered = sorted(points, key=lambda p: p.date)
    if len(ordered) < slow + 2:
        return []
    closes = [p.close for p in ordered]
    returns: list[SeriesPoint] = []
    prev_exposure = 0.0
    for i in range(1, len(ordered)):
        prev_c = closes[i - 1]
        cur_c = closes[i]
        if not math.isfinite(prev_c) or not math.isfinite(cur_c) or prev_c <= 0:
            continue
        spot = cur_c / prev_c - 1
        returns.append(SeriesPoint(date=ordered[i].date, value=spot * prev_exposure))
        if i + 1 < slow:
            prev_exposure = 0.0
            continue
        fast_slice = closes[max(0, i - fast + 1) : i + 1]
        slow_slice = closes[max(0, i - slow + 1) : i + 1]
        prev_exposure = 1.0 if mean(fast_slice) > mean(slow_slice) else 0.0
    return returns


def _ema(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    alpha = 2.0 / (period + 1)
    output: list[float] = []
    prev = values[0]
    for v in values:
        prev = alpha * v + (1 - alpha) * prev
        output.append(prev)
    return output


def _group_key_by_timeframe(date: str, timeframe: ChartTimeframe) -> str:
    if timeframe == "1D":
        return date
    try:
        parts = date[:10].split("-")
        y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
    except (ValueError, IndexError):
        return date
    if timeframe == "1W":
        import datetime
        dt = datetime.date(y, m, d)
        iso_year, iso_week, _ = dt.isocalendar()
        return f"{iso_year}-W{iso_week:02d}"
    return f"{y}-{m:02d}"


__all__ = [
    "VolatilityRegime",
    "PortfolioConstructionMethod",
    "MarketRegime",
    "ChartTimeframe",
    "PricePoint",
    "_clamp",
    "_to_fixed",
    "_normalize_weights",
    "_rolling_std",
    "_make_deterministic_noise",
    "_compare",
    "_parse_date",
    "_price_to_returns",
    "_build_ma_strategy_returns",
    "_ema",
    "_group_key_by_timeframe",
]
