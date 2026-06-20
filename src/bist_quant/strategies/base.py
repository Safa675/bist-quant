"""
Tactical overlay protocol, factory, and shared helpers.

A tactical overlay is a strategy that produces an exposure scalar in [0, 1]
for each trading date. This scalar multiplies the regime allocation inside
the Backtester event loop, allowing macro or cross-asset signals to gate
equity exposure.

Overlay lifecycle:
    1. build_tactical_overlay(name, loader, config) creates the instance.
    2. Backtester.run() calls overlay.exposure(date) once per trading day.
    3. The return value multiplies the regime allocation.

Unlike SignalBuilder (which produces dates × tickers panels), overlays are
not cross-sectional — they operate at the portfolio level.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class TacticalOverlay(Protocol):
    """Protocol for tactical exposure overlays."""

    def exposure(self, date: pd.Timestamp) -> float:
        """Return equity exposure scalar in [0, 1] as of *date*.

        The caller (Backtester) must only pass data available at ``date``.
        Implementations must not use future information.

        Returns:
            Float in [0.0, 1.0]. 1.0 = fully invested in equities,
            0.0 = fully defensive (gold/cash).
        """
        ...


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_OVERLAY_REGISTRY: dict[str, type] = {}


def register_overlay(name: str):
    """Decorator that registers an overlay class by name."""
    def _decorator(cls):
        _OVERLAY_REGISTRY[name] = cls
        return cls
    return _decorator


def build_tactical_overlay(
    name: str,
    loader: Any,
    config: dict[str, Any],
) -> TacticalOverlay | None:
    """Instantiate a tactical overlay by name.

    Args:
        name: Overlay identifier (e.g. ``"paired_switching"``).
        loader: Data loader instance for fetching prices/macro data.
        config: Strategy configuration dict. The ``parameters`` sub-dict
            is passed through to the overlay constructor.

    Returns:
        A TacticalOverlay instance, or ``None`` if *name* is not
        recognised (silently degrades to no overlay).
    """
    # Lazy import to avoid circular dependencies at module level.
    from bist_quant.strategies.tactical.paired_switching import (  # noqa: F811
        PairedSwitchingOverlay,
    )
    from bist_quant.strategies.tactical.oil_equity import (  # noqa: F811
        OilEquityOverlay,
    )

    _OVERLAY_REGISTRY.update({
        "paired_switching": PairedSwitchingOverlay,
        "oil_equity": OilEquityOverlay,
    })

    params = config.get("parameters", config.get("signal_params", {}))
    cls = _OVERLAY_REGISTRY.get(name)
    if cls is None:
        logger.warning("Unknown tactical overlay '%s' — skipping", name)
        return None

    logger.info("Building tactical overlay: %s", name)
    return cls(loader=loader, config=config, params=params)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def quarterly_rebalance_dates(trading_days: pd.DatetimeIndex) -> set[pd.Timestamp]:
    """Return the first trading day of each quarter (Jan, Apr, Jul, Oct).

    This mirrors the QC paired-switching logic which rebalances in months
    where ``month % 3 == 0`` on the first bar.
    """
    rebalance_days: set[pd.Timestamp] = set()
    seen_months: set[tuple[int, int]] = set()
    for ts in trading_days:
        ym = (ts.year, ts.month)
        if ts.month % 3 == 0 and ym not in seen_months:
            rebalance_days.add(ts)
            seen_months.add(ym)
    return rebalance_days


def monthly_rebalance_dates(trading_days: pd.DatetimeIndex) -> set[pd.Timestamp]:
    """Return the first trading day of each month."""
    rebalance_days: set[pd.Timestamp] = set()
    seen_months: set[tuple[int, int]] = set()
    for ts in trading_days:
        ym = (ts.year, ts.month)
        if ym not in seen_months:
            rebalance_days.add(ts)
            seen_months.add(ym)
    return rebalance_days


def rolling_ols_predict(
    x: np.ndarray,
    y: np.ndarray,
    latest_x: float,
    min_obs: int = 13,
) -> float | None:
    """Expanding-window OLS prediction.

    Regresses ``y`` on ``x`` and returns the predicted *y* for the latest
    observation of *x*. Returns ``None`` when there are fewer than
    ``min_obs`` observations.

    Args:
        x: Independent variable observations (e.g. lagged oil returns).
        y: Dependent variable observations (e.g. equity returns).
        latest_x: Most recent value of *x* to predict from.
        min_obs: Minimum observations before regression is valid.

    Returns:
        Predicted *y* value, or ``None`` if insufficient data.
    """
    n = len(x)
    if n < min_obs:
        return None
    if not np.isfinite(latest_x):
        return None

    # Simple OLS via numpy (no scipy dependency)
    x_col = x.reshape(-1, 1)
    ones = np.ones((n, 1))
    X = np.hstack([ones, x_col])

    try:
        # np.linalg.lstsq is numerically stable
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        intercept, slope = beta
        return float(intercept + slope * latest_x)
    except np.linalg.LinAlgError:
        return None


__all__ = [
    "TacticalOverlay",
    "build_tactical_overlay",
    "register_overlay",
    "quarterly_rebalance_dates",
    "monthly_rebalance_dates",
    "rolling_ols_predict",
]
