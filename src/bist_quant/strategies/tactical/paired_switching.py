"""
Paired switching tactical overlay.

Source: https://quantpedia.com/strategies/paired-switching/

Logic:
    Every quarter (Jan, Apr, Jul, Oct), compare the lookback-period return
    of two negatively-correlated assets (XU100 vs gold).  Hold the winner
    for the next quarter.  Maps to an equity exposure toggle:
    ``1.0`` when XU100 wins (risk-on), ``0.0`` when gold wins (defensive).

Return formula (matching QC source):
    ``performance = (current_price - lookback_price) / current_price``

BIST adaptation:
    - Asset A: XU100 index (``DataLoader.load_xu100_prices()``)
    - Asset B: XAU/TRY gold (``DataLoader.load_xautry_prices()``)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from bist_quant.strategies.base import (
    TacticalOverlay,
    quarterly_rebalance_dates,
)

logger = logging.getLogger(__name__)


class PairedSwitchingOverlay(TacticalOverlay):
    """Quarterly paired-switching overlay between XU100 and gold.

    Args:
        loader: Data loader instance.
        config: Full strategy config dict.
        params: Strategy parameters dict (``lookback_days``, ``defensive_asset``).
    """

    def __init__(
        self,
        loader: Any,
        config: dict[str, Any],
        params: dict[str, Any] | None = None,
    ) -> None:
        params = params or {}
        self._lookback_days: int = int(params.get("lookback_days", 63))
        self._defensive_asset: str = str(params.get("defensive_asset", "gold"))

        # Load data from the data loader
        self._equity: pd.Series = loader.load_xu100_prices()
        self._gold: pd.Series = loader.load_xautry_prices()

        # Build a combined date index for trading days
        all_dates = self._equity.index.union(self._gold.index).sort_values()
        self._rebalance_days: set[pd.Timestamp] = quarterly_rebalance_dates(all_dates)

        # Cache: date → exposure decision
        self._cache: dict[pd.Timestamp, float] = {}
        self._last_exposure: float = 1.0  # Default: fully invested

    def _compute_performance(
        self,
        series: pd.Series,
        date: pd.Timestamp,
    ) -> float | None:
        """Compute lookback performance: (current - lookback) / current.

        Uses only data available at or before *date*.  Returns ``None`` if
        the lookback window is incomplete.
        """
        # Select data up to and including date (no lookahead)
        valid = series.loc[:date]
        if len(valid) < self._lookback_days:
            return None

        current = float(valid.iloc[-1])
        lookback = float(valid.iloc[-self._lookback_days])

        if current <= 0:
            return None

        return (current - lookback) / current

    def exposure(self, date: pd.Timestamp) -> float:
        """Return equity exposure scalar.

        On a rebalance date, compare XU100 vs gold lookback returns.
        Between rebalance dates, return the last decision.
        """
        # Return cached exposure for non-rebalance dates
        if date not in self._rebalance_days:
            return self._last_exposure

        equity_perf = self._compute_performance(self._equity, date)
        gold_perf = self._compute_performance(self._gold, date)

        if equity_perf is None or gold_perf is None:
            # Not enough data yet — stay fully invested
            self._last_exposure = 1.0
            return self._last_exposure

        if equity_perf > gold_perf:
            self._last_exposure = 1.0  # XU100 wins → risk-on
        else:
            self._last_exposure = 0.0  # Gold wins → defensive

        self._cache[date] = self._last_exposure
        return self._last_exposure


__all__ = ["PairedSwitchingOverlay"]
