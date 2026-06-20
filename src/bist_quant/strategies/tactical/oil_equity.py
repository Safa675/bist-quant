"""
Oil-predicts-equity tactical overlay.

Source: https://quantpedia.com/strategies/crude-oil-predicts-equity-returns/

Logic:
    Turkey is a net oil importer. Rising oil prices pressure the current
    account and corporate margins, depressing equity returns.  This overlay
    runs a monthly expanding-window OLS regression of equity returns on
    lagged oil returns and allocates to equities when the predicted return
    exceeds the risk-free rate.

    Regression:  equity_ret_{t} = α + β × oil_ret_{t-1} + ε
    Signal:      predicted = α + β × last_oil_ret
    Allocation:  equity if predicted > rf, else cash (exposure = 0.0)

BIST adaptation:
    - Equity: XU100 index
    - Oil: WTI crude (CL=F), fetched via fetch_oil_prices.py
    - Risk-free rate: TCMB policy rate (FixedIncomeProvider, fallback 28%)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from bist_quant.strategies.base import (
    TacticalOverlay,
    monthly_rebalance_dates,
    rolling_ols_predict,
)

logger = logging.getLogger(__name__)


class OilEquityOverlay(TacticalOverlay):
    """Monthly oil→equity regression overlay.

    Args:
        loader: Data loader instance.
        config: Full strategy config dict.
        params: Strategy parameters (``min_observations``, ``oil_symbol``,
            ``rf_fallback``).
    """

    def __init__(
        self,
        loader: Any,
        config: dict[str, Any],
        params: dict[str, Any] | None = None,
    ) -> None:
        params = params or {}
        self._min_obs: int = int(params.get("min_observations", 13))
        self._oil_symbol: str = str(params.get("oil_symbol", "WTI"))
        self._rf_fallback: float = float(params.get("rf_fallback", 0.28))

        # Load equity index prices
        self._equity: pd.Series = loader.load_xu100_prices()

        # Load oil prices
        self._oil: pd.Series = self._load_oil(loader)

        # Load risk-free rate
        self._rf_rate: float = self._load_risk_free_rate(loader)

        # Compute daily returns on aligned dates
        self._equity_ret: pd.Series | None = None
        self._oil_ret: pd.Series | None = None
        self._aligned_dates: pd.DatetimeIndex | None = None
        self._compute_returns()

        # Rebalance schedule
        if self._aligned_dates is not None:
            self._rebalance_days: set[pd.Timestamp] = monthly_rebalance_dates(
                self._aligned_dates
            )
        else:
            self._rebalance_days: set[pd.Timestamp] = set()

        # Cache
        self._cache: dict[pd.Timestamp, float] = {}
        self._last_exposure: float = 1.0

    def _load_oil(self, loader: Any) -> pd.Series:
        """Load oil prices from DataLoader (falls back to empty series)."""
        if hasattr(loader, "load_oil_prices"):
            try:
                oil_df = loader.load_oil_prices()
                if oil_df is not None and not oil_df.empty:
                    col = self._oil_symbol
                    if col not in oil_df.columns:
                        # Try first column as fallback
                        col = oil_df.columns[0]
                        logger.info(
                            "Oil symbol '%s' not found, using '%s'", self._oil_symbol, col
                        )
                    return oil_df[col].astype(float)
            except Exception as exc:
                logger.warning("Failed to load oil prices: %s", exc)

        logger.warning("Oil prices not available — oil_equity overlay will be inactive")
        return pd.Series(dtype=float)

    def _load_risk_free_rate(self, loader: Any) -> float:
        """Load TCMB policy rate from fixed income provider."""
        try:
            if hasattr(loader, "load_fixed_income"):
                fi = loader.load_fixed_income()
                if fi is not None:
                    rate = float(fi)
                    logger.info("Using risk-free rate: %.2f%%", rate * 100)
                    return rate
        except Exception as exc:
            logger.debug("Could not load risk-free rate: %s", exc)

        logger.info("Using fallback risk-free rate: %.2f%%", self._rf_fallback * 100)
        return self._rf_fallback

    def _compute_returns(self) -> None:
        """Align equity and oil prices and compute daily returns."""
        if self._equity.empty or self._oil.empty:
            return

        # Inner join on date — only dates where both series have data
        merged = pd.DataFrame(
            {"equity": self._equity, "oil": self._oil}
        ).dropna()

        if len(merged) < self._min_obs + 1:
            logger.warning(
                "Insufficient aligned observations for oil_equity: %d", len(merged)
            )
            return

        self._equity_ret = merged["equity"].pct_change().dropna()
        self._oil_ret = merged["oil"].pct_change().dropna()
        self._aligned_dates = self._equity_ret.index

    def exposure(self, date: pd.Timestamp) -> float:
        """Return equity exposure based on oil→equity regression prediction.

        On rebalance days (first of each month), run expanding-window OLS
        using all data up to *date*. Between rebalance dates, return the
        last decision.
        """
        if self._equity_ret is None or self._oil_ret is None:
            self._last_exposure = 1.0
            return self._last_exposure

        # Use last decision for non-rebalance days
        if date not in self._rebalance_days:
            return self._last_exposure

        # Slice data up to and including date (no lookahead)
        eq_mask = self._equity_ret.index <= date
        oil_mask = self._oil_ret.index <= date
        eq = self._equity_ret.loc[eq_mask]
        oil = self._oil_ret.loc[oil_mask]

        # Align for the regression: oil[t-1] predicts equity[t]
        common_idx = eq.index.intersection(oil.index)
        eq_aligned = eq.loc[common_idx]
        oil_aligned = oil.loc[common_idx]

        if len(eq_aligned) < self._min_obs:
            self._last_exposure = 1.0
            return self._last_exposure

        # lagged oil returns as X, equity returns as Y
        # Regression: equity_ret[t] = α + β * oil_ret[t-1]
        x = oil_aligned.values[:-1]   # oil_ret[0..n-2]
        y = eq_aligned.values[1:]    # equity_ret[1..n-1]
        latest_x = float(oil_aligned.iloc[-1])

        predicted = rolling_ols_predict(
            x=x,
            y=y,
            latest_x=latest_x,
            min_obs=self._min_obs,
        )

        if predicted is None:
            self._last_exposure = 1.0
        elif predicted > self._rf_rate:
            self._last_exposure = 1.0  # Predicted return > rf → equity
        else:
            self._last_exposure = 0.0  # Predicted return <= rf → defensive

        self._cache[date] = self._last_exposure
        return self._last_exposure


__all__ = ["OilEquityOverlay"]
