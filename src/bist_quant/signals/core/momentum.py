"""Price momentum primitives shared across BUILDERS, five_factor_rotation, and standalone."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from bist_quant.signals.core.constants import (
    DOWNSIDE_VOL_LOOKBACK,
    MOMENTUM_LOOKBACK,
    MOMENTUM_SKIP,
)

MomentumMode = Literal["prod", "rotation", "legacy_standalone"]

MIN_PROD_VOL = 0.001


def compute_price_momentum(
    close: pd.DataFrame,
    *,
    lookback: int,
    skip: int,
    mode: MomentumMode = "prod",
    use_log_returns: bool = False,
) -> pd.DataFrame:
    """Compute lookback momentum with explicit indexing mode.

    Modes
    -----
    ``prod`` (BUILDERS / ``momentum_signals``):
        ``close.shift(skip) / close.shift(lookback) - 1``
    ``rotation`` (``five_factor_rotation`` axis):
        ``close.shift(skip) / close.shift(lookback + skip) - 1``
    ``legacy_standalone`` (pre-refactor research notebooks):
        Same indexing as ``prod``; use with ``lookback=126`` for old 126d raw momentum.
    """
    if lookback <= 0 or skip < 0:
        raise ValueError(f"lookback must be > 0 and skip >= 0, got lookback={lookback}, skip={skip}")

    if use_log_returns:
        log_price = np.log(close.astype(float))
        if mode == "rotation":
            return log_price.shift(skip) - log_price.shift(lookback + skip)
        return log_price.shift(skip) - log_price.shift(lookback)

    if mode == "rotation":
        return close.shift(skip) / close.shift(lookback + skip) - 1.0
    # prod and legacy_standalone share BUILDERS-style denominator indexing
    return close.shift(skip) / close.shift(lookback) - 1.0


def compute_prod_downside_volatility(
    close_df: pd.DataFrame,
    *,
    lookback: int = DOWNSIDE_VOL_LOOKBACK,
    skip: int = MOMENTUM_SKIP,
) -> pd.DataFrame:
    """Downside vol for BUILDERS risk-adjusted momentum (prod path only)."""
    daily_returns = close_df.pct_change()
    shifted_returns = daily_returns.shift(skip)
    effective_lookback = lookback - skip
    min_periods = int(effective_lookback * 0.5)
    negative_only = shifted_returns.where(shifted_returns < 0.0)
    total_counts = shifted_returns.rolling(effective_lookback, min_periods=min_periods).count()
    negative_counts = negative_only.rolling(effective_lookback, min_periods=1).count()
    downside_vol = negative_only.rolling(effective_lookback, min_periods=1).std()
    return downside_vol.where((total_counts >= min_periods) & (negative_counts > 2))


def compute_risk_adjusted_momentum(
    close_df: pd.DataFrame,
    *,
    lookback: int = MOMENTUM_LOOKBACK,
    skip: int = MOMENTUM_SKIP,
    vol_lookback: int = DOWNSIDE_VOL_LOOKBACK,
    mode: MomentumMode = "prod",
) -> pd.DataFrame:
    """BUILDERS momentum: price momentum / downside volatility."""
    momentum = compute_price_momentum(close_df, lookback=lookback, skip=skip, mode=mode)
    downside_vol = compute_prod_downside_volatility(close_df, lookback=vol_lookback, skip=skip)
    score = momentum / downside_vol.clip(lower=MIN_PROD_VOL)
    return score.replace([np.inf, -np.inf], np.nan)


def compute_simple_vol_adjusted_momentum(
    close_df: pd.DataFrame,
    *,
    lookback: int,
    skip: int,
    mode: MomentumMode = "rotation",
) -> pd.DataFrame:
    """Standalone optional vol-adjust: total-return vol divisor."""
    momentum = compute_price_momentum(close_df, lookback=lookback, skip=skip, mode=mode)
    vol = close_df.pct_change().rolling(lookback, min_periods=lookback // 2).std()
    return momentum / vol.replace(0, np.nan)


__all__ = [
    "MomentumMode",
    "MOMENTUM_LOOKBACK",
    "MOMENTUM_SKIP",
    "DOWNSIDE_VOL_LOOKBACK",
    "compute_price_momentum",
    "compute_prod_downside_volatility",
    "compute_risk_adjusted_momentum",
    "compute_simple_vol_adjusted_momentum",
]
