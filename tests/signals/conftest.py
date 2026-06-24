"""Shared fixtures and helpers for signal golden tests."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

SIGNAL_TICKERS = ("AAA", "BBB", "CCC", "DDD", "EEE")
SIGNAL_N_DAYS = 300
SIGNAL_LOOKBACK = 252
SIGNAL_SKIP = 21


@pytest.fixture
def signal_dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=SIGNAL_N_DAYS)


@pytest.fixture
def signal_close_df(signal_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Deterministic 5-ticker × 300-day close panel."""
    rng = np.random.default_rng(42)
    tickers = list(SIGNAL_TICKERS)
    base = rng.uniform(20.0, 120.0, size=len(tickers))
    daily_returns = rng.normal(0.0004, 0.015, size=(len(signal_dates), len(tickers)))
    prices = base * np.cumprod(1.0 + daily_returns, axis=0)
    return pd.DataFrame(prices, index=signal_dates, columns=tickers)


def panel_checksum(df: pd.DataFrame, *, decimals: int = 6) -> str:
    """Stable MD5 of rounded float values (NaNs preserved)."""
    rounded = np.round(df.to_numpy(dtype=float), decimals)
    return hashlib.md5(rounded.tobytes()).hexdigest()


def finite_slice_checksum(df: pd.DataFrame, row: int = -1) -> str:
    """Checksum of one row's finite values (useful when most panel is NaN)."""
    values = df.iloc[row].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "empty"
    rounded = np.round(finite, 6)
    return hashlib.md5(rounded.tobytes()).hexdigest()
