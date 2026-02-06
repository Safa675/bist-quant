"""
Breakout + Value Composite Signal

Combines Donchian breakout (trend) with fundamental value.
Only buys value stocks that are breaking out of their range.

Logic:
1. Calculate Donchian position (where price is in 20-day range)
2. Calculate Value composite score (cheapness)
3. Filter: Only consider stocks near top of Donchian channel (breakout)
4. Rank by Value score among breakout candidates

This should:
- Catch value stocks starting to move
- Avoid stagnant cheap stocks
- Combine momentum with fundamentals
"""

import pandas as pd
import numpy as np
from pathlib import Path


def build_breakout_value_signals(
    close_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader,
    value_signals: pd.DataFrame = None,
    donchian_lookback: int = 20,
    breakout_threshold: float = 0.7,  # Must be in top 30% of range
) -> pd.DataFrame:
    """
    Build Breakout + Value composite signals.

    Args:
        close_df: Close prices DataFrame (dates x tickers)
        high_df: High prices DataFrame
        low_df: Low prices DataFrame
        dates: Target dates for signals
        data_loader: DataLoader instance
        value_signals: Pre-computed value signals (optional)
        donchian_lookback: Lookback for Donchian channel (default 20)
        breakout_threshold: Minimum position in channel (0-1, default 0.7)

    Returns:
        DataFrame (dates x tickers) with composite scores
    """
    print(f"\n🔧 Building Breakout + Value composite signals...")
    print(f"   Donchian: {donchian_lookback} days, Breakout threshold: {breakout_threshold*100:.0f}%")

    # Step 1: Calculate Donchian channel position
    high_max = high_df.rolling(donchian_lookback, min_periods=donchian_lookback).max()
    low_min = low_df.rolling(donchian_lookback, min_periods=donchian_lookback).min()

    # Position in channel: 0 = at low, 1 = at high
    channel_range = high_max - low_min
    channel_position = (close_df - low_min) / channel_range.replace(0, np.nan)

    # Step 2: Get or compute value signals
    if value_signals is None:
        from signals.value_signals import build_value_signals
        fundamentals = data_loader.load_fundamentals()
        value_signals = build_value_signals(fundamentals, close_df, dates, data_loader)

    # Step 3: Combine - only keep value scores where price is breaking out
    breakout_mask = channel_position >= breakout_threshold

    composite = value_signals.copy()
    composite = composite.where(breakout_mask.reindex(composite.index, method='ffill'), np.nan)

    # Count coverage
    total_signals = value_signals.notna().sum().sum()
    filtered_signals = composite.notna().sum().sum()
    filter_pct = (1 - filtered_signals / total_signals) * 100 if total_signals > 0 else 0

    print(f"  ✅ Breakout + Value signals: {composite.shape[0]} days × {composite.shape[1]} tickers")
    print(f"     Breakout filter removed {filter_pct:.1f}% of value signals")

    return composite
