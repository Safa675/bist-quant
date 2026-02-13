"""
Donchian Channel Breakout Signal Construction

Calculates Donchian Channel breakout scores based on:
- N-day high (upper band)
- N-day low (lower band)
- Position of close price relative to channel

Score interpretation:
- +100: At upper band (strong breakout/buy signal)
- 0: At middle of channel (neutral)
- -100: At lower band (breakdown/sell signal)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict


def build_donchian_signals(
    close_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader,
    lookback: int = 20,
) -> pd.DataFrame:
    """
    Build Donchian Channel breakout signals.
    
    Args:
        close_df: Close prices DataFrame (dates x tickers)
        high_df: High prices DataFrame (dates x tickers)
        low_df: Low prices DataFrame (dates x tickers)
        dates: Target dates for signals
        data_loader: DataLoader instance (unused, for consistency)
        lookback: Channel lookback period (default 20)
    
    Returns:
        DataFrame (dates x tickers) with Donchian scores (-100 to +100)
        
    Signal Interpretation:
        - +100: Price at upper band (strong breakout, buy signal)
        - +50: Price in upper half of channel (bullish)
        - 0: Price at middle of channel (neutral)
        - -50: Price in lower half of channel (bearish)
        - -100: Price at lower band (breakdown, sell signal)
    """
    print(f"\n🔧 Building {lookback}-day Donchian Channel signals...")
    
    # Calculate Donchian bands (shift by 1 to avoid lookahead bias)
    upper_band = high_df.rolling(lookback, min_periods=lookback).max().shift(1)
    lower_band = low_df.rolling(lookback, min_periods=lookback).min().shift(1)
    middle_band = (upper_band + lower_band) / 2
    
    # Calculate channel width
    channel_width = upper_band - lower_band
    
    # Position in channel (0 to 1)
    # 0 = at lower band, 0.5 = at middle, 1 = at upper band
    position_in_channel = (close_df - lower_band) / channel_width.replace(0, np.nan)
    
    # Scale to -100 to +100
    # -100 = at lower band, 0 = at middle, +100 = at upper band
    donchian_score = (position_in_channel - 0.5) * 200
    
    # Clip to valid range (handle edge cases)
    donchian_score = donchian_score.clip(-100, 100)
    
    # Reindex to target dates
    result = donchian_score.reindex(dates, method='ffill')
    
    # Count valid signals
    valid_count = result.notna().sum().sum()
    total_count = result.shape[0] * result.shape[1]
    coverage = valid_count / total_count if total_count > 0 else 0
    
    print(f"  ✅ Donchian signals: {result.shape[0]} days × {result.shape[1]} tickers")
    print(f"     Coverage: {coverage*100:.1f}% ({valid_count:,} / {total_count:,})")
    
    return result
