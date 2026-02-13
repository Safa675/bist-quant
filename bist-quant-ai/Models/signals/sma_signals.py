"""
SMA Crossover Signal Construction

Calculates SMA crossover scores based on:
- Short-term SMA (default 10 days)
- Long-term SMA (default 30 days)
- Score = (SMA_short / SMA_long - 1) × 100

Higher score = stronger bullish signal (short MA above long MA)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict


def build_sma_signals(
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader,
    short_period: int = 10,
    long_period: int = 30,
) -> pd.DataFrame:
    """
    Build SMA crossover signals.
    
    Args:
        close_df: Close prices DataFrame (dates x tickers)
        dates: Target dates for signals
        data_loader: DataLoader instance (unused, for consistency)
        short_period: Short SMA period (default 10)
        long_period: Long SMA period (default 30)
    
    Returns:
        DataFrame (dates x tickers) with SMA scores
        
    Signal Interpretation:
        - Positive score: Short MA above long MA (bullish)
        - Negative score: Short MA below long MA (bearish)
        - Higher absolute value: Stronger signal
    """
    print(f"\n🔧 Building {short_period}/{long_period} SMA signals...")
    
    # Calculate SMAs
    sma_short = close_df.rolling(short_period, min_periods=short_period).mean()
    sma_long = close_df.rolling(long_period, min_periods=long_period).mean()
    
    # SMA ratio: how much short SMA is above/below long SMA (%)
    # Positive = bullish (short > long), Negative = bearish (short < long)
    sma_score = (sma_short / sma_long - 1.0) * 100
    
    # Reindex to target dates
    result = sma_score.reindex(dates, method='ffill')
    
    # Count valid signals
    valid_count = result.notna().sum().sum()
    total_count = result.shape[0] * result.shape[1]
    coverage = valid_count / total_count if total_count > 0 else 0
    
    print(f"  ✅ SMA signals: {result.shape[0]} days × {result.shape[1]} tickers")
    print(f"     Coverage: {coverage*100:.1f}% ({valid_count:,} / {total_count:,})")
    
    return result
