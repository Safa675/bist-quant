"""
MACD (Moving Average Convergence Divergence) Signal Construction

Uses borsapy MACD indicator for momentum signals:
- MACD histogram: (MACD line - Signal line)
- Positive histogram = bullish momentum
- Higher histogram = stronger upward momentum

Signal: MACD histogram value (cross-sectionally comparable)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from signals.borsapy_indicators import BorsapyIndicators


def build_macd_signals(
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    Build MACD momentum signals.

    Args:
        close_df: Close prices DataFrame (dates x tickers)
        dates: Target dates for signals
        data_loader: DataLoader instance (unused, for consistency)
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)

    Returns:
        DataFrame (dates x tickers) with MACD histogram scores
        Higher score = stronger bullish momentum

    Signal Interpretation:
        - Positive: MACD line above signal line (bullish)
        - Negative: MACD line below signal line (bearish)
        - Magnitude: Strength of the trend
    """
    print(f"\n🔧 Building MACD({fast},{slow},{signal}) momentum signals...")

    # Build MACD histogram panel using borsapy
    macd_panel = BorsapyIndicators.build_macd_panel(
        close_df, fast=fast, slow=slow, signal=signal, output="histogram"
    )

    # Normalize by price to make cross-sectionally comparable
    # MACD histogram is in price units, so divide by price
    normalized_macd = macd_panel / close_df.replace(0, np.nan) * 100

    # Reindex to target dates
    result = normalized_macd.reindex(dates, method='ffill')

    # Handle infinities
    result = result.replace([np.inf, -np.inf], np.nan)

    # Summary stats
    valid_count = result.notna().sum().sum()
    total_count = result.shape[0] * result.shape[1]
    coverage = valid_count / total_count if total_count > 0 else 0

    print(f"  ✅ MACD signals: {result.shape[0]} days × {result.shape[1]} tickers")
    print(f"     Coverage: {coverage*100:.1f}% ({valid_count:,} / {total_count:,})")

    return result
