"""
OBV (On-Balance Volume) Signal Construction

Uses borsapy OBV indicator for volume-momentum signals:
- OBV trend (rate of change) indicates smart money flow
- Rising OBV = accumulation (bullish)
- Falling OBV = distribution (bearish)

Signal: OBV momentum (rate of change over lookback period)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from signals.borsapy_indicators import BorsapyIndicators


def build_obv_signals(
    close_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
    momentum_lookback: int = 20,
) -> pd.DataFrame:
    """
    Build OBV momentum signals.

    Args:
        close_df: Close prices DataFrame (dates x tickers)
        volume_df: Volume DataFrame (dates x tickers)
        dates: Target dates for signals
        data_loader: DataLoader instance (unused, for consistency)
        momentum_lookback: Lookback for OBV rate of change (default 20)

    Returns:
        DataFrame (dates x tickers) with OBV momentum scores
        Higher = stronger accumulation (bullish)
    """
    print(f"\n🔧 Building OBV momentum({momentum_lookback}) signals...")

    obv_panel = BorsapyIndicators.build_obv_panel(close_df, volume_df)

    # OBV rate of change: how much OBV changed over lookback
    # Normalize by average volume to make cross-sectionally comparable
    avg_vol = volume_df.rolling(momentum_lookback).mean().replace(0, np.nan)
    obv_momentum = (obv_panel - obv_panel.shift(momentum_lookback)) / avg_vol

    result = obv_momentum.reindex(dates, method='ffill')
    result = result.replace([np.inf, -np.inf], np.nan)

    valid_count = result.notna().sum().sum()
    total_count = result.shape[0] * result.shape[1]
    coverage = valid_count / total_count if total_count > 0 else 0

    print(f"  ✅ OBV signals: {result.shape[0]} days × {result.shape[1]} tickers")
    print(f"     Coverage: {coverage*100:.1f}% ({valid_count:,} / {total_count:,})")

    return result
