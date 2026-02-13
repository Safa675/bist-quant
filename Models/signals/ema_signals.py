"""
EMA Crossover Signal Construction

Uses borsapy EMA indicator for trend-following signals:
- Short EMA vs Long EMA crossover
- Score = (EMA_short / EMA_long - 1) × 100
- Positive = bullish (short above long), Negative = bearish

Same concept as SMA crossover but faster-reacting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from signals.borsapy_indicators import BorsapyIndicators


def build_ema_signals(
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
    short_period: int = 12,
    long_period: int = 26,
) -> pd.DataFrame:
    """
    Build EMA crossover signals.

    Args:
        close_df: Close prices DataFrame (dates x tickers)
        dates: Target dates for signals
        data_loader: DataLoader instance (unused, for consistency)
        short_period: Short EMA period (default 12)
        long_period: Long EMA period (default 26)

    Returns:
        DataFrame (dates x tickers) with EMA crossover scores
        Positive = bullish (short EMA > long EMA)
    """
    print(f"\n🔧 Building EMA({short_period}/{long_period}) crossover signals...")

    ema_short = BorsapyIndicators.build_ema_panel(close_df, period=short_period)
    ema_long = BorsapyIndicators.build_ema_panel(close_df, period=long_period)

    # Score: how far short EMA is above/below long EMA (%)
    ema_score = (ema_short / ema_long - 1.0) * 100

    result = ema_score.reindex(dates, method='ffill')
    result = result.replace([np.inf, -np.inf], np.nan)

    valid_count = result.notna().sum().sum()
    total_count = result.shape[0] * result.shape[1]
    coverage = valid_count / total_count if total_count > 0 else 0

    print(f"  ✅ EMA signals: {result.shape[0]} days × {result.shape[1]} tickers")
    print(f"     Coverage: {coverage*100:.1f}% ({valid_count:,} / {total_count:,})")

    return result
