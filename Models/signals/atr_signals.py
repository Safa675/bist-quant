"""
ATR (Average True Range) Signal Construction

Uses borsapy ATR indicator for volatility-adjusted trend signals:
- ATR normalized by price gives volatility percentage
- Inverted: lower volatility stocks rank higher (low-vol anomaly)
- Stocks with small ATR/Price ratio are less volatile, tend to outperform

Signal: Inverted normalized ATR — calmer stocks rank highest
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from signals.borsapy_indicators import BorsapyIndicators


def build_atr_signals(
    close_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
    period: int = 14,
) -> pd.DataFrame:
    """
    Build ATR volatility signals (inverted — low vol ranks highest).

    Args:
        close_df: Close prices DataFrame (dates x tickers)
        high_df: High prices DataFrame (dates x tickers)
        low_df: Low prices DataFrame (dates x tickers)
        dates: Target dates for signals
        data_loader: DataLoader instance (unused, for consistency)
        period: ATR period (default 14)

    Returns:
        DataFrame (dates x tickers) with inverted ATR% scores
        Higher score = lower volatility = calmer stock
    """
    print(f"\n🔧 Building ATR({period}) low-volatility signals...")

    atr_panel = BorsapyIndicators.build_atr_panel(high_df, low_df, close_df, period=period)

    # Normalize ATR by price to get volatility percentage
    atr_pct = atr_panel / close_df.replace(0, np.nan) * 100

    # Invert: lower ATR% (calmer) = higher score
    # Use cross-sectional rank (inverted)
    inverted_atr = -atr_pct

    result = inverted_atr.reindex(dates, method='ffill')
    result = result.replace([np.inf, -np.inf], np.nan)

    valid_count = result.notna().sum().sum()
    total_count = result.shape[0] * result.shape[1]
    coverage = valid_count / total_count if total_count > 0 else 0

    print(f"  ✅ ATR signals: {result.shape[0]} days × {result.shape[1]} tickers")
    print(f"     Coverage: {coverage*100:.1f}% ({valid_count:,} / {total_count:,})")

    return result
