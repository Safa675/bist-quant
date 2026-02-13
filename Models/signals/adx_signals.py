"""
ADX (Average Directional Index) Signal Construction

Uses borsapy ADX indicator for trend-strength signals:
- ADX measures trend strength (0-100)
- DI+ and DI- measure directional movement
- Signal combines direction with strength:
  (DI+ - DI-) × (ADX / 100)

Signal: Positive = strong uptrend, Negative = strong downtrend
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from signals.borsapy_indicators import BorsapyIndicators


def build_adx_signals(
    close_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
    period: int = 14,
) -> pd.DataFrame:
    """
    Build ADX directional trend signals.

    Args:
        close_df: Close prices DataFrame (dates x tickers)
        high_df: High prices DataFrame (dates x tickers)
        low_df: Low prices DataFrame (dates x tickers)
        dates: Target dates for signals
        data_loader: DataLoader instance (unused, for consistency)
        period: ADX period (default 14)

    Returns:
        DataFrame (dates x tickers) with ADX-weighted directional scores
        Higher score = stronger uptrend

    Signal Interpretation:
        - Large positive: Strong uptrend (DI+ >> DI-, high ADX)
        - Near zero: No trend or weak trend
        - Large negative: Strong downtrend (DI- >> DI+, high ADX)
    """
    print(f"\n🔧 Building ADX({period}) directional trend signals...")

    tickers = close_df.columns
    signal_data = {}

    for ticker in tickers:
        if ticker not in high_df.columns or ticker not in low_df.columns:
            continue

        try:
            adx_df = BorsapyIndicators.calculate_adx(
                high_df[ticker], low_df[ticker], close_df[ticker], period
            )

            # Directional signal weighted by trend strength
            # (DI+ - DI-) gives direction, ADX/100 gives confidence
            directional_signal = (adx_df["di_plus"] - adx_df["di_minus"]) * (adx_df["adx"] / 100)
            signal_data[ticker] = directional_signal
        except Exception:
            continue

    adx_signal_panel = pd.DataFrame(signal_data, index=close_df.index)

    # Reindex to target dates
    result = adx_signal_panel.reindex(dates, method='ffill')

    # Handle infinities
    result = result.replace([np.inf, -np.inf], np.nan)

    # Summary stats
    valid_count = result.notna().sum().sum()
    total_count = result.shape[0] * result.shape[1]
    coverage = valid_count / total_count if total_count > 0 else 0

    print(f"  ✅ ADX signals: {result.shape[0]} days × {result.shape[1]} tickers")
    print(f"     Coverage: {coverage*100:.1f}% ({valid_count:,} / {total_count:,})")

    return result
