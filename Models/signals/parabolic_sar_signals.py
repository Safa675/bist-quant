"""
Parabolic SAR Signal Construction

Uses borsapy Parabolic SAR indicator for trend-following signals:
- SAR below price = bullish (uptrend) → direction = +1
- SAR above price = bearish (downtrend) → direction = -1

Signal: Parabolic SAR direction (+1 uptrend / -1 downtrend)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from signals.borsapy_indicators import BorsapyIndicators


def build_parabolic_sar_signals(
    close_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
    af_start: float = 0.02,
    af_step: float = 0.02,
    af_max: float = 0.20,
) -> pd.DataFrame:
    """
    Build Parabolic SAR direction signals.

    Args:
        close_df: Close prices DataFrame (dates x tickers)
        high_df: High prices DataFrame (dates x tickers)
        low_df: Low prices DataFrame (dates x tickers)
        dates: Target dates for signals
        data_loader: DataLoader instance (unused, for consistency)
        af_start: Acceleration factor start (default 0.02)
        af_step: Acceleration factor step (default 0.02)
        af_max: Acceleration factor max (default 0.20)

    Returns:
        DataFrame (dates x tickers) with SAR direction
        +1 = uptrend (buy), -1 = downtrend (avoid)
    """
    print(f"\n🔧 Building Parabolic SAR({af_start},{af_step},{af_max}) signals...")

    tickers = close_df.columns
    signal_data = {}

    for ticker in tickers:
        if ticker not in high_df.columns or ticker not in low_df.columns:
            continue

        try:
            sar_df = BorsapyIndicators.calculate_parabolic_sar(
                high_df[ticker], low_df[ticker], close_df[ticker],
                af_start, af_step, af_max
            )
            signal_data[ticker] = sar_df["direction"]
        except Exception:
            continue

    sar_panel = pd.DataFrame(signal_data, index=close_df.index)

    result = sar_panel.reindex(dates, method='ffill')

    valid_count = result.notna().sum().sum()
    total_count = result.shape[0] * result.shape[1]
    coverage = valid_count / total_count if total_count > 0 else 0

    if not result.empty:
        latest = result.iloc[-1].dropna()
        n_up = (latest > 0).sum()
        n_down = (latest < 0).sum()
        print(f"     Latest: {n_up} bullish, {n_down} bearish")

    print(f"  ✅ Parabolic SAR signals: {result.shape[0]} days × {result.shape[1]} tickers")
    print(f"     Coverage: {coverage*100:.1f}% ({valid_count:,} / {total_count:,})")

    return result
