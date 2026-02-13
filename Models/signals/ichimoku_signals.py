"""
Ichimoku Cloud Signal Construction

Uses borsapy Ichimoku Cloud indicator for trend-following signals:
- Composite score based on 5 Ichimoku signals:
  1. Price vs Cloud (above cloud = bullish)
  2. Conversion vs Base (TK cross)
  3. Cloud color (span_a > span_b = bullish)

Signal: Multi-factor Ichimoku score (-3 to +3)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from signals.borsapy_indicators import BorsapyIndicators


def build_ichimoku_signals(
    close_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
    conversion_period: int = 9,
    base_period: int = 26,
    span_b_period: int = 52,
) -> pd.DataFrame:
    """
    Build Ichimoku Cloud composite signals.

    Args:
        close_df: Close prices DataFrame (dates x tickers)
        high_df: High prices DataFrame (dates x tickers)
        low_df: Low prices DataFrame (dates x tickers)
        dates: Target dates for signals
        data_loader: DataLoader instance (unused, for consistency)
        conversion_period: Tenkan-sen period (default 9)
        base_period: Kijun-sen period (default 26)
        span_b_period: Senkou Span B period (default 52)

    Returns:
        DataFrame (dates x tickers) with Ichimoku composite scores
        Score range: -3 (all bearish) to +3 (all bullish)
    """
    print(f"\n🔧 Building Ichimoku({conversion_period},{base_period},{span_b_period}) signals...")

    tickers = close_df.columns
    signal_data = {}

    for ticker in tickers:
        if ticker not in high_df.columns or ticker not in low_df.columns:
            continue

        try:
            ichi = BorsapyIndicators.calculate_ichimoku(
                high_df[ticker], low_df[ticker], close_df[ticker],
                conversion_period, base_period, span_b_period
            )

            score = pd.Series(0.0, index=close_df.index)

            # Signal 1: Price above cloud (both spans) = +1
            cloud_top = ichi[["span_a", "span_b"]].max(axis=1)
            cloud_bot = ichi[["span_a", "span_b"]].min(axis=1)
            score += (close_df[ticker] > cloud_top).astype(float)
            score -= (close_df[ticker] < cloud_bot).astype(float)

            # Signal 2: TK Cross (conversion > base = +1)
            score += (ichi["conversion"] > ichi["base"]).astype(float)
            score -= (ichi["conversion"] < ichi["base"]).astype(float)

            # Signal 3: Cloud color (span_a > span_b = bullish cloud)
            score += (ichi["span_a"] > ichi["span_b"]).astype(float)
            score -= (ichi["span_a"] < ichi["span_b"]).astype(float)

            signal_data[ticker] = score
        except Exception:
            continue

    ichimoku_panel = pd.DataFrame(signal_data, index=close_df.index)

    result = ichimoku_panel.reindex(dates, method='ffill')

    valid_count = result.notna().sum().sum()
    total_count = result.shape[0] * result.shape[1]
    coverage = valid_count / total_count if total_count > 0 else 0

    print(f"  ✅ Ichimoku signals: {result.shape[0]} days × {result.shape[1]} tickers")
    print(f"     Coverage: {coverage*100:.1f}% ({valid_count:,} / {total_count:,})")

    return result
