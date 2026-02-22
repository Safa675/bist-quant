"""
Supertrend Signal Construction

Uses borsapy Supertrend indicator for trend-following signals:
- Supertrend direction: +1 (uptrend) or -1 (downtrend)
- Binary trend signal based on ATR-based dynamic support/resistance

Signal: +1 (uptrend, buy) or -1 (downtrend, avoid)
"""

import logging

import pandas as pd

# Add paths
from bist_quant.signals.borsapy_indicators import BorsapyIndicators

logger = logging.getLogger(__name__)
def build_supertrend_signals(
    close_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
    period: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """
    Build Supertrend direction signals.

    Args:
        close_df: Close prices DataFrame (dates x tickers)
        high_df: High prices DataFrame (dates x tickers)
        low_df: Low prices DataFrame (dates x tickers)
        dates: Target dates for signals
        data_loader: DataLoader instance (unused, for consistency)
        period: ATR period for Supertrend (default 10)
        multiplier: ATR multiplier (default 3.0)

    Returns:
        DataFrame (dates x tickers) with Supertrend direction
        +1 = uptrend (buy signal), -1 = downtrend (avoid)

    Signal Interpretation:
        - +1: Price above Supertrend line (bullish, hold/buy)
        - -1: Price below Supertrend line (bearish, avoid/sell)
    """
    logger.info(f"\n🔧 Building Supertrend({period}, {multiplier}) direction signals...")

    # Build Supertrend direction panel using borsapy
    st_panel = BorsapyIndicators.build_supertrend_panel(
        high_df, low_df, close_df,
        period=period, multiplier=multiplier, output="direction"
    )

    # Reindex to target dates
    result = st_panel.reindex(dates, method='ffill')

    # Summary stats
    valid_count = result.notna().sum().sum()
    total_count = result.shape[0] * result.shape[1]
    coverage = valid_count / total_count if total_count > 0 else 0

    # Count uptrend vs downtrend at latest date
    if not result.empty:
        latest = result.iloc[-1].dropna()
        n_up = (latest > 0).sum()
        n_down = (latest < 0).sum()
        logger.info(f"     Latest: {n_up} uptrend, {n_down} downtrend")

    logger.info(f"  ✅ Supertrend signals: {result.shape[0]} days × {result.shape[1]} tickers")
    logger.info(f"     Coverage: {coverage*100:.1f}% ({valid_count:,} / {total_count:,})")

    return result
