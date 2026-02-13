"""
Trend + Value Composite Signal

Combines technical trend (SMA crossover) with fundamental value.
Only buys value stocks that are also in uptrends.

Logic:
1. Calculate SMA crossover score (trend strength)
2. Calculate Value composite score (cheapness)
3. Filter: Only consider stocks where SMA > 0 (in uptrend)
4. Rank by Value score among uptrending stocks

This should:
- Avoid value traps (cheap stocks in downtrends)
- Capture mean reversion only when trend confirms
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict


def build_trend_value_signals(
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader,
    value_signals: pd.DataFrame = None,
    sma_short: int = 10,
    sma_long: int = 30,
) -> pd.DataFrame:
    """
    Build Trend + Value composite signals.

    Args:
        close_df: Close prices DataFrame (dates x tickers)
        dates: Target dates for signals
        data_loader: DataLoader instance
        value_signals: Pre-computed value signals (optional, will compute if None)
        sma_short: Short SMA period (default 10)
        sma_long: Long SMA period (default 30)

    Returns:
        DataFrame (dates x tickers) with composite scores
    """
    print(f"\n🔧 Building Trend + Value composite signals...")
    print(f"   SMA: {sma_short}/{sma_long}, Value filter: uptrend only")

    # Step 1: Calculate SMA signals (trend)
    sma_short_ma = close_df.rolling(sma_short, min_periods=sma_short).mean()
    sma_long_ma = close_df.rolling(sma_long, min_periods=sma_long).mean()
    sma_score = (sma_short_ma / sma_long_ma - 1) * 100  # Percentage above/below

    # Step 2: Get or compute value signals
    if value_signals is None:
        from signals.value_signals import build_value_signals
        fundamentals = data_loader.load_fundamentals()
        value_signals = build_value_signals(fundamentals, close_df, dates, data_loader)

    # Step 3: Combine - only keep value scores where trend is bullish
    # Uptrend filter: SMA score > 0 (short MA above long MA)
    uptrend_mask = sma_score > 0

    # Composite score: Value score, but only for uptrending stocks
    composite = value_signals.copy()
    composite = composite.where(uptrend_mask.reindex(composite.index, method='ffill'), np.nan)

    # Count coverage
    total_signals = value_signals.notna().sum().sum()
    filtered_signals = composite.notna().sum().sum()
    filter_pct = (1 - filtered_signals / total_signals) * 100 if total_signals > 0 else 0

    print(f"  ✅ Trend + Value signals: {composite.shape[0]} days × {composite.shape[1]} tickers")
    print(f"     Uptrend filter removed {filter_pct:.1f}% of value signals")

    return composite


def build_trend_value_signals_v2(
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader,
    value_signals: pd.DataFrame = None,
    sma_short: int = 10,
    sma_long: int = 30,
    trend_weight: float = 0.3,
    value_weight: float = 0.7,
) -> pd.DataFrame:
    """
    Alternative: Weighted combination of trend and value.

    Instead of filtering, combines scores with weights.
    This allows some value exposure even in mild downtrends.

    Args:
        trend_weight: Weight for trend score (default 0.3)
        value_weight: Weight for value score (default 0.7)
    """
    print(f"\n🔧 Building Trend + Value weighted signals...")
    print(f"   Weights: Trend={trend_weight}, Value={value_weight}")

    # Calculate SMA score
    sma_short_ma = close_df.rolling(sma_short, min_periods=sma_short).mean()
    sma_long_ma = close_df.rolling(sma_long, min_periods=sma_long).mean()
    sma_score = (sma_short_ma / sma_long_ma - 1) * 100

    # Get value signals
    if value_signals is None:
        from signals.value_signals import build_value_signals
        fundamentals = data_loader.load_fundamentals()
        value_signals = build_value_signals(fundamentals, close_df, dates, data_loader)

    # Normalize both to z-scores for fair combination
    def zscore_by_date(df):
        return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)

    sma_z = zscore_by_date(sma_score.reindex(value_signals.index, method='ffill'))
    value_z = zscore_by_date(value_signals)

    # Weighted combination
    composite = trend_weight * sma_z + value_weight * value_z

    print(f"  ✅ Trend + Value weighted signals: {composite.shape[0]} days × {composite.shape[1]} tickers")

    return composite
