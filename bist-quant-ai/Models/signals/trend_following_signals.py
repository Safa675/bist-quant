"""
Trend Following Signal Construction

Identifies stocks in strong uptrends:
- Stocks trading near all-time highs
- Uses ATR-based trailing stop concept for signal strength

Based on Quantpedia strategy: Trend Following Effect in Stocks
Signal: Closeness to all-time high + momentum strength
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# TREND FOLLOWING PARAMETERS
# ============================================================================

MAX_LOOKBACK = 252  # ~1 year for "all-time high" (practical limit)
ATR_PERIOD = 10     # Average True Range period for volatility
MOMENTUM_LOOKBACK = 63  # ~3 months momentum confirmation


# ============================================================================
# CORE CALCULATIONS
# ============================================================================

def calculate_rolling_max(
    close_df: pd.DataFrame,
    lookback: int = MAX_LOOKBACK,
) -> pd.DataFrame:
    """
    Calculate rolling maximum close price.

    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        lookback: Lookback period for max calculation

    Returns:
        pd.DataFrame: Rolling maximum for each stock
    """
    rolling_max = close_df.rolling(lookback, min_periods=lookback // 4).max()
    return rolling_max


def calculate_closeness_to_high(
    close_df: pd.DataFrame,
    lookback: int = MAX_LOOKBACK,
) -> pd.DataFrame:
    """
    Calculate how close current price is to rolling high.

    1.0 = At the high
    0.9 = 10% below the high
    etc.

    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        lookback: Lookback period for max calculation

    Returns:
        pd.DataFrame: Closeness ratio (0 to 1)
    """
    rolling_max = calculate_rolling_max(close_df, lookback)
    closeness = close_df / rolling_max
    return closeness


def calculate_momentum_strength(
    close_df: pd.DataFrame,
    lookback: int = MOMENTUM_LOOKBACK,
) -> pd.DataFrame:
    """
    Calculate momentum strength as confirmation.

    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        lookback: Lookback period for momentum

    Returns:
        pd.DataFrame: Momentum returns
    """
    momentum = close_df / close_df.shift(lookback) - 1.0
    return momentum


def calculate_trend_scores(
    close_df: pd.DataFrame,
    high_df: pd.DataFrame = None,
    low_df: pd.DataFrame = None,
    max_lookback: int = MAX_LOOKBACK,
    momentum_lookback: int = MOMENTUM_LOOKBACK,
) -> pd.DataFrame:
    """
    Calculate trend following scores.

    Score = Closeness to High * (1 + Momentum)

    Stocks at new highs with positive momentum get highest scores.

    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        high_df: DataFrame of high prices (optional, for ATR)
        low_df: DataFrame of low prices (optional, for ATR)
        max_lookback: Lookback for all-time high
        momentum_lookback: Lookback for momentum confirmation

    Returns:
        pd.DataFrame: Trend following scores (dates x tickers)
    """
    # Calculate closeness to rolling high
    closeness = calculate_closeness_to_high(close_df, max_lookback)

    # Calculate momentum confirmation
    momentum = calculate_momentum_strength(close_df, momentum_lookback)

    # Combine: stocks at highs with positive momentum score highest
    # Closeness is 0-1, momentum can be negative
    # We want: near high (closeness ~1) AND positive momentum

    # Normalize momentum to 0-1 scale (cross-sectional rank)
    momentum_rank = momentum.rank(axis=1, pct=True)

    # Combined score: weighted average of closeness and momentum rank
    trend_score = 0.6 * closeness + 0.4 * momentum_rank

    # Bonus for stocks at new highs (closeness > 0.95)
    at_new_high = closeness >= 0.95
    trend_score = trend_score.where(~at_new_high, trend_score * 1.2)

    # Handle infinities and NaNs
    trend_score = trend_score.replace([np.inf, -np.inf], np.nan)

    return trend_score


# ============================================================================
# SIGNAL BUILDER (MAIN INTERFACE)
# ============================================================================

def build_trend_following_signals(
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
) -> pd.DataFrame:
    """
    Build trend following signal panel.

    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        dates: DatetimeIndex to align signals to
        data_loader: Optional DataLoader instance

    Returns:
        DataFrame (dates x tickers) with trend following scores
    """
    print("\n🔧 Building trend following signals...")
    print(f"  Max Lookback: {MAX_LOOKBACK} days")
    print(f"  Momentum Lookback: {MOMENTUM_LOOKBACK} days")

    # Calculate trend following scores
    trend_scores = calculate_trend_scores(close_df)

    # Align to requested dates
    result = trend_scores.reindex(dates)

    # Summary stats
    valid_scores = result.dropna(how='all')
    if not valid_scores.empty:
        latest = valid_scores.iloc[-1].dropna()
        if len(latest) > 0:
            print(f"  Latest scores - Mean: {latest.mean():.4f}, Std: {latest.std():.4f}")
            # Count stocks at new highs
            n_at_high = (latest > 0.95 * 1.2).sum()  # Bonus threshold
            print(f"  Stocks at/near highs: {n_at_high}")

    print(f"  ✅ Trend following signals: {result.shape[0]} days × {result.shape[1]} tickers")

    return result
