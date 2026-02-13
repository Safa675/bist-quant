"""
Momentum + Reversal + Volatility Combined Signal

Combines three anomalies into a composite signal:
1. Medium-term momentum (3-12 month returns)
2. Short-term reversal (weekly losers outperform)
3. Low volatility (lower vol = higher returns)

Based on Quantpedia strategy: Momentum and Reversal Combined with Volatility Effect in Stocks
Signal: Composite z-score of all three factors
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# PARAMETERS
# ============================================================================

# Momentum parameters
MOMENTUM_LOOKBACK = 252  # 12 months
MOMENTUM_SKIP = 21       # Skip last month

# Reversal parameters
REVERSAL_LOOKBACK = 5    # 1 week

# Volatility parameters
VOLATILITY_LOOKBACK = 252  # 12 months for volatility calculation


# ============================================================================
# CORE CALCULATIONS
# ============================================================================

def calculate_momentum_component(
    close_df: pd.DataFrame,
    lookback: int = MOMENTUM_LOOKBACK,
    skip: int = MOMENTUM_SKIP,
) -> pd.DataFrame:
    """Calculate 12-1 momentum component."""
    momentum = close_df.shift(skip) / close_df.shift(lookback) - 1.0
    return momentum


def calculate_reversal_component(
    close_df: pd.DataFrame,
    lookback: int = REVERSAL_LOOKBACK,
) -> pd.DataFrame:
    """Calculate weekly reversal component (negative of weekly return)."""
    weekly_return = close_df / close_df.shift(lookback) - 1.0
    reversal = -weekly_return  # Negative: losers get high scores
    return reversal


def calculate_volatility_component(
    close_df: pd.DataFrame,
    lookback: int = VOLATILITY_LOOKBACK,
) -> pd.DataFrame:
    """
    Calculate low volatility component.

    Lower volatility = Higher score (we want low-vol stocks)
    """
    daily_returns = close_df.pct_change(fill_method=None)
    rolling_vol = daily_returns.rolling(lookback, min_periods=lookback // 2).std()

    # Convert to "low vol score": negative of volatility
    # Higher volatility = lower score
    low_vol_score = -rolling_vol

    return low_vol_score


def cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize each row (date) to z-scores cross-sectionally.

    This ensures all three components are on the same scale.
    """
    row_mean = df.mean(axis=1)
    row_std = df.std(axis=1).replace(0, np.nan)
    z_scored = df.sub(row_mean, axis=0).div(row_std, axis=0)
    return z_scored


def calculate_combined_scores(
    close_df: pd.DataFrame,
    momentum_weight: float = 0.4,
    reversal_weight: float = 0.3,
    volatility_weight: float = 0.3,
) -> pd.DataFrame:
    """
    Calculate combined momentum + reversal + volatility score.

    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        momentum_weight: Weight for momentum component
        reversal_weight: Weight for reversal component
        volatility_weight: Weight for volatility component

    Returns:
        pd.DataFrame: Combined scores (dates x tickers)
    """
    # Calculate each component
    momentum = calculate_momentum_component(close_df)
    reversal = calculate_reversal_component(close_df)
    volatility = calculate_volatility_component(close_df)

    # Z-score normalize each component
    momentum_z = cross_sectional_zscore(momentum)
    reversal_z = cross_sectional_zscore(reversal)
    volatility_z = cross_sectional_zscore(volatility)

    # Weighted combination
    combined = (
        momentum_weight * momentum_z +
        reversal_weight * reversal_z +
        volatility_weight * volatility_z
    )

    # Handle infinities and NaNs
    combined = combined.replace([np.inf, -np.inf], np.nan)

    return combined


# ============================================================================
# SIGNAL BUILDER (MAIN INTERFACE)
# ============================================================================

def build_momentum_reversal_volatility_signals(
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
) -> pd.DataFrame:
    """
    Build combined momentum + reversal + volatility signal panel.

    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        dates: DatetimeIndex to align signals to
        data_loader: Optional DataLoader instance

    Returns:
        DataFrame (dates x tickers) with combined scores
    """
    print("\n🔧 Building momentum + reversal + volatility signals...")
    print(f"  Momentum: {MOMENTUM_LOOKBACK}-{MOMENTUM_SKIP} days (40% weight)")
    print(f"  Reversal: {REVERSAL_LOOKBACK} days (30% weight)")
    print(f"  Volatility: {VOLATILITY_LOOKBACK} days (30% weight)")

    # Calculate combined scores
    combined_scores = calculate_combined_scores(close_df)

    # Align to requested dates
    result = combined_scores.reindex(dates)

    # Summary stats
    valid_scores = result.dropna(how='all')
    if not valid_scores.empty:
        latest = valid_scores.iloc[-1].dropna()
        if len(latest) > 0:
            print(f"  Latest scores - Mean: {latest.mean():.4f}, Std: {latest.std():.4f}")
            print(f"  Latest scores - Min: {latest.min():.4f}, Max: {latest.max():.4f}")

    print(f"  ✅ Combined signals: {result.shape[0]} days × {result.shape[1]} tickers")

    return result
