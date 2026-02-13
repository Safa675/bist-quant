"""
Momentum Signal Construction

Calculates risk-adjusted momentum scores based on:
- 12-1 Momentum: Past 12 months return excluding the last month
- Downside Volatility: Rolling standard deviation of negative returns
- Risk-Adjusted Score: (12-1 Return) / Downside Volatility

This follows the Fama-French momentum methodology with downside risk adjustment.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# MOMENTUM CALCULATION PARAMETERS
# ============================================================================

MOMENTUM_LOOKBACK = 252  # ~12 months trading days
MOMENTUM_SKIP = 21  # ~1 month skip (most recent month excluded)
DOWNSIDE_VOL_LOOKBACK = 252  # 12-month lookback for downside volatility


# ============================================================================
# CORE MOMENTUM CALCULATIONS
# ============================================================================

def calculate_12_minus_1_momentum(
    close_df: pd.DataFrame,
    lookback: int = MOMENTUM_LOOKBACK,
    skip: int = MOMENTUM_SKIP,
) -> pd.DataFrame:
    """
    Calculate 12-1 Momentum: Past 12 months return excluding the last month.
    
    This is the classic Fama-French momentum calculation that avoids
    short-term reversal effects.
    
    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        lookback: Total lookback period in trading days (~252 = 12 months)
        skip: Days to skip from most recent (~21 = 1 month)
    
    Returns:
        pd.DataFrame: 12-1 returns indexed by date, columns are tickers
    
    Example:
        If lookback=252, skip=21:
        We want: price[t-21] / price[t-252] - 1
        This gives the 11-month return ending 1 month ago
    """
    # Calculate the return from lookback days ago to skip days ago
    momentum = close_df.shift(skip) / close_df.shift(lookback) - 1.0
    
    return momentum


def calculate_downside_volatility(
    close_df: pd.DataFrame,
    lookback: int = DOWNSIDE_VOL_LOOKBACK,
    skip: int = MOMENTUM_SKIP,
) -> pd.DataFrame:
    """
    Calculate rolling downside volatility (standard deviation of negative returns only).
    
    Downside volatility is a better risk measure for momentum strategies
    because it only penalizes downside movements, not upside.
    
    IMPORTANT: We skip the most recent month to match the 12-1 momentum calculation.
    This ensures both the return and volatility use the same time window.
    
    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        lookback: Lookback period in trading days
        skip: Days to skip from most recent (to match momentum skip)
    
    Returns:
        pd.DataFrame: Rolling downside volatility indexed by date, columns are tickers
    """
    # Calculate daily returns
    daily_returns = close_df.pct_change(fill_method=None)
    
    # Shift returns to skip the most recent month (match 12-1 momentum)
    shifted_returns = daily_returns.shift(skip)
    
    # Calculate rolling downside volatility on shifted returns
    def calc_downside_vol(window):
        """Calculate downside volatility for a rolling window."""
        negative_rets = window[window < 0]
        if len(negative_rets) > 2:
            return negative_rets.std()
        return np.nan
    
    # Use adjusted lookback since we're skipping the recent period
    effective_lookback = lookback - skip
    downside_vol = shifted_returns.rolling(
        effective_lookback, min_periods=int(effective_lookback * 0.5)
    ).apply(calc_downside_vol, raw=False)
    
    return downside_vol


def calculate_momentum_scores(
    close_df: pd.DataFrame,
    lookback: int = MOMENTUM_LOOKBACK,
    skip: int = MOMENTUM_SKIP,
    vol_lookback: int = DOWNSIDE_VOL_LOOKBACK,
) -> pd.DataFrame:
    """
    Calculate risk-adjusted momentum scores:
    
    Momentum Score = (12-1 Return) / Downside Volatility
    
    Higher score = Better risk-adjusted momentum
    
    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        lookback: Lookback for 12-1 return calculation
        skip: Days to skip from most recent month
        vol_lookback: Lookback for downside volatility
    
    Returns:
        pd.DataFrame: Risk-adjusted momentum scores (dates x tickers)
    """
    # Calculate 12-1 momentum
    momentum_12_1 = calculate_12_minus_1_momentum(close_df, lookback, skip)
    
    # Calculate downside volatility
    downside_vol = calculate_downside_volatility(close_df, lookback=vol_lookback, skip=skip)
    
    # Apply minimum volatility threshold to prevent extreme scores
    # from division by near-zero volatility
    MIN_VOL = 0.001  # 0.1% minimum daily volatility
    downside_vol_safe = downside_vol.clip(lower=MIN_VOL)
    
    # Risk-adjusted momentum = (12-1 return) / downside volatility
    # Higher is better: good momentum with low downside risk
    momentum_score = momentum_12_1 / downside_vol_safe
    
    # Handle infinities and NaNs
    momentum_score = momentum_score.replace([np.inf, -np.inf], np.nan)
    
    return momentum_score



# ============================================================================
# SIGNAL BUILDER (MAIN INTERFACE)
# ============================================================================

def build_momentum_signals(
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
    lookback: int = MOMENTUM_LOOKBACK,
    skip: int = MOMENTUM_SKIP,
    vol_lookback: int = DOWNSIDE_VOL_LOOKBACK,
) -> pd.DataFrame:
    """
    Build momentum signal panel with risk-adjusted scores.
    
    This is the main interface function that follows the same pattern
    as other signal builders (profitability, value, size, investment).
    
    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        dates: DatetimeIndex to align signals to
        data_loader: Optional DataLoader instance (for consistency with other signals)
    
    Returns:
        DataFrame (dates x tickers) with risk-adjusted momentum scores
    """
    print("\n🔧 Building momentum signals...")
    print(f"  Momentum: {lookback} days lookback, {skip} days skip")
    print(f"  Downside Vol: {vol_lookback} days lookback")
    
    # Calculate momentum scores
    momentum_scores = calculate_momentum_scores(
        close_df,
        lookback=lookback,
        skip=skip,
        vol_lookback=vol_lookback,
    )
    
    # Align to requested dates
    result = momentum_scores.reindex(dates)
    
    # Summary stats
    valid_scores = result.dropna(how='all')
    if not valid_scores.empty:
        latest = valid_scores.iloc[-1].dropna()
        if len(latest) > 0:
            print(f"  Latest scores - Mean: {latest.mean():.2f}, Std: {latest.std():.2f}")
            print(f"  Latest scores - Min: {latest.min():.2f}, Max: {latest.max():.2f}")
    
    print(f"  ✅ Momentum signals: {result.shape[0]} days × {result.shape[1]} tickers")
    
    return result
