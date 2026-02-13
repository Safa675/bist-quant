"""
Low Volatility Factor Signal Construction

Exploits the low volatility anomaly:
- Low volatility stocks tend to outperform high volatility stocks
- Opposite of traditional finance theory (higher risk = higher return)
- Also known as "betting against volatility"

Based on Quantpedia strategy: Low Volatility Factor Effect in Stocks
Signal: Negative of rolling volatility (lower vol = higher signal)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# LOW VOLATILITY PARAMETERS
# ============================================================================

VOLATILITY_LOOKBACK = 252  # 12 months of daily returns
WEEKLY_VOLATILITY = True   # Use weekly returns for vol calculation (more stable)


# ============================================================================
# CORE CALCULATIONS
# ============================================================================

def calculate_daily_volatility(
    close_df: pd.DataFrame,
    lookback: int = VOLATILITY_LOOKBACK,
) -> pd.DataFrame:
    """
    Calculate rolling volatility using daily returns.

    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        lookback: Rolling window for volatility calculation

    Returns:
        pd.DataFrame: Rolling volatility for each stock
    """
    daily_returns = close_df.pct_change(fill_method=None)
    rolling_vol = daily_returns.rolling(lookback, min_periods=lookback // 2).std()
    return rolling_vol


def calculate_weekly_volatility(
    close_df: pd.DataFrame,
    lookback: int = VOLATILITY_LOOKBACK,
) -> pd.DataFrame:
    """
    Calculate rolling volatility using weekly returns.

    Weekly returns are less noisy and provide more stable volatility estimates.

    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        lookback: Rolling window in days (will be converted to weeks)

    Returns:
        pd.DataFrame: Rolling weekly volatility for each stock
    """
    # Calculate weekly returns (every 5 days)
    weekly_returns = close_df / close_df.shift(5) - 1.0

    # Number of weeks in lookback period
    weeks_lookback = lookback // 5

    # Rolling standard deviation of weekly returns
    rolling_vol = weekly_returns.rolling(weeks_lookback, min_periods=weeks_lookback // 2).std()

    return rolling_vol


def calculate_downside_volatility(
    close_df: pd.DataFrame,
    lookback: int = VOLATILITY_LOOKBACK,
) -> pd.DataFrame:
    """
    Calculate rolling downside volatility (only negative returns).

    This focuses on the "bad" volatility that investors care about.

    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        lookback: Rolling window for volatility calculation

    Returns:
        pd.DataFrame: Rolling downside volatility for each stock
    """
    daily_returns = close_df.pct_change(fill_method=None)

    def calc_downside_vol(window):
        negative_rets = window[window < 0]
        if len(negative_rets) > 2:
            return negative_rets.std()
        return np.nan

    downside_vol = daily_returns.rolling(lookback, min_periods=lookback // 2).apply(
        calc_downside_vol, raw=False
    )

    return downside_vol


def calculate_low_volatility_scores(
    close_df: pd.DataFrame,
    use_weekly: bool = WEEKLY_VOLATILITY,
    use_downside: bool = False,
    lookback: int = VOLATILITY_LOOKBACK,
) -> pd.DataFrame:
    """
    Calculate low volatility scores.

    Lower volatility = Higher score

    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        use_weekly: Use weekly returns for volatility (more stable)
        use_downside: Use downside volatility only
        lookback: Lookback period in days

    Returns:
        pd.DataFrame: Low volatility scores (dates x tickers)
    """
    if use_downside:
        volatility = calculate_downside_volatility(close_df, lookback)
    elif use_weekly:
        volatility = calculate_weekly_volatility(close_df, lookback)
    else:
        volatility = calculate_daily_volatility(close_df, lookback)

    # Convert to low-vol score: negative of volatility
    # Lower vol = higher score
    low_vol_score = -volatility

    # Handle infinities and NaNs
    low_vol_score = low_vol_score.replace([np.inf, -np.inf], np.nan)

    return low_vol_score


# ============================================================================
# SIGNAL BUILDER (MAIN INTERFACE)
# ============================================================================

def build_low_volatility_signals(
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
) -> pd.DataFrame:
    """
    Build low volatility signal panel.

    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        dates: DatetimeIndex to align signals to
        data_loader: Optional DataLoader instance

    Returns:
        DataFrame (dates x tickers) with low volatility scores
    """
    print("\n🔧 Building low volatility signals...")
    print(f"  Volatility Lookback: {VOLATILITY_LOOKBACK} days")
    print(f"  Using Weekly Returns: {WEEKLY_VOLATILITY}")

    # Calculate low volatility scores
    low_vol_scores = calculate_low_volatility_scores(close_df)

    # Align to requested dates
    result = low_vol_scores.reindex(dates)

    # Summary stats
    valid_scores = result.dropna(how='all')
    if not valid_scores.empty:
        latest = valid_scores.iloc[-1].dropna()
        if len(latest) > 0:
            # Convert back to positive vol for display
            latest_vol = -latest
            print(f"  Latest volatility range: {latest_vol.min()*100:.2f}% to {latest_vol.max()*100:.2f}%")
            print(f"  Median volatility: {latest_vol.median()*100:.2f}%")

    print(f"  ✅ Low volatility signals: {result.shape[0]} days × {result.shape[1]} tickers")

    return result
