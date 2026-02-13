"""
Short-Term Reversal Signal Construction

Exploits the short-term reversal anomaly in stock returns:
- Weekly losers tend to outperform in the following week
- Weekly winners tend to underperform in the following week

Based on Quantpedia strategy: Short-Term Reversal in Stocks
Signal: Negative of weekly return (lower past return = higher signal)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# REVERSAL CALCULATION PARAMETERS
# ============================================================================

WEEKLY_LOOKBACK = 5  # 5 trading days = 1 week
MONTHLY_LOOKBACK = 21  # 21 trading days = 1 month


# ============================================================================
# CORE REVERSAL CALCULATIONS
# ============================================================================

def calculate_weekly_return(close_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate weekly (5-day) returns.

    Args:
        close_df: DataFrame of close prices (Date x Ticker)

    Returns:
        pd.DataFrame: Weekly returns (current / 5 days ago - 1)
    """
    weekly_return = close_df / close_df.shift(WEEKLY_LOOKBACK) - 1.0
    return weekly_return


def calculate_monthly_return(close_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate monthly (21-day) returns.

    Args:
        close_df: DataFrame of close prices (Date x Ticker)

    Returns:
        pd.DataFrame: Monthly returns (current / 21 days ago - 1)
    """
    monthly_return = close_df / close_df.shift(MONTHLY_LOOKBACK) - 1.0
    return monthly_return


def calculate_reversal_scores(
    close_df: pd.DataFrame,
    use_weekly: bool = True,
    use_monthly: bool = False,
) -> pd.DataFrame:
    """
    Calculate reversal scores: negative of past returns.

    Lower past return = Higher signal score (expecting reversal upward)

    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        use_weekly: Use weekly returns for reversal
        use_monthly: Use monthly returns for reversal

    Returns:
        pd.DataFrame: Reversal scores (dates x tickers)
    """
    if use_weekly and not use_monthly:
        # Pure weekly reversal
        weekly_ret = calculate_weekly_return(close_df)
        reversal_score = -weekly_ret  # Negative: losers get high scores
    elif use_monthly and not use_weekly:
        # Pure monthly reversal
        monthly_ret = calculate_monthly_return(close_df)
        reversal_score = -monthly_ret
    else:
        # Combined: average of weekly and monthly reversal
        weekly_ret = calculate_weekly_return(close_df)
        monthly_ret = calculate_monthly_return(close_df)
        reversal_score = (-weekly_ret + -monthly_ret) / 2

    # Handle infinities and NaNs
    reversal_score = reversal_score.replace([np.inf, -np.inf], np.nan)

    return reversal_score


# ============================================================================
# SIGNAL BUILDER (MAIN INTERFACE)
# ============================================================================

def build_short_term_reversal_signals(
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
) -> pd.DataFrame:
    """
    Build short-term reversal signal panel.

    This is the main interface function that follows the same pattern
    as other signal builders (momentum, value, quality, etc.).

    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        dates: DatetimeIndex to align signals to
        data_loader: Optional DataLoader instance (for consistency with other signals)

    Returns:
        DataFrame (dates x tickers) with reversal scores
    """
    print("\n🔧 Building short-term reversal signals...")
    print(f"  Weekly Lookback: {WEEKLY_LOOKBACK} days")

    # Calculate reversal scores (weekly reversal - most effective)
    reversal_scores = calculate_reversal_scores(close_df, use_weekly=True, use_monthly=False)

    # Align to requested dates
    result = reversal_scores.reindex(dates)

    # Summary stats
    valid_scores = result.dropna(how='all')
    if not valid_scores.empty:
        latest = valid_scores.iloc[-1].dropna()
        if len(latest) > 0:
            print(f"  Latest scores - Mean: {latest.mean():.4f}, Std: {latest.std():.4f}")
            print(f"  Latest scores - Min: {latest.min():.4f}, Max: {latest.max():.4f}")

    print(f"  ✅ Short-term reversal signals: {result.shape[0]} days × {result.shape[1]} tickers")

    return result
