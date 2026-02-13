"""
Consistent Momentum Signal Construction

Identifies stocks with consistent momentum across multiple timeframes:
- Stocks that are winners in BOTH short and medium-term periods
- More robust than single-period momentum

Based on Quantpedia strategy: Consistent Momentum Strategy
Signal: Stocks must be in top performers for both t-7 to t-1 AND t-6 to t periods
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# CONSISTENT MOMENTUM PARAMETERS
# ============================================================================

PERIOD_1_START = 147  # ~7 months ago (7 * 21 trading days)
PERIOD_1_END = 21     # ~1 month ago
PERIOD_2_START = 126  # ~6 months ago (6 * 21 trading days)
PERIOD_2_END = 0      # Current

QUANTILE_THRESHOLD = 0.20  # Top 20% in both periods = consistent winner


# ============================================================================
# CORE CALCULATIONS
# ============================================================================

def calculate_period_return(
    close_df: pd.DataFrame,
    start_lag: int,
    end_lag: int,
) -> pd.DataFrame:
    """
    Calculate returns over a specific period.

    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        start_lag: Days ago for period start
        end_lag: Days ago for period end

    Returns:
        pd.DataFrame: Period returns
    """
    if end_lag == 0:
        end_prices = close_df
    else:
        end_prices = close_df.shift(end_lag)

    start_prices = close_df.shift(start_lag)

    period_return = end_prices / start_prices - 1.0
    return period_return


def calculate_consistent_momentum_scores(
    close_df: pd.DataFrame,
    period1_start: int = PERIOD_1_START,
    period1_end: int = PERIOD_1_END,
    period2_start: int = PERIOD_2_START,
    period2_end: int = PERIOD_2_END,
    quantile_threshold: float = QUANTILE_THRESHOLD,
) -> pd.DataFrame:
    """
    Calculate consistent momentum scores.

    Stocks get high scores only if they are top performers in BOTH periods.
    This filters out one-hit wonders and identifies persistent momentum.

    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        period1_start: Start of first period (days ago)
        period1_end: End of first period (days ago)
        period2_start: Start of second period (days ago)
        period2_end: End of second period (days ago)
        quantile_threshold: Threshold for top performers (0.20 = top 20%)

    Returns:
        pd.DataFrame: Consistent momentum scores (dates x tickers)
    """
    # Calculate returns for both periods
    ret_period1 = calculate_period_return(close_df, period1_start, period1_end)
    ret_period2 = calculate_period_return(close_df, period2_start, period2_end)

    # Rank stocks within each period (percentile ranking)
    rank_period1 = ret_period1.rank(axis=1, pct=True)
    rank_period2 = ret_period2.rank(axis=1, pct=True)

    # Identify consistent winners: top quantile in BOTH periods
    is_winner_p1 = rank_period1 >= (1 - quantile_threshold)
    is_winner_p2 = rank_period2 >= (1 - quantile_threshold)
    is_consistent_winner = is_winner_p1 & is_winner_p2

    # Score: average rank for consistent winners, 0 for others
    # This creates separation between consistent winners
    avg_rank = (rank_period1 + rank_period2) / 2
    consistent_score = avg_rank.where(is_consistent_winner, np.nan)

    # For non-consistent stocks, give them lower scores based on their average rank
    # but penalized
    penalty_factor = 0.5
    non_consistent_score = avg_rank * penalty_factor

    # Combine: consistent winners keep full score, others get penalized score
    final_score = consistent_score.fillna(non_consistent_score)

    return final_score


# ============================================================================
# SIGNAL BUILDER (MAIN INTERFACE)
# ============================================================================

def build_consistent_momentum_signals(
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
) -> pd.DataFrame:
    """
    Build consistent momentum signal panel.

    This is the main interface function that follows the same pattern
    as other signal builders.

    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        dates: DatetimeIndex to align signals to
        data_loader: Optional DataLoader instance

    Returns:
        DataFrame (dates x tickers) with consistent momentum scores
    """
    print("\n🔧 Building consistent momentum signals...")
    print(f"  Period 1: t-{PERIOD_1_START} to t-{PERIOD_1_END} (~7mo to ~1mo ago)")
    print(f"  Period 2: t-{PERIOD_2_START} to t-{PERIOD_2_END} (~6mo to current)")
    print(f"  Consistency threshold: Top {QUANTILE_THRESHOLD*100:.0f}% in both periods")

    # Calculate consistent momentum scores
    momentum_scores = calculate_consistent_momentum_scores(close_df)

    # Align to requested dates
    result = momentum_scores.reindex(dates)

    # Summary stats
    valid_scores = result.dropna(how='all')
    if not valid_scores.empty:
        latest = valid_scores.iloc[-1].dropna()
        if len(latest) > 0:
            print(f"  Latest scores - Mean: {latest.mean():.4f}, Std: {latest.std():.4f}")
            # Count consistent winners
            n_consistent = (latest > 0.5).sum()  # Above 0.5 = likely consistent winners
            print(f"  Consistent winners (latest): {n_consistent}")

    print(f"  ✅ Consistent momentum signals: {result.shape[0]} days × {result.shape[1]} tickers")

    return result
