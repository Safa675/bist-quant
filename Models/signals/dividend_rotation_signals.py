"""
Dividend Rotation Signal Construction

Identifies high-quality dividend stocks using cross-sectional ranking.

Logic:
- Ranks stocks by dividend payout ratio (moderate payout preferred)
- Ranks stocks by volatility (lower is better)
- Ranks stocks by earnings growth (higher is better)
- Combines ranks to identify quality dividend stocks

Uses cross-sectional percentile ranks to avoid overfitting to specific
threshold values. This approach adapts to market conditions automatically.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.utils import (
    SignalDataError,
    assert_has_cross_section,
    assert_panel_not_constant,
    raise_signal_data_error,
)


# ============================================================================
# DIVIDEND ROTATION PARAMETERS (minimal)
# ============================================================================

VOLATILITY_LOOKBACK = 252  # 1 year for volatility calculation


# ============================================================================
# VOLATILITY CALCULATION
# ============================================================================

def calculate_annualized_volatility(
    close_df: pd.DataFrame,
    lookback: int = VOLATILITY_LOOKBACK,
) -> pd.DataFrame:
    """
    Calculate annualized volatility for each stock.
    
    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        lookback: Lookback period in trading days
    
    Returns:
        DataFrame of annualized volatility (Date x Ticker)
    """
    # Calculate daily returns
    daily_returns = close_df.pct_change(fill_method=None)
    
    # Calculate rolling standard deviation
    rolling_std = daily_returns.rolling(
        lookback, min_periods=int(lookback * 0.5)
    ).std()
    
    # Annualize (assuming 252 trading days per year)
    annualized_vol = rolling_std * np.sqrt(252)
    
    return annualized_vol


# ============================================================================
# CROSS-SECTIONAL RANKING UTILITIES
# ============================================================================

def cross_sectional_rank(series: pd.Series) -> pd.Series:
    """
    Convert values to cross-sectional percentile ranks (0-100).
    Higher rank = higher value in the cross-section.
    """
    valid = series.dropna()
    if len(valid) < 2:
        return pd.Series(np.nan, index=series.index, dtype=float)
    ranks = valid.rank(pct=True) * 100
    return ranks.reindex(series.index)


def inverse_rank(series: pd.Series) -> pd.Series:
    """
    Inverse percentile rank (0-100). Higher rank = lower value.
    Used for metrics where lower is better (e.g., volatility, debt).
    """
    valid = series.dropna()
    if len(valid) < 2:
        return pd.Series(np.nan, index=series.index, dtype=float)
    ranks = (1 - valid.rank(pct=True)) * 100
    return ranks.reindex(series.index)


def moderate_rank(series: pd.Series, optimal_pct: float = 0.5) -> pd.Series:
    """
    Rank that favors moderate values (closer to optimal percentile).
    Used for metrics where extremes are bad (e.g., payout ratio).
    """
    valid = series.dropna()
    if len(valid) < 2:
        return pd.Series(np.nan, index=series.index, dtype=float)
    pct_ranks = valid.rank(pct=True)
    # Distance from optimal (0 = at optimal, 1 = at extreme)
    distance = (pct_ranks - optimal_pct).abs()
    # Convert to score (100 = at optimal, 0 = at extreme)
    scores = (1 - distance * 2).clip(0, 1) * 100
    return scores.reindex(series.index)


# ============================================================================
# SIGNAL BUILDER (MAIN INTERFACE)
# ============================================================================

def build_dividend_rotation_signals(
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
) -> pd.DataFrame:
    """
    Build dividend rotation signal using cross-sectional ranking.

    Uses percentile ranks instead of absolute thresholds to reduce overfitting:
    - Payout ratio: moderate rank (favors middle of distribution)
    - Earnings growth: direct rank (higher is better)
    - Volatility: inverse rank (lower is better)

    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        dates: DatetimeIndex to align signals to
        data_loader: DataLoader instance with fundamental data

    Returns:
        DataFrame (dates x tickers) with dividend rotation scores (0-100)
    """
    print("\n Building dividend rotation signals (cross-sectional ranking)...")
    print(f"  Volatility lookback: {VOLATILITY_LOOKBACK} days")

    if data_loader is None:
        raise_signal_data_error(
            "dividend_rotation",
            "no data_loader provided for required fundamental metrics",
        )

    # Calculate volatility for all stocks
    volatility = calculate_annualized_volatility(close_df, VOLATILITY_LOOKBACK)

    # Load fundamental metrics
    try:
        metrics_df = data_loader.load_fundamental_metrics()

        if metrics_df.empty:
            raise_signal_data_error(
                "dividend_rotation",
                "fundamental metrics are empty; run calculate_fundamental_metrics.py",
            )

        # Check for required metrics
        required_metrics = ['dividend_payout_ratio', 'earnings_growth_yoy']
        available_metrics = metrics_df.columns.tolist()
        missing_metrics = [m for m in required_metrics if m not in available_metrics]

        if missing_metrics:
            raise_signal_data_error(
                "dividend_rotation",
                f"missing required metrics: {missing_metrics}; available: {available_metrics}",
            )

        print(f"  Loaded metrics for {len(metrics_df.index.get_level_values(0).unique())} tickers")

        # Build raw metric panels (dates x tickers)
        payout_panel = pd.DataFrame(index=dates, columns=close_df.columns, dtype=float)
        growth_panel = pd.DataFrame(index=dates, columns=close_df.columns, dtype=float)

        for ticker in close_df.columns:
            if ticker not in metrics_df.index.get_level_values(0):
                continue

            ticker_metrics = metrics_df.loc[ticker]
            if ticker_metrics.empty:
                continue

            # Reindex to daily dates and forward-fill
            ticker_metrics = ticker_metrics.reindex(dates, method='ffill')

            payout_panel[ticker] = ticker_metrics['dividend_payout_ratio']
            growth_panel[ticker] = ticker_metrics['earnings_growth_yoy']

        # Align volatility to dates
        volatility = volatility.reindex(dates)
        assert_has_cross_section(
            payout_panel,
            "dividend_rotation",
            "dividend_payout_ratio panel",
            min_valid_tickers=5,
        )
        assert_has_cross_section(
            growth_panel,
            "dividend_rotation",
            "earnings_growth_yoy panel",
            min_valid_tickers=5,
        )
        assert_has_cross_section(
            volatility,
            "dividend_rotation",
            "volatility panel",
            min_valid_tickers=5,
        )

        # Calculate cross-sectional ranks for each date
        result = pd.DataFrame(np.nan, index=dates, columns=close_df.columns, dtype=float)

        for date in dates:
            if date not in payout_panel.index:
                continue

            # Get cross-section for this date
            payout_cs = payout_panel.loc[date]
            growth_cs = growth_panel.loc[date]
            vol_cs = volatility.loc[date] if date in volatility.index else pd.Series(dtype=float)

            # Skip if not enough data
            valid_mask = payout_cs.notna() & growth_cs.notna() & vol_cs.notna()
            valid_count = int(valid_mask.sum())
            if valid_count < 5:
                continue

            # Cross-sectional ranks
            # Payout: moderate is best (favor ~50th percentile)
            payout_rank = moderate_rank(payout_cs, optimal_pct=0.5)

            # Growth: higher is better
            growth_rank = cross_sectional_rank(growth_cs)

            # Volatility: lower is better
            vol_rank = inverse_rank(vol_cs)

            # Combine ranks (equal weight)
            combined = pd.concat(
                [payout_rank, growth_rank, vol_rank],
                axis=1,
            ).mean(axis=1, skipna=False)
            combined = combined.where(valid_mask)

            result.loc[date] = combined

        assert_has_cross_section(
            result,
            "dividend_rotation",
            "final score panel",
            min_valid_tickers=5,
        )
        assert_panel_not_constant(result, "dividend_rotation", "final score panel")

        latest_valid = int(result.iloc[-1].notna().sum()) if len(result.index) else 0
        if latest_valid < 5:
            raise_signal_data_error(
                "dividend_rotation",
                f"latest date has insufficient coverage: {latest_valid} valid names (< 5)",
            )

    except SignalDataError:
        raise
    except Exception as e:
        raise_signal_data_error(
            "dividend_rotation",
            f"unexpected error while building signal: {e}",
        )

    # Summary stats
    valid_scores = result.dropna(how='all')
    if not valid_scores.empty:
        latest = valid_scores.iloc[-1].dropna()
        if len(latest) > 0:
            print(f"  Latest scores - Mean: {latest.mean():.1f}, Std: {latest.std():.1f}")
            print(f"  Latest scores - Min: {latest.min():.1f}, Max: {latest.max():.1f}")

            # Show top dividend stocks
            top_5 = latest.nlargest(5)
            if len(top_5) > 0:
                print(f"  Top 5 dividend stocks: {', '.join(top_5.index.tolist())}")

    print(f"  Dividend rotation signals: {result.shape[0]} days x {result.shape[1]} tickers")

    return result
