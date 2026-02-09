"""
Macro Hedge Signal Construction

Identifies fortress balance sheet stocks using cross-sectional ranking.

Logic:
- Ranks stocks by debt-to-equity (lower is better)
- Ranks stocks by cash ratio (higher is better)
- Ranks stocks by current ratio (higher is better)
- Ranks stocks by operating cash flow (higher is better)

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
    Used for metrics where lower is better (e.g., debt-to-equity).
    """
    valid = series.dropna()
    if len(valid) < 2:
        return pd.Series(np.nan, index=series.index, dtype=float)
    ranks = (1 - valid.rank(pct=True)) * 100
    return ranks.reindex(series.index)


# ============================================================================
# SIGNAL BUILDER (MAIN INTERFACE)
# ============================================================================

def build_macro_hedge_signals(
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
) -> pd.DataFrame:
    """
    Build macro hedge signal using cross-sectional ranking.

    Uses percentile ranks instead of absolute thresholds to reduce overfitting:
    - Debt-to-equity: inverse rank (lower is better)
    - Cash ratio: direct rank (higher is better)
    - Current ratio: direct rank (higher is better)
    - Operating cash flow: direct rank (higher is better)

    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        dates: DatetimeIndex to align signals to
        data_loader: DataLoader instance with fundamental data

    Returns:
        DataFrame (dates x tickers) with macro hedge scores (0-100)
    """
    print("\n Building macro hedge signals (cross-sectional ranking)...")

    if data_loader is None:
        raise_signal_data_error(
            "macro_hedge",
            "no data_loader provided for required fundamental metrics",
        )

    # Load fundamental metrics
    try:
        metrics_df = data_loader.load_fundamental_metrics()

        if metrics_df.empty:
            raise_signal_data_error(
                "macro_hedge",
                "fundamental metrics are empty; run calculate_fundamental_metrics.py",
            )

        # Check for required metrics
        required_metrics = ['debt_to_equity', 'cash_ratio', 'current_ratio', 'operating_cash_flow']
        available_metrics = metrics_df.columns.tolist()
        missing_metrics = [m for m in required_metrics if m not in available_metrics]

        if missing_metrics:
            raise_signal_data_error(
                "macro_hedge",
                f"missing required metrics: {missing_metrics}; available: {available_metrics}",
            )

        print(f"  Loaded metrics for {len(metrics_df.index.get_level_values(0).unique())} tickers")

        # Build raw metric panels (dates x tickers)
        de_panel = pd.DataFrame(index=dates, columns=close_df.columns, dtype=float)
        cash_panel = pd.DataFrame(index=dates, columns=close_df.columns, dtype=float)
        current_panel = pd.DataFrame(index=dates, columns=close_df.columns, dtype=float)
        ocf_panel = pd.DataFrame(index=dates, columns=close_df.columns, dtype=float)

        for ticker in close_df.columns:
            if ticker not in metrics_df.index.get_level_values(0):
                continue

            ticker_metrics = metrics_df.loc[ticker]
            if ticker_metrics.empty:
                continue

            # Reindex to daily dates and forward-fill
            ticker_metrics = ticker_metrics.reindex(dates, method='ffill')

            de_panel[ticker] = ticker_metrics['debt_to_equity']
            cash_panel[ticker] = ticker_metrics['cash_ratio']
            current_panel[ticker] = ticker_metrics['current_ratio']
            ocf_panel[ticker] = ticker_metrics['operating_cash_flow']

        assert_has_cross_section(
            de_panel,
            "macro_hedge",
            "debt_to_equity panel",
            min_valid_tickers=5,
        )
        assert_has_cross_section(
            cash_panel,
            "macro_hedge",
            "cash_ratio panel",
            min_valid_tickers=5,
        )
        assert_has_cross_section(
            current_panel,
            "macro_hedge",
            "current_ratio panel",
            min_valid_tickers=5,
        )
        assert_has_cross_section(
            ocf_panel,
            "macro_hedge",
            "operating_cash_flow panel",
            min_valid_tickers=5,
        )

        # Calculate cross-sectional ranks for each date
        result = pd.DataFrame(np.nan, index=dates, columns=close_df.columns, dtype=float)

        for date in dates:
            if date not in de_panel.index:
                continue

            # Get cross-section for this date
            de_cs = de_panel.loc[date]
            cash_cs = cash_panel.loc[date]
            current_cs = current_panel.loc[date]
            ocf_cs = ocf_panel.loc[date]

            # Skip if not enough data
            valid_mask = de_cs.notna() & cash_cs.notna() & current_cs.notna() & ocf_cs.notna()
            valid_count = int(valid_mask.sum())
            if valid_count < 5:
                continue

            # Cross-sectional ranks
            # Debt-to-equity: lower is better (inverse rank)
            de_rank = inverse_rank(de_cs)

            # Cash ratio: higher is better
            cash_rank = cross_sectional_rank(cash_cs)

            # Current ratio: higher is better
            current_rank = cross_sectional_rank(current_cs)

            # Operating cash flow: higher is better
            ocf_rank = cross_sectional_rank(ocf_cs)

            # Combine ranks (equal weight)
            combined = pd.concat(
                [de_rank, cash_rank, current_rank, ocf_rank],
                axis=1,
            ).mean(axis=1, skipna=False)
            combined = combined.where(valid_mask)

            result.loc[date] = combined

        assert_has_cross_section(
            result,
            "macro_hedge",
            "final score panel",
            min_valid_tickers=5,
        )
        assert_panel_not_constant(result, "macro_hedge", "final score panel")

        latest_valid = int(result.iloc[-1].notna().sum()) if len(result.index) else 0
        if latest_valid < 5:
            raise_signal_data_error(
                "macro_hedge",
                f"latest date has insufficient coverage: {latest_valid} valid names (< 5)",
            )

    except SignalDataError:
        raise
    except Exception as e:
        raise_signal_data_error(
            "macro_hedge",
            f"unexpected error while building signal: {e}",
        )

    # Summary stats
    valid_scores = result.dropna(how='all')
    if not valid_scores.empty:
        latest = valid_scores.iloc[-1].dropna()
        if len(latest) > 0:
            print(f"  Latest scores - Mean: {latest.mean():.1f}, Std: {latest.std():.1f}")
            print(f"  Latest scores - Min: {latest.min():.1f}, Max: {latest.max():.1f}")

            # Show top fortress stocks
            top_5 = latest.nlargest(5)
            if len(top_5) > 0:
                print(f"  Top 5 fortress stocks: {', '.join(top_5.index.tolist())}")

    print(f"  Macro hedge signals: {result.shape[0]} days x {result.shape[1]} tickers")

    return result
