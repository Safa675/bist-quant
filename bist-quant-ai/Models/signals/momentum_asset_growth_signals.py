"""
Momentum + Asset Growth Combined Signal Construction

Combines momentum with asset growth effect:
- First filter: Select high asset growth stocks
- Second filter: Within high-growth stocks, rank by momentum

Strategy: Momentum within the high-growth universe
This exploits the interaction between growth and momentum anomalies.

Based on Quantpedia strategy: Momentum Factor Combined with Asset Growth Effect
Signal: Momentum score for stocks in top asset growth decile
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.utils import (
    coerce_quarter_cols,
    get_consolidated_sheet,
    pick_row_from_sheet,
    apply_lag,
)


# ============================================================================
# PARAMETERS
# ============================================================================

MOMENTUM_LOOKBACK = 252  # 12 months
MOMENTUM_SKIP = 21       # Skip last month
ASSET_GROWTH_TOP_PERCENTILE = 0.30  # Top 30% by asset growth

# Fundamental data keys
BALANCE_SHEET = "Bilanço"
TOTAL_ASSETS_KEYS = (
    "Toplam Varlıklar",
    "Toplam Aktifler",
)


# ============================================================================
# ASSET GROWTH CALCULATION
# ============================================================================

def calculate_asset_growth_for_ticker(
    ticker: str,
    data_loader,
    fundamentals_parquet: pd.DataFrame = None,
) -> pd.Series:
    """
    Calculate YoY asset growth for a ticker.

    Asset Growth = (Total Assets_t / Total Assets_t-1) - 1

    Returns:
        pd.Series: Asset growth indexed by quarter
    """
    if fundamentals_parquet is None:
        return pd.Series(dtype=float)

    bs = get_consolidated_sheet(fundamentals_parquet, ticker, BALANCE_SHEET)
    if bs.empty:
        return pd.Series(dtype=float)

    assets_row = pick_row_from_sheet(bs, TOTAL_ASSETS_KEYS)
    if assets_row is None:
        return pd.Series(dtype=float)

    assets = coerce_quarter_cols(assets_row)
    if assets.empty or len(assets) < 5:
        return pd.Series(dtype=float)

    # YoY asset growth (4 quarters ago)
    asset_growth = assets / assets.shift(4) - 1.0
    asset_growth = asset_growth.replace([np.inf, -np.inf], np.nan).dropna()

    return asset_growth


# ============================================================================
# SIGNAL BUILDER
# ============================================================================

def build_momentum_asset_growth_signals(
    fundamentals: Dict,
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader,
) -> pd.DataFrame:
    """
    Build Momentum + Asset Growth combined signal panel.

    Strategy:
    1. Identify high asset growth stocks (top percentile)
    2. Within those, rank by 12-1 momentum
    3. Combined score favors high-growth + high-momentum stocks

    Args:
        fundamentals: Dict of fundamental data by ticker
        close_df: DataFrame of close prices
        dates: DatetimeIndex to align signals to
        data_loader: DataLoader instance

    Returns:
        DataFrame (dates x tickers) with combined scores
    """
    print("\n🔧 Building Momentum + Asset Growth signals...")
    print(f"  Momentum Lookback: {MOMENTUM_LOOKBACK} days, Skip: {MOMENTUM_SKIP} days")
    print(f"  Asset Growth Top Percentile: {ASSET_GROWTH_TOP_PERCENTILE*100:.0f}%")

    fundamentals_parquet = data_loader.load_fundamentals_parquet()

    # Calculate asset growth for all tickers
    asset_growth_panel = {}
    count = 0

    for ticker in fundamentals.keys():
        if ticker not in close_df.columns:
            continue

        ag = calculate_asset_growth_for_ticker(ticker, data_loader, fundamentals_parquet)
        if not ag.empty:
            lagged = apply_lag(ag, dates)
            if not lagged.empty:
                asset_growth_panel[ticker] = lagged

        count += 1
        if count % 50 == 0:
            print(f"  Processed {count} tickers for asset growth...")

    # Build asset growth DataFrame
    asset_growth_df = pd.DataFrame(asset_growth_panel, index=dates)

    # Calculate 12-1 momentum
    momentum = close_df.shift(MOMENTUM_SKIP) / close_df.shift(MOMENTUM_LOOKBACK) - 1.0
    momentum_df = momentum.reindex(dates)

    # Cross-sectional ranking
    print("  Calculating cross-sectional ranks...")

    # Asset growth rank (higher growth = higher rank)
    ag_rank = asset_growth_df.rank(axis=1, pct=True)

    # Momentum rank (higher momentum = higher rank)
    mom_rank = momentum_df.rank(axis=1, pct=True)

    # Identify high-growth stocks (top percentile)
    is_high_growth = ag_rank >= (1 - ASSET_GROWTH_TOP_PERCENTILE)

    # Combined score:
    # For high-growth stocks: momentum rank (they get scored by momentum)
    # For low-growth stocks: penalized score (lower priority)
    print("  Combining asset growth and momentum signals...")

    combined = pd.DataFrame(index=dates, columns=close_df.columns, dtype=float)

    for ticker in close_df.columns:
        if ticker in ag_rank.columns and ticker in mom_rank.columns:
            # High-growth stocks: use momentum rank as score
            # Low-growth stocks: heavily penalized
            high_growth_mask = is_high_growth[ticker].fillna(False)

            # Score = momentum rank for high-growth, penalized for others
            score = mom_rank[ticker].where(high_growth_mask, mom_rank[ticker] * 0.3)
            combined[ticker] = score
        elif ticker in mom_rank.columns:
            # No asset growth data, use momentum with penalty
            combined[ticker] = mom_rank[ticker] * 0.5

    result = combined.replace([np.inf, -np.inf], np.nan)

    # Summary stats
    valid_scores = result.dropna(how='all')
    if not valid_scores.empty:
        latest = valid_scores.iloc[-1].dropna()
        if len(latest) > 0:
            n_high_growth = (latest > 0.7).sum()  # Approximate high-growth count
            print(f"  High-growth stocks (latest): ~{n_high_growth}")

    print(f"  ✅ Momentum + Asset Growth signals: {result.shape[0]} days × {result.shape[1]} tickers")

    return result
