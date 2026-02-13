"""
Sector Rotation Signal Construction

Rotates into the strongest sectors based on momentum:
- Groups stocks by BIST sector
- Calculates sector momentum (12-month returns)
- Overweights stocks in top-performing sectors

Based on Quantpedia strategy: Sector Momentum Rotational System
Signal: Sector momentum score assigned to each stock in that sector
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# SECTOR ROTATION PARAMETERS
# ============================================================================

SECTOR_MOMENTUM_LOOKBACK = 252  # 12 months for sector momentum
TOP_SECTORS = 3  # Number of top sectors to overweight


# ============================================================================
# BIST SECTOR MAPPINGS
# ============================================================================

# Major BIST sector groupings based on common Turkish stock classifications
BIST_SECTORS = {
    'Banking': [
        'AKBNK', 'GARAN', 'ISCTR', 'YKBNK', 'HALKB', 'VAKBN', 'QNBFB', 'TSKB', 'ALBRK', 'KLNMA'
    ],
    'Holding': [
        'SAHOL', 'KCHOL', 'DOHOL', 'GLYHO', 'ECZYT', 'KOZAL', 'TAVHL', 'TTKOM', 'TCELL'
    ],
    'Industry': [
        'TUPRS', 'EREGL', 'KRDMD', 'TOASO', 'FROTO', 'OTKAR', 'ASELS', 'VESTL', 'ARCLK',
        'PETKM', 'SASA', 'BRISA', 'GUBRF', 'BAGFS', 'TMSN', 'CEMTS', 'KARTN', 'KORDS'
    ],
    'Construction': [
        'ENKAI', 'EKGYO', 'ISGYO', 'HLGYO', 'KLGYO', 'YEOTK', 'OYAYO', 'NTHOL'
    ],
    'Retail_Consumer': [
        'BIMAS', 'MGROS', 'SOKM', 'MAVI', 'BIZIM', 'ULKER', 'BANVT', 'CCOLA', 'AEFES', 'PGSUS'
    ],
    'Technology': [
        'LOGO', 'INDES', 'KAREL', 'ARENA', 'NETAS', 'DGATE', 'PAPIL', 'SMART'
    ],
    'Energy': [
        'AKSEN', 'ODAS', 'ZOREN', 'AYEN', 'AKSA', 'ENJSA', 'AKENR', 'GESAN'
    ],
    'Mining_Metals': [
        'KOZAA', 'IPEKE', 'KLMSN', 'ALKIM', 'SARKY', 'BAKAB', 'DOKTA'
    ],
    'Financials_Other': [
        'SKBNK', 'ALGYO', 'ANHYT', 'ANSGR', 'TURSG', 'AKGRT', 'AGYO', 'ATAGY', 'KAYSE'
    ],
}


def get_sector_for_ticker(ticker: str) -> str:
    """Get sector for a given ticker."""
    for sector, tickers in BIST_SECTORS.items():
        if ticker in tickers:
            return sector
    return 'Other'


def build_sector_mapping(tickers: list) -> Dict[str, str]:
    """Build ticker to sector mapping."""
    return {ticker: get_sector_for_ticker(ticker) for ticker in tickers}


# ============================================================================
# CORE CALCULATIONS
# ============================================================================

def calculate_sector_returns(
    close_df: pd.DataFrame,
    sector_mapping: Dict[str, str],
    lookback: int = SECTOR_MOMENTUM_LOOKBACK,
) -> pd.DataFrame:
    """
    Calculate sector-level momentum returns.

    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        sector_mapping: Dict mapping tickers to sectors
        lookback: Lookback period for momentum

    Returns:
        pd.DataFrame: Sector returns (Date x Sector)
    """
    # Group tickers by sector
    sectors = {}
    for ticker, sector in sector_mapping.items():
        if ticker in close_df.columns:
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(ticker)

    # Calculate equal-weighted sector returns
    sector_prices = {}
    for sector, tickers in sectors.items():
        if tickers:
            # Equal-weighted average price for sector
            sector_prices[sector] = close_df[tickers].mean(axis=1)

    sector_df = pd.DataFrame(sector_prices)

    # Calculate momentum returns
    sector_returns = sector_df / sector_df.shift(lookback) - 1.0

    return sector_returns


def calculate_sector_rotation_scores(
    close_df: pd.DataFrame,
    lookback: int = SECTOR_MOMENTUM_LOOKBACK,
    top_n_sectors: int = TOP_SECTORS,
) -> pd.DataFrame:
    """
    Calculate sector rotation scores for each stock.

    Stocks in top-performing sectors get higher scores.

    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        lookback: Lookback for sector momentum
        top_n_sectors: Number of top sectors to favor

    Returns:
        pd.DataFrame: Sector rotation scores (dates x tickers)
    """
    # Build sector mapping
    sector_mapping = build_sector_mapping(close_df.columns.tolist())

    # Calculate sector returns
    sector_returns = calculate_sector_returns(close_df, sector_mapping, lookback)

    # Rank sectors by momentum (higher = better)
    sector_ranks = sector_returns.rank(axis=1, pct=True)

    # Assign scores to each stock based on its sector's rank
    scores = pd.DataFrame(index=close_df.index, columns=close_df.columns, dtype=float)

    for ticker in close_df.columns:
        sector = sector_mapping.get(ticker, 'Other')
        if sector in sector_ranks.columns:
            scores[ticker] = sector_ranks[sector]
        else:
            scores[ticker] = 0.5  # Neutral for unclassified stocks

    # Bonus for stocks in top sectors
    for date in sector_ranks.index:
        if pd.isna(sector_ranks.loc[date]).all():
            continue

        # Get top sectors for this date
        top_sectors = sector_ranks.loc[date].nlargest(top_n_sectors).index.tolist()

        for ticker in close_df.columns:
            sector = sector_mapping.get(ticker, 'Other')
            if sector in top_sectors:
                scores.loc[date, ticker] = scores.loc[date, ticker] * 1.5

    # Handle infinities and NaNs
    scores = scores.replace([np.inf, -np.inf], np.nan)

    return scores


# ============================================================================
# SIGNAL BUILDER (MAIN INTERFACE)
# ============================================================================

def build_sector_rotation_signals(
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
) -> pd.DataFrame:
    """
    Build sector rotation signal panel.

    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        dates: DatetimeIndex to align signals to
        data_loader: Optional DataLoader instance

    Returns:
        DataFrame (dates x tickers) with sector rotation scores
    """
    print("\n🔧 Building sector rotation signals...")
    print(f"  Sector Momentum Lookback: {SECTOR_MOMENTUM_LOOKBACK} days")
    print(f"  Top Sectors: {TOP_SECTORS}")

    # Build sector mapping and count coverage
    sector_mapping = build_sector_mapping(close_df.columns.tolist())
    sector_counts = {}
    for ticker, sector in sector_mapping.items():
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    print(f"  Sector coverage:")
    for sector, count in sorted(sector_counts.items(), key=lambda x: -x[1]):
        print(f"    {sector}: {count} stocks")

    # Calculate sector rotation scores
    sector_scores = calculate_sector_rotation_scores(close_df)

    # Align to requested dates
    result = sector_scores.reindex(dates)

    # Summary stats
    valid_scores = result.dropna(how='all')
    if not valid_scores.empty:
        latest = valid_scores.iloc[-1].dropna()
        if len(latest) > 0:
            print(f"  Latest scores - Mean: {latest.mean():.4f}, Std: {latest.std():.4f}")

    print(f"  ✅ Sector rotation signals: {result.shape[0]} days × {result.shape[1]} tickers")

    return result
