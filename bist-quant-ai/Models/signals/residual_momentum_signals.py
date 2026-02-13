"""
Residual Momentum Signal Construction

Calculates momentum adjusted for market exposure:
- Regresses stock returns against market returns
- Uses residual (alpha) returns for momentum signal
- Captures stock-specific momentum, not market-driven momentum

Based on Quantpedia strategy: Residual Momentum Factor
Signal: 12-month residual returns / residual volatility
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# RESIDUAL MOMENTUM PARAMETERS
# ============================================================================

MOMENTUM_LOOKBACK = 252  # ~12 months trading days
MOMENTUM_SKIP = 21       # ~1 month skip (most recent month excluded)
REGRESSION_LOOKBACK = 252  # Rolling window for beta estimation


# ============================================================================
# CORE CALCULATIONS
# ============================================================================

def calculate_rolling_beta(
    stock_returns: pd.DataFrame,
    market_returns: pd.Series,
    lookback: int = REGRESSION_LOOKBACK,
) -> pd.DataFrame:
    """
    Calculate rolling beta for each stock vs market.

    Args:
        stock_returns: DataFrame of daily stock returns (Date x Ticker)
        market_returns: Series of daily market returns
        lookback: Rolling window for beta estimation

    Returns:
        pd.DataFrame: Rolling beta for each stock
    """
    betas = pd.DataFrame(index=stock_returns.index, columns=stock_returns.columns, dtype=float)

    # Align market returns
    market_aligned = market_returns.reindex(stock_returns.index)

    for ticker in stock_returns.columns:
        stock_ret = stock_returns[ticker]

        # Rolling covariance and variance
        cov = stock_ret.rolling(lookback, min_periods=lookback // 2).cov(market_aligned)
        var = market_aligned.rolling(lookback, min_periods=lookback // 2).var()

        # Beta = Cov(stock, market) / Var(market)
        beta = cov / var.replace(0, np.nan)
        betas[ticker] = beta

    return betas


def calculate_residual_returns(
    stock_returns: pd.DataFrame,
    market_returns: pd.Series,
    betas: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate residual returns (alpha) for each stock.

    Residual = Stock Return - Beta * Market Return

    Args:
        stock_returns: DataFrame of daily stock returns
        market_returns: Series of daily market returns
        betas: DataFrame of rolling betas

    Returns:
        pd.DataFrame: Residual returns for each stock
    """
    market_aligned = market_returns.reindex(stock_returns.index)

    # Residual = Stock - Beta * Market
    residuals = stock_returns - betas.multiply(market_aligned, axis=0)

    return residuals


def calculate_residual_momentum_scores(
    close_df: pd.DataFrame,
    market_close: pd.Series,
    lookback: int = MOMENTUM_LOOKBACK,
    skip: int = MOMENTUM_SKIP,
) -> pd.DataFrame:
    """
    Calculate residual momentum scores.

    Score = Cumulative Residual Return / Residual Volatility

    Higher score = Better risk-adjusted residual momentum

    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        market_close: Series of market index close prices
        lookback: Lookback for momentum calculation
        skip: Days to skip from most recent month

    Returns:
        pd.DataFrame: Residual momentum scores (dates x tickers)
    """
    # Calculate returns
    stock_returns = close_df.pct_change(fill_method=None)
    market_returns = market_close.pct_change(fill_method=None)

    # Calculate rolling betas
    betas = calculate_rolling_beta(stock_returns, market_returns, lookback)

    # Calculate residual returns
    residuals = calculate_residual_returns(stock_returns, market_returns, betas)

    # Shift to skip most recent month
    residuals_shifted = residuals.shift(skip)

    # Calculate cumulative residual return over lookback-skip period
    effective_lookback = lookback - skip
    cum_residual = residuals_shifted.rolling(effective_lookback, min_periods=effective_lookback // 2).sum()

    # Calculate residual volatility
    residual_vol = residuals_shifted.rolling(effective_lookback, min_periods=effective_lookback // 2).std()

    # Apply minimum volatility threshold
    MIN_VOL = 0.001
    residual_vol_safe = residual_vol.clip(lower=MIN_VOL)

    # Residual momentum score = cumulative residual / residual volatility
    residual_momentum = cum_residual / residual_vol_safe

    # Handle infinities and NaNs
    residual_momentum = residual_momentum.replace([np.inf, -np.inf], np.nan)

    return residual_momentum


# ============================================================================
# SIGNAL BUILDER (MAIN INTERFACE)
# ============================================================================

def build_residual_momentum_signals(
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
) -> pd.DataFrame:
    """
    Build residual momentum signal panel.

    Uses XU100 as the market proxy for BIST stocks.

    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        dates: DatetimeIndex to align signals to
        data_loader: DataLoader instance (for loading XU100)

    Returns:
        DataFrame (dates x tickers) with residual momentum scores
    """
    print("\n🔧 Building residual momentum signals...")
    print(f"  Momentum Lookback: {MOMENTUM_LOOKBACK} days")
    print(f"  Skip Period: {MOMENTUM_SKIP} days")
    print(f"  Beta Estimation Window: {REGRESSION_LOOKBACK} days")

    # Load market proxy (XU100)
    if data_loader is not None:
        script_dir = Path(__file__).parent.parent.parent
        xu100_file = script_dir / "data" / "xu100_prices.csv"
        if xu100_file.exists():
            xu100_df = pd.read_csv(xu100_file)
            xu100_df['Date'] = pd.to_datetime(xu100_df['Date'])
            xu100_df = xu100_df.set_index('Date').sort_index()
            market_close = xu100_df['Close'] if 'Close' in xu100_df.columns else xu100_df['Open']
            print("  Using XU100 as market proxy")
        else:
            # Fallback: use equal-weighted average of all stocks
            market_close = close_df.mean(axis=1)
            print("  Using equal-weighted market as proxy (XU100 not found)")
    else:
        market_close = close_df.mean(axis=1)
        print("  Using equal-weighted market as proxy")

    # Align market to close_df index
    market_close = market_close.reindex(close_df.index).ffill()

    # Calculate residual momentum scores
    residual_scores = calculate_residual_momentum_scores(close_df, market_close)

    # Align to requested dates
    result = residual_scores.reindex(dates)

    # Summary stats
    valid_scores = result.dropna(how='all')
    if not valid_scores.empty:
        latest = valid_scores.iloc[-1].dropna()
        if len(latest) > 0:
            print(f"  Latest scores - Mean: {latest.mean():.2f}, Std: {latest.std():.2f}")
            print(f"  Latest scores - Min: {latest.min():.2f}, Max: {latest.max():.2f}")

    print(f"  ✅ Residual momentum signals: {result.shape[0]} days × {result.shape[1]} tickers")

    return result
