"""
Betting Against Beta Signal Construction

Low-beta stocks tend to outperform high-beta stocks on a risk-adjusted basis.
This exploits the empirical finding that the Security Market Line is flatter than CAPM predicts.

Formula:
Beta = Cov(Stock Returns, Market Returns) / Var(Market Returns)

Lower beta = Higher score (defensive stocks that outperform)
Higher beta = Lower score (aggressive stocks that underperform)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def calculate_rolling_beta(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    window: int = 252,  # 1 year
) -> float:
    """Calculate beta using rolling window"""
    if len(stock_returns) < window or len(market_returns) < window:
        return np.nan
    
    # Align the series
    aligned = pd.DataFrame({
        'stock': stock_returns,
        'market': market_returns
    }).dropna()
    
    if len(aligned) < window:
        return np.nan
    
    # Use last 'window' observations
    recent = aligned.tail(window)
    
    # Calculate beta: Cov(stock, market) / Var(market)
    cov = recent['stock'].cov(recent['market'])
    var = recent['market'].var()
    
    if var == 0 or pd.isna(var):
        return np.nan
    
    beta = cov / var
    return beta


def build_betting_against_beta_signals(
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
    beta_window: int = 252,  # 1 year rolling window
) -> pd.DataFrame:
    """
    Build betting against beta signal panel
    
    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        dates: DatetimeIndex to align signals to
        data_loader: DataLoader instance for XU100 benchmark
        beta_window: Rolling window for beta calculation (default 252 days = 1 year)
    
    Returns:
        DataFrame (dates x tickers) with beta scores
        Higher score = Lower beta = Better (defensive stocks)
    """
    print("\n🔧 Building betting against beta signals...")
    print(f"  Beta calculation window: {beta_window} days (~1 year)")
    print("  Lower beta = Higher score = Defensive stocks")
    
    # Load XU100 benchmark
    xu100_file = Path(data_loader.data_dir) / "xu100_prices.csv"
    xu100_prices = data_loader.load_xu100_prices(xu100_file)
    
    if xu100_prices is None or xu100_prices.empty:
        print("  ⚠️  No XU100 benchmark data available")
        return pd.DataFrame(0.0, index=dates, columns=close_df.columns)
    
    # Calculate market and stock returns
    market_returns = xu100_prices.pct_change(fill_method=None)
    stock_returns = close_df.pct_change(fill_method=None)

    # Work on market trading days to avoid NaN gaps from union calendars.
    common_dates = stock_returns.index.intersection(market_returns.index)
    if len(common_dates) < beta_window:
        print("  ⚠️  Not enough overlap with market returns for beta window")
        return pd.DataFrame(0.0, index=dates, columns=close_df.columns)

    stock_aligned = stock_returns.reindex(common_dates)
    market_aligned = market_returns.reindex(common_dates)

    print("  Calculating rolling betas (vectorized)...")

    # Cov(stock, market) for all stocks and dates in one pass.
    rolling_cov = stock_aligned.rolling(beta_window, min_periods=beta_window).cov(market_aligned)

    # Compute pairwise market variance to match stock-specific missing-data windows.
    market_panel = (
        stock_aligned.notna()
        .astype(float)
        .replace(0.0, np.nan)
        .mul(market_aligned, axis=0)
    )
    rolling_var = market_panel.rolling(beta_window, min_periods=beta_window).var()

    # Beta = Cov / Var; then invert score so lower beta ranks higher.
    beta = rolling_cov.div(rolling_var).replace([np.inf, -np.inf], np.nan)
    beta_panel = (-beta).reindex(index=dates, columns=close_df.columns).ffill().fillna(0.0)
    
    # Summary stats
    if not beta_panel.empty:
        latest = beta_panel.iloc[-1].dropna()
        latest = latest[latest != 0]
        if len(latest) > 0:
            # Convert back to actual beta for display (multiply by -1)
            actual_betas = -latest
            print(f"  Latest betas - Mean: {actual_betas.mean():.3f}, Std: {actual_betas.std():.3f}")
            print(f"  Latest betas - Min: {actual_betas.min():.3f}, Max: {actual_betas.max():.3f}")
            
            # Show top 5 low-beta stocks (highest scores)
            top_5 = latest.nlargest(5)
            print(f"  Top 5 low-beta stocks: {', '.join(top_5.index.tolist())}")
            top_5_betas = -top_5
            for ticker, beta in top_5_betas.items():
                print(f"    {ticker}: β = {beta:.3f}")
    
    print(f"  ✅ Betting against beta signals: {beta_panel.shape[0]} days × {beta_panel.shape[1]} tickers")
    return beta_panel
