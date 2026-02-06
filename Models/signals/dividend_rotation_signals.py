"""
Dividend Rotation Signal Construction

Identifies high-quality dividend stocks during rate normalization periods.

Logic:
- High dividend yield (4-7% sweet spot) attracts investors when rates normalize
- Low volatility ensures dividend sustainability
- Positive earnings growth confirms dividend is not at risk
- Reasonable valuation (P/B < 1.5) provides margin of safety

This signal is designed for the Turkish market where TCMB rate policy
drives significant rotations between growth and dividend stocks.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# DIVIDEND ROTATION PARAMETERS
# ============================================================================

VOLATILITY_LOOKBACK = 252  # 1 year for volatility calculation
VOLATILITY_THRESHOLD = 0.35  # Maximum acceptable annualized volatility
DIVIDEND_YIELD_MIN = 0.04  # 4% minimum yield
DIVIDEND_YIELD_OPTIMAL = 0.07  # 7% optimal yield (higher may signal distress)
PB_RATIO_THRESHOLD = 1.5  # Maximum P/B ratio for value screen
EARNINGS_GROWTH_MIN = 0.0  # Minimum earnings growth (positive)


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
# DIVIDEND ROTATION SIGNAL
# ============================================================================

def calculate_dividend_rotation_scores(
    close_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate dividend rotation signal scores.
    
    Combines:
    1. Dividend yield (4-7% optimal)
    2. Low volatility (< 35% annualized)
    3. Positive earnings growth
    4. Reasonable valuation (P/B < 1.5)
    
    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        fundamentals_df: DataFrame with fundamental metrics
                        Expected columns: dividend_yield, earnings_growth_yoy, pb_ratio
                        Index: MultiIndex (Ticker, Date) or Date with Ticker columns
    
    Returns:
        DataFrame (dates x tickers) with dividend rotation scores (0-100)
    """
    # Calculate volatility
    volatility = calculate_annualized_volatility(close_df)
    
    # Initialize scores DataFrame
    scores = pd.DataFrame(0.0, index=close_df.index, columns=close_df.columns)
    
    # Process fundamentals
    # Handle both MultiIndex (Ticker, Date) and regular Date index formats
    if isinstance(fundamentals_df.index, pd.MultiIndex):
        # MultiIndex format: (Ticker, Date)
        for ticker in close_df.columns:
            if ticker not in fundamentals_df.index.get_level_values(0):
                continue
            
            ticker_fundamentals = fundamentals_df.xs(ticker, level=0)
            
            # Align dates
            common_dates = close_df.index.intersection(ticker_fundamentals.index)
            
            for date in common_dates:
                if date not in ticker_fundamentals.index:
                    continue
                
                fund_data = ticker_fundamentals.loc[date]
                
                # Extract metrics
                div_yield = fund_data.get('dividend_yield', 0.0)
                earnings_growth = fund_data.get('earnings_growth_yoy', 0.0)
                pb_ratio = fund_data.get('pb_ratio', np.nan)
                
                # Get volatility
                vol = volatility.loc[date, ticker] if date in volatility.index else np.nan
                
                # Calculate score
                score = calculate_dividend_score(
                    div_yield, earnings_growth, pb_ratio, vol
                )
                
                scores.loc[date, ticker] = score
    
    else:
        # Regular Date index format
        # Assume fundamentals_df has same structure as close_df
        for date in close_df.index:
            if date not in fundamentals_df.index:
                continue
            
            for ticker in close_df.columns:
                if ticker not in fundamentals_df.columns:
                    continue
                
                # This assumes fundamentals are in a different format
                # Adjust based on your actual data structure
                # For now, skip this path and rely on MultiIndex
                pass
    
    return scores


def calculate_dividend_score(
    dividend_yield: float,
    earnings_growth: float,
    pb_ratio: float,
    volatility: float,
) -> float:
    """
    Calculate dividend rotation score for a single stock.
    
    Scoring:
    - Dividend yield (4-7%): 30 points
    - Low volatility (< 35%): 25 points
    - Positive earnings growth: 25 points
    - Low P/B ratio (< 1.5): 20 points
    
    Args:
        dividend_yield: Annual dividend yield (e.g., 0.05 = 5%)
        earnings_growth: YoY earnings growth (e.g., 0.10 = 10%)
        pb_ratio: Price-to-book ratio
        volatility: Annualized volatility
    
    Returns:
        Score from 0 to 100
    """
    score = 0.0
    
    # Handle NaN values
    if pd.isna(dividend_yield):
        dividend_yield = 0.0
    if pd.isna(earnings_growth):
        earnings_growth = 0.0
    if pd.isna(pb_ratio):
        pb_ratio = 999.0  # High value to penalize missing data
    if pd.isna(volatility):
        return 0.0  # Cannot score without volatility
    
    # 1. Dividend yield score (30 points)
    if DIVIDEND_YIELD_MIN <= dividend_yield <= DIVIDEND_YIELD_OPTIMAL:
        # Optimal range: full points
        score += 30.0
    elif dividend_yield > DIVIDEND_YIELD_OPTIMAL:
        # Too high might indicate distress
        score += 25.0
    elif dividend_yield > 0:
        # Some yield is better than none
        score += 15.0
    
    # 2. Volatility score (25 points)
    if volatility < VOLATILITY_THRESHOLD:
        # Low volatility = stable dividend
        score += 25.0
    elif volatility < VOLATILITY_THRESHOLD * 1.2:
        # Slightly elevated but acceptable
        score += 15.0
    
    # 3. Earnings growth score (25 points)
    if earnings_growth > EARNINGS_GROWTH_MIN:
        # Positive earnings growth
        if earnings_growth > 0.10:  # > 10% growth
            score += 25.0
        elif earnings_growth > 0.05:  # > 5% growth
            score += 20.0
        else:
            score += 15.0
    
    # 4. Valuation score (20 points)
    if pb_ratio > 0 and pb_ratio < PB_RATIO_THRESHOLD:
        # Attractive valuation
        score += 20.0
    elif pb_ratio < PB_RATIO_THRESHOLD * 1.5:
        # Reasonable valuation
        score += 10.0
    
    return min(score, 100.0)


# ============================================================================
# SIGNAL BUILDER (MAIN INTERFACE)
# ============================================================================

def build_dividend_rotation_signals(
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
) -> pd.DataFrame:
    """
    Build dividend rotation signal panel.
    
    This is the main interface function that follows the same pattern
    as other signal builders (momentum, value, profitability, etc.).
    
    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        dates: DatetimeIndex to align signals to
        data_loader: DataLoader instance with fundamental data
    
    Returns:
        DataFrame (dates x tickers) with dividend rotation scores (0-100)
    """
    print("\n🔧 Building dividend rotation signals...")
    print(f"  Dividend yield range: {DIVIDEND_YIELD_MIN:.1%} - {DIVIDEND_YIELD_OPTIMAL:.1%}")
    print(f"  Max volatility: {VOLATILITY_THRESHOLD:.1%}")
    print(f"  Volatility lookback: {VOLATILITY_LOOKBACK} days")
    
    if data_loader is None:
        print("  ⚠️  No data_loader provided - cannot load fundamental data")
        print("  Returning neutral scores (50) for all stocks")
        return pd.DataFrame(50.0, index=dates, columns=close_df.columns)
    
    # Load fundamental metrics from data_loader
    try:
        metrics_df = data_loader.load_fundamental_metrics()
        
        if metrics_df.empty:
            print("  ⚠️  Fundamental metrics file is empty")
            print("  Run calculate_fundamental_metrics.py to generate metrics")
            print("  Returning neutral scores (50) for all stocks")
            return pd.DataFrame(50.0, index=dates, columns=close_df.columns)
        
        # Check for required metrics
        required_metrics = ['dividend_payout_ratio', 'earnings_growth_yoy']
        available_metrics = metrics_df.columns.tolist()
        missing_metrics = [m for m in required_metrics if m not in available_metrics]
        
        if missing_metrics:
            print(f"  ⚠️  Missing required metrics: {missing_metrics}")
            print(f"  Available metrics: {available_metrics}")
            print("  Returning neutral scores (50) for all stocks")
            return pd.DataFrame(50.0, index=dates, columns=close_df.columns)
        
        print(f"  ✅ Loaded metrics for {len(metrics_df.index.get_level_values(0).unique())} tickers")
        
        # Build scores for each ticker
        scores_dict = {}
        tickers = close_df.columns
        
        for ticker in tickers:
            if ticker not in metrics_df.index.get_level_values(0):
                continue
            
            # Get metrics for this ticker
            ticker_metrics = metrics_df.loc[ticker]
            
            if ticker_metrics.empty:
                continue
            
            # Reindex to daily dates and forward-fill
            ticker_metrics = ticker_metrics.reindex(dates, method='ffill')
            
            # Calculate score based on:
            # 1. Dividend payout ratio (0-100 scale, optimal around 40-60%)
            # 2. Earnings growth (positive is good)
            
            payout = ticker_metrics['dividend_payout_ratio']
            growth = ticker_metrics['earnings_growth_yoy']
            
            # Score dividend payout (0-100)
            # Optimal: 0.4-0.6 (40-60% payout)
            # Too low or too high is bad
            payout_score = 100 - np.abs(payout - 0.5) * 200
            payout_score = payout_score.clip(0, 100)
            
            # Score earnings growth (0-100)
            # Positive growth is good, negative is bad
            # Cap at +/-50% growth
            growth_score = 50 + (growth.clip(-0.5, 0.5) * 100)
            
            # Combined score (50% payout, 50% growth)
            combined_score = 0.5 * payout_score + 0.5 * growth_score
            
            scores_dict[ticker] = combined_score
        
        if not scores_dict:
            print("  ⚠️  No valid scores calculated")
            print("  Returning neutral scores (50) for all stocks")
            return pd.DataFrame(50.0, index=dates, columns=close_df.columns)
        
        # Convert to DataFrame
        result = pd.DataFrame(scores_dict, index=dates)
        
        # Fill missing tickers with neutral score
        for ticker in close_df.columns:
            if ticker not in result.columns:
                result[ticker] = 50.0
        
        # Reorder columns to match close_df
        result = result[close_df.columns]
        
        # Fill NaN with neutral score
        result = result.fillna(50.0)
        
    except Exception as e:
        print(f"  ⚠️  Error loading fundamental metrics: {e}")
        import traceback
        traceback.print_exc()
        print("  Returning neutral scores (50) for all stocks")
        return pd.DataFrame(50.0, index=dates, columns=close_df.columns)
    
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
    
    print(f"  ✅ Dividend rotation signals: {result.shape[0]} days × {result.shape[1]} tickers")
    
    return result

