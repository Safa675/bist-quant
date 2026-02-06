"""
Currency Rotation Signal Construction

Exploits USD/TRY mean reversion to identify sector rotation opportunities.

Logic:
- USD/TRY exhibits mean-reverting behavior around its moving average
- When USD/TRY > MA + threshold: Export-heavy stocks benefit (currency tailwind)
- When USD/TRY < MA - threshold: Domestic consumption stocks benefit
- Combines currency regime with stock-level momentum for confirmation

This signal is designed for the Turkish market where USD/TRY volatility
drives significant sector rotations between exporters and domestic plays.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# CURRENCY ROTATION PARAMETERS
# ============================================================================

USDTRY_MA_LOOKBACK = 60  # 60-day moving average for USD/TRY
USDTRY_STD_THRESHOLD = 1.5  # Standard deviation threshold for mean reversion
STOCK_MOMENTUM_LOOKBACK = 60  # Stock momentum confirmation period
MIN_MOMENTUM_THRESHOLD = 0.02  # Minimum 2% momentum for confirmation


# ============================================================================
# CURRENCY REGIME DETECTION
# ============================================================================

def calculate_usdtry_regime(
    usdtry_df: pd.DataFrame,
    lookback: int = USDTRY_MA_LOOKBACK,
    std_threshold: float = USDTRY_STD_THRESHOLD,
) -> pd.DataFrame:
    """
    Calculate USD/TRY regime: oversold TRY, neutral, or overbought TRY.
    
    Args:
        usdtry_df: DataFrame with USD/TRY rates (Date index, 'Close' column)
        lookback: Days for moving average calculation
        std_threshold: Standard deviation threshold for regime classification
    
    Returns:
        DataFrame with columns:
        - 'ma': Moving average
        - 'std': Standard deviation
        - 'z_score': Z-score (current - ma) / std
        - 'regime': 'weak_try', 'neutral', or 'strong_try'
    """
    result = pd.DataFrame(index=usdtry_df.index)
    
    # Calculate moving average and standard deviation
    result['ma'] = usdtry_df['Close'].rolling(lookback, min_periods=int(lookback * 0.5)).mean()
    result['std'] = usdtry_df['Close'].rolling(lookback, min_periods=int(lookback * 0.5)).std()
    
    # Calculate z-score
    result['z_score'] = (usdtry_df['Close'] - result['ma']) / result['std']
    
    # Classify regime
    # Positive z-score = USD/TRY above average = Weak TRY (good for exporters)
    # Negative z-score = USD/TRY below average = Strong TRY (good for domestic)
    result['regime'] = 'neutral'
    result.loc[result['z_score'] > std_threshold, 'regime'] = 'weak_try'
    result.loc[result['z_score'] < -std_threshold, 'regime'] = 'strong_try'
    
    return result


def calculate_stock_momentum(
    close_df: pd.DataFrame,
    lookback: int = STOCK_MOMENTUM_LOOKBACK,
) -> pd.DataFrame:
    """
    Calculate simple price momentum for stocks.
    
    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        lookback: Lookback period in trading days
    
    Returns:
        DataFrame of momentum scores (Date x Ticker)
    """
    # Calculate return over lookback period
    momentum = close_df.pct_change(lookback)
    
    return momentum


# ============================================================================
# CURRENCY ROTATION SIGNAL
# ============================================================================

def calculate_currency_rotation_scores(
    close_df: pd.DataFrame,
    usdtry_df: pd.DataFrame,
    sector_mapping: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Calculate currency rotation signal scores.
    
    Combines USD/TRY regime with stock momentum to generate rotation signals:
    - Weak TRY regime + Positive stock momentum = High score (export plays)
    - Strong TRY regime + Negative stock momentum = Low score (avoid exports)
    
    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        usdtry_df: DataFrame with USD/TRY rates (Date index, 'Close' column)
        sector_mapping: Optional dict mapping tickers to sectors
                       (e.g., {'THYAO': 'export', 'BIMAS': 'domestic'})
    
    Returns:
        DataFrame (dates x tickers) with currency rotation scores
        Higher score = Better opportunity given current currency regime
    """
    # Calculate USD/TRY regime
    currency_regime = calculate_usdtry_regime(usdtry_df)
    
    # Calculate stock momentum
    stock_momentum = calculate_stock_momentum(close_df)
    
    # Align dates
    common_dates = close_df.index.intersection(currency_regime.index)
    currency_regime = currency_regime.loc[common_dates]
    stock_momentum = stock_momentum.loc[common_dates]
    
    # Initialize scores DataFrame
    scores = pd.DataFrame(0.0, index=common_dates, columns=close_df.columns)
    
    # If no sector mapping provided, use heuristic based on momentum
    # In reality, you'd want to provide actual sector classifications
    if sector_mapping is None:
        # Heuristic: stocks with consistently positive momentum in weak TRY are likely exporters
        # This is a simplification - ideally you'd have actual sector data
        sector_mapping = {}
    
    # Calculate scores for each date
    for date in common_dates:
        regime = currency_regime.loc[date, 'regime']
        z_score = currency_regime.loc[date, 'z_score']
        
        if pd.isna(regime) or pd.isna(z_score):
            continue
        
        # Get stock momentums for this date
        momentums = stock_momentum.loc[date]
        
        for ticker in close_df.columns:
            if ticker not in momentums or pd.isna(momentums[ticker]):
                continue
            
            momentum = momentums[ticker]
            
            # Determine if stock is likely an exporter (simplified heuristic)
            # In production, use actual sector classification
            is_exporter = sector_mapping.get(ticker, 'unknown')
            
            # Score calculation based on regime and momentum
            if regime == 'weak_try':  # USD strong, TRY weak
                # Favor stocks with positive momentum (likely exporters benefiting)
                if momentum > MIN_MOMENTUM_THRESHOLD:
                    scores.loc[date, ticker] = 1.0 + min(z_score / 2, 0.5)  # Boost by z-score
                elif momentum > 0:
                    scores.loc[date, ticker] = 0.6
                else:
                    scores.loc[date, ticker] = 0.3  # Negative momentum in favorable regime
            
            elif regime == 'strong_try':  # USD weak, TRY strong
                # Favor domestic stocks, penalize exporters
                if momentum < -MIN_MOMENTUM_THRESHOLD:
                    scores.loc[date, ticker] = 0.2  # Avoid stocks with negative momentum
                elif momentum < 0:
                    scores.loc[date, ticker] = 0.4
                else:
                    scores.loc[date, ticker] = 0.7  # Positive momentum in strong TRY
            
            else:  # Neutral regime
                # Use momentum as primary signal
                if momentum > MIN_MOMENTUM_THRESHOLD:
                    scores.loc[date, ticker] = 0.6
                elif momentum > 0:
                    scores.loc[date, ticker] = 0.5
                else:
                    scores.loc[date, ticker] = 0.4
    
    return scores


# ============================================================================
# SIGNAL BUILDER (MAIN INTERFACE)
# ============================================================================

def build_currency_rotation_signals(
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
) -> pd.DataFrame:
    """
    Build currency rotation signal panel.
    
    This is the main interface function that follows the same pattern
    as other signal builders (momentum, value, profitability, etc.).
    
    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        dates: DatetimeIndex to align signals to
        data_loader: DataLoader instance with USD/TRY data
    
    Returns:
        DataFrame (dates x tickers) with currency rotation scores
    """
    print("\n🔧 Building currency rotation signals...")
    print(f"  USD/TRY MA: {USDTRY_MA_LOOKBACK} days")
    print(f"  Z-score threshold: ±{USDTRY_STD_THRESHOLD}")
    print(f"  Stock momentum: {STOCK_MOMENTUM_LOOKBACK} days")
    
    if data_loader is None:
        print("  ⚠️  No data_loader provided - cannot load USD/TRY data")
        print("  Returning neutral scores (0.5) for all stocks")
        return pd.DataFrame(0.5, index=dates, columns=close_df.columns)
    
    # Load USD/TRY data from data_loader
    try:
        usdtry_df = data_loader.load_usdtry()
        
        if usdtry_df.empty:
            print("  ⚠️  USD/TRY data is empty")
            print("  Returning neutral scores (0.5) for all stocks")
            return pd.DataFrame(0.5, index=dates, columns=close_df.columns)
        
        # Ensure USD/TRY data has 'Close' column
        if 'Close' not in usdtry_df.columns:
            print(f"  ⚠️  USD/TRY data missing 'Close' column. Available: {usdtry_df.columns.tolist()}")
            return pd.DataFrame(0.5, index=dates, columns=close_df.columns)
        
    except Exception as e:
        print(f"  ⚠️  Error loading USD/TRY data: {e}")
        print("  Returning neutral scores (0.5) for all stocks")
        return pd.DataFrame(0.5, index=dates, columns=close_df.columns)
    
    # Calculate currency rotation scores
    rotation_scores = calculate_currency_rotation_scores(close_df, usdtry_df)
    
    # Align to requested dates
    result = rotation_scores.reindex(dates)
    
    # Fill NaN with neutral score
    result = result.fillna(0.5)
    
    # Summary stats
    valid_scores = result.dropna(how='all')
    if not valid_scores.empty:
        latest = valid_scores.iloc[-1].dropna()
        if len(latest) > 0:
            print(f"  Latest scores - Mean: {latest.mean():.2f}, Std: {latest.std():.2f}")
            print(f"  Latest scores - Min: {latest.min():.2f}, Max: {latest.max():.2f}")
            
            # Show current USD/TRY regime
            if not usdtry_df.empty:
                latest_date = valid_scores.index[-1]
                if latest_date in usdtry_df.index:
                    regime_info = calculate_usdtry_regime(usdtry_df)
                    if latest_date in regime_info.index:
                        regime = regime_info.loc[latest_date, 'regime']
                        z_score = regime_info.loc[latest_date, 'z_score']
                        print(f"  Current USD/TRY regime: {regime} (z-score: {z_score:.2f})")
    
    print(f"  ✅ Currency rotation signals: {result.shape[0]} days × {result.shape[1]} tickers")
    
    return result
