"""
Macro Hedge Signal Construction

Identifies fortress balance sheet stocks that outperform during macro uncertainty.

Logic:
- Low debt-to-equity (< 0.5) reduces bankruptcy risk
- High cash ratio (> 30%) provides financial flexibility
- Positive operating cash flow ensures self-sufficiency
- Sustainable dividend payout (< 60%) indicates financial health

This signal is designed for the Turkish market where macro shocks
(geopolitical risk, global recession, currency crises) are frequent.
Companies with strong balance sheets significantly outperform during stress.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# MACRO HEDGE PARAMETERS
# ============================================================================

DEBT_EQUITY_THRESHOLD = 0.5  # Maximum debt-to-equity ratio
DEBT_EQUITY_OPTIMAL = 0.3  # Optimal debt-to-equity (conservative)
CASH_RATIO_THRESHOLD = 0.3  # Minimum cash ratio (cash / current liabilities)
CURRENT_RATIO_THRESHOLD = 1.5  # Minimum current ratio (current assets / current liabilities)
DIVIDEND_PAYOUT_MAX = 0.6  # Maximum sustainable payout ratio


# ============================================================================
# MACRO HEDGE SIGNAL
# ============================================================================

def calculate_macro_hedge_scores(
    fundamentals_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate macro hedge signal scores.
    
    Combines:
    1. Balance sheet strength (low debt, high cash)
    2. Cash flow generation (positive operating CF)
    3. Dividend sustainability (payout < 60%)
    4. Earnings stability (positive growth)
    
    Args:
        fundamentals_df: DataFrame with fundamental metrics
                        Expected columns: debt_to_equity, cash_ratio, current_ratio,
                                        operating_cash_flow, free_cash_flow,
                                        dividend_payout_ratio, earnings_growth_yoy
                        Index: MultiIndex (Ticker, Date) or Date with Ticker columns
    
    Returns:
        DataFrame (dates x tickers) with macro hedge scores (0-100)
    """
    # Initialize scores DataFrame
    # We'll build this incrementally as we process fundamentals
    scores_dict = {}
    
    # Process fundamentals
    # Handle both MultiIndex (Ticker, Date) and regular Date index formats
    if isinstance(fundamentals_df.index, pd.MultiIndex):
        # MultiIndex format: (Ticker, Date)
        tickers = fundamentals_df.index.get_level_values(0).unique()
        
        for ticker in tickers:
            ticker_fundamentals = fundamentals_df.xs(ticker, level=0)
            ticker_scores = pd.Series(0.0, index=ticker_fundamentals.index)
            
            for date in ticker_fundamentals.index:
                fund_data = ticker_fundamentals.loc[date]
                
                # Extract metrics
                debt_equity = fund_data.get('debt_to_equity', np.nan)
                cash_ratio = fund_data.get('cash_ratio', np.nan)
                current_ratio = fund_data.get('current_ratio', np.nan)
                operating_cf = fund_data.get('operating_cash_flow', np.nan)
                free_cf = fund_data.get('free_cash_flow', np.nan)
                dividend_payout = fund_data.get('dividend_payout_ratio', np.nan)
                earnings_growth = fund_data.get('earnings_growth_yoy', np.nan)
                
                # Calculate score
                score = calculate_hedge_score(
                    debt_equity, cash_ratio, current_ratio,
                    operating_cf, free_cf, dividend_payout, earnings_growth
                )
                
                ticker_scores.loc[date] = score
            
            scores_dict[ticker] = ticker_scores
        
        # Combine into DataFrame
        scores = pd.DataFrame(scores_dict)
    
    else:
        # Regular Date index format - return empty for now
        # This would need adjustment based on actual data structure
        scores = pd.DataFrame()
    
    return scores


def calculate_hedge_score(
    debt_equity: float,
    cash_ratio: float,
    current_ratio: float,
    operating_cf: float,
    free_cf: float,
    dividend_payout: float,
    earnings_growth: float,
) -> float:
    """
    Calculate macro hedge score for a single stock.
    
    Scoring:
    - Low debt-to-equity (< 0.5): 20 points
    - High cash ratio (> 30%): 15 points
    - Strong current ratio (> 1.5): 10 points
    - Positive operating cash flow: 15 points
    - Positive free cash flow: 10 points
    - Sustainable dividend payout (< 60%): 15 points
    - Positive earnings growth: 15 points
    
    Args:
        debt_equity: Debt-to-equity ratio
        cash_ratio: Cash / current liabilities
        current_ratio: Current assets / current liabilities
        operating_cf: Operating cash flow
        free_cf: Free cash flow
        dividend_payout: Dividend payout ratio
        earnings_growth: YoY earnings growth
    
    Returns:
        Score from 0 to 100
    """
    score = 50.0  # Start with neutral base
    
    # 1. Debt-to-equity score (20 points)
    if not pd.isna(debt_equity):
        if debt_equity < DEBT_EQUITY_OPTIMAL:
            # Very conservative leverage
            score += 20.0
        elif debt_equity < DEBT_EQUITY_THRESHOLD:
            # Acceptable leverage
            score += 15.0
        elif debt_equity < 1.0:
            # Moderate leverage
            score += 5.0
        else:
            # High leverage - penalize
            score -= 10.0
    
    # 2. Cash ratio score (15 points)
    if not pd.isna(cash_ratio):
        if cash_ratio > CASH_RATIO_THRESHOLD:
            # Strong cash position
            score += 15.0
        elif cash_ratio > CASH_RATIO_THRESHOLD * 0.7:
            # Adequate cash
            score += 10.0
        elif cash_ratio > 0:
            # Some cash
            score += 5.0
    
    # 3. Current ratio score (10 points)
    if not pd.isna(current_ratio):
        if current_ratio > CURRENT_RATIO_THRESHOLD:
            # Healthy liquidity
            score += 10.0
        elif current_ratio > 1.0:
            # Adequate liquidity
            score += 5.0
    
    # 4. Operating cash flow score (15 points)
    if not pd.isna(operating_cf):
        if operating_cf > 0:
            # Positive cash generation
            score += 15.0
        else:
            # Negative cash flow - penalize
            score -= 10.0
    
    # 5. Free cash flow score (10 points)
    if not pd.isna(free_cf):
        if free_cf > 0:
            # Positive free cash flow
            score += 10.0
    
    # 6. Dividend sustainability score (15 points)
    if not pd.isna(dividend_payout):
        if 0 < dividend_payout < DIVIDEND_PAYOUT_MAX:
            # Sustainable payout
            score += 15.0
        elif dividend_payout < 0.8:
            # Slightly elevated but manageable
            score += 10.0
        elif dividend_payout > 1.0:
            # Unsustainable - penalize
            score -= 10.0
    
    # 7. Earnings growth score (15 points)
    if not pd.isna(earnings_growth):
        if earnings_growth > 0.10:  # > 10% growth
            score += 15.0
        elif earnings_growth > 0:
            score += 10.0
        else:
            # Negative growth
            score += 0.0
    
    return np.clip(score, 0.0, 100.0)


# ============================================================================
# SIGNAL BUILDER (MAIN INTERFACE)
# ============================================================================

def build_macro_hedge_signals(
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
) -> pd.DataFrame:
    """
    Build macro hedge signal panel.
    
    This is the main interface function that follows the same pattern
    as other signal builders (momentum, value, profitability, etc.).
    
    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        dates: DatetimeIndex to align signals to
        data_loader: DataLoader instance with fundamental data
    
    Returns:
        DataFrame (dates x tickers) with macro hedge scores (0-100)
    """
    print("\n🔧 Building macro hedge signals...")
    print(f"  Max debt-to-equity: {DEBT_EQUITY_THRESHOLD:.1%}")
    print(f"  Min cash ratio: {CASH_RATIO_THRESHOLD:.1%}")
    print(f"  Min current ratio: {CURRENT_RATIO_THRESHOLD:.1f}")
    
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
        required_metrics = ['debt_to_equity', 'cash_ratio', 'current_ratio', 'operating_cash_flow']
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
            
            # Calculate score based on fortress balance sheet criteria:
            # 1. Low debt-to-equity (< 0.5 is excellent)
            # 2. High cash ratio (> 0.3 is excellent)
            # 3. High current ratio (> 1.5 is excellent)
            # 4. Positive operating cash flow
            
            de_ratio = ticker_metrics['debt_to_equity']
            cash_ratio = ticker_metrics['cash_ratio']
            current_ratio = ticker_metrics['current_ratio']
            ocf = ticker_metrics['operating_cash_flow']
            
            # Score debt-to-equity (0-100, lower is better)
            # Excellent: < 0.5, Good: 0.5-1.0, Bad: > 1.0
            de_score = 100 - (de_ratio.clip(0, 2) * 50)
            de_score = de_score.clip(0, 100)
            
            # Score cash ratio (0-100, higher is better)
            # Excellent: > 0.3, Good: 0.1-0.3, Bad: < 0.1
            cash_score = (cash_ratio.clip(0, 0.5) / 0.5) * 100
            cash_score = cash_score.clip(0, 100)
            
            # Score current ratio (0-100, higher is better)
            # Excellent: > 1.5, Good: 1.0-1.5, Bad: < 1.0
            current_score = ((current_ratio.clip(0, 3) - 1) / 2) * 100
            current_score = current_score.clip(0, 100)
            
            # Score operating cash flow (0-100, positive is good)
            # Normalize by absolute value (capped at 1B)
            ocf_normalized = ocf / 1e9  # Convert to billions
            ocf_score = 50 + (ocf_normalized.clip(-1, 1) * 50)
            ocf_score = ocf_score.clip(0, 100)
            
            # Combined score (weighted average)
            # Debt-to-equity: 30%
            # Cash ratio: 25%
            # Current ratio: 25%
            # Operating cash flow: 20%
            combined_score = (
                0.30 * de_score +
                0.25 * cash_score +
                0.25 * current_score +
                0.20 * ocf_score
            )
            
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
            
            # Show top fortress stocks
            top_5 = latest.nlargest(5)
            if len(top_5) > 0:
                print(f"  Top 5 fortress stocks: {', '.join(top_5.index.tolist())}")
    
    print(f"  ✅ Macro hedge signals: {result.shape[0]} days × {result.shape[1]} tickers")
    
    return result

