"""
Small Cap Signal Construction

Calculates composite small cap scores (smaller = better) based on:
1. Market Cap = log(Price × Shares) - INVERTED
2. Enterprise Value = log(Market Cap + Debt - Cash) - INVERTED
3. Revenue Size = log(Sales TTM) - INVERTED
4. Liquidity-Adjusted Size = log(Market Cap × ADV ratio) - INVERTED

This is the pure "size premium" factor - favors smaller companies.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.utils import (
    pick_row,
    coerce_quarter_cols,
    sum_ttm,
    get_consolidated_sheet,
    pick_row_from_sheet,
    apply_lag,
)


# Fundamental data keys
INCOME_SHEET = "Gelir Tablosu (Çeyreklik)"
BALANCE_SHEET = "Bilanço"

REVENUE_KEYS = (
    "Satış Gelirleri",
    "Toplam Hasılat",
    "Hasılat",
    "Net Satışlar",
)
TOTAL_DEBT_KEYS = (
    "Finansal Borçlar",
)
CASH_KEYS = (
    "Nakit ve Nakit Benzerleri",
)


def calculate_size_metrics_for_ticker(
    xlsx_path: Path | None,
    ticker: str,
    data_loader,
    fundamentals_parquet: pd.DataFrame | None = None,
) -> Dict:
    """Calculate size metrics for a single ticker"""
    if fundamentals_parquet is not None:
        inc = get_consolidated_sheet(fundamentals_parquet, ticker, INCOME_SHEET)
        bs = get_consolidated_sheet(fundamentals_parquet, ticker, BALANCE_SHEET)
        if inc.empty and bs.empty:
            return {}
        rev_row = pick_row_from_sheet(inc, REVENUE_KEYS)
        debt_row = pick_row_from_sheet(bs, TOTAL_DEBT_KEYS)
        cash_row = pick_row_from_sheet(bs, CASH_KEYS)
    else:
        if xlsx_path is None:
            return {}
        try:
            inc = pd.read_excel(xlsx_path, sheet_name=INCOME_SHEET)
            bs = pd.read_excel(xlsx_path, sheet_name=BALANCE_SHEET)
        except Exception:
            return {}
        
        rev_row = pick_row(inc, REVENUE_KEYS)
        debt_row = pick_row(bs, TOTAL_DEBT_KEYS)
        cash_row = pick_row(bs, CASH_KEYS)
    
    rev = coerce_quarter_cols(rev_row) if rev_row is not None else pd.Series(dtype=float)
    debt = coerce_quarter_cols(debt_row) if debt_row is not None else pd.Series(dtype=float)
    cash = coerce_quarter_cols(cash_row) if cash_row is not None else pd.Series(dtype=float)
    
    rev_ttm = sum_ttm(rev)
    shares = data_loader.load_shares_outstanding(ticker)
    
    return {
        'revenue_ttm': rev_ttm,
        'total_debt': debt,
        'cash': cash,
        'shares_outstanding': shares,
    }


def build_small_cap_signals(
    fundamentals: Dict,
    close_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader,
) -> pd.DataFrame:
    """
    Build composite small cap signal panel (smaller = higher score)

    Returns:
        DataFrame (dates x tickers) with inverted size scores (small caps score higher)
    """
    print("\n🔧 Building small cap signals...")
    
    panels = {
        'market_cap': {},
        'enterprise_value': {},
        'revenue_size': {},
        'liquidity_adj_size': {},
    }
    
    fundamentals_parquet = data_loader.load_fundamentals_parquet()
    
    # OPTIMIZATION: Pre-compute volume percentiles for all dates (do once, not per ticker!)
    print("  Pre-computing volume percentiles...")
    volume_percentiles = {}
    for date in volume_df.index:
        date_volumes = volume_df.loc[date].dropna()
        if len(date_volumes) > 1:
            # Store the sorted volumes for quick percentile lookup
            volume_percentiles[date] = date_volumes.sort_values()
    print(f"  Cached percentiles for {len(volume_percentiles)} dates")

    count = 0
    for ticker, fund_data in fundamentals.items():
        if ticker not in close_df.columns:
            continue
        
        xlsx_path = fund_data['path']
        metrics = calculate_size_metrics_for_ticker(
            xlsx_path,
            ticker,
            data_loader,
            fundamentals_parquet,
        )
        
        if not metrics:
            continue
        
        price_series = close_df[ticker].dropna()
        shares = metrics.get('shares_outstanding', pd.Series(dtype=float))
        
        # Calculate market cap - SKIP if no shares data (Bug #1 fix)
        if shares.empty:
            # Cannot calculate proper market cap without shares data
            # Skip this ticker to avoid using price as market cap
            continue
        
        # Remove duplicates before reindexing
        shares = shares[~shares.index.duplicated(keep='last')]
        shares_aligned = shares.reindex(price_series.index).ffill()
        market_cap = price_series * shares_aligned

        
        # 1. Market Cap Size (log-transformed) - No lag needed (uses daily price × shares)
        log_mc = np.log(market_cap.replace(0, np.nan).dropna())
        if not log_mc.empty:
            panels['market_cap'][ticker] = log_mc.reindex(dates, method='ffill')
        
        # 2. Enterprise Value - Need to lag debt/cash before using in EV calculation
        # Debt and cash are from quarterly reports with reporting delay
        debt = metrics.get('total_debt', pd.Series(dtype=float))
        cash = metrics.get('cash', pd.Series(dtype=float))
        
        # Apply reporting lag to fundamental components FIRST
        if not debt.empty:
            debt_lagged = apply_lag(debt, dates)
        else:
            debt_lagged = pd.Series(dtype=float, index=dates)
        
        if not cash.empty:
            cash_lagged = apply_lag(cash, dates)
        else:
            cash_lagged = pd.Series(dtype=float, index=dates)
        
        # Now compute EV using lagged fundamentals + daily market cap
        mc_aligned = log_mc.reindex(dates, method='ffill')
        if not mc_aligned.empty:
            mc_for_ev = market_cap.reindex(dates, method='ffill')
            ev = mc_for_ev + debt_lagged.fillna(0) - cash_lagged.fillna(0)
            log_ev = np.log(ev.replace(0, np.nan).dropna())
            if not log_ev.empty:
                panels['enterprise_value'][ticker] = log_ev
        
        # 3. Revenue Size - Apply reporting lag
        rev_ttm = metrics.get('revenue_ttm', pd.Series(dtype=float))
        if not rev_ttm.empty:
            log_rev = np.log(rev_ttm.replace(0, np.nan).dropna())
            if not log_rev.empty:
                lagged_log_rev = apply_lag(log_rev, dates)
                if not lagged_log_rev.empty:
                    panels['revenue_size'][ticker] = lagged_log_rev
        
        # 4. Liquidity-Adjusted Size (OPTIMIZED - uses pre-computed percentiles)
        if ticker in volume_df.columns:
            adv = volume_df[ticker]
            adv_aligned = adv.reindex(market_cap.index).dropna()
            
            if not adv_aligned.empty:
                liquidity_adj_series = pd.Series(dtype=float, index=adv_aligned.index)
                
                for date in adv_aligned.index:
                    if date in volume_percentiles:
                        sorted_volumes = volume_percentiles[date]
                        ticker_volume = adv_aligned.loc[date]
                        
                        # Fast percentile lookup using searchsorted
                        percentile = max((sorted_volumes < ticker_volume).sum() / len(sorted_volumes), 0.01)
                        adj_size = market_cap.loc[date] / percentile
                        
                        if adj_size > 0:
                            liquidity_adj_series.loc[date] = np.log(adj_size)
                
                liquidity_adj_series = liquidity_adj_series.dropna()
                if not liquidity_adj_series.empty:
                    panels['liquidity_adj_size'][ticker] = liquidity_adj_series.reindex(dates, method='ffill')
        
        count += 1
        if count % 50 == 0:
            print(f"  Processed {count} tickers...")
    
    # Combine and INVERT (smaller = higher score)
    print("  Combining and inverting size scores...")
    composite_panel = {}
    
    for ticker in close_df.columns:
        scores_list = []
        for panel_name, panel_dict in panels.items():
            if ticker in panel_dict:
                scores = panel_dict[ticker]
                if not scores.empty:
                    scores_list.append(scores)
        
        if scores_list:
            # Average size metrics
            avg_size = pd.concat(scores_list, axis=1).mean(axis=1)
            # Invert: smaller = better (multiply by -1 so smaller values become larger)
            composite_panel[ticker] = -avg_size
    
    result = pd.DataFrame(composite_panel, index=dates)
    print(f"  ✅ Size signals: {result.shape[0]} days × {result.shape[1]} tickers")
    return result
