"""
F-Score + Short-Term Reversal Signal Construction

Combines Piotroski F-Score with short-term reversal:
- F-Score: 9-point fundamental quality score (higher = better)
- Reversal: Monthly losers outperform winners

Strategy: Long losers with high F-Score (>=7), avoid winners with low F-Score (<=3)

Based on Quantpedia strategy: Combining Fundamental F-Score and Equity Short-Term Reversals
Signal: Composite of F-Score rank and negative monthly return
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.utils import (
    coerce_quarter_cols,
    sum_ttm,
    get_consolidated_sheet,
    pick_row_from_sheet,
    apply_lag,
)


# ============================================================================
# F-SCORE PARAMETERS
# ============================================================================

REVERSAL_LOOKBACK = 21  # 1 month for reversal
FSCORE_HIGH_THRESHOLD = 7  # F-Score >= 7 is "strong"
FSCORE_LOW_THRESHOLD = 3   # F-Score <= 3 is "weak"

# Fundamental data keys
INCOME_SHEET = "Gelir Tablosu (Çeyreklik)"
BALANCE_SHEET = "Bilanço"
CASH_FLOW_SHEET = "Nakit Akış (Çeyreklik)"

NET_INCOME_KEYS = (
    "Dönem Net Karı veya Zararı",
    "Dönem Karı (Zararı)",
    "Net Dönem Karı (Zararı)",
)
OPERATING_CF_KEYS = (
    "İşletme Faaliyetlerinden Nakit Akışları",
    "İşletme Faaliyetlerinden Kaynaklanan Net Nakit",
)
TOTAL_ASSETS_KEYS = (
    "Toplam Varlıklar",
    "Toplam Aktifler",
)
LONG_TERM_DEBT_KEYS = (
    "Uzun Vadeli Borçlanmalar",
    "Uzun Vadeli Yükümlülükler",
)
CURRENT_ASSETS_KEYS = (
    "Dönen Varlıklar",
)
CURRENT_LIABILITIES_KEYS = (
    "Kısa Vadeli Yükümlülükler",
)
REVENUE_KEYS = (
    "Satış Gelirleri",
    "Hasılat",
    "Net Satışlar",
)
GROSS_PROFIT_KEYS = (
    "Brüt Kar (Zarar)",
    "Brüt Kar",
)
SHARES_OUTSTANDING_KEYS = (
    "Ödenmiş Sermaye",
)


# ============================================================================
# F-SCORE CALCULATION
# ============================================================================

def calculate_fscore_for_ticker(
    ticker: str,
    data_loader,
    fundamentals_parquet: pd.DataFrame = None,
) -> pd.Series:
    """
    Calculate Piotroski F-Score (0-9) for a ticker.

    F-Score components:
    1. ROA positive (+1)
    2. Operating Cash Flow positive (+1)
    3. ROA change positive (+1)
    4. Accruals (CFO > Net Income) (+1)
    5. Leverage decrease (+1)
    6. Liquidity (current ratio) increase (+1)
    7. No equity dilution (+1)
    8. Gross margin increase (+1)
    9. Asset turnover increase (+1)

    Returns:
        pd.Series: F-Score indexed by quarter
    """
    if fundamentals_parquet is None:
        return pd.Series(dtype=float)

    inc = get_consolidated_sheet(fundamentals_parquet, ticker, INCOME_SHEET)
    bs = get_consolidated_sheet(fundamentals_parquet, ticker, BALANCE_SHEET)
    cf = get_consolidated_sheet(fundamentals_parquet, ticker, CASH_FLOW_SHEET)

    if inc.empty and bs.empty:
        return pd.Series(dtype=float)

    # Extract rows
    net_income_row = pick_row_from_sheet(inc, NET_INCOME_KEYS)
    opcf_row = pick_row_from_sheet(cf, OPERATING_CF_KEYS) if not cf.empty else None
    assets_row = pick_row_from_sheet(bs, TOTAL_ASSETS_KEYS)
    lt_debt_row = pick_row_from_sheet(bs, LONG_TERM_DEBT_KEYS)
    curr_assets_row = pick_row_from_sheet(bs, CURRENT_ASSETS_KEYS)
    curr_liab_row = pick_row_from_sheet(bs, CURRENT_LIABILITIES_KEYS)
    revenue_row = pick_row_from_sheet(inc, REVENUE_KEYS)
    gross_profit_row = pick_row_from_sheet(inc, GROSS_PROFIT_KEYS)
    shares_row = pick_row_from_sheet(bs, SHARES_OUTSTANDING_KEYS)

    # Coerce to series
    net_income = coerce_quarter_cols(net_income_row) if net_income_row is not None else pd.Series(dtype=float)
    opcf = coerce_quarter_cols(opcf_row) if opcf_row is not None else pd.Series(dtype=float)
    assets = coerce_quarter_cols(assets_row) if assets_row is not None else pd.Series(dtype=float)
    lt_debt = coerce_quarter_cols(lt_debt_row) if lt_debt_row is not None else pd.Series(dtype=float)
    curr_assets = coerce_quarter_cols(curr_assets_row) if curr_assets_row is not None else pd.Series(dtype=float)
    curr_liab = coerce_quarter_cols(curr_liab_row) if curr_liab_row is not None else pd.Series(dtype=float)
    revenue = coerce_quarter_cols(revenue_row) if revenue_row is not None else pd.Series(dtype=float)
    gross_profit = coerce_quarter_cols(gross_profit_row) if gross_profit_row is not None else pd.Series(dtype=float)
    shares = coerce_quarter_cols(shares_row) if shares_row is not None else pd.Series(dtype=float)

    # Calculate TTM for flow items
    net_income_ttm = sum_ttm(net_income)
    opcf_ttm = sum_ttm(opcf)
    revenue_ttm = sum_ttm(revenue)
    gross_profit_ttm = sum_ttm(gross_profit)

    # Align all series to common index
    common_idx = net_income_ttm.index.intersection(assets.index)
    if len(common_idx) < 2:
        return pd.Series(dtype=float)

    # Calculate ratios
    roa = net_income_ttm / assets.reindex(net_income_ttm.index).ffill()
    leverage = lt_debt / assets.reindex(lt_debt.index).ffill() if not lt_debt.empty else pd.Series(0, index=assets.index)
    current_ratio = curr_assets / curr_liab.reindex(curr_assets.index).ffill() if not curr_liab.empty else pd.Series(1, index=curr_assets.index)
    gross_margin = gross_profit_ttm / revenue_ttm.reindex(gross_profit_ttm.index).ffill() if not revenue_ttm.empty else pd.Series(dtype=float)
    asset_turnover = revenue_ttm / assets.reindex(revenue_ttm.index).ffill() if not revenue_ttm.empty else pd.Series(dtype=float)

    # Calculate F-Score components
    fscore = pd.Series(0, index=common_idx, dtype=float)

    # 1. ROA positive
    if not roa.empty:
        fscore = fscore.add((roa.reindex(common_idx) > 0).astype(int), fill_value=0)

    # 2. CFO positive
    if not opcf_ttm.empty:
        fscore = fscore.add((opcf_ttm.reindex(common_idx) > 0).astype(int), fill_value=0)

    # 3. ROA change positive (YoY)
    if not roa.empty:
        roa_change = roa.diff(4)  # 4 quarters = YoY
        fscore = fscore.add((roa_change.reindex(common_idx) > 0).astype(int), fill_value=0)

    # 4. Accruals: CFO > Net Income
    if not opcf_ttm.empty and not net_income_ttm.empty:
        accrual_quality = opcf_ttm > net_income_ttm.reindex(opcf_ttm.index).ffill()
        fscore = fscore.add(
            accrual_quality.reindex(common_idx).fillna(False).astype(int),
            fill_value=0,
        )

    # 5. Leverage decrease (YoY)
    if not leverage.empty:
        lev_change = leverage.diff(4)
        fscore = fscore.add((lev_change.reindex(common_idx) < 0).astype(int), fill_value=0)

    # 6. Liquidity increase (YoY)
    if not current_ratio.empty:
        liq_change = current_ratio.diff(4)
        fscore = fscore.add((liq_change.reindex(common_idx) > 0).astype(int), fill_value=0)

    # 7. No equity dilution (YoY)
    if not shares.empty:
        shares_change = shares.diff(4)
        fscore = fscore.add((shares_change.reindex(common_idx) <= 0).astype(int), fill_value=0)

    # 8. Gross margin increase (YoY)
    if not gross_margin.empty:
        gm_change = gross_margin.diff(4)
        fscore = fscore.add((gm_change.reindex(common_idx) > 0).astype(int), fill_value=0)

    # 9. Asset turnover increase (YoY)
    if not asset_turnover.empty:
        at_change = asset_turnover.diff(4)
        fscore = fscore.add((at_change.reindex(common_idx) > 0).astype(int), fill_value=0)

    return fscore


# ============================================================================
# SIGNAL BUILDER
# ============================================================================

def build_fscore_reversal_signals(
    fundamentals: Dict,
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader,
) -> pd.DataFrame:
    """
    Build F-Score + Reversal combined signal panel.

    Strategy: Long losers with high F-Score, avoid winners with low F-Score
    Signal = F-Score rank * (-monthly return rank)

    Args:
        fundamentals: Dict of fundamental data by ticker
        close_df: DataFrame of close prices
        dates: DatetimeIndex to align signals to
        data_loader: DataLoader instance

    Returns:
        DataFrame (dates x tickers) with combined scores
    """
    print("\n🔧 Building F-Score + Reversal signals...")
    print(f"  Reversal Lookback: {REVERSAL_LOOKBACK} days")
    print(f"  F-Score High Threshold: >= {FSCORE_HIGH_THRESHOLD}")
    print(f"  F-Score Low Threshold: <= {FSCORE_LOW_THRESHOLD}")

    fundamentals_parquet = data_loader.load_fundamentals_parquet()

    # Calculate F-Scores for all tickers
    fscore_panel = {}
    count = 0

    for ticker in fundamentals.keys():
        if ticker not in close_df.columns:
            continue

        fscore = calculate_fscore_for_ticker(ticker, data_loader, fundamentals_parquet)
        if not fscore.empty:
            # Apply lag and align to daily dates
            lagged = apply_lag(fscore, dates)
            if not lagged.empty:
                fscore_panel[ticker] = lagged

        count += 1
        if count % 50 == 0:
            print(f"  Processed {count} tickers...")

    # Build F-Score DataFrame
    fscore_df = pd.DataFrame(fscore_panel, index=dates)

    # Calculate monthly reversal (negative of monthly return)
    monthly_return = close_df / close_df.shift(REVERSAL_LOOKBACK) - 1.0
    reversal_score = -monthly_return  # Negative: losers get high scores

    # Align to dates
    reversal_df = reversal_score.reindex(dates)

    # Cross-sectional z-score normalization
    print("  Normalizing F-Score and reversal (z-score per date)...")

    def zscore_df(df):
        row_mean = df.mean(axis=1)
        row_std = df.std(axis=1).replace(0, np.nan)
        return df.sub(row_mean, axis=0).div(row_std, axis=0)

    fscore_z = zscore_df(fscore_df)
    reversal_z = zscore_df(reversal_df)

    # Combined score: average of F-Score and reversal z-scores
    # High F-Score + High Reversal (loser) = Best
    print("  Combining F-Score and reversal signals...")
    combined = pd.DataFrame(index=dates, columns=close_df.columns, dtype=float)

    for ticker in close_df.columns:
        if ticker in fscore_z.columns and ticker in reversal_z.columns:
            combined[ticker] = (fscore_z[ticker] + reversal_z[ticker]) / 2
        elif ticker in reversal_z.columns:
            # No F-Score data, use reversal only with penalty
            combined[ticker] = reversal_z[ticker] * 0.5

    result = combined.replace([np.inf, -np.inf], np.nan)
    print(f"  ✅ F-Score + Reversal signals: {result.shape[0]} days × {result.shape[1]} tickers")

    return result
