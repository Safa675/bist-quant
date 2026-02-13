"""
Earnings Quality Signal Construction

Multi-factor quality metric combining:
1. Accruals (low = better quality)
2. ROE (high = better)
3. Cash Flow to Assets (high = better)
4. Debt to Assets (low = better)

Based on Quantpedia strategy: Earnings Quality Factor
Signal: Composite quality score (percentile sum across all 4 metrics)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
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


# ============================================================================
# FUNDAMENTAL DATA KEYS
# ============================================================================

INCOME_SHEET = "Gelir Tablosu (Çeyreklik)"
BALANCE_SHEET = "Bilanço"
CASH_FLOW_SHEET = "Nakit Akış (Çeyreklik)"

# Net Income
NET_INCOME_KEYS = (
    "Dönem Net Karı veya Zararı",
    "Dönem Karı (Zararı)",
    "Net Dönem Karı (Zararı)",
    "Dönem Net Kar/Zararı",
)

# Operating Cash Flow
OPERATING_CF_KEYS = (
    "İşletme Faaliyetlerinden Nakit Akışları",
    "İşletme Faaliyetlerinden Kaynaklanan Net Nakit",
)

# Total Assets
TOTAL_ASSETS_KEYS = (
    "Toplam Varlıklar",
    "Toplam Aktifler",
)

# Total Equity
TOTAL_EQUITY_KEYS = (
    "Özkaynaklar",
    "Toplam Özkaynaklar",
    "Ana Ortaklığa Ait Özkaynaklar",
)

# Total Liabilities (for debt calculation)
TOTAL_LIABILITIES_KEYS = (
    "Toplam Yükümlülükler",
    "Toplam Borçlar",
)


# ============================================================================
# CORE CALCULATIONS
# ============================================================================

def calculate_quality_metrics_for_ticker(
    ticker: str,
    data_loader,
    fundamentals_parquet: pd.DataFrame = None,
) -> Dict:
    """
    Calculate all quality metrics for a single ticker.

    Returns dict with:
    - accruals: (Net Income - Operating CF) / Total Assets (lower = better)
    - roe: Net Income / Equity (higher = better)
    - cfa: Operating CF / Total Assets (higher = better)
    - debt_ratio: Total Liabilities / Total Assets (lower = better)
    """
    if fundamentals_parquet is None:
        return {}

    inc = get_consolidated_sheet(fundamentals_parquet, ticker, INCOME_SHEET)
    bs = get_consolidated_sheet(fundamentals_parquet, ticker, BALANCE_SHEET)
    cf = get_consolidated_sheet(fundamentals_parquet, ticker, CASH_FLOW_SHEET)

    if inc.empty and bs.empty:
        return {}

    # Extract rows
    net_income_row = pick_row_from_sheet(inc, NET_INCOME_KEYS)
    assets_row = pick_row_from_sheet(bs, TOTAL_ASSETS_KEYS)
    equity_row = pick_row_from_sheet(bs, TOTAL_EQUITY_KEYS)
    liabilities_row = pick_row_from_sheet(bs, TOTAL_LIABILITIES_KEYS)
    opcf_row = pick_row_from_sheet(cf, OPERATING_CF_KEYS) if not cf.empty else None

    # Coerce to series
    net_income = coerce_quarter_cols(net_income_row) if net_income_row is not None else pd.Series(dtype=float)
    assets = coerce_quarter_cols(assets_row) if assets_row is not None else pd.Series(dtype=float)
    equity = coerce_quarter_cols(equity_row) if equity_row is not None else pd.Series(dtype=float)
    liabilities = coerce_quarter_cols(liabilities_row) if liabilities_row is not None else pd.Series(dtype=float)
    opcf = coerce_quarter_cols(opcf_row) if opcf_row is not None else pd.Series(dtype=float)

    # TTM for income/cash flow items
    net_income_ttm = sum_ttm(net_income)
    opcf_ttm = sum_ttm(opcf)

    return {
        'net_income_ttm': net_income_ttm,
        'opcf_ttm': opcf_ttm,
        'total_assets': assets,
        'equity': equity,
        'liabilities': liabilities,
    }


def build_earnings_quality_signals(
    fundamentals: Dict,
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader,
) -> pd.DataFrame:
    """
    Build composite earnings quality signal panel.

    Combines 4 quality metrics:
    1. Accruals (lower = better)
    2. ROE (higher = better)
    3. Cash Flow / Assets (higher = better)
    4. Debt / Assets (lower = better)

    Args:
        fundamentals: Dict of fundamental data by ticker
        close_df: DataFrame of close prices
        dates: DatetimeIndex to align signals to
        data_loader: DataLoader instance

    Returns:
        DataFrame (dates x tickers) with quality scores
    """
    print("\n🔧 Building earnings quality signals...")

    panels = {
        'accruals': {},      # Lower = better (will be negated)
        'roe': {},           # Higher = better
        'cfa': {},           # Higher = better (Cash Flow / Assets)
        'debt_ratio': {},    # Lower = better (will be negated)
    }

    fundamentals_parquet = data_loader.load_fundamentals_parquet()

    count = 0
    for ticker, fund_data in fundamentals.items():
        if ticker not in close_df.columns:
            continue

        metrics = calculate_quality_metrics_for_ticker(
            ticker, data_loader, fundamentals_parquet
        )

        if not metrics:
            continue

        net_income = metrics.get('net_income_ttm', pd.Series(dtype=float))
        opcf = metrics.get('opcf_ttm', pd.Series(dtype=float))
        assets = metrics.get('total_assets', pd.Series(dtype=float))
        equity = metrics.get('equity', pd.Series(dtype=float))
        liabilities = metrics.get('liabilities', pd.Series(dtype=float))

        # 1. Accruals = (Net Income - Operating CF) / Assets
        # Lower is better (high accruals = low quality earnings)
        if not net_income.empty and not opcf.empty and not assets.empty:
            accruals = (net_income - opcf.reindex(net_income.index).ffill()) / assets.reindex(net_income.index).ffill()
            accruals = accruals.replace([np.inf, -np.inf], np.nan).dropna()
            if not accruals.empty:
                lagged = apply_lag(accruals, dates)
                if not lagged.empty:
                    panels['accruals'][ticker] = -lagged  # Negate: lower accruals = higher score

        # 2. ROE = Net Income / Equity
        # Higher is better
        if not net_income.empty and not equity.empty:
            roe = net_income / equity.reindex(net_income.index).ffill()
            roe = roe.replace([np.inf, -np.inf], np.nan).dropna()
            roe = roe.clip(-1, 1)  # Clip extreme values
            if not roe.empty:
                lagged = apply_lag(roe, dates)
                if not lagged.empty:
                    panels['roe'][ticker] = lagged

        # 3. Cash Flow / Assets
        # Higher is better
        if not opcf.empty and not assets.empty:
            cfa = opcf / assets.reindex(opcf.index).ffill()
            cfa = cfa.replace([np.inf, -np.inf], np.nan).dropna()
            cfa = cfa.clip(-1, 1)
            if not cfa.empty:
                lagged = apply_lag(cfa, dates)
                if not lagged.empty:
                    panels['cfa'][ticker] = lagged

        # 4. Debt Ratio = Liabilities / Assets
        # Lower is better
        if not liabilities.empty and not assets.empty:
            debt_ratio = liabilities / assets.reindex(liabilities.index).ffill()
            debt_ratio = debt_ratio.replace([np.inf, -np.inf], np.nan).dropna()
            debt_ratio = debt_ratio.clip(0, 2)
            if not debt_ratio.empty:
                lagged = apply_lag(debt_ratio, dates)
                if not lagged.empty:
                    panels['debt_ratio'][ticker] = -lagged  # Negate: lower debt = higher score

        count += 1
        if count % 50 == 0:
            print(f"  Processed {count} tickers...")

    # Cross-sectional z-score normalization
    print("  Normalizing quality metrics (z-score per date)...")
    normalized_panels = {}
    for panel_name, panel_dict in panels.items():
        if panel_dict:
            df = pd.DataFrame(panel_dict, index=dates)
            row_mean = df.mean(axis=1)
            row_std = df.std(axis=1).replace(0, np.nan)
            df_zscore = df.sub(row_mean, axis=0).div(row_std, axis=0)
            normalized_panels[panel_name] = df_zscore

    # Combine into composite score
    print("  Combining into composite quality score...")
    composite_panel = {}

    for ticker in close_df.columns:
        scores_list = []
        for panel_name, panel_df in normalized_panels.items():
            if ticker in panel_df.columns:
                scores_list.append(panel_df[ticker])

        if scores_list:
            composite = pd.concat(scores_list, axis=1).mean(axis=1)
            composite_panel[ticker] = composite

    result = pd.DataFrame(composite_panel, index=dates)
    print(f"  ✅ Earnings quality signals: {result.shape[0]} days × {result.shape[1]} tickers")
    return result
