"""
Value Signal Construction

Calculates composite value scores based on 5 ratios:
1. E/P (Earnings / Price)
2. FCF/P (Free Cash Flow / Price)
3. OCF/EV (Operating Cash Flow / Enterprise Value)
4. S/P (Sales / Price)
5. EBITDA/EV (EBITDA / Enterprise Value)
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
CASH_FLOW_SHEET = "Nakit Akış (Çeyreklik)"

NET_INCOME_KEYS = (
    "Dönem Net Karı (Zararı)",
    "Net Dönem Karı (Zararı)",
    "Ana Ortaklık Payları",
    "Dönem Karı (Zararı)",
)
REVENUE_KEYS = (
    "Satış Gelirleri",
    "Toplam Hasılat",
    "Hasılat",
    "Net Satışlar",
)
EBITDA_KEYS = (
    "FAVÖK",
    "Faiz Amortisman ve Vergi Öncesi Kar",
)
OPERATING_CASH_FLOW_KEYS = (
    "Faaliyetlerden Elde Edilen Nakit Akışları",
    "İşletme Faaliyetlerinden Nakit Akışları",
)
CAPEX_KEYS = (
    "Maddi ve Maddi Olmayan Duran Varlıkların Alımından Kaynaklanan Nakit Çıkışları",
)
TOTAL_DEBT_KEYS = (
    "Finansal Borçlar",
)
CASH_KEYS = (
    "Nakit ve Nakit Benzerleri",
)


def calculate_value_metrics_for_ticker(
    xlsx_path: Path | None,
    ticker: str,
    data_loader,
    fundamentals_parquet: pd.DataFrame | None = None,
) -> Dict:
    """Calculate value metrics for a single ticker"""
    if fundamentals_parquet is not None:
        inc = get_consolidated_sheet(fundamentals_parquet, ticker, INCOME_SHEET)
        bs = get_consolidated_sheet(fundamentals_parquet, ticker, BALANCE_SHEET)
        cf = get_consolidated_sheet(fundamentals_parquet, ticker, CASH_FLOW_SHEET)
        if inc.empty and bs.empty and cf.empty:
            return {}
        ni_row = pick_row_from_sheet(inc, NET_INCOME_KEYS)
        rev_row = pick_row_from_sheet(inc, REVENUE_KEYS)
        ebitda_row = pick_row_from_sheet(inc, EBITDA_KEYS)
        ocf_row = pick_row_from_sheet(cf, OPERATING_CASH_FLOW_KEYS)
        capex_row = pick_row_from_sheet(cf, CAPEX_KEYS)
        debt_row = pick_row_from_sheet(bs, TOTAL_DEBT_KEYS)
        cash_row = pick_row_from_sheet(bs, CASH_KEYS)
    else:
        if xlsx_path is None:
            return {}
        try:
            inc = pd.read_excel(xlsx_path, sheet_name=INCOME_SHEET)
            bs = pd.read_excel(xlsx_path, sheet_name=BALANCE_SHEET)
            cf = pd.read_excel(xlsx_path, sheet_name=CASH_FLOW_SHEET)
        except Exception:
            return {}
        
        # Extract rows
        ni_row = pick_row(inc, NET_INCOME_KEYS)
        rev_row = pick_row(inc, REVENUE_KEYS)
        ebitda_row = pick_row(inc, EBITDA_KEYS)
        ocf_row = pick_row(cf, OPERATING_CASH_FLOW_KEYS)
        capex_row = pick_row(cf, CAPEX_KEYS)
        debt_row = pick_row(bs, TOTAL_DEBT_KEYS)
        cash_row = pick_row(bs, CASH_KEYS)
    
    # Convert to series
    ni = coerce_quarter_cols(ni_row) if ni_row is not None else pd.Series(dtype=float)
    rev = coerce_quarter_cols(rev_row) if rev_row is not None else pd.Series(dtype=float)
    ebitda = coerce_quarter_cols(ebitda_row) if ebitda_row is not None else pd.Series(dtype=float)
    ocf = coerce_quarter_cols(ocf_row) if ocf_row is not None else pd.Series(dtype=float)
    capex = coerce_quarter_cols(capex_row) if capex_row is not None else pd.Series(dtype=float)
    debt = coerce_quarter_cols(debt_row) if debt_row is not None else pd.Series(dtype=float)
    cash = coerce_quarter_cols(cash_row) if cash_row is not None else pd.Series(dtype=float)
    
    # Calculate TTM
    ni_ttm = sum_ttm(ni)
    rev_ttm = sum_ttm(rev)
    ebitda_ttm = sum_ttm(ebitda)
    ocf_ttm = sum_ttm(ocf)
    capex_ttm = sum_ttm(capex)
    
    # FCF = OCF - CapEx
    fcf_ttm = ocf_ttm - capex_ttm.reindex(ocf_ttm.index).ffill().fillna(0)
    
    # Load shares outstanding
    shares = data_loader.load_shares_outstanding(ticker)
    
    return {
        'net_income_ttm': ni_ttm,
        'revenue_ttm': rev_ttm,
        'ebitda_ttm': ebitda_ttm,
        'ocf_ttm': ocf_ttm,
        'fcf_ttm': fcf_ttm,
        'debt': debt,
        'cash': cash,
        'shares_outstanding': shares,
    }


def build_value_signals(
    fundamentals: Dict,
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader,
    metric_weights: Dict[str, float] | None = None,
    enabled_metrics: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build composite value signal panel
    
    Returns:
        DataFrame (dates x tickers) with composite value scores
    """
    print("\n🔧 Building value signals...")
    if enabled_metrics:
        print(f"  Enabled metrics: {', '.join(enabled_metrics)}")
    
    # Build individual ratio panels
    panels = {
        'ep': {},
        'fcfp': {},
        'ocfev': {},
        'sp': {},
        'ebitdaev': {},
    }
    
    fundamentals_parquet = data_loader.load_fundamentals_parquet()

    count = 0
    for ticker, fund_data in fundamentals.items():
        if ticker not in close_df.columns:
            continue
        
        xlsx_path = fund_data['path']
        metrics = calculate_value_metrics_for_ticker(
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
        
        # Get fundamentals
        ni_ttm = metrics.get('net_income_ttm', pd.Series(dtype=float))
        rev_ttm = metrics.get('revenue_ttm', pd.Series(dtype=float))
        ebitda_ttm = metrics.get('ebitda_ttm', pd.Series(dtype=float))
        ocf_ttm = metrics.get('ocf_ttm', pd.Series(dtype=float))
        fcf_ttm = metrics.get('fcf_ttm', pd.Series(dtype=float))
        debt = metrics.get('debt', pd.Series(dtype=float))
        cash = metrics.get('cash', pd.Series(dtype=float))
        
        # Calculate EV
        debt_aligned = debt.reindex(market_cap.index).ffill().fillna(0)
        cash_aligned = cash.reindex(market_cap.index).ffill().fillna(0)
        ev = market_cap + debt_aligned - cash_aligned
        
        # Calculate ratios with proper reporting lag
        # Fundamentals are only known after reporting delay (45/75 days)
        for metric_name, numerator in [
            ('ep', ni_ttm),
            ('fcfp', fcf_ttm),
            ('sp', rev_ttm),
        ]:
            if not numerator.empty:
                # Calculate ratio at quarter dates
                ratio = numerator / market_cap.reindex(numerator.index).ffill()
                ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
                if not ratio.empty:
                    # Apply reporting lag before forward-filling
                    lagged_ratio = apply_lag(ratio, dates)
                    if not lagged_ratio.empty:
                        panels[metric_name][ticker] = lagged_ratio
        
        # OCF/EV and EBITDA/EV
        if not ocf_ttm.empty:
            ratio = ocf_ttm / ev.reindex(ocf_ttm.index).ffill()
            ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
            if not ratio.empty:
                lagged_ratio = apply_lag(ratio, dates)
                if not lagged_ratio.empty:
                    panels['ocfev'][ticker] = lagged_ratio
        
        if not ebitda_ttm.empty:
            ratio = ebitda_ttm / ev.reindex(ebitda_ttm.index).ffill()
            ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
            if not ratio.empty:
                lagged_ratio = apply_lag(ratio, dates)
                if not lagged_ratio.empty:
                    panels['ebitdaev'][ticker] = lagged_ratio
        
        count += 1
        if count % 50 == 0:
            print(f"  Processed {count} tickers...")
    
    # Cross-sectional z-score normalization (Bug #2 fix)
    # Normalize each ratio type before combining to prevent scale bias
    print("  Normalizing ratios (z-score per date)...")
    normalized_panels = {}
    for panel_name, panel_dict in panels.items():
        if panel_dict:
            df = pd.DataFrame(panel_dict, index=dates)
            # Z-score: (x - mean) / std, computed cross-sectionally per date
            row_mean = df.mean(axis=1)
            row_std = df.std(axis=1)
            # Avoid division by zero
            row_std = row_std.replace(0, np.nan)
            df_zscore = df.sub(row_mean, axis=0).div(row_std, axis=0)
            normalized_panels[panel_name] = df_zscore
    
    # Combine into composite score
    print("  Combining into composite value score...")
    composite_panel = {}

    default_weights = {
        "ep": 1.0,
        "fcfp": 1.0,
        "ocfev": 1.0,
        "sp": 1.0,
        "ebitdaev": 1.0,
    }
    if metric_weights:
        for key, value in metric_weights.items():
            if key in default_weights and isinstance(value, (int, float)):
                default_weights[key] = float(value)
    enabled_set = set(enabled_metrics) if enabled_metrics else set(default_weights.keys())
    
    for ticker in close_df.columns:
        scores_list = []
        for panel_name, panel_df in normalized_panels.items():
            if panel_name not in enabled_set:
                continue
            if ticker in panel_df.columns:
                weight = default_weights.get(panel_name, 1.0)
                if weight > 0:
                    scores_list.append(panel_df[ticker] * weight)

        if scores_list:
            # Average across all available normalized ratios
            stacked = pd.concat(scores_list, axis=1)
            composite = stacked.mean(axis=1)
            composite_panel[ticker] = composite
    
    result = pd.DataFrame(composite_panel, index=dates)
    print(f"  ✅ Value signals: {result.shape[0]} days × {result.shape[1]} tickers")
    return result
