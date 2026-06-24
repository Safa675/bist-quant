"""
Value Signal Construction

Calculates composite value scores based on 5 ratios:
1. E/P (Earnings / Price)
2. FCF/P (Free Cash Flow / Price)
3. OCF/EV (Operating Cash Flow / Enterprise Value)
4. S/P (Sales / Price)
5. EBITDA/EV (EBITDA / Enterprise Value)
"""

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from bist_quant.common.utils import (
    apply_lag,
    apply_staleness_weighting,
    coerce_quarter_cols,
    get_consolidated_sheet,
    pick_row,
    pick_row_from_sheet,
    sum_ttm,
    validate_signal_panel_schema,
)
from bist_quant.signals.core.constants import (
    BALANCE_SHEET,
    CASH_FLOW_SHEET,
    INCOME_SHEET,
)
from bist_quant.signals.core.value import calculate_value_metrics_for_ticker
from bist_quant.signals.fundamental_keys import (
    CAPEX_KEYS,
    CASH_KEYS,
    EBITDA_KEYS,
    NET_INCOME_KEYS,
    OPERATING_CF_KEYS as OPERATING_CASH_FLOW_KEYS,
    REVENUE_KEYS,
    TOTAL_DEBT_KEYS,
)

logger = logging.getLogger(__name__)


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
    logger.info("\n🔧 Building value signals...")
    if enabled_metrics:
        logger.info(f"  Enabled metrics: {', '.join(enabled_metrics)}")
    
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
        shares_aligned = shares.reindex(price_series.index, method='ffill')
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
        debt_aligned = debt.reindex(market_cap.index, method='ffill').fillna(0) if not debt.empty else pd.Series(0.0, index=market_cap.index)
        cash_aligned = cash.reindex(market_cap.index, method='ffill').fillna(0) if not cash.empty else pd.Series(0.0, index=market_cap.index)
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
                ratio = numerator / market_cap.reindex(numerator.index, method='ffill')
                ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
                if not ratio.empty:
                    # Apply reporting lag before forward-filling
                    lagged_ratio = apply_lag(ratio, dates)
                    if not lagged_ratio.empty:
                        panels[metric_name][ticker] = lagged_ratio
        
        # OCF/EV and EBITDA/EV
        if not ocf_ttm.empty:
            ratio = ocf_ttm / ev.reindex(ocf_ttm.index, method='ffill')
            ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
            if not ratio.empty:
                lagged_ratio = apply_lag(ratio, dates)
                if not lagged_ratio.empty:
                    panels['ocfev'][ticker] = lagged_ratio
        
        if not ebitda_ttm.empty:
            ratio = ebitda_ttm / ev.reindex(ebitda_ttm.index, method='ffill')
            ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
            if not ratio.empty:
                lagged_ratio = apply_lag(ratio, dates)
                if not lagged_ratio.empty:
                    panels['ebitdaev'][ticker] = lagged_ratio
        
        count += 1
        if count % 50 == 0:
            logger.info(f"  Processed {count} tickers...")
    
    # Cross-sectional z-score normalization (Bug #2 fix)
    # Normalize each ratio type before combining to prevent scale bias
    logger.info("  Normalizing ratios (z-score per date)...")
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
    logger.info("  Combining into composite value score...")
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

    # Apply staleness-based down-weighting (Part D)
    result = apply_staleness_weighting(result)

    result = validate_signal_panel_schema(
        result,
        dates=dates,
        tickers=close_df.columns,
        signal_name="value",
        context="final score panel",
        dtype=np.float32,
    )
    logger.info(f"  ✅ Value signals: {result.shape[0]} days × {result.shape[1]} tickers")
    return result
