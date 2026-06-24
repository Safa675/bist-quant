"""
Factor Panel Builders

Internal panel builders for ``five_factor_rotation`` and related composites.
Prefer ``build_signal()`` or ``signals.core`` for new code.

Functions build raw factor panels from price data and fundamentals.
These panels are cached to avoid recomputation.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from bist_quant.common.utils import (
    align_numeric_panel,
    apply_lag,
    coerce_quarter_cols,
    # Consolidated utilities (Phase 2 refactoring)
    debug_enabled,
    debug_log,
    get_consolidated_sheet,
    pick_row,
    pick_row_from_sheet,
    sum_ttm,
    validate_reference_axes,
)
from bist_quant.signals.fundamental_keys import (
    CFO_KEYS,
    CURRENT_ASSETS_KEYS,
    CURRENT_LIABILITIES_KEYS,
    DIVIDENDS_PAID_KEYS,
    GROSS_PROFIT_KEYS,
    LONG_TERM_DEBT_KEYS,
    NET_INCOME_KEYS,
    OPERATING_INCOME_KEYS,
    REVENUE_KEYS,
    TOTAL_ASSETS_KEYS,
    TOTAL_EQUITY_KEYS,
)
from bist_quant.signals.core.constants import (
    AMIHUD_LOOKBACK_DAYS,
    BALANCE_SHEET,
    BETA_LOOKBACK_DAYS,
    BETA_MIN_OBS,
    CASH_FLOW_SHEET,
    EARNINGS_STABILITY_QUARTERS,
    INCOME_SHEET,
    MARGIN_CHANGE_QUARTERS,
    MIN_ROLLING_OBS_RATIO,
    PRICE_ACCELERATION_FAST,
    PRICE_ACCELERATION_SLOW,
    REVERSAL_LOOKBACK_DAYS,
    SALES_ACCEL_QUARTERS,
    TURNOVER_LOOKBACK_DAYS,
    VALUE_GROWTH_LOOKBACK_DAYS,
    VOLATILITY_LOOKBACK_DAYS,
)
from bist_quant.signals.value_signals import build_value_signals

logger = logging.getLogger(__name__)


# ============================================================================
# FACTOR PANEL CONTRACT / VALIDATION
# ============================================================================

FACTOR_PANEL_CONTRACT: Dict[str, Tuple[str, ...]] = {
    "quality": (
        "quality_roe",
        "quality_roa",
        "quality_accruals",
        "quality_piotroski",
    ),
    "liquidity": (
        "liquidity_amihud",
        "liquidity_turnover",
        "liquidity_spread_proxy",
    ),
    "trading_intensity": (
        "trading_intensity_relative_volume",
        "trading_intensity_volume_trend",
        "trading_intensity_turnover_velocity",
    ),
    "sentiment": (
        "sentiment_52w_high_pct",
        "sentiment_price_acceleration",
        "sentiment_reversal",
    ),
    "fundmom": (
        "fundmom_margin_change",
        "fundmom_sales_accel",
    ),
    "carry": (
        "carry_dividend_yield",
        "carry_shareholder_yield",
    ),
    "defensive": (
        "defensive_earnings_stability",
        "defensive_beta_to_market",
    ),
    "risk": (
        "realized_volatility",
        "market_beta",
    ),
}


# Aliases for backward compatibility (use consolidated utils)
_debug_enabled = debug_enabled
_validate_reference_axes = validate_reference_axes
_align_numeric_panel = align_numeric_panel


def _debug_log(msg: str) -> None:
    debug_log(msg, prefix="FACTOR_DEBUG")


def get_factor_panel_contract() -> Dict[str, Tuple[str, ...]]:
    """Return expected raw panel keys grouped by factor family."""
    return {name: tuple(keys) for name, keys in FACTOR_PANEL_CONTRACT.items()}


def _finalize_builder_outputs(
    builder_name: str,
    raw_panels: Dict[str, pd.DataFrame],
    expected_keys: Tuple[str, ...],
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
) -> Dict[str, pd.DataFrame]:
    """Validate builder key schema and panel contract before returning."""
    missing = [name for name in expected_keys if name not in raw_panels]
    extra = [name for name in raw_panels.keys() if name not in expected_keys]
    if missing or extra:
        raise KeyError(
            f"{builder_name}: panel keys mismatch; missing={missing or 'none'}, extra={extra or 'none'}"
        )

    finalized: Dict[str, pd.DataFrame] = {}
    for key in expected_keys:
        finalized[key] = _align_numeric_panel(raw_panels[key], key, dates, tickers)
    return finalized


# ============================================================================
# SHARED PANEL HELPERS (USED BY FIVE-FACTOR ROTATION)
# ============================================================================

def _build_metric_panel(
    metrics_df: pd.DataFrame,
    metric_name: str,
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
) -> pd.DataFrame:
    """Convert (ticker, date) metric series into a daily Date x Ticker panel."""
    panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)

    if metrics_df.empty or metric_name not in metrics_df.columns:
        return panel

    available_tickers = set(metrics_df.index.get_level_values(0))
    for ticker in tickers:
        if ticker not in available_tickers:
            continue
        try:
            series = metrics_df.loc[ticker, metric_name]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            series = pd.to_numeric(series, errors="coerce")
            series = series.sort_index()
            series = series[~series.index.duplicated(keep="last")]
            series.index = pd.to_datetime(series.index)
            panel[ticker] = apply_lag(series, dates)
        except KeyError:
            continue
        except (ValueError, TypeError) as exc:
            logger.info(f"    Warning: Could not process {metric_name} for {ticker}: {exc}")
            continue

    return panel




def _build_value_level_growth_panels(
    fundamentals: Dict,
    close: pd.DataFrame,
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
    data_loader,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build value level and lagged YoY value-growth panels."""
    level_df = build_value_signals(
        fundamentals,
        close,
        dates,
        data_loader,
    ).reindex(index=dates, columns=tickers)

    growth_df = level_df.diff(VALUE_GROWTH_LOOKBACK_DAYS)
    growth_df = growth_df.replace([np.inf, -np.inf], np.nan)
    return level_df, growth_df


# ============================================================================
# QUALITY FACTOR PANELS
# ============================================================================

def _load_ticker_fundamental_sheets(
    ticker: str,
    fundamentals_parquet: pd.DataFrame | None,
    fundamentals: Dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool] | None:
    """Load (income, balance, cash_flow) sheets for one ticker.

    Returns ``None`` if no data is available. The ``use_parquet`` flag
    indicates whether sheets came from the consolidated parquet (row-indexed,
    picked via ``pick_row_from_sheet``) or from per-ticker Excel files
    (picked via ``pick_row``).
    """
    use_parquet = fundamentals_parquet is not None
    if use_parquet:
        inc = get_consolidated_sheet(fundamentals_parquet, ticker, INCOME_SHEET)
        bs = get_consolidated_sheet(fundamentals_parquet, ticker, BALANCE_SHEET)
        cf = get_consolidated_sheet(fundamentals_parquet, ticker, CASH_FLOW_SHEET)
    else:
        fund_data = fundamentals.get(str(ticker), {}) if isinstance(fundamentals, dict) else {}
        xlsx_path = fund_data.get("path") if isinstance(fund_data, dict) else None
        if xlsx_path is None:
            return None
        try:
            inc = pd.read_excel(xlsx_path, sheet_name=INCOME_SHEET)
            bs = pd.read_excel(xlsx_path, sheet_name=BALANCE_SHEET)
            try:
                cf = pd.read_excel(xlsx_path, sheet_name=CASH_FLOW_SHEET)
            except Exception:
                cf = pd.DataFrame()
        except Exception:
            return None

    if inc.empty or bs.empty:
        return None
    return inc, bs, cf, use_parquet


def build_quality_panels(
    fundamentals: Dict,
    close: pd.DataFrame,
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
    data_loader,
) -> Dict[str, pd.DataFrame]:
    """Build Quality factor panels (delegates to ``signals.core.panels.quality``)."""
    from bist_quant.signals.core.panels.quality import build_quality_panels as _build_quality_panels

    return _build_quality_panels(fundamentals, close, dates, tickers, data_loader)

# ============================================================================
# PANEL BUILDERS (facade → signals.core.panels)
# ============================================================================

def _calculate_margin_level_growth_for_ticker(
    xlsx_path, ticker, fundamentals_parquet=None,
):
    from bist_quant.signals.core.panels.profit_margin import (
        calculate_margin_level_growth_for_ticker as _fn,
    )
    return _fn(xlsx_path, ticker, fundamentals_parquet)


def _build_profitability_margin_panels(fundamentals, dates, data_loader):
    from bist_quant.signals.core.panels.profit_margin import (
        build_profitability_margin_panels as _fn,
    )
    return _fn(fundamentals, dates, data_loader)


def _load_shares_panel(data_loader, dates, tickers):
    """Load and align shares-outstanding panel for turnover-based metrics."""
    empty_panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)
    if data_loader is None or not hasattr(data_loader, "load_shares_outstanding_panel"):
        return empty_panel, False

    panel_cache = getattr(data_loader, "panel_cache", None)
    cache_key = None
    if panel_cache is not None:
        cache_key = panel_cache.make_key(
            "shares_outstanding_aligned",
            start=dates[0] if len(dates) else None,
            end=dates[-1] if len(dates) else None,
            rows=int(len(dates)),
            tickers=tuple(str(t) for t in tickers),
        )
        cached_panel = panel_cache.get(cache_key)
        if isinstance(cached_panel, pd.DataFrame):
            return cached_panel, True

    try:
        shares_outstanding = data_loader.load_shares_outstanding_panel()
    except Exception as exc:
        logger.warning(f"    ⚠️  Failed to load shares outstanding panel: {exc}")
        return empty_panel, False

    if shares_outstanding is None or shares_outstanding.empty:
        return empty_panel, False

    shares = shares_outstanding.copy()
    shares.index = pd.to_datetime(shares.index, errors="coerce")
    shares = shares.sort_index()
    shares.columns = pd.Index([str(c).upper() for c in shares.columns])
    aligned = _align_numeric_panel(shares, "shares_outstanding", dates, tickers).ffill()
    if panel_cache is not None and cache_key is not None:
        panel_cache.set(cache_key, aligned)
    return aligned, True


def build_liquidity_panels(close, volume_df, dates, tickers, data_loader=None):
    from bist_quant.signals.core.panels.liquidity import build_liquidity_panels as _fn
    return _fn(close, volume_df, dates, tickers, data_loader)


def build_trading_intensity_panels(close, volume_df, dates, tickers, data_loader=None):
    from bist_quant.signals.core.panels.trading_intensity import (
        build_trading_intensity_panels as _fn,
    )
    return _fn(close, volume_df, dates, tickers, data_loader)


def build_sentiment_panels(close, dates, tickers):
    from bist_quant.signals.core.panels.sentiment import build_sentiment_panels as _fn
    return _fn(close, dates, tickers)


def build_fundamental_momentum_panels(fundamentals, dates, tickers, data_loader):
    from bist_quant.signals.core.panels.fundamental_momentum import (
        build_fundamental_momentum_panels as _fn,
    )
    return _fn(fundamentals, dates, tickers, data_loader)


def build_carry_panels(fundamentals, close, dates, tickers, data_loader):
    from bist_quant.signals.core.panels.carry import build_carry_panels as _fn
    return _fn(fundamentals, close, dates, tickers, data_loader)


def build_defensive_panels(fundamentals, close, dates, tickers, data_loader):
    from bist_quant.signals.core.panels.defensive import build_defensive_panels as _fn
    return _fn(fundamentals, close, dates, tickers, data_loader)


def build_realized_volatility_panel(close, dates, lookback=VOLATILITY_LOOKBACK_DAYS):
    from bist_quant.signals.core.panels.vol_beta import build_realized_volatility_panel as _fn
    return _fn(close, dates, lookback)


def build_market_beta_panel(close, dates, data_loader=None, lookback=BETA_LOOKBACK_DAYS):
    from bist_quant.signals.core.panels.vol_beta import build_market_beta_panel as _fn
    return _fn(close, dates, data_loader, lookback)


def build_volatility_beta_panels(close, dates, data_loader=None):
    from bist_quant.signals.core.panels.vol_beta import build_volatility_beta_panels as _fn
    return _fn(close, dates, data_loader)

