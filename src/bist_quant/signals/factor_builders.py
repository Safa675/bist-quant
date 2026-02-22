"""
Factor Panel Builders

Functions to build raw factor panels from price data and fundamentals.
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
from bist_quant.signals.value_signals import build_value_signals

logger = logging.getLogger(__name__)

# ============================================================================
# TURKISH FUNDAMENTAL FIELD KEYS
# ============================================================================

# Sheet names - MUST match actual parquet data
INCOME_SHEET = "Gelir Tablosu (Çeyreklik)"
BALANCE_SHEET = "Bilanço"  # NOT "Finansal Durum Tablosu (Çeyreklik)"
CASH_FLOW_SHEET = "Nakit Akış (Çeyreklik)"  # NOT "Nakit Akış Tablosu (Çeyreklik)"

REVENUE_KEYS = (
    "Satış Gelirleri",
    "Toplam Hasılat",
    "Hasılat",
    "Net Satışlar",
)
OPERATING_INCOME_KEYS = (
    "Faaliyet Karı (Zararı)",
    "Finansman Geliri (Gideri) Öncesi Faaliyet Karı (Zararı)",
)
GROSS_PROFIT_KEYS = (
    "Brüt Kar (Zarar)",
    "Ticari Faaliyetlerden Brüt Kar (Zarar)",
)
# Row name keys - ordered by preference (first match wins)
NET_INCOME_KEYS = (
    "Dönem Karı (Zararı)",  # Most common
    "Dönem Net Karı (Zararı)",
    "Net Dönem Karı (Zararı)",
    "Ana Ortaklık Payları",  # Net income to parent
)
TOTAL_ASSETS_KEYS = (
    "Toplam Varlıklar",
    "TOPLAM VARLIKLAR",
    "Varlık Toplamı",
)
TOTAL_EQUITY_KEYS = (
    "Özkaynaklar",
    "Toplam Özkaynaklar",
    "TOPLAM ÖZKAYNAKLAR",
    "Ana Ortaklığa Ait Özkaynaklar",
)
CURRENT_ASSETS_KEYS = (
    "Dönen Varlıklar",
    "DÖNEN VARLIKLAR",
)
CURRENT_LIABILITIES_KEYS = (
    "Kısa Vadeli Yükümlülükler",
    "KISA VADELİ YÜKÜMLÜLÜKLER",
    "Toplam Kısa Vadeli Yükümlülükler",
)
LONG_TERM_DEBT_KEYS = (
    "Uzun Vadeli Yükümlülükler",  # Use total long-term liabilities as proxy
    "Uzun Vadeli Borçlanmalar",
    "Uzun Vadeli Finansal Borçlar",
    "Finansal Borçlar",
)
CFO_KEYS = (
    "İşletme Faaliyetlerinden Nakit Akışları",
    "Faaliyetlerden Elde Edilen Nakit Akışları",
    "Esas Faaliyetlerden Kaynaklanan Net Nakit",
)
DIVIDENDS_PAID_KEYS = (
    "Ödenen Temettüler",
    "Ödenen Kar Payları",
    "Temettü Ödemeleri",
    "Kar Payı Ödemeleri",
)


# ============================================================================
# PARAMETERS
# ============================================================================

MIN_ROLLING_OBS_RATIO = 0.5

# Volatility/Beta
VOLATILITY_LOOKBACK_DAYS = 63
BETA_LOOKBACK_DAYS = 252
BETA_MIN_OBS = 126
VALUE_GROWTH_LOOKBACK_DAYS = 252

# Liquidity
AMIHUD_LOOKBACK_DAYS = 21
TURNOVER_LOOKBACK_DAYS = 63

# Sentiment
PRICE_ACCELERATION_FAST = 21
PRICE_ACCELERATION_SLOW = 63
REVERSAL_LOOKBACK_DAYS = 5

# Fundamental Momentum
MARGIN_CHANGE_QUARTERS = 4
SALES_ACCEL_QUARTERS = 4

# Defensive
EARNINGS_STABILITY_QUARTERS = 8


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


def _calculate_margin_level_growth_for_ticker(
    xlsx_path: Path | None,
    ticker: str,
    fundamentals_parquet: pd.DataFrame | None = None,
) -> tuple[pd.Series | None, pd.Series | None]:
    """Return profitability margin level and YoY margin growth series for one ticker."""
    if fundamentals_parquet is not None:
        inc = get_consolidated_sheet(fundamentals_parquet, ticker, INCOME_SHEET)
        if inc.empty:
            return None, None
        rev_row = pick_row_from_sheet(inc, REVENUE_KEYS)
        op_row = pick_row_from_sheet(inc, OPERATING_INCOME_KEYS)
        gp_row = pick_row_from_sheet(inc, GROSS_PROFIT_KEYS)
    else:
        if xlsx_path is None:
            return None, None
        try:
            inc = pd.read_excel(xlsx_path, sheet_name=INCOME_SHEET)
        except Exception:
            return None, None
        rev_row = pick_row(inc, REVENUE_KEYS)
        op_row = pick_row(inc, OPERATING_INCOME_KEYS)
        gp_row = pick_row(inc, GROSS_PROFIT_KEYS)

    if rev_row is None or op_row is None or gp_row is None:
        return None, None

    rev = coerce_quarter_cols(rev_row)
    op = coerce_quarter_cols(op_row)
    gp = coerce_quarter_cols(gp_row)
    if rev.empty or op.empty or gp.empty:
        return None, None

    rev_ttm = sum_ttm(rev)
    op_ttm = sum_ttm(op)
    gp_ttm = sum_ttm(gp)
    if rev_ttm.empty or op_ttm.empty or gp_ttm.empty:
        return None, None

    combined = pd.concat([rev_ttm, op_ttm, gp_ttm], axis=1, join="inner").dropna()
    if combined.empty:
        return None, None
    combined.columns = ["RevenueTTM", "OperatingIncomeTTM", "GrossProfitTTM"]

    revenue = combined["RevenueTTM"].replace(0.0, np.nan)
    op_margin = combined["OperatingIncomeTTM"] / revenue
    gp_margin = combined["GrossProfitTTM"] / revenue
    op_margin_weight = 0.60
    gp_margin_weight = 0.40
    margin_level = (
        op_margin_weight * op_margin + gp_margin_weight * gp_margin
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if margin_level.empty:
        return None, None

    margin_growth = margin_level.diff(4).replace([np.inf, -np.inf], np.nan).dropna()
    return margin_level.sort_index(), margin_growth.sort_index() if not margin_growth.empty else None


def _build_profitability_margin_panels(
    fundamentals: Dict,
    dates: pd.DatetimeIndex,
    data_loader,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build lagged margin-level and margin-growth panels."""
    fundamentals_parquet = data_loader.load_fundamentals_parquet() if data_loader is not None else None
    level_panel: Dict[str, pd.Series] = {}
    growth_panel: Dict[str, pd.Series] = {}

    count = 0
    for ticker, fund_data in fundamentals.items():
        xlsx_path = fund_data.get("path") if isinstance(fund_data, dict) else None
        margin_level, margin_growth = _calculate_margin_level_growth_for_ticker(
            xlsx_path=xlsx_path,
            ticker=ticker,
            fundamentals_parquet=fundamentals_parquet,
        )

        if margin_level is not None and not margin_level.empty:
            lagged_level = apply_lag(margin_level, dates)
            if not lagged_level.empty:
                level_panel[ticker] = lagged_level

        if margin_growth is not None and not margin_growth.empty:
            lagged_growth = apply_lag(margin_growth, dates)
            if not lagged_growth.empty:
                growth_panel[ticker] = lagged_growth

        count += 1
        if count % 100 == 0:
            logger.info(f"  Profitability margin progress: {count}/{len(fundamentals)}")

    level_df = pd.DataFrame(level_panel, index=dates)
    growth_df = pd.DataFrame(growth_panel, index=dates)
    return level_df, growth_df


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

def build_quality_panels(
    fundamentals: Dict,
    close: pd.DataFrame,
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
    data_loader,
) -> Dict[str, pd.DataFrame]:
    """
    Build Quality factor panels: ROE, ROA, Accruals, Piotroski F-score.
    """
    del close
    fundamentals = fundamentals or {}
    dates, tickers = _validate_reference_axes(dates, tickers, "build_quality_panels")

    logger.info("  Building quality factor panels...")
    fundamentals_parquet = data_loader.load_fundamentals_parquet() if data_loader is not None else None

    roe_panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)
    roa_panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)
    accruals_panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)
    piotroski_panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)

    if fundamentals_parquet is None:
        logger.warning("    ⚠️  No fundamentals parquet - falling back to per-ticker Excel files")

    count = 0
    success_count = 0
    for ticker in tickers:
        try:
            use_parquet = fundamentals_parquet is not None
            if use_parquet:
                inc = get_consolidated_sheet(fundamentals_parquet, ticker, INCOME_SHEET)
                bs = get_consolidated_sheet(fundamentals_parquet, ticker, BALANCE_SHEET)
                cf = get_consolidated_sheet(fundamentals_parquet, ticker, CASH_FLOW_SHEET)
            else:
                fund_data = fundamentals.get(str(ticker), {}) if isinstance(fundamentals, dict) else {}
                xlsx_path = fund_data.get("path") if isinstance(fund_data, dict) else None
                if xlsx_path is None:
                    continue
                try:
                    inc = pd.read_excel(xlsx_path, sheet_name=INCOME_SHEET)
                    bs = pd.read_excel(xlsx_path, sheet_name=BALANCE_SHEET)
                    try:
                        cf = pd.read_excel(xlsx_path, sheet_name=CASH_FLOW_SHEET)
                    except Exception:
                        cf = pd.DataFrame()
                except Exception:
                    continue

            if inc.empty or bs.empty:
                continue

            # Extract key rows
            row_picker = pick_row_from_sheet if use_parquet else pick_row
            net_income_row = row_picker(inc, NET_INCOME_KEYS)
            revenue_row = row_picker(inc, REVENUE_KEYS)
            total_assets_row = row_picker(bs, TOTAL_ASSETS_KEYS)
            total_equity_row = row_picker(bs, TOTAL_EQUITY_KEYS)
            current_assets_row = row_picker(bs, CURRENT_ASSETS_KEYS)
            current_liab_row = row_picker(bs, CURRENT_LIABILITIES_KEYS)
            long_debt_row = row_picker(bs, LONG_TERM_DEBT_KEYS)

            cfo_row = None
            if not cf.empty:
                cfo_row = row_picker(cf, CFO_KEYS)

            ticker_has_data = False

            # ROE = Net Income TTM / Average Equity
            if net_income_row is not None and total_equity_row is not None:
                ni = coerce_quarter_cols(net_income_row)
                eq = coerce_quarter_cols(total_equity_row)
                if not ni.empty and not eq.empty:
                    ni_ttm = sum_ttm(ni)
                    eq_avg = eq.rolling(4, min_periods=2).mean()
                    # Align indices before division
                    ni_ttm, eq_avg = ni_ttm.align(eq_avg, join="inner")
                    roe = (ni_ttm / eq_avg.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
                    roe = roe.dropna()
                    if not roe.empty:
                        roe_panel[ticker] = apply_lag(roe, dates)
                        ticker_has_data = True

            # ROA = Net Income TTM / Average Total Assets
            if net_income_row is not None and total_assets_row is not None:
                ni = coerce_quarter_cols(net_income_row)
                ta = coerce_quarter_cols(total_assets_row)
                if not ni.empty and not ta.empty:
                    ni_ttm = sum_ttm(ni)
                    ta_avg = ta.rolling(4, min_periods=2).mean()
                    # Align indices before division
                    ni_ttm, ta_avg = ni_ttm.align(ta_avg, join="inner")
                    roa = (ni_ttm / ta_avg.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
                    roa = roa.dropna()
                    if not roa.empty:
                        roa_panel[ticker] = apply_lag(roa, dates)
                        ticker_has_data = True

            # Accruals = (Net Income - CFO) / Total Assets (lower is better quality)
            if net_income_row is not None and cfo_row is not None and total_assets_row is not None:
                ni = coerce_quarter_cols(net_income_row)
                cfo = coerce_quarter_cols(cfo_row)
                ta = coerce_quarter_cols(total_assets_row)
                if not ni.empty and not cfo.empty and not ta.empty:
                    ni_ttm = sum_ttm(ni)
                    cfo_ttm = sum_ttm(cfo)
                    ta_avg = ta.rolling(4, min_periods=2).mean()
                    # Align all three series
                    ni_ttm, cfo_ttm = ni_ttm.align(cfo_ttm, join="inner")
                    ni_ttm, ta_avg = ni_ttm.align(ta_avg, join="inner")
                    cfo_ttm = cfo_ttm.reindex(ni_ttm.index)
                    accruals = ((ni_ttm - cfo_ttm) / ta_avg.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
                    accruals = accruals.dropna()
                    if not accruals.empty:
                        accruals_panel[ticker] = apply_lag(accruals, dates)
                        ticker_has_data = True

            # Piotroski F-score (simplified 0-8)
            if net_income_row is not None and total_assets_row is not None:
                ni = coerce_quarter_cols(net_income_row)
                ta = coerce_quarter_cols(total_assets_row)
                if not ni.empty and not ta.empty:
                    ni_ttm = sum_ttm(ni)
                    ta_avg = ta.rolling(4, min_periods=2).mean()
                    roa_check = ni_ttm / ta_avg.replace(0, np.nan)

                    # Start with ROA > 0
                    # Use reindex_like to ensure all components align properly
                    base_index = roa_check.index
                    piotroski_score = (roa_check > 0).astype(float)

                    # Positive CFO
                    if cfo_row is not None:
                        cfo = coerce_quarter_cols(cfo_row)
                        if not cfo.empty:
                            cfo_ttm = sum_ttm(cfo).reindex(base_index)
                            piotroski_score = piotroski_score + (cfo_ttm > 0).fillna(False).astype(float)
                            # CFO > Net Income (quality of earnings)
                            ni_ttm_aligned = ni_ttm.reindex(base_index)
                            piotroski_score = piotroski_score + (cfo_ttm > ni_ttm_aligned).fillna(False).astype(float)

                    # Improving ROA
                    piotroski_score = piotroski_score + (roa_check.diff(4) > 0).fillna(False).astype(float)

                    # Decreasing leverage
                    if long_debt_row is not None:
                        debt = coerce_quarter_cols(long_debt_row)
                        if not debt.empty:
                            ta_avg_aligned = ta_avg.reindex(debt.index)
                            leverage = debt / ta_avg_aligned.replace(0, np.nan)
                            leverage = leverage.reindex(base_index)
                            piotroski_score = piotroski_score + (leverage.diff(4) < 0).fillna(False).astype(float)

                    # Improving current ratio
                    if current_assets_row is not None and current_liab_row is not None:
                        ca = coerce_quarter_cols(current_assets_row)
                        cl = coerce_quarter_cols(current_liab_row)
                        if not ca.empty and not cl.empty:
                            # Align ca and cl first
                            ca, cl = ca.align(cl, join="inner")
                            cr = ca / cl.replace(0, np.nan)
                            cr = cr.reindex(base_index)
                            piotroski_score = piotroski_score + (cr.diff(4) > 0).fillna(False).astype(float)

                    # Improving margin
                    if revenue_row is not None:
                        rev = coerce_quarter_cols(revenue_row)
                        if not rev.empty:
                            rev_ttm = sum_ttm(rev).reindex(base_index)
                            margin = ni_ttm.reindex(base_index) / rev_ttm.replace(0, np.nan)
                            piotroski_score = piotroski_score + (margin.diff(4) > 0).fillna(False).astype(float)

                            # Improving asset turnover
                            turnover = rev_ttm / ta_avg.reindex(base_index).replace(0, np.nan)
                            piotroski_score = piotroski_score + (turnover.diff(4) > 0).fillna(False).astype(float)

                    piotroski_score = piotroski_score.replace([np.inf, -np.inf], np.nan).dropna()
                    if not piotroski_score.empty:
                        piotroski_panel[ticker] = apply_lag(piotroski_score, dates)
                        ticker_has_data = True

            if ticker_has_data:
                success_count += 1

        except KeyError:
            # Ticker not found in fundamentals
            continue
        except (ValueError, TypeError):
            # Data conversion issues - skip this ticker
            continue

        count += 1
        if count % 50 == 0:
            logger.info(f"    Quality progress: {count}/{len(tickers)} ({success_count} with data)")

    logger.info(f"    Quality panels built: {success_count}/{len(tickers)} tickers with data")

    return _finalize_builder_outputs(
        "build_quality_panels",
        {
            "quality_roe": roe_panel,
            "quality_roa": roa_panel,
            "quality_accruals": accruals_panel,
            "quality_piotroski": piotroski_panel,
        },
        FACTOR_PANEL_CONTRACT["quality"],
        dates,
        tickers,
    )


# ============================================================================
# LIQUIDITY FACTOR PANELS
# ============================================================================

def _load_shares_panel(
    data_loader,
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
) -> tuple[pd.DataFrame, bool]:
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


def build_liquidity_panels(
    close: pd.DataFrame,
    volume_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
    data_loader=None,
) -> Dict[str, pd.DataFrame]:
    """
    Build Liquidity factor panels: Amihud illiquidity, real turnover, bid-ask spread proxy.

    Real turnover = volume / shares_outstanding (from SERMAYE column in isyatirim data)
    This is distinct from trading intensity which uses relative volume measures.
    """
    dates, tickers = _validate_reference_axes(dates, tickers, "build_liquidity_panels")
    close = _align_numeric_panel(close, "close", dates, tickers)

    logger.info("  Building liquidity factor panels...")

    daily_returns = close.pct_change().abs()
    volume = _align_numeric_panel(volume_df, "volume_df", dates, tickers)

    # Amihud illiquidity = |return| / dollar volume (lower = more liquid)
    dollar_volume = close * volume
    amihud_daily = daily_returns / dollar_volume.replace(0, np.nan)
    amihud_panel = amihud_daily.rolling(AMIHUD_LOOKBACK_DAYS, min_periods=10).mean()
    # Log transform to reduce skewness
    amihud_panel = np.log1p(amihud_panel * 1e6).replace([np.inf, -np.inf], np.nan)

    # Real Turnover = volume / shares_outstanding
    turnover_panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)

    shares_aligned, has_shares = _load_shares_panel(data_loader, dates, tickers)
    if has_shares:
        # Real turnover = daily volume / shares outstanding
        real_turnover = volume / shares_aligned.replace(0, np.nan)
        # Smooth with 63-day rolling mean
        turnover_panel = real_turnover.rolling(TURNOVER_LOOKBACK_DAYS, min_periods=21).mean()
        turnover_panel = turnover_panel.replace([np.inf, -np.inf], np.nan)
        logger.info("    Real turnover computed using shares outstanding")
    else:
        logger.warning("    ⚠️  Shares outstanding not available - turnover panel will be empty")

    # Bid-ask spread proxy: high-low range relative to close
    # This captures the spread cost component of liquidity
    # Note: requires high/low data which may not be in close df
    spread_proxy_panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)

    logger.info(f"    Liquidity panels: Amihud {amihud_panel.notna().sum().sum()}, "
          f"Turnover {turnover_panel.notna().sum().sum()} data points")

    return _finalize_builder_outputs(
        "build_liquidity_panels",
        {
            "liquidity_amihud": amihud_panel,
            "liquidity_turnover": turnover_panel,
            "liquidity_spread_proxy": spread_proxy_panel,
        },
        FACTOR_PANEL_CONTRACT["liquidity"],
        dates,
        tickers,
    )


# ============================================================================
# TRADING INTENSITY PANELS
# ============================================================================

def build_trading_intensity_panels(
    close: pd.DataFrame,
    volume_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
    data_loader=None,
) -> Dict[str, pd.DataFrame]:
    """
    Build Trading Intensity panels: relative volume, volume trend, turnover velocity.

    Trading intensity measures how actively a stock is being traded relative to:
    1. Its own historical volume (relative volume)
    2. Recent vs older trading activity (volume trend)
    3. How fast shares change hands (turnover velocity using shares outstanding)

    This is conceptually different from liquidity:
    - Liquidity = ease of trading without price impact (Amihud, spreads)
    - Trading Intensity = level of trading activity / attention
    """
    dates, tickers = _validate_reference_axes(dates, tickers, "build_trading_intensity_panels")
    del close  # Not needed for this panel family.

    logger.info("  Building trading intensity panels...")

    volume = _align_numeric_panel(volume_df, "volume_df", dates, tickers)

    # Relative Volume = volume / avg volume (252-day baseline)
    # High relative volume indicates unusual trading activity
    avg_volume = volume.rolling(252, min_periods=63).mean()
    relative_volume = (volume / avg_volume.replace(0, np.nan)).rolling(
        TURNOVER_LOOKBACK_DAYS, min_periods=21
    ).mean()
    relative_volume = relative_volume.replace([np.inf, -np.inf], np.nan)

    # Volume Trend = short-term avg / long-term avg - 1
    # Positive trend means increasing trading activity
    short_vol = volume.rolling(21, min_periods=10).mean()
    long_vol = volume.rolling(126, min_periods=42).mean()
    volume_trend = (short_vol / long_vol.replace(0, np.nan) - 1.0).replace([np.inf, -np.inf], np.nan)

    # Turnover Velocity = real turnover rate (volume / shares outstanding)
    # How fast the float is turning over
    turnover_velocity = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)

    shares_aligned, has_shares = _load_shares_panel(data_loader, dates, tickers)
    if has_shares:
        # Daily turnover rate
        daily_turnover = volume / shares_aligned.replace(0, np.nan)
        # Smooth and annualize (252 trading days)
        turnover_velocity = daily_turnover.rolling(21, min_periods=10).mean() * 252
        turnover_velocity = turnover_velocity.replace([np.inf, -np.inf], np.nan)
        logger.info("    Turnover velocity computed using shares outstanding")
    else:
        logger.warning("    ⚠️  Shares outstanding not available - turnover velocity will be empty")

    logger.info(f"    Trading intensity panels: RelVol {relative_volume.notna().sum().sum()}, "
          f"VolTrend {volume_trend.notna().sum().sum()}, "
          f"TurnoverVel {turnover_velocity.notna().sum().sum()} data points")

    return _finalize_builder_outputs(
        "build_trading_intensity_panels",
        {
            "trading_intensity_relative_volume": relative_volume,
            "trading_intensity_volume_trend": volume_trend,
            "trading_intensity_turnover_velocity": turnover_velocity,
        },
        FACTOR_PANEL_CONTRACT["trading_intensity"],
        dates,
        tickers,
    )


# ============================================================================
# SENTIMENT / PRICE ACTION PANELS
# ============================================================================

def build_sentiment_panels(
    close: pd.DataFrame,
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
) -> Dict[str, pd.DataFrame]:
    """
    Build Sentiment/Price Action panels: 52-week high proximity, price acceleration, reversal.
    """
    dates, tickers = _validate_reference_axes(dates, tickers, "build_sentiment_panels")
    close = _align_numeric_panel(close, "close", dates, tickers)

    logger.info("  Building sentiment/price action panels...")

    # 52-week high proximity: current price / 52-week high
    rolling_high = close.rolling(252, min_periods=126).max()
    high_proximity = close / rolling_high.replace(0, np.nan)
    high_proximity = high_proximity.replace([np.inf, -np.inf], np.nan)

    # Price acceleration: short-term momentum - medium-term momentum
    mom_fast = close.pct_change(PRICE_ACCELERATION_FAST)
    mom_slow = close.pct_change(PRICE_ACCELERATION_SLOW)
    price_acceleration = mom_fast - mom_slow

    # Short-term reversal: negative of very short-term return (mean reversion)
    reversal = -close.pct_change(REVERSAL_LOOKBACK_DAYS)

    logger.info(f"    Sentiment panels: 52wHigh {high_proximity.notna().sum().sum()}, "
          f"Accel {price_acceleration.notna().sum().sum()}, "
          f"Reversal {reversal.notna().sum().sum()} data points")

    return _finalize_builder_outputs(
        "build_sentiment_panels",
        {
            "sentiment_52w_high_pct": high_proximity,
            "sentiment_price_acceleration": price_acceleration,
            "sentiment_reversal": reversal,
        },
        FACTOR_PANEL_CONTRACT["sentiment"],
        dates,
        tickers,
    )


# ============================================================================
# FUNDAMENTAL MOMENTUM PANELS
# ============================================================================

def build_fundamental_momentum_panels(
    fundamentals: Dict,
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
    data_loader,
) -> Dict[str, pd.DataFrame]:
    """
    Build Fundamental Momentum panels: margin change, sales growth acceleration.
    """
    del fundamentals  # Unused in current implementation.
    dates, tickers = _validate_reference_axes(dates, tickers, "build_fundamental_momentum_panels")

    logger.info("  Building fundamental momentum panels...")
    fundamentals_parquet = data_loader.load_fundamentals_parquet() if data_loader is not None else None

    margin_change_panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)
    sales_accel_panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)

    if fundamentals_parquet is None:
        logger.warning("    ⚠️  No fundamentals parquet - fundmom panels will be empty")
        return _finalize_builder_outputs(
            "build_fundamental_momentum_panels",
            {
                "fundmom_margin_change": margin_change_panel,
                "fundmom_sales_accel": sales_accel_panel,
            },
            FACTOR_PANEL_CONTRACT["fundmom"],
            dates,
            tickers,
        )

    count = 0
    success_count = 0
    for ticker in tickers:
        try:
            inc = get_consolidated_sheet(fundamentals_parquet, ticker, INCOME_SHEET)
            if inc.empty:
                continue

            revenue_row = pick_row_from_sheet(inc, REVENUE_KEYS)
            op_income_row = pick_row_from_sheet(inc, OPERATING_INCOME_KEYS)

            if revenue_row is not None and op_income_row is not None:
                rev = coerce_quarter_cols(revenue_row)
                op = coerce_quarter_cols(op_income_row)

                if not rev.empty and not op.empty:
                    rev_ttm = sum_ttm(rev)
                    op_ttm = sum_ttm(op)

                    # Operating margin and its YoY change
                    op_margin = op_ttm / rev_ttm.replace(0, np.nan)
                    margin_change = op_margin.diff(MARGIN_CHANGE_QUARTERS)
                    margin_change = margin_change.replace([np.inf, -np.inf], np.nan)
                    if not margin_change.dropna().empty:
                        margin_change_panel[ticker] = apply_lag(margin_change, dates)

                    # Sales growth and its acceleration
                    sales_growth = rev_ttm.pct_change(SALES_ACCEL_QUARTERS)
                    sales_accel = sales_growth.diff(SALES_ACCEL_QUARTERS)
                    sales_accel = sales_accel.replace([np.inf, -np.inf], np.nan)
                    if not sales_accel.dropna().empty:
                        sales_accel_panel[ticker] = apply_lag(sales_accel, dates)

                    success_count += 1

        except KeyError:
            # Ticker not found in fundamentals
            continue
        except (ValueError, TypeError):
            # Data conversion issues
            continue

        count += 1
        if count % 50 == 0:
            logger.info(f"    FundMom progress: {count}/{len(tickers)} ({success_count} with data)")

    logger.info(f"    FundMom panels built: {success_count}/{len(tickers)} tickers with data")

    return _finalize_builder_outputs(
        "build_fundamental_momentum_panels",
        {
            "fundmom_margin_change": margin_change_panel,
            "fundmom_sales_accel": sales_accel_panel,
        },
        FACTOR_PANEL_CONTRACT["fundmom"],
        dates,
        tickers,
    )


# ============================================================================
# CARRY FACTOR PANELS
# ============================================================================

def build_carry_panels(
    fundamentals: Dict,
    close: pd.DataFrame,
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
    data_loader,
) -> Dict[str, pd.DataFrame]:
    """
    Build Carry factor panels: dividend yield.

    Note: In Turkey, buyback data is rarely available, so we focus on dividends.
    """
    del fundamentals  # Unused in current implementation.
    dates, tickers = _validate_reference_axes(dates, tickers, "build_carry_panels")
    close = _align_numeric_panel(close, "close", dates, tickers)

    logger.info("  Building carry factor panels...")

    div_yield_panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)

    # Try to load dividend yield from fundamental metrics first
    try:
        metrics_df = data_loader.load_fundamental_metrics()
        if not metrics_df.empty and "dividend_yield" in metrics_df.columns:
            logger.info("    Loading dividend yield from fundamental metrics...")
            available_tickers = set(metrics_df.index.get_level_values(0))
            for ticker in tickers:
                if ticker not in available_tickers:
                    continue
                try:
                    series = metrics_df.loc[ticker, "dividend_yield"]
                    if isinstance(series, pd.DataFrame):
                        series = series.iloc[:, 0]
                    series = series.sort_index()
                    series = series[~series.index.duplicated(keep="last")]
                    series.index = pd.to_datetime(series.index)
                    div_yield_panel[ticker] = apply_lag(series, dates)
                except Exception:
                    continue
    except Exception as e:
        logger.warning(f"    ⚠️  Could not load dividend metrics: {e}")

    # If we got data from metrics, we're done
    metrics_count = div_yield_panel.notna().sum().sum()
    if metrics_count > 1000:
        logger.info(f"    Carry panels from metrics: {metrics_count} data points")
        return _finalize_builder_outputs(
            "build_carry_panels",
            {
                "carry_dividend_yield": div_yield_panel,
                "carry_shareholder_yield": div_yield_panel.copy(),  # Same as div yield for Turkey
            },
            FACTOR_PANEL_CONTRACT["carry"],
            dates,
            tickers,
        )

    # Otherwise try to build from cash flow statements
    logger.info("    Building dividend yield from cash flow statements + shares outstanding...")
    fundamentals_parquet = data_loader.load_fundamentals_parquet() if data_loader is not None else None

    if fundamentals_parquet is None:
        logger.warning("    ⚠️  No fundamentals parquet - carry panels will be empty")
        return _finalize_builder_outputs(
            "build_carry_panels",
            {
                "carry_dividend_yield": div_yield_panel,
                "carry_shareholder_yield": div_yield_panel.copy(),
            },
            FACTOR_PANEL_CONTRACT["carry"],
            dates,
            tickers,
        )

    count = 0
    success_count = 0
    for ticker in tickers:
        # Skip if already have data
        if div_yield_panel[ticker].notna().sum() > 100:
            success_count += 1
            continue

        try:
            cf = get_consolidated_sheet(fundamentals_parquet, ticker, CASH_FLOW_SHEET)
            if cf.empty:
                continue

            div_row = pick_row_from_sheet(cf, DIVIDENDS_PAID_KEYS)
            if div_row is not None:
                divs = coerce_quarter_cols(div_row).abs()  # dividends paid are usually negative
                if not divs.empty:
                    div_ttm = sum_ttm(divs)
                    if div_ttm.empty:
                        continue

                    # Get shares outstanding (check method exists first)
                    shares = None
                    if data_loader and hasattr(data_loader, "load_shares_outstanding"):
                        shares = data_loader.load_shares_outstanding(ticker)
                    if shares is None or shares.empty:
                        continue

                    shares = shares.sort_index()
                    shares = shares[~shares.index.duplicated(keep="last")]

                    # Get price for this ticker
                    if ticker not in close.columns:
                        continue
                    price = close[ticker].reindex(dates)

                    # Apply lag to div_ttm to get it on daily dates
                    div_ttm_daily = apply_lag(div_ttm, dates)
                    if div_ttm_daily.empty or div_ttm_daily.isna().all():
                        continue

                    # Align shares to dates (using ffill() instead of deprecated method param)
                    shares_daily = shares.reindex(dates).ffill()

                    # Market cap = price * shares
                    mcap = price * shares_daily

                    # Dividend yield = TTM dividends / market cap
                    div_yield = div_ttm_daily / mcap.replace(0, np.nan)
                    div_yield = div_yield.replace([np.inf, -np.inf], np.nan)

                    # Clip extreme values (yield > 50% is suspicious)
                    div_yield = div_yield.clip(upper=0.5)

                    if div_yield.notna().sum() > 50:
                        div_yield_panel[ticker] = div_yield
                        success_count += 1

        except KeyError:
            # Ticker not found
            continue
        except (ValueError, TypeError):
            # Data conversion issues
            continue

        count += 1
        if count % 50 == 0:
            logger.info(f"    Carry progress: {count}/{len(tickers)} ({success_count} with data)")

    logger.info(f"    Carry panels built: {success_count}/{len(tickers)} tickers with data")

    return _finalize_builder_outputs(
        "build_carry_panels",
        {
            "carry_dividend_yield": div_yield_panel,
            "carry_shareholder_yield": div_yield_panel.copy(),
        },
        FACTOR_PANEL_CONTRACT["carry"],
        dates,
        tickers,
    )


# ============================================================================
# DEFENSIVE FACTOR PANELS
# ============================================================================

def build_defensive_panels(
    fundamentals: Dict,
    close: pd.DataFrame,
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
    data_loader,
) -> Dict[str, pd.DataFrame]:
    """
    Build Defensive/Cyclical panels: earnings stability, beta to market.
    """
    del fundamentals  # Unused in current implementation.
    dates, tickers = _validate_reference_axes(dates, tickers, "build_defensive_panels")
    close = _align_numeric_panel(close, "close", dates, tickers)

    logger.info("  Building defensive factor panels...")
    fundamentals_parquet = data_loader.load_fundamentals_parquet() if data_loader is not None else None

    stability_panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)

    if fundamentals_parquet is not None:
        count = 0
        success_count = 0
        for ticker in tickers:
            try:
                inc = get_consolidated_sheet(fundamentals_parquet, ticker, INCOME_SHEET)
                if inc.empty:
                    continue

                net_income_row = pick_row_from_sheet(inc, NET_INCOME_KEYS)
                if net_income_row is not None:
                    ni = coerce_quarter_cols(net_income_row)
                    if not ni.empty:
                        ni_ttm = sum_ttm(ni)

                        # Rolling earnings stability (inverse of coefficient of variation)
                        rolling_mean = ni_ttm.rolling(EARNINGS_STABILITY_QUARTERS, min_periods=4).mean()
                        rolling_std = ni_ttm.rolling(EARNINGS_STABILITY_QUARTERS, min_periods=4).std()
                        cv = rolling_std / rolling_mean.abs().replace(0, np.nan)
                        stability = 1.0 / cv.replace(0, np.nan)
                        stability = stability.replace([np.inf, -np.inf], np.nan).clip(-10, 10)

                        if not stability.dropna().empty:
                            stability_panel[ticker] = apply_lag(stability, dates)
                            success_count += 1

            except KeyError:
                # Ticker not found
                continue
            except (ValueError, TypeError):
                # Data conversion issues
                continue

            count += 1
            if count % 50 == 0:
                logger.info(f"    Defensive progress: {count}/{len(tickers)} ({success_count} with data)")

        logger.info(f"    Earnings stability built: {success_count}/{len(tickers)} tickers with data")

    # Beta to market
    beta_panel = build_market_beta_panel(close, dates, data_loader)

    return _finalize_builder_outputs(
        "build_defensive_panels",
        {
            "defensive_earnings_stability": stability_panel,
            "defensive_beta_to_market": beta_panel,
        },
        FACTOR_PANEL_CONTRACT["defensive"],
        dates,
        tickers,
    )


# ============================================================================
# VOLATILITY AND BETA PANELS
# ============================================================================

def build_realized_volatility_panel(
    close: pd.DataFrame,
    dates: pd.DatetimeIndex,
    lookback: int = VOLATILITY_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Build rolling realized volatility panel (annualized)."""
    dates, tickers = _validate_reference_axes(dates, close.columns, "build_realized_volatility_panel")
    close = _align_numeric_panel(close, "close", dates, tickers)

    daily_returns = close.pct_change()
    min_obs = max(int(lookback * MIN_ROLLING_OBS_RATIO), 21)
    vol = daily_returns.rolling(lookback, min_periods=min_obs).std() * np.sqrt(252)
    return _align_numeric_panel(vol, "realized_volatility", dates, tickers)


def build_market_beta_panel(
    close: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
    lookback: int = BETA_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Build rolling market beta panel.

    Note: This uses a loop over tickers which can be slow for large universes.
    Consider vectorization if performance is critical.
    """
    del data_loader  # unused
    dates, tickers = _validate_reference_axes(dates, close.columns, "build_market_beta_panel")
    close = _align_numeric_panel(close, "close", dates, tickers)

    daily_returns = close.pct_change()
    min_obs = max(int(lookback * MIN_ROLLING_OBS_RATIO), BETA_MIN_OBS)

    # Equal-weighted market return as benchmark
    market_return = daily_returns.mean(axis=1)
    market_var = market_return.rolling(lookback, min_periods=min_obs).var()

    beta_panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)

    for ticker in tickers:
        stock_ret = daily_returns[ticker]
        cov = stock_ret.rolling(lookback, min_periods=min_obs).cov(market_return)
        beta = cov / market_var.replace(0.0, np.nan)
        beta_panel[ticker] = beta.reindex(dates)

    beta_panel = beta_panel.clip(lower=-2.0, upper=5.0)
    return _align_numeric_panel(beta_panel, "market_beta", dates, tickers)


def build_volatility_beta_panels(
    close: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build realized volatility and market beta panels for risk axis."""
    dates, tickers = _validate_reference_axes(dates, close.columns, "build_volatility_beta_panels")
    close = _align_numeric_panel(close, "close", dates, tickers)

    logger.info("  Building volatility panel...")
    vol_panel = build_realized_volatility_panel(close, dates)

    logger.info("  Building market beta panel...")
    beta_panel = build_market_beta_panel(close, dates, data_loader)

    return vol_panel, beta_panel
