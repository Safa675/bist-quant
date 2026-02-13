"""
Factor Panel Builders

Functions to build raw factor panels from price data and fundamentals.
These panels are cached to avoid recomputation.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.utils import (
    apply_lag,
    coerce_quarter_cols,
    get_consolidated_sheet,
    pick_row_from_sheet,
    sum_ttm,
)


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
    print("  Building quality factor panels...")
    fundamentals_parquet = data_loader.load_fundamentals_parquet() if data_loader is not None else None

    roe_panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)
    roa_panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)
    accruals_panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)
    piotroski_panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)

    if fundamentals_parquet is None:
        print("    ⚠️  No fundamentals parquet - quality panels will be empty")
        return {
            "quality_roe": roe_panel,
            "quality_roa": roa_panel,
            "quality_accruals": accruals_panel,
            "quality_piotroski": piotroski_panel,
        }

    count = 0
    success_count = 0
    for ticker in tickers:
        try:
            # Get financial statements
            inc = get_consolidated_sheet(fundamentals_parquet, ticker, INCOME_SHEET)
            bs = get_consolidated_sheet(fundamentals_parquet, ticker, BALANCE_SHEET)
            cf = get_consolidated_sheet(fundamentals_parquet, ticker, CASH_FLOW_SHEET)

            if inc.empty or bs.empty:
                continue

            # Extract key rows
            net_income_row = pick_row_from_sheet(inc, NET_INCOME_KEYS)
            revenue_row = pick_row_from_sheet(inc, REVENUE_KEYS)
            total_assets_row = pick_row_from_sheet(bs, TOTAL_ASSETS_KEYS)
            total_equity_row = pick_row_from_sheet(bs, TOTAL_EQUITY_KEYS)
            current_assets_row = pick_row_from_sheet(bs, CURRENT_ASSETS_KEYS)
            current_liab_row = pick_row_from_sheet(bs, CURRENT_LIABILITIES_KEYS)
            long_debt_row = pick_row_from_sheet(bs, LONG_TERM_DEBT_KEYS)

            cfo_row = None
            if not cf.empty:
                cfo_row = pick_row_from_sheet(cf, CFO_KEYS)

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
            print(f"    Quality progress: {count}/{len(tickers)} ({success_count} with data)")

    print(f"    Quality panels built: {success_count}/{len(tickers)} tickers with data")

    return {
        "quality_roe": roe_panel,
        "quality_roa": roa_panel,
        "quality_accruals": accruals_panel,
        "quality_piotroski": piotroski_panel,
    }


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

    try:
        shares_outstanding = data_loader.load_shares_outstanding_panel()
    except Exception as exc:
        print(f"    ⚠️  Failed to load shares outstanding panel: {exc}")
        return empty_panel, False

    if shares_outstanding is None or shares_outstanding.empty:
        return empty_panel, False

    shares = shares_outstanding.copy()
    shares.index = pd.to_datetime(shares.index, errors="coerce")
    shares = shares.sort_index()
    shares.columns = pd.Index([str(c).upper() for c in shares.columns])
    aligned = shares.reindex(index=dates, columns=tickers).ffill()
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
    print("  Building liquidity factor panels...")

    daily_returns = close.pct_change().abs()
    volume = volume_df.reindex(index=dates, columns=tickers)

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
        print(f"    Real turnover computed using shares outstanding")
    else:
        print(f"    ⚠️  Shares outstanding not available - turnover panel will be empty")

    # Bid-ask spread proxy: high-low range relative to close
    # This captures the spread cost component of liquidity
    # Note: requires high/low data which may not be in close df
    spread_proxy_panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)

    print(f"    Liquidity panels: Amihud {amihud_panel.notna().sum().sum()}, "
          f"Turnover {turnover_panel.notna().sum().sum()} data points")

    return {
        "liquidity_amihud": amihud_panel.reindex(dates),
        "liquidity_turnover": turnover_panel.reindex(dates),
        "liquidity_spread_proxy": spread_proxy_panel.reindex(dates),
    }


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
    print("  Building trading intensity panels...")

    volume = volume_df.reindex(index=dates, columns=tickers)

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
        print(f"    Turnover velocity computed using shares outstanding")
    else:
        print(f"    ⚠️  Shares outstanding not available - turnover velocity will be empty")

    print(f"    Trading intensity panels: RelVol {relative_volume.notna().sum().sum()}, "
          f"VolTrend {volume_trend.notna().sum().sum()}, "
          f"TurnoverVel {turnover_velocity.notna().sum().sum()} data points")

    return {
        "trading_intensity_relative_volume": relative_volume.reindex(dates),
        "trading_intensity_volume_trend": volume_trend.reindex(dates),
        "trading_intensity_turnover_velocity": turnover_velocity.reindex(dates),
    }


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
    print("  Building sentiment/price action panels...")

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

    print(f"    Sentiment panels: 52wHigh {high_proximity.notna().sum().sum()}, "
          f"Accel {price_acceleration.notna().sum().sum()}, "
          f"Reversal {reversal.notna().sum().sum()} data points")

    return {
        "sentiment_52w_high_pct": high_proximity.reindex(dates),
        "sentiment_price_acceleration": price_acceleration.reindex(dates),
        "sentiment_reversal": reversal.reindex(dates),
    }


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
    print("  Building fundamental momentum panels...")
    fundamentals_parquet = data_loader.load_fundamentals_parquet() if data_loader is not None else None

    margin_change_panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)
    sales_accel_panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)

    if fundamentals_parquet is None:
        print("    ⚠️  No fundamentals parquet - fundmom panels will be empty")
        return {
            "fundmom_margin_change": margin_change_panel,
            "fundmom_sales_accel": sales_accel_panel,
        }

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
            print(f"    FundMom progress: {count}/{len(tickers)} ({success_count} with data)")

    print(f"    FundMom panels built: {success_count}/{len(tickers)} tickers with data")

    return {
        "fundmom_margin_change": margin_change_panel,
        "fundmom_sales_accel": sales_accel_panel,
    }


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
    print("  Building carry factor panels...")

    div_yield_panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)

    # Try to load dividend yield from fundamental metrics first
    try:
        metrics_df = data_loader.load_fundamental_metrics()
        if not metrics_df.empty and "dividend_yield" in metrics_df.columns:
            print("    Loading dividend yield from fundamental metrics...")
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
        print(f"    ⚠️  Could not load dividend metrics: {e}")

    # If we got data from metrics, we're done
    metrics_count = div_yield_panel.notna().sum().sum()
    if metrics_count > 1000:
        print(f"    Carry panels from metrics: {metrics_count} data points")
        return {
            "carry_dividend_yield": div_yield_panel,
            "carry_shareholder_yield": div_yield_panel.copy(),  # Same as div yield for Turkey
        }

    # Otherwise try to build from cash flow statements
    print("    Building dividend yield from cash flow statements + shares outstanding...")
    fundamentals_parquet = data_loader.load_fundamentals_parquet() if data_loader is not None else None

    if fundamentals_parquet is None:
        print("    ⚠️  No fundamentals parquet - carry panels will be empty")
        return {
            "carry_dividend_yield": div_yield_panel,
            "carry_shareholder_yield": div_yield_panel.copy(),
        }

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
            print(f"    Carry progress: {count}/{len(tickers)} ({success_count} with data)")

    print(f"    Carry panels built: {success_count}/{len(tickers)} tickers with data")

    return {
        "carry_dividend_yield": div_yield_panel,
        "carry_shareholder_yield": div_yield_panel.copy(),
    }


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
    print("  Building defensive factor panels...")
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
                print(f"    Defensive progress: {count}/{len(tickers)} ({success_count} with data)")

        print(f"    Earnings stability built: {success_count}/{len(tickers)} tickers with data")

    # Beta to market
    beta_panel = build_market_beta_panel(close, dates, data_loader)

    return {
        "defensive_earnings_stability": stability_panel,
        "defensive_beta_to_market": beta_panel,
    }


# ============================================================================
# VOLATILITY AND BETA PANELS
# ============================================================================

def build_realized_volatility_panel(
    close: pd.DataFrame,
    dates: pd.DatetimeIndex,
    lookback: int = VOLATILITY_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Build rolling realized volatility panel (annualized)."""
    daily_returns = close.pct_change()
    min_obs = max(int(lookback * MIN_ROLLING_OBS_RATIO), 21)
    vol = daily_returns.rolling(lookback, min_periods=min_obs).std() * np.sqrt(252)
    # Ensure both rows and columns are properly aligned
    return vol.reindex(index=dates, columns=close.columns)


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
    daily_returns = close.pct_change()
    min_obs = max(int(lookback * MIN_ROLLING_OBS_RATIO), BETA_MIN_OBS)

    # Equal-weighted market return as benchmark
    market_return = daily_returns.mean(axis=1)
    market_var = market_return.rolling(lookback, min_periods=min_obs).var()

    beta_panel = pd.DataFrame(np.nan, index=dates, columns=close.columns, dtype=float)

    for ticker in close.columns:
        stock_ret = daily_returns[ticker]
        cov = stock_ret.rolling(lookback, min_periods=min_obs).cov(market_return)
        beta = cov / market_var.replace(0.0, np.nan)
        beta_panel[ticker] = beta.reindex(dates)

    beta_panel = beta_panel.clip(lower=-2.0, upper=5.0)
    return beta_panel


def build_volatility_beta_panels(
    close: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build realized volatility and market beta panels for risk axis."""
    print("  Building volatility panel...")
    vol_panel = build_realized_volatility_panel(close, dates)

    print("  Building market beta panel...")
    beta_panel = build_market_beta_panel(close, dates, data_loader)

    return vol_panel, beta_panel
