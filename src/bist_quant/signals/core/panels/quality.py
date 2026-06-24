"""Quality factor panels: ROE, ROA, accruals, Piotroski F-score."""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

from bist_quant.common.utils import (
    apply_lag,
    coerce_quarter_cols,
    pick_row,
    pick_row_from_sheet,
    sum_ttm,
)
from bist_quant.signals.fundamental_keys import (
    CFO_KEYS,
    CURRENT_ASSETS_KEYS,
    CURRENT_LIABILITIES_KEYS,
    LONG_TERM_DEBT_KEYS,
    NET_INCOME_KEYS,
    REVENUE_KEYS,
    TOTAL_ASSETS_KEYS,
    TOTAL_EQUITY_KEYS,
)

logger = logging.getLogger(__name__)


def build_quality_panels(
    fundamentals: Dict,
    close: pd.DataFrame,
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
    data_loader,
) -> Dict[str, pd.DataFrame]:
    """Build Quality factor panels: ROE, ROA, Accruals, Piotroski F-score."""
    # Lazy imports avoid circular dependency with factor_builders facade.
    from bist_quant.signals.factor_builders import (
        FACTOR_PANEL_CONTRACT,
        _finalize_builder_outputs,
        _load_ticker_fundamental_sheets,
        _validate_reference_axes,
    )

    del close
    fundamentals = fundamentals or {}
    dates, tickers = _validate_reference_axes(dates, tickers, "build_quality_panels")

    logger.info("  Building quality factor panels...")
    fundamentals_parquet = data_loader.load_fundamentals_parquet() if data_loader is not None else None

    if fundamentals_parquet is None:
        logger.warning("    ⚠️  No fundamentals parquet - falling back to per-ticker Excel files")

    roe_panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)
    roa_panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)
    accruals_panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)
    piotroski_panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)

    count = 0
    success_count = 0
    for ticker in tickers:
        try:
            sheets = _load_ticker_fundamental_sheets(ticker, fundamentals_parquet, fundamentals)
            if sheets is None:
                continue
            inc, bs, cf, use_parquet = sheets

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

            if net_income_row is not None and total_equity_row is not None:
                ni = coerce_quarter_cols(net_income_row)
                eq = coerce_quarter_cols(total_equity_row)
                if not ni.empty and not eq.empty:
                    ni_ttm = sum_ttm(ni)
                    eq_avg = eq.rolling(4, min_periods=2).mean()
                    ni_ttm, eq_avg = ni_ttm.align(eq_avg, join="inner")
                    roe = (ni_ttm / eq_avg.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
                    roe = roe.dropna()
                    if not roe.empty:
                        roe_panel[ticker] = apply_lag(roe, dates)
                        ticker_has_data = True

            if net_income_row is not None and total_assets_row is not None:
                ni = coerce_quarter_cols(net_income_row)
                ta = coerce_quarter_cols(total_assets_row)
                if not ni.empty and not ta.empty:
                    ni_ttm = sum_ttm(ni)
                    ta_avg = ta.rolling(4, min_periods=2).mean()
                    ni_ttm, ta_avg = ni_ttm.align(ta_avg, join="inner")
                    roa = (ni_ttm / ta_avg.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
                    roa = roa.dropna()
                    if not roa.empty:
                        roa_panel[ticker] = apply_lag(roa, dates)
                        ticker_has_data = True

            if net_income_row is not None and cfo_row is not None and total_assets_row is not None:
                ni = coerce_quarter_cols(net_income_row)
                cfo = coerce_quarter_cols(cfo_row)
                ta = coerce_quarter_cols(total_assets_row)
                if not ni.empty and not cfo.empty and not ta.empty:
                    ni_ttm = sum_ttm(ni)
                    cfo_ttm = sum_ttm(cfo)
                    ta_avg = ta.rolling(4, min_periods=2).mean()
                    ni_ttm, cfo_ttm = ni_ttm.align(cfo_ttm, join="inner")
                    ni_ttm, ta_avg = ni_ttm.align(ta_avg, join="inner")
                    cfo_ttm = cfo_ttm.reindex(ni_ttm.index)
                    accruals = ((ni_ttm - cfo_ttm) / ta_avg.replace(0, np.nan)).replace(
                        [np.inf, -np.inf], np.nan
                    )
                    accruals = accruals.dropna()
                    if not accruals.empty:
                        accruals_panel[ticker] = apply_lag(accruals, dates)
                        ticker_has_data = True

            if net_income_row is not None and total_assets_row is not None:
                ni = coerce_quarter_cols(net_income_row)
                ta = coerce_quarter_cols(total_assets_row)
                if not ni.empty and not ta.empty:
                    ni_ttm = sum_ttm(ni)
                    ta_avg = ta.rolling(4, min_periods=2).mean()
                    roa_check = ni_ttm / ta_avg.replace(0, np.nan)

                    base_index = roa_check.index
                    piotroski_score = (roa_check > 0).astype(float)

                    if cfo_row is not None:
                        cfo = coerce_quarter_cols(cfo_row)
                        if not cfo.empty:
                            cfo_ttm = sum_ttm(cfo).reindex(base_index)
                            piotroski_score = piotroski_score + (cfo_ttm > 0).fillna(False).astype(float)
                            ni_ttm_aligned = ni_ttm.reindex(base_index)
                            piotroski_score = piotroski_score + (
                                (cfo_ttm > ni_ttm_aligned).fillna(False).astype(float)
                            )

                    piotroski_score = piotroski_score + (roa_check.diff(4) > 0).fillna(False).astype(float)

                    if long_debt_row is not None:
                        debt = coerce_quarter_cols(long_debt_row)
                        if not debt.empty:
                            ta_avg_aligned = ta_avg.reindex(debt.index)
                            leverage = debt / ta_avg_aligned.replace(0, np.nan)
                            leverage = leverage.reindex(base_index)
                            piotroski_score = piotroski_score + (leverage.diff(4) < 0).fillna(False).astype(float)

                    if current_assets_row is not None and current_liab_row is not None:
                        ca = coerce_quarter_cols(current_assets_row)
                        cl = coerce_quarter_cols(current_liab_row)
                        if not ca.empty and not cl.empty:
                            ca, cl = ca.align(cl, join="inner")
                            cr = ca / cl.replace(0, np.nan)
                            cr = cr.reindex(base_index)
                            piotroski_score = piotroski_score + (cr.diff(4) > 0).fillna(False).astype(float)

                    if revenue_row is not None:
                        rev = coerce_quarter_cols(revenue_row)
                        if not rev.empty:
                            rev_ttm = sum_ttm(rev).reindex(base_index)
                            margin = ni_ttm.reindex(base_index) / rev_ttm.replace(0, np.nan)
                            piotroski_score = piotroski_score + (margin.diff(4) > 0).fillna(False).astype(float)

                            turnover = rev_ttm / ta_avg.reindex(base_index).replace(0, np.nan)
                            piotroski_score = piotroski_score + (turnover.diff(4) > 0).fillna(False).astype(float)

                    piotroski_score = piotroski_score.replace([np.inf, -np.inf], np.nan).dropna()
                    if not piotroski_score.empty:
                        piotroski_panel[ticker] = apply_lag(piotroski_score, dates)
                        ticker_has_data = True

            if ticker_has_data:
                success_count += 1

        except KeyError:
            continue
        except (ValueError, TypeError):
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


__all__ = ["build_quality_panels"]
