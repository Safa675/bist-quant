"""
Quality Factor Signal

Economic Intuition:
------------------
Quality investing focuses on companies with sustainable competitive advantages,
strong balance sheets, and consistent profitability. Quality stocks tend to
outperform because:
1. They have pricing power and defensible margins
2. Less prone to earnings manipulation
3. Better able to weather economic downturns
4. Compound returns more reliably

Quality Components:
1. ROE - Return on Equity (profitability of shareholder capital)
2. ROA - Return on Assets (overall capital efficiency)
3. Accruals - Low accruals indicate high earnings quality (cash-backed)
4. Piotroski F-Score - 8-point fundamental strength score

Mathematical Construction:
-------------------------
Quality Score = mean(z(ROE), z(ROA), -z(Accruals), z(Piotroski))

Lower accruals = higher quality (inverted sign).

Input Data Requirements:
-----------------------
- Net Income TTM
- Total Equity
- Total Assets
- Operating Cash Flow TTM
- Various balance sheet items for Piotroski score

Normalization:
-------------
Each component z-scored cross-sectionally, then averaged.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import (
    FactorSignal,
    FactorData,
    FactorParams,
    cross_sectional_zscore,
    combine_components_zscore,
)


@dataclass
class QualityParams:
    """Quality-specific parameters."""
    enabled_components: List[str] = field(default_factory=lambda: ["roe", "roa", "accruals", "piotroski"])
    component_weights: Dict[str, float] = field(default_factory=lambda: {
        "roe": 1.0,
        "roa": 1.0,
        "accruals": 1.0,  # Will be inverted
        "piotroski": 1.0,
    })
    min_component_data_points: int = 100
    reporting_lag_days: int = 45


class QualitySignal(FactorSignal):
    """
    Quality factor: favors high-quality companies.

    Combines ROE, ROA, accruals quality, and Piotroski F-score
    into a composite quality measure.
    """

    @property
    def name(self) -> str:
        return "quality"

    @property
    def description(self) -> str:
        return (
            "Quality factor. Companies with high ROE, high ROA, low accruals, "
            "and high Piotroski F-scores receive higher scores. Captures "
            "fundamental strength and earnings quality."
        )

    @property
    def category(self) -> str:
        return "quality"

    @property
    def higher_is_better(self) -> bool:
        return True  # Higher quality = expected outperformance

    def compute_raw_signal(
        self,
        data: FactorData,
        params: FactorParams,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Compute composite quality signal.

        Returns:
            Tuple of (raw_scores, metadata)
        """
        dates = data.dates
        tickers = data.tickers

        quality_params = QualityParams(**params.custom) if params.custom else QualityParams()

        # Build quality component panels
        panels = self._build_quality_panels(data, quality_params)

        # Combine components
        components = []
        enabled = set(quality_params.enabled_components)

        for name, panel in panels.items():
            if name not in enabled:
                continue
            if panel.notna().sum().sum() < quality_params.min_component_data_points:
                continue

            weight = quality_params.component_weights.get(name, 1.0)
            if weight == 0:
                continue

            z_panel = cross_sectional_zscore(panel)

            # Invert accruals (low accruals = high quality)
            if name == "accruals":
                z_panel = -z_panel

            components.append((name, z_panel * weight))

        if not components:
            raw_scores = pd.DataFrame(np.nan, index=dates, columns=tickers)
            return raw_scores, {"warning": "No quality components available"}

        raw_scores = combine_components_zscore(components, dates, tickers)

        metadata = {
            "components": [c[0] for c in components],
            "n_components": len(components),
            "coverage_pct": float(raw_scores.notna().sum().sum()) / max(raw_scores.size, 1) * 100,
        }

        return raw_scores, metadata

    def _build_quality_panels(
        self,
        data: FactorData,
        quality_params: QualityParams,
    ) -> Dict[str, pd.DataFrame]:
        """Build quality component panels from fundamental data."""
        dates = data.dates
        tickers = data.tickers

        panels = {
            "roe": pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float),
            "roa": pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float),
            "accruals": pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float),
            "piotroski": pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float),
        }

        if data.data_loader is None:
            return panels

        try:
            fundamentals_parquet = data.data_loader.load_fundamentals_parquet()
        except Exception:
            return panels

        if fundamentals_parquet is None:
            return panels

        # Import utils
        try:
            from ..factor_builders import (
                INCOME_SHEET,
                BALANCE_SHEET,
                CASH_FLOW_SHEET,
                NET_INCOME_KEYS,
                TOTAL_ASSETS_KEYS,
                TOTAL_EQUITY_KEYS,
                REVENUE_KEYS,
                CURRENT_ASSETS_KEYS,
                CURRENT_LIABILITIES_KEYS,
                LONG_TERM_DEBT_KEYS,
                CFO_KEYS,
            )
            from ...common.utils import (
                get_consolidated_sheet,
                pick_row_from_sheet,
                coerce_quarter_cols,
                sum_ttm,
                apply_lag,
            )
        except ImportError:
            return panels

        for ticker in tickers:
            try:
                inc = get_consolidated_sheet(fundamentals_parquet, ticker, INCOME_SHEET)
                bs = get_consolidated_sheet(fundamentals_parquet, ticker, BALANCE_SHEET)
                cf = get_consolidated_sheet(fundamentals_parquet, ticker, CASH_FLOW_SHEET)

                if inc.empty or bs.empty:
                    continue

                # Extract key rows
                ni_row = pick_row_from_sheet(inc, NET_INCOME_KEYS)
                rev_row = pick_row_from_sheet(inc, REVENUE_KEYS)
                ta_row = pick_row_from_sheet(bs, TOTAL_ASSETS_KEYS)
                eq_row = pick_row_from_sheet(bs, TOTAL_EQUITY_KEYS)
                ca_row = pick_row_from_sheet(bs, CURRENT_ASSETS_KEYS)
                cl_row = pick_row_from_sheet(bs, CURRENT_LIABILITIES_KEYS)
                debt_row = pick_row_from_sheet(bs, LONG_TERM_DEBT_KEYS)
                cfo_row = pick_row_from_sheet(cf, CFO_KEYS) if not cf.empty else None

                # ROE = Net Income TTM / Average Equity
                if ni_row is not None and eq_row is not None:
                    ni = coerce_quarter_cols(ni_row)
                    eq = coerce_quarter_cols(eq_row)
                    if not ni.empty and not eq.empty:
                        ni_ttm = sum_ttm(ni)
                        eq_avg = eq.rolling(4, min_periods=2).mean()
                        ni_ttm, eq_avg = ni_ttm.align(eq_avg, join="inner")
                        roe = (ni_ttm / eq_avg.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
                        if not roe.empty:
                            panels["roe"][ticker] = apply_lag(roe, dates)

                # ROA = Net Income TTM / Average Total Assets
                if ni_row is not None and ta_row is not None:
                    ni = coerce_quarter_cols(ni_row)
                    ta = coerce_quarter_cols(ta_row)
                    if not ni.empty and not ta.empty:
                        ni_ttm = sum_ttm(ni)
                        ta_avg = ta.rolling(4, min_periods=2).mean()
                        ni_ttm, ta_avg = ni_ttm.align(ta_avg, join="inner")
                        roa = (ni_ttm / ta_avg.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
                        if not roa.empty:
                            panels["roa"][ticker] = apply_lag(roa, dates)

                # Accruals = (Net Income - CFO) / Total Assets
                if ni_row is not None and cfo_row is not None and ta_row is not None:
                    ni = coerce_quarter_cols(ni_row)
                    cfo = coerce_quarter_cols(cfo_row)
                    ta = coerce_quarter_cols(ta_row)
                    if not ni.empty and not cfo.empty and not ta.empty:
                        ni_ttm = sum_ttm(ni)
                        cfo_ttm = sum_ttm(cfo)
                        ta_avg = ta.rolling(4, min_periods=2).mean()
                        ni_ttm, cfo_ttm = ni_ttm.align(cfo_ttm, join="inner")
                        ni_ttm, ta_avg = ni_ttm.align(ta_avg, join="inner")
                        cfo_ttm = cfo_ttm.reindex(ni_ttm.index)
                        accruals = ((ni_ttm - cfo_ttm) / ta_avg.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
                        if not accruals.empty:
                            panels["accruals"][ticker] = apply_lag(accruals, dates)

                # Piotroski F-Score (simplified)
                if ni_row is not None and ta_row is not None:
                    ni = coerce_quarter_cols(ni_row)
                    ta = coerce_quarter_cols(ta_row)
                    if not ni.empty and not ta.empty:
                        ni_ttm = sum_ttm(ni)
                        ta_avg = ta.rolling(4, min_periods=2).mean()
                        roa_check = ni_ttm / ta_avg.replace(0, np.nan)
                        base_index = roa_check.index

                        # Start with ROA > 0
                        score = (roa_check > 0).astype(float)

                        # CFO > 0
                        if cfo_row is not None:
                            cfo = coerce_quarter_cols(cfo_row)
                            if not cfo.empty:
                                cfo_ttm = sum_ttm(cfo).reindex(base_index)
                                score = score + (cfo_ttm > 0).fillna(False).astype(float)
                                # CFO > Net Income
                                ni_aligned = ni_ttm.reindex(base_index)
                                score = score + (cfo_ttm > ni_aligned).fillna(False).astype(float)

                        # Improving ROA
                        score = score + (roa_check.diff(4) > 0).fillna(False).astype(float)

                        # Other Piotroski components...
                        if debt_row is not None:
                            debt = coerce_quarter_cols(debt_row)
                            if not debt.empty:
                                ta_aligned = ta_avg.reindex(debt.index)
                                leverage = debt / ta_aligned.replace(0, np.nan)
                                leverage = leverage.reindex(base_index)
                                score = score + (leverage.diff(4) < 0).fillna(False).astype(float)

                        if ca_row is not None and cl_row is not None:
                            ca = coerce_quarter_cols(ca_row)
                            cl = coerce_quarter_cols(cl_row)
                            if not ca.empty and not cl.empty:
                                ca, cl = ca.align(cl, join="inner")
                                cr = ca / cl.replace(0, np.nan)
                                cr = cr.reindex(base_index)
                                score = score + (cr.diff(4) > 0).fillna(False).astype(float)

                        if rev_row is not None:
                            rev = coerce_quarter_cols(rev_row)
                            if not rev.empty:
                                rev_ttm = sum_ttm(rev).reindex(base_index)
                                margin = ni_ttm.reindex(base_index) / rev_ttm.replace(0, np.nan)
                                score = score + (margin.diff(4) > 0).fillna(False).astype(float)
                                # Asset turnover
                                turnover = rev_ttm / ta_avg.reindex(base_index).replace(0, np.nan)
                                score = score + (turnover.diff(4) > 0).fillna(False).astype(float)

                        score = score.replace([np.inf, -np.inf], np.nan).dropna()
                        if not score.empty:
                            panels["piotroski"][ticker] = apply_lag(score, dates)

            except Exception:
                continue

        return panels

    def get_default_params(self) -> FactorParams:
        """Return quality-specific default parameters."""
        return FactorParams(
            lookback_days=1,
            lag_days=0,
            winsorize_pct=2.0,
            custom={
                "enabled_components": ["roe", "roa", "accruals", "piotroski"],
                "min_component_data_points": 100,
            },
        )


# =============================================================================
# POTENTIAL IMPROVEMENTS
# =============================================================================
"""
1. Gross Profitability:
   - Gross Profit / Assets (Novy-Marx)
   - Simpler and more robust than ROE/ROA

2. Cash Flow ROE:
   - CFO / Equity instead of Net Income / Equity
   - Less manipulable

3. Earnings Persistence:
   - AR(1) coefficient of earnings
   - Persistent earnings are higher quality

4. Beneish M-Score:
   - Earnings manipulation detection
   - Negative M-score = lower manipulation risk

5. Quality Momentum:
   - Change in quality metrics over time
   - Improving quality may predict returns

6. Sector-Adjusted Quality:
   - Compare to sector median
   - Some sectors naturally have different quality profiles
"""
