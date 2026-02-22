"""
Fundamental Momentum Factor Signal

Economic Intuition:
------------------
Fundamental momentum captures the trajectory of business fundamentals,
as opposed to price momentum. Improving fundamentals often lead price
because:
1. Markets underreact to fundamental improvements initially
2. Analysts revise estimates gradually
3. Earnings momentum persists (stickiness)

This is distinct from price momentum:
- Price momentum = past price performance
- Fundamental momentum = improving business metrics

Fundamental Momentum Components:
1. Margin Change - YoY change in operating margin
2. Sales Acceleration - Change in revenue growth rate

Mathematical Construction:
-------------------------
FundMom = mean(z(MarginChange), z(SalesAcceleration))

Input Data Requirements:
-----------------------
- Revenue TTM (quarterly)
- Operating Income TTM (quarterly)

Normalization:
-------------
Each component z-scored cross-sectionally, then averaged.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

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
class FundamentalMomentumParams:
    """Fundamental momentum-specific parameters."""
    margin_change_quarters: int = 4    # YoY comparison
    sales_accel_quarters: int = 4      # YoY comparison
    reporting_lag_days: int = 45


class FundamentalMomentumSignal(FactorSignal):
    """
    Fundamental Momentum factor: captures improving fundamentals.

    Higher scores indicate improving margins and accelerating sales.
    Leading indicator compared to price momentum.
    """

    @property
    def name(self) -> str:
        return "fundamental_momentum"

    @property
    def description(self) -> str:
        return (
            "Fundamental momentum factor. Captures improving business fundamentals "
            "including margin improvement and sales acceleration. Often leads price "
            "momentum as markets gradually incorporate fundamental changes."
        )

    @property
    def category(self) -> str:
        return "fundamental_momentum"

    @property
    def higher_is_better(self) -> bool:
        return True  # Improving fundamentals = expected outperformance

    def compute_raw_signal(
        self,
        data: FactorData,
        params: FactorParams,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Compute fundamental momentum signal.

        Returns:
            Tuple of (raw_scores, metadata)
        """
        dates = data.dates
        tickers = data.tickers

        fm_params = FundamentalMomentumParams(**params.custom) if params.custom else FundamentalMomentumParams()

        # Build fundamental momentum panels
        panels = self._build_fundmom_panels(data, fm_params)

        # Combine components
        components = []
        for name, panel in panels.items():
            if panel.notna().sum().sum() > 100:
                components.append((name, cross_sectional_zscore(panel)))

        if not components:
            return pd.DataFrame(np.nan, index=dates, columns=tickers), {"warning": "No fundmom data"}

        raw_scores = combine_components_zscore(components, dates, tickers)

        metadata = {
            "components": [c[0] for c in components],
            "margin_change_quarters": fm_params.margin_change_quarters,
            "sales_accel_quarters": fm_params.sales_accel_quarters,
            "coverage_pct": float(raw_scores.notna().sum().sum()) / max(raw_scores.size, 1) * 100,
        }

        return raw_scores, metadata

    def _build_fundmom_panels(
        self,
        data: FactorData,
        fm_params: FundamentalMomentumParams,
    ) -> Dict[str, pd.DataFrame]:
        """Build fundamental momentum panels."""
        dates = data.dates
        tickers = data.tickers

        panels = {
            "margin_change": pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float),
            "sales_accel": pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float),
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
                REVENUE_KEYS,
                OPERATING_INCOME_KEYS,
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
                if inc.empty:
                    continue

                rev_row = pick_row_from_sheet(inc, REVENUE_KEYS)
                op_row = pick_row_from_sheet(inc, OPERATING_INCOME_KEYS)

                if rev_row is None or op_row is None:
                    continue

                rev = coerce_quarter_cols(rev_row)
                op = coerce_quarter_cols(op_row)

                if rev.empty or op.empty:
                    continue

                rev_ttm = sum_ttm(rev)
                op_ttm = sum_ttm(op)

                # Operating margin and YoY change
                op_ttm, rev_aligned = op_ttm.align(rev_ttm, join="inner")
                op_margin = op_ttm / rev_aligned.replace(0, np.nan)
                margin_change = op_margin.diff(fm_params.margin_change_quarters)
                margin_change = margin_change.replace([np.inf, -np.inf], np.nan)

                if not margin_change.dropna().empty:
                    panels["margin_change"][ticker] = apply_lag(margin_change, dates)

                # Sales growth acceleration
                sales_growth = rev_ttm.pct_change(fm_params.sales_accel_quarters)
                sales_accel = sales_growth.diff(fm_params.sales_accel_quarters)
                sales_accel = sales_accel.replace([np.inf, -np.inf], np.nan)

                if not sales_accel.dropna().empty:
                    panels["sales_accel"][ticker] = apply_lag(sales_accel, dates)

            except Exception:
                continue

        return panels

    def get_default_params(self) -> FactorParams:
        """Return fundamental momentum-specific default parameters."""
        return FactorParams(
            lookback_days=1,
            lag_days=0,
            winsorize_pct=2.0,
            custom={
                "margin_change_quarters": 4,
                "sales_accel_quarters": 4,
                "reporting_lag_days": 45,
            },
        )


class EarningsRevisionSignal(FactorSignal):
    """
    Earnings Revision: based on analyst estimate changes.

    Note: Requires analyst estimate data which may not be available.
    """

    @property
    def name(self) -> str:
        return "earnings_revision"

    @property
    def description(self) -> str:
        return (
            "Earnings revision factor. Stocks with upward analyst estimate "
            "revisions receive higher scores. Strong predictor of future returns."
        )

    @property
    def category(self) -> str:
        return "fundamental_momentum"

    @property
    def higher_is_better(self) -> bool:
        return True

    def compute_raw_signal(
        self,
        data: FactorData,
        params: FactorParams,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Compute earnings revision signal (placeholder)."""
        # Note: This would require analyst estimate data
        dates = data.dates
        tickers = data.tickers

        return pd.DataFrame(np.nan, index=dates, columns=tickers), {
            "warning": "Earnings revision requires analyst estimate data"
        }


# =============================================================================
# POTENTIAL IMPROVEMENTS
# =============================================================================
"""
1. Earnings Surprise:
   - Actual EPS vs. consensus estimate
   - Post-earnings announcement drift

2. Revenue Surprise:
   - Actual revenue vs. estimates
   - Less followed but often more informative

3. Guidance Momentum:
   - Change in company's own guidance
   - Management's view of future

4. SUE (Standardized Unexpected Earnings):
   - Earnings surprise / historical surprise volatility
   - Normalized for comparability

5. Analyst Rating Changes:
   - Upgrade/downgrade momentum
   - Consensus shifts

6. Estimate Dispersion:
   - Standard deviation of analyst estimates
   - High dispersion = more uncertainty
"""
