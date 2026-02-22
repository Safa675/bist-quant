"""
Profitability Factor Signal

Economic Intuition:
------------------
Profitable companies tend to outperform unprofitable ones. This is the RMW
(Robust Minus Weak) factor from Fama-French 5-factor model. The intuition:
1. Profitability is a sign of competitive advantage
2. Profitable firms can reinvest for growth
3. Market may undervalue profitability growth potential

This factor has two sides:
- Margin Level: Current profitability (operating margin, gross margin)
- Margin Growth: Improving profitability trajectory

Mathematical Construction:
-------------------------
Margin Level = (Operating Margin × 0.6 + Gross Margin × 0.4)
Margin Growth = YoY change in Margin Level

The axis compares stocks favoring either:
- High current margins (value-like, defensive)
- Improving margins (growth-like, momentum)

Input Data Requirements:
-----------------------
- Revenue TTM
- Operating Income TTM
- Gross Profit TTM

Normalization:
-------------
Cross-sectional z-score of margin metrics.
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
)


@dataclass
class ProfitabilityParams:
    """Profitability-specific parameters."""
    operating_margin_weight: float = 0.6
    gross_margin_weight: float = 0.4
    margin_growth_lookback_quarters: int = 4  # YoY comparison
    reporting_lag_days: int = 45


class ProfitabilitySignal(FactorSignal):
    """
    Profitability factor: favors profitable companies.

    Higher scores indicate higher profit margins.
    This is the "margin level" component.
    """

    @property
    def name(self) -> str:
        return "profitability"

    @property
    def description(self) -> str:
        return (
            "Profitability factor (RMW). Companies with higher operating and gross "
            "margins receive higher scores. Based on the empirical finding that "
            "profitable firms outperform unprofitable ones (Fama-French RMW)."
        )

    @property
    def category(self) -> str:
        return "profitability"

    @property
    def higher_is_better(self) -> bool:
        return True  # Higher margin = expected outperformance

    def compute_raw_signal(
        self,
        data: FactorData,
        params: FactorParams,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Compute raw profitability signal from margin levels.

        Returns:
            Tuple of (raw_scores, metadata)
        """
        dates = data.dates
        tickers = data.tickers

        # Get profitability-specific params
        prof_params = ProfitabilityParams(**params.custom) if params.custom else ProfitabilityParams()

        # Build margin panel from fundamentals
        margin_panel = self._build_margin_panel(data, prof_params)

        raw_scores = margin_panel.reindex(index=dates, columns=tickers)

        metadata = {
            "components": ["operating_margin", "gross_margin"],
            "op_margin_weight": prof_params.operating_margin_weight,
            "gp_margin_weight": prof_params.gross_margin_weight,
            "coverage_pct": float(raw_scores.notna().sum().sum()) / max(raw_scores.size, 1) * 100,
        }

        return raw_scores, metadata

    def _build_margin_panel(
        self,
        data: FactorData,
        prof_params: ProfitabilityParams,
    ) -> pd.DataFrame:
        """Build margin panel from fundamental data."""
        dates = data.dates
        tickers = data.tickers
        margin_panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)

        if data.fundamentals_parquet is None and data.data_loader is None:
            return margin_panel

        # Load from data_loader if available
        if data.data_loader is not None:
            try:
                fundamentals_parquet = data.data_loader.load_fundamentals_parquet()
            except Exception:
                fundamentals_parquet = None
        else:
            fundamentals_parquet = data.fundamentals_parquet

        if fundamentals_parquet is None:
            return margin_panel

        # Import utils for parsing
        try:
            from ..factor_builders import (
                INCOME_SHEET,
                REVENUE_KEYS,
                OPERATING_INCOME_KEYS,
                GROSS_PROFIT_KEYS,
            )
            from ...common.utils import (
                get_consolidated_sheet,
                pick_row_from_sheet,
                coerce_quarter_cols,
                sum_ttm,
                apply_lag,
            )
        except ImportError:
            return margin_panel

        for ticker in tickers:
            try:
                inc = get_consolidated_sheet(fundamentals_parquet, ticker, INCOME_SHEET)
                if inc.empty:
                    continue

                rev_row = pick_row_from_sheet(inc, REVENUE_KEYS)
                op_row = pick_row_from_sheet(inc, OPERATING_INCOME_KEYS)
                gp_row = pick_row_from_sheet(inc, GROSS_PROFIT_KEYS)

                if rev_row is None or (op_row is None and gp_row is None):
                    continue

                rev = coerce_quarter_cols(rev_row)
                rev_ttm = sum_ttm(rev)

                if rev_ttm.empty:
                    continue

                margins = []

                # Operating margin
                if op_row is not None:
                    op = coerce_quarter_cols(op_row)
                    op_ttm = sum_ttm(op)
                    op_ttm, rev_aligned = op_ttm.align(rev_ttm, join="inner")
                    op_margin = op_ttm / rev_aligned.replace(0, np.nan)
                    margins.append(("op", op_margin, prof_params.operating_margin_weight))

                # Gross margin
                if gp_row is not None:
                    gp = coerce_quarter_cols(gp_row)
                    gp_ttm = sum_ttm(gp)
                    gp_ttm, rev_aligned = gp_ttm.align(rev_ttm, join="inner")
                    gp_margin = gp_ttm / rev_aligned.replace(0, np.nan)
                    margins.append(("gp", gp_margin, prof_params.gross_margin_weight))

                if not margins:
                    continue

                # Weighted average of margins
                combined = None
                total_weight = 0
                for name, margin, weight in margins:
                    if combined is None:
                        combined = margin * weight
                    else:
                        combined, margin = combined.align(margin, join="outer")
                        combined = combined.fillna(0) + margin.fillna(0) * weight
                    total_weight += weight

                if combined is not None and total_weight > 0:
                    combined = combined / total_weight
                    combined = combined.replace([np.inf, -np.inf], np.nan).dropna()

                    if not combined.empty:
                        margin_panel[ticker] = apply_lag(combined, dates)

            except Exception:
                continue

        return margin_panel

    def get_default_params(self) -> FactorParams:
        """Return profitability-specific default parameters."""
        return FactorParams(
            lookback_days=1,
            lag_days=0,
            winsorize_pct=2.0,
            custom={
                "operating_margin_weight": 0.6,
                "gross_margin_weight": 0.4,
            },
        )


class MarginGrowthSignal(FactorSignal):
    """
    Margin Growth factor: favors companies with improving margins.

    Captures the trajectory of profitability rather than level.
    """

    @property
    def name(self) -> str:
        return "margin_growth"

    @property
    def description(self) -> str:
        return (
            "Margin growth factor. Companies with improving profit margins "
            "(YoY change in operating/gross margins) receive higher scores. "
            "Captures profitability momentum."
        )

    @property
    def category(self) -> str:
        return "profitability"

    @property
    def higher_is_better(self) -> bool:
        return True

    def compute_raw_signal(
        self,
        data: FactorData,
        params: FactorParams,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Compute margin growth signal (YoY change in margins)."""
        # First compute margin levels
        prof_signal = ProfitabilitySignal()
        margin_panel, _ = prof_signal.compute_raw_signal(data, params)

        # Compute YoY change (4 quarters = ~252 trading days)
        margin_growth = margin_panel.diff(252)

        metadata = {
            "base_metric": "margin_level",
            "growth_window_days": 252,
        }

        return margin_growth, metadata


# =============================================================================
# POTENTIAL IMPROVEMENTS
# =============================================================================
"""
1. Margin Stability:
   - Standard deviation of margins over time
   - Stable high margins > volatile high margins

2. ROE/ROA Based Profitability:
   - Return metrics rather than margin metrics
   - Captures capital efficiency

3. Sector-Relative Margins:
   - Compare to sector median
   - Some sectors naturally have lower margins

4. Cash Flow Profitability:
   - OCF/Assets or FCF/Assets
   - Less manipulable than accounting margins

5. DuPont Decomposition:
   - Margin × Turnover × Leverage
   - Identifies source of profitability

6. Earnings Quality Adjustment:
   - Weight margins by earnings persistence
   - Sustainable margins > one-time gains
"""
