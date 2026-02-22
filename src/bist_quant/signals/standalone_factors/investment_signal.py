"""
Investment Factor Signal

Economic Intuition:
------------------
The investment factor (CMA - Conservative Minus Aggressive) captures the tendency
of low-investment firms to outperform high-investment firms. This is because:
1. Aggressive investment often indicates overconfidence or empire building
2. Conservative firms return capital to shareholders
3. High investment may dilute existing shareholders

This factor has two sides:
- Conservative: Low debt, high cash, dividend paying
- Reinvestment: High R&D, capex, asset growth

Mathematical Construction:
-------------------------
Conservative Profile = -0.55×Debt/Equity + 0.30×Cash Ratio + 0.30×Current Ratio + 0.20×Payout

Reinvestment Signal = R&D Intensity + Investment Efficiency

Input Data Requirements:
-----------------------
- Debt/Equity ratio
- Cash ratio
- Current ratio
- Dividend payout ratio
- R&D expenses
- Revenue and Total Assets (for efficiency)

Normalization:
-------------
Each component is squashed via tanh to bounded range, then combined.
Cross-sectional z-score of final score.
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
class InvestmentParams:
    """Investment-specific parameters."""
    debt_weight: float = -0.55       # Negative: high debt is bad
    cash_weight: float = 0.30
    current_ratio_weight: float = 0.30
    payout_weight: float = 0.20
    reporting_lag_days: int = 45


def _squash_metric(
    panel: pd.DataFrame,
    scale: float,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
) -> pd.DataFrame:
    """Clip and squash a metric panel to bounded range via tanh."""
    clipped = panel
    if lower is not None or upper is not None:
        clipped = clipped.clip(lower=lower, upper=upper)
    return np.tanh(clipped / scale)


class InvestmentSignal(FactorSignal):
    """
    Investment factor: captures conservative vs aggressive investment styles.

    This signal measures the "reinvestment" style - how aggressively
    a company is investing in growth vs. returning capital to shareholders.
    """

    @property
    def name(self) -> str:
        return "investment"

    @property
    def description(self) -> str:
        return (
            "Investment factor (CMA). Measures R&D intensity, payout yield, and "
            "investment efficiency. High scores indicate companies actively "
            "reinvesting for growth. Low scores indicate conservative capital "
            "allocation."
        )

    @property
    def category(self) -> str:
        return "investment"

    @property
    def higher_is_better(self) -> bool:
        return True  # Higher = more reinvestment-oriented

    def compute_raw_signal(
        self,
        data: FactorData,
        params: FactorParams,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Compute raw investment signal.

        Returns:
            Tuple of (raw_scores, metadata)
        """
        dates = data.dates
        tickers = data.tickers

        # Get investment-specific params
        inv_params = InvestmentParams(**params.custom) if params.custom else InvestmentParams()

        # Build investment metrics from data loader
        raw_scores = self._build_investment_panel(data, inv_params)

        metadata = {
            "components": ["rd_intensity", "payout_yield", "inv_efficiency"],
            "coverage_pct": float(raw_scores.notna().sum().sum()) / max(raw_scores.size, 1) * 100,
        }

        return raw_scores, metadata

    def _build_investment_panel(
        self,
        data: FactorData,
        inv_params: InvestmentParams,
    ) -> pd.DataFrame:
        """Build investment panel from fundamental data."""
        dates = data.dates
        tickers = data.tickers
        close = data.close.reindex(index=dates, columns=tickers)

        panels = {
            "rd_intensity": pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float),
            "payout_yield": pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float),
            "inv_efficiency": pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float),
        }

        if data.data_loader is None:
            return pd.DataFrame(np.nan, index=dates, columns=tickers)

        try:
            fundamentals_parquet = data.data_loader.load_fundamentals_parquet()
        except Exception:
            fundamentals_parquet = None

        if fundamentals_parquet is None:
            return pd.DataFrame(np.nan, index=dates, columns=tickers)

        # Import utils
        try:
            from ..factor_builders import (
                INCOME_SHEET,
                BALANCE_SHEET,
                CASH_FLOW_SHEET,
                REVENUE_KEYS,
                TOTAL_ASSETS_KEYS,
            )
            from ...common.utils import (
                get_consolidated_sheet,
                pick_row_from_sheet,
                coerce_quarter_cols,
                sum_ttm,
                apply_lag,
            )
        except ImportError:
            return pd.DataFrame(np.nan, index=dates, columns=tickers)

        # R&D keys
        RD_KEYS = (
            "Araştırma ve Geliştirme Giderleri (-)",
            "Araştırma ve Geliştirme Giderleri",
        )
        DIVIDENDS_KEYS = ("Ödenen Temettüler",)

        # Get market cap for yield calculations
        if data.market_cap is not None:
            market_cap = data.market_cap.reindex(index=dates, columns=tickers)
        elif data.shares_outstanding is not None:
            shares = data.shares_outstanding.reindex(index=dates, columns=tickers)
            market_cap = close * shares
        else:
            market_cap = None

        for ticker in tickers:
            try:
                inc = get_consolidated_sheet(fundamentals_parquet, ticker, INCOME_SHEET)
                bs = get_consolidated_sheet(fundamentals_parquet, ticker, BALANCE_SHEET)
                cf = get_consolidated_sheet(fundamentals_parquet, ticker, CASH_FLOW_SHEET)

                if inc.empty and bs.empty:
                    continue

                rev_row = pick_row_from_sheet(inc, REVENUE_KEYS)
                rd_row = pick_row_from_sheet(inc, RD_KEYS)
                assets_row = pick_row_from_sheet(bs, TOTAL_ASSETS_KEYS)
                div_row = pick_row_from_sheet(cf, DIVIDENDS_KEYS) if not cf.empty else None

                # R&D Intensity = R&D / Revenue
                if rd_row is not None and rev_row is not None:
                    rd = coerce_quarter_cols(rd_row)
                    rev = coerce_quarter_cols(rev_row)
                    rd_ttm = sum_ttm(rd)
                    rev_ttm = sum_ttm(rev)
                    rd_ttm, rev_ttm = rd_ttm.align(rev_ttm, join="inner")
                    rd_intensity = abs(rd_ttm) / rev_ttm.replace(0, np.nan)
                    rd_intensity = rd_intensity.replace([np.inf, -np.inf], np.nan).dropna()
                    if not rd_intensity.empty:
                        panels["rd_intensity"][ticker] = apply_lag(rd_intensity, dates)

                # Payout Yield = Dividends / Market Cap
                if div_row is not None and market_cap is not None:
                    div = coerce_quarter_cols(div_row)
                    div_ttm = sum_ttm(div)
                    if not div_ttm.empty:
                        mc = market_cap[ticker].dropna()
                        div_ttm_lagged = apply_lag(abs(div_ttm), dates)
                        payout = div_ttm_lagged / mc.replace(0, np.nan)
                        payout = payout.replace([np.inf, -np.inf], np.nan)
                        panels["payout_yield"][ticker] = payout

                # Investment Efficiency = ΔRevenue / ΔAssets (YoY)
                if rev_row is not None and assets_row is not None:
                    rev = coerce_quarter_cols(rev_row)
                    assets = coerce_quarter_cols(assets_row)
                    rev_ttm = sum_ttm(rev)
                    rev_yoy = rev_ttm.diff(periods=4)
                    assets_yoy = assets.diff(periods=4)
                    # Floor denominator
                    assets_floor = assets_yoy.abs().clip(lower=assets.rolling(4).mean() * 0.01)
                    rev_yoy, assets_floor = rev_yoy.align(assets_floor, join="inner")
                    efficiency = rev_yoy / assets_floor.replace(0, np.nan)
                    efficiency = efficiency.replace([np.inf, -np.inf], np.nan).clip(-5, 5).dropna()
                    if not efficiency.empty:
                        panels["inv_efficiency"][ticker] = apply_lag(efficiency, dates)

            except Exception:
                continue

        # Combine panels with z-score
        components = []
        for name, panel in panels.items():
            if panel.notna().sum().sum() > 100:
                components.append((name, cross_sectional_zscore(panel)))

        if not components:
            return pd.DataFrame(np.nan, index=dates, columns=tickers)

        # Average across components
        stacked = np.stack([c[1].values for _, c in enumerate(components) for c in [components[_]]], axis=0)
        result = pd.DataFrame(
            np.nanmean(stacked, axis=0),
            index=dates,
            columns=tickers,
        )

        return result

    def get_default_params(self) -> FactorParams:
        """Return investment-specific default parameters."""
        return FactorParams(
            lookback_days=1,
            lag_days=0,
            winsorize_pct=2.0,
            custom={
                "reporting_lag_days": 45,
            },
        )


class ConservativeSignal(FactorSignal):
    """
    Conservative investment profile: low debt, high cash, shareholder friendly.

    This is the opposite side of the investment axis from reinvestment.
    """

    @property
    def name(self) -> str:
        return "conservative"

    @property
    def description(self) -> str:
        return (
            "Conservative capital allocation factor. Companies with low debt, "
            "high cash reserves, and strong dividend payout receive higher scores. "
            "Captures the 'CMA' factor from Fama-French."
        )

    @property
    def category(self) -> str:
        return "investment"

    @property
    def higher_is_better(self) -> bool:
        return True

    def compute_raw_signal(
        self,
        data: FactorData,
        params: FactorParams,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Compute conservative profile from balance sheet metrics."""
        dates = data.dates
        tickers = data.tickers

        inv_params = InvestmentParams(**params.custom) if params.custom else InvestmentParams()

        # Load fundamental metrics
        if data.data_loader is None:
            return pd.DataFrame(np.nan, index=dates, columns=tickers), {"warning": "No data_loader"}

        try:
            metrics_df = data.data_loader.load_fundamental_metrics()
        except Exception:
            return pd.DataFrame(np.nan, index=dates, columns=tickers), {"warning": "No metrics"}

        if metrics_df.empty:
            return pd.DataFrame(np.nan, index=dates, columns=tickers), {"warning": "Empty metrics"}

        # Build metric panels
        debt_panel = self._build_metric_panel(metrics_df, "debt_to_equity", dates, tickers)
        cash_panel = self._build_metric_panel(metrics_df, "cash_ratio", dates, tickers)
        current_panel = self._build_metric_panel(metrics_df, "current_ratio", dates, tickers)
        payout_panel = self._build_metric_panel(metrics_df, "dividend_payout_ratio", dates, tickers)

        # Squash and combine
        debt_score = _squash_metric(debt_panel, scale=1.5, lower=-2.0, upper=6.0).fillna(0)
        cash_score = _squash_metric(cash_panel, scale=1.0, lower=0.0, upper=8.0).fillna(0)
        current_score = _squash_metric(current_panel, scale=2.0, lower=0.0, upper=12.0).fillna(0)
        payout_score = _squash_metric(payout_panel, scale=0.5, lower=-1.0, upper=2.0).fillna(0)

        # Conservative profile
        conservative = (
            inv_params.debt_weight * debt_score +
            inv_params.cash_weight * cash_score +
            inv_params.current_ratio_weight * current_score +
            inv_params.payout_weight * payout_score
        )

        # Only keep where we have some data
        has_data = debt_panel.notna() | cash_panel.notna() | current_panel.notna() | payout_panel.notna()
        raw_scores = conservative.where(has_data)

        metadata = {
            "components": ["debt_to_equity", "cash_ratio", "current_ratio", "payout_ratio"],
            "weights": {
                "debt": inv_params.debt_weight,
                "cash": inv_params.cash_weight,
                "current": inv_params.current_ratio_weight,
                "payout": inv_params.payout_weight,
            },
        }

        return raw_scores, metadata

    def _build_metric_panel(
        self,
        metrics_df: pd.DataFrame,
        metric_name: str,
        dates: pd.DatetimeIndex,
        tickers: pd.Index,
    ) -> pd.DataFrame:
        """Build panel for a single metric."""
        panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)

        if metric_name not in metrics_df.columns:
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
                panel[ticker] = series.reindex(dates).ffill()
            except Exception:
                continue

        return panel


# =============================================================================
# POTENTIAL IMPROVEMENTS
# =============================================================================
"""
1. Asset Growth Factor:
   - YoY change in total assets
   - Simpler measure than component-based

2. CapEx Intensity:
   - CapEx / Revenue or CapEx / Assets
   - Direct measure of investment aggressiveness

3. Acquisition Activity:
   - M&A spending / Total Assets
   - Aggressive acquisitions often destroy value

4. Share Issuance:
   - Net share issuance (dilution)
   - Equity issuance often predicts underperformance

5. Debt Issuance:
   - Change in debt levels
   - Rapid debt growth can be risky

6. Industry-Adjusted Investment:
   - Compare to sector median
   - Some industries require high investment
"""
