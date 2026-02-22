"""
Value Factor Signal

Economic Intuition:
------------------
Value investing is based on the premise that stocks trading below their intrinsic
value will eventually revert to fair value. The value premium (HML - High Minus Low)
captures the tendency of "cheap" stocks to outperform "expensive" stocks.

Key value ratios:
1. E/P (Earnings/Price) - earnings yield
2. B/P (Book/Price) - book value relative to market price
3. FCF/P (Free Cash Flow/Price) - cash generation yield
4. EBITDA/EV - operating earnings relative to enterprise value
5. S/P (Sales/Price) - revenue multiple

Mathematical Construction:
-------------------------
Composite Value Score = mean(z-score(E/P), z-score(FCF/P), z-score(EBITDA/EV), ...)

Each ratio is z-scored cross-sectionally, then averaged to create a composite.
Higher values indicate cheaper stocks.

Input Data Requirements:
-----------------------
- Daily close prices
- Shares outstanding (for market cap)
- Net Income TTM, Revenue TTM, EBITDA TTM, OCF TTM, FCF TTM
- Total Debt, Cash (for EV calculation)

Normalization:
-------------
Each component ratio is z-scored cross-sectionally before combining.
Final composite is also z-scored.
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
class ValueParams:
    """Value-specific parameters."""
    enabled_metrics: List[str] = field(default_factory=lambda: ["ep", "fcfp", "ebitdaev", "sp"])
    metric_weights: Dict[str, float] = field(default_factory=lambda: {
        "ep": 1.0,       # Earnings/Price
        "fcfp": 1.0,     # FCF/Price
        "ebitdaev": 1.0, # EBITDA/EV
        "sp": 1.0,       # Sales/Price
        "ocfev": 1.0,    # OCF/EV
    })
    use_ttm: bool = True           # Use trailing twelve months
    reporting_lag_days: int = 45   # Lag for fundamental data


class ValueSignal(FactorSignal):
    """
    Value factor: favors cheap stocks (value premium).

    Combines multiple valuation ratios into a composite score.
    Higher scores indicate cheaper/more attractive valuations.
    """

    @property
    def name(self) -> str:
        return "value"

    @property
    def description(self) -> str:
        return (
            "Value premium factor. Stocks with low price multiples (high E/P, FCF/P, etc.) "
            "receive higher scores based on the empirical finding that 'cheap' stocks tend "
            "to outperform 'expensive' stocks over the long term (Fama-French HML factor)."
        )

    @property
    def category(self) -> str:
        return "value"

    @property
    def higher_is_better(self) -> bool:
        return True  # Higher value score = cheaper = expected outperformance

    def compute_raw_signal(
        self,
        data: FactorData,
        params: FactorParams,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Compute raw value signal from multiple valuation ratios.

        Returns:
            Tuple of (raw_scores, metadata)
        """
        dates = data.dates
        tickers = data.tickers
        close = data.close.reindex(index=dates, columns=tickers)

        # Get value-specific params
        value_params = ValueParams(**params.custom) if params.custom else ValueParams()

        # Calculate market cap
        if data.market_cap is not None:
            market_cap = data.market_cap.reindex(index=dates, columns=tickers)
        elif data.shares_outstanding is not None:
            shares = data.shares_outstanding.reindex(index=dates, columns=tickers)
            market_cap = close * shares
        else:
            raise ValueError("Value signal requires market_cap or shares_outstanding")

        # Build component panels
        component_panels = {}

        # If we have fundamental data from data_loader, use it
        if data.data_loader is not None:
            component_panels = self._build_components_from_loader(
                data, market_cap, value_params
            )

        # Combine components
        components = []
        enabled = set(value_params.enabled_metrics)
        for name, panel in component_panels.items():
            if name in enabled and panel.notna().sum().sum() > 100:
                weight = value_params.metric_weights.get(name, 1.0)
                if weight > 0:
                    z_panel = cross_sectional_zscore(panel)
                    components.append((name, z_panel * weight))

        if not components:
            # Fallback: use simple B/M if no fundamentals
            raw_scores = pd.DataFrame(np.nan, index=dates, columns=tickers)
            metadata = {"components": [], "warning": "No fundamental data available"}
            return raw_scores, metadata

        # Combine with proper NaN handling
        raw_scores = combine_components_zscore(components, dates, tickers)

        metadata = {
            "components": [c[0] for c in components],
            "n_components": len(components),
            "coverage_pct": float(raw_scores.notna().sum().sum()) / max(raw_scores.size, 1) * 100,
        }

        return raw_scores, metadata

    def _build_components_from_loader(
        self,
        data: FactorData,
        market_cap: pd.DataFrame,
        value_params: ValueParams,
    ) -> Dict[str, pd.DataFrame]:
        """Build value ratio panels from data loader."""
        panels = {}
        dates = data.dates
        tickers = data.tickers

        try:
            metrics_df = data.data_loader.load_fundamental_metrics()
        except Exception:
            return panels

        if metrics_df.empty:
            return panels

        available_tickers = set(metrics_df.index.get_level_values(0))

        # Build each ratio panel
        for ratio_name in ["ep", "fcfp", "ebitdaev", "sp", "ocfev"]:
            panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)

            metric_map = {
                "ep": "earnings_yield",
                "fcfp": "fcf_yield",
                "ebitdaev": "ebitda_ev",
                "sp": "sales_price",
                "ocfev": "ocf_ev",
            }

            metric_name = metric_map.get(ratio_name)
            if metric_name and metric_name in metrics_df.columns:
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
                        # Apply reporting lag
                        lagged = self._apply_lag(series, dates, value_params.reporting_lag_days)
                        panel[ticker] = lagged
                    except Exception:
                        continue

            if panel.notna().sum().sum() > 0:
                panels[ratio_name] = panel

        return panels

    def _apply_lag(
        self,
        series: pd.Series,
        dates: pd.DatetimeIndex,
        lag_days: int,
    ) -> pd.Series:
        """Apply reporting lag and forward-fill to daily dates."""
        if series.empty:
            return pd.Series(np.nan, index=dates)

        # Shift index by lag_days
        lagged_index = series.index + pd.Timedelta(days=lag_days)
        lagged_series = pd.Series(series.values, index=lagged_index)

        # Reindex to daily dates with forward fill
        return lagged_series.reindex(dates).ffill()

    def get_default_params(self) -> FactorParams:
        """Return value-specific default parameters."""
        return FactorParams(
            lookback_days=1,   # Point-in-time valuation
            lag_days=0,        # Lag handled in component calculation
            winsorize_pct=2.0, # Value ratios can have outliers
            custom={
                "enabled_metrics": ["ep", "fcfp", "ebitdaev", "sp"],
                "reporting_lag_days": 45,
            },
        )


# =============================================================================
# POTENTIAL IMPROVEMENTS
# =============================================================================
"""
1. Value-Quality Interaction:
   - Cheap stocks with high quality metrics are safer
   - Could filter for "quality value" (cheap + profitable)

2. Value Momentum:
   - YoY change in valuation ratios
   - Improving value (ratios getting cheaper) may predict returns

3. Relative Value:
   - Compare to sector median or historical average
   - Stock that's cheap vs. its history may be more actionable

4. Earnings Revision Adjusted:
   - Weight E/P by earnings estimate revisions
   - Avoid value traps where earnings are declining

5. Deep Value vs. Moderate Value:
   - Extreme cheapness may indicate distress
   - Could use non-linear transformation (e.g., winsorize heavily)

6. Cash-Adjusted E/P:
   - Subtract net cash from market cap in denominator
   - More accurate for cash-rich companies
"""
