"""
Defensive Factor Signal

Economic Intuition:
------------------
Defensive stocks exhibit lower volatility and more stable fundamentals.
They tend to outperform in bear markets and provide downside protection:

1. Low Beta - Less sensitive to market movements
2. Earnings Stability - Consistent earnings reduce uncertainty
3. Stable Business Models - Utilities, consumer staples, healthcare

The low-volatility anomaly suggests that defensive stocks actually
outperform on a risk-adjusted basis, challenging traditional CAPM.

Defensive Components:
1. Earnings Stability - Inverse of earnings coefficient of variation
2. Low Beta - Negative of market beta

Mathematical Construction:
-------------------------
Defensive Score = mean(z(EarningsStability), -z(Beta))

Note: Beta is inverted (low beta = high defensive score).

Input Data Requirements:
-----------------------
- Net Income TTM (for earnings stability)
- Daily prices (for beta calculation)

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
class DefensiveParams:
    """Defensive-specific parameters."""
    earnings_stability_quarters: int = 8   # 2 years of quarterly data
    beta_lookback_days: int = 252
    beta_min_observations: int = 126
    stability_clip: Tuple[float, float] = (-10.0, 10.0)


class DefensiveSignal(FactorSignal):
    """
    Defensive factor: favors stable, low-beta stocks.

    Higher scores indicate more defensive characteristics
    (low beta, stable earnings).
    """

    @property
    def name(self) -> str:
        return "defensive"

    @property
    def description(self) -> str:
        return (
            "Defensive factor. Combines earnings stability and low market beta. "
            "Higher scores indicate more stable, defensive stocks that tend to "
            "outperform in bear markets and provide downside protection."
        )

    @property
    def category(self) -> str:
        return "defensive"

    @property
    def higher_is_better(self) -> bool:
        return True  # Higher defensive score = more stable

    def compute_raw_signal(
        self,
        data: FactorData,
        params: FactorParams,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Compute defensive signal.

        Returns:
            Tuple of (raw_scores, metadata)
        """
        dates = data.dates
        tickers = data.tickers

        def_params = DefensiveParams(**params.custom) if params.custom else DefensiveParams()

        panels = {}

        # 1. Earnings Stability
        stability_panel = self._build_earnings_stability_panel(data, def_params)
        if stability_panel.notna().sum().sum() > 100:
            panels["earnings_stability"] = stability_panel

        # 2. Low Beta (inverted)
        beta_panel = self._build_beta_panel(data, def_params)
        if beta_panel.notna().sum().sum() > 100:
            panels["low_beta"] = -beta_panel  # Invert: low beta = high score

        # Combine components
        components = []
        for name, panel in panels.items():
            components.append((name, cross_sectional_zscore(panel)))

        if not components:
            return pd.DataFrame(np.nan, index=dates, columns=tickers), {"warning": "No components"}

        raw_scores = combine_components_zscore(components, dates, tickers)

        metadata = {
            "components": [c[0] for c in components],
            "earnings_stability_quarters": def_params.earnings_stability_quarters,
            "beta_lookback": def_params.beta_lookback_days,
            "coverage_pct": float(raw_scores.notna().sum().sum()) / max(raw_scores.size, 1) * 100,
        }

        return raw_scores, metadata

    def _build_earnings_stability_panel(
        self,
        data: FactorData,
        def_params: DefensiveParams,
    ) -> pd.DataFrame:
        """Build earnings stability panel."""
        dates = data.dates
        tickers = data.tickers

        stability_panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)

        if data.data_loader is None:
            return stability_panel

        try:
            fundamentals_parquet = data.data_loader.load_fundamentals_parquet()
        except Exception:
            return stability_panel

        if fundamentals_parquet is None:
            return stability_panel

        try:
            from ..factor_builders import (
                INCOME_SHEET,
                NET_INCOME_KEYS,
            )
            from ...common.utils import (
                get_consolidated_sheet,
                pick_row_from_sheet,
                coerce_quarter_cols,
                sum_ttm,
                apply_lag,
            )
        except ImportError:
            return stability_panel

        for ticker in tickers:
            try:
                inc = get_consolidated_sheet(fundamentals_parquet, ticker, INCOME_SHEET)
                if inc.empty:
                    continue

                ni_row = pick_row_from_sheet(inc, NET_INCOME_KEYS)
                if ni_row is None:
                    continue

                ni = coerce_quarter_cols(ni_row)
                if ni.empty:
                    continue

                ni_ttm = sum_ttm(ni)
                if ni_ttm.empty:
                    continue

                # Rolling earnings stability = 1 / CV (coefficient of variation)
                # CV = std / |mean|
                n_quarters = def_params.earnings_stability_quarters
                rolling_mean = ni_ttm.rolling(n_quarters, min_periods=4).mean()
                rolling_std = ni_ttm.rolling(n_quarters, min_periods=4).std()

                cv = rolling_std / rolling_mean.abs().replace(0, np.nan)
                stability = 1.0 / cv.replace(0, np.nan)
                stability = stability.replace([np.inf, -np.inf], np.nan)
                stability = stability.clip(
                    lower=def_params.stability_clip[0],
                    upper=def_params.stability_clip[1]
                )

                if not stability.dropna().empty:
                    stability_panel[ticker] = apply_lag(stability, dates)

            except Exception:
                continue

        return stability_panel

    def _build_beta_panel(
        self,
        data: FactorData,
        def_params: DefensiveParams,
    ) -> pd.DataFrame:
        """Build market beta panel."""
        dates = data.dates
        tickers = data.tickers
        close = data.close.reindex(index=dates, columns=tickers).astype(float)

        lookback = def_params.beta_lookback_days
        min_obs = def_params.beta_min_observations

        daily_returns = close.pct_change()

        # Equal-weighted market return
        market_return = daily_returns.mean(axis=1)
        market_var = market_return.rolling(lookback, min_periods=min_obs).var()

        beta_panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)

        for ticker in tickers:
            stock_ret = daily_returns[ticker]
            cov = stock_ret.rolling(lookback, min_periods=min_obs).cov(market_return)
            beta = cov / market_var.replace(0.0, np.nan)
            beta_panel[ticker] = beta

        # Clip extreme betas
        beta_panel = beta_panel.clip(lower=-2.0, upper=5.0)

        return beta_panel

    def get_default_params(self) -> FactorParams:
        """Return defensive-specific default parameters."""
        return FactorParams(
            lookback_days=252,
            lag_days=0,
            winsorize_pct=2.0,
            custom={
                "earnings_stability_quarters": 8,
                "beta_lookback_days": 252,
                "beta_min_observations": 126,
            },
        )


class LowVolatilitySignal(FactorSignal):
    """
    Low Volatility factor: favors stocks with lower realized volatility.

    Based on the low-volatility anomaly.
    """

    @property
    def name(self) -> str:
        return "low_volatility"

    @property
    def description(self) -> str:
        return (
            "Low volatility factor. Stocks with lower realized volatility "
            "receive higher scores. Based on the low-volatility anomaly - "
            "less volatile stocks tend to outperform on risk-adjusted basis."
        )

    @property
    def category(self) -> str:
        return "defensive"

    @property
    def higher_is_better(self) -> bool:
        return True  # Higher score = lower volatility

    def compute_raw_signal(
        self,
        data: FactorData,
        params: FactorParams,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Compute low volatility signal (inverted volatility)."""
        dates = data.dates
        tickers = data.tickers
        close = data.close.reindex(index=dates, columns=tickers).astype(float)

        lookback = params.lookback_days if params.lookback_days > 1 else 63
        min_obs = max(int(lookback * 0.5), 21)

        daily_returns = close.pct_change()
        volatility = daily_returns.rolling(lookback, min_periods=min_obs).std() * np.sqrt(252)

        # Invert: low volatility = high score
        raw_scores = -volatility

        metadata = {
            "lookback_days": lookback,
            "annualized": True,
            "inverted": True,
        }

        return raw_scores, metadata

    def get_default_params(self) -> FactorParams:
        return FactorParams(lookback_days=63)


# =============================================================================
# POTENTIAL IMPROVEMENTS
# =============================================================================
"""
1. Downside Volatility:
   - Only consider negative returns
   - More relevant for downside protection

2. Maximum Drawdown:
   - Worst peak-to-trough decline
   - Captures tail risk

3. Revenue Stability:
   - Coefficient of variation of revenues
   - More stable than earnings

4. Business Model Stability:
   - Recurring revenue percentage
   - Contract length/visibility

5. Sector-Based Defensiveness:
   - Utility, consumer staples, healthcare
   - Industry classification as defensive indicator

6. Dividend Stability:
   - Years of consecutive dividend increases
   - Dividend aristocrats
"""
