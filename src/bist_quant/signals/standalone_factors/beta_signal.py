"""
Beta (Market Sensitivity) Factor Signal

Economic Intuition:
------------------
Market beta measures a stock's sensitivity to overall market movements.
This factor captures the cyclicality of stocks:

High Beta:
- More volatile than market
- Outperform in bull markets, underperform in bear markets
- Typically growth stocks, cyclicals, high leverage

Low Beta:
- Less volatile than market
- Defensive characteristics
- Typically utilities, consumer staples, healthcare

The factor allows rotation between aggressive (high beta) and defensive (low beta)
positions based on market regime.

Mathematical Construction:
-------------------------
Beta = Cov(r_stock, r_market) / Var(r_market)

Calculated using rolling 252-day window with minimum 126 observations.
Market return is the equal-weighted average of all stocks.

Input Data Requirements:
-----------------------
- Daily close prices

Normalization:
-------------
Cross-sectional z-score of beta values.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .base import (
    FactorSignal,
    FactorData,
    FactorParams,
)


@dataclass
class BetaParams:
    """Beta-specific parameters."""
    lookback_days: int = 252       # Rolling window for beta calculation
    min_observations: int = 126    # Minimum observations required
    market_proxy: str = "equal_weight"  # "equal_weight" or "cap_weight"
    clip_beta: Tuple[float, float] = (-2.0, 5.0)  # Clip extreme betas


class BetaSignal(FactorSignal):
    """
    Beta factor: measures market sensitivity.

    Higher scores indicate higher beta (more cyclical/aggressive).
    Can be used to rotate between risk-on and risk-off exposures.
    """

    @property
    def name(self) -> str:
        return "beta"

    @property
    def description(self) -> str:
        return (
            "Market beta factor. Measures stock sensitivity to market movements. "
            "High beta stocks are more volatile and cyclical (outperform in bull markets). "
            "Low beta stocks are defensive (outperform in bear markets)."
        )

    @property
    def category(self) -> str:
        return "risk"

    @property
    def higher_is_better(self) -> bool:
        # Depends on market regime - neutral by default
        return True  # In this axis, high beta = aggressive = one end of spectrum

    def compute_raw_signal(
        self,
        data: FactorData,
        params: FactorParams,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Compute rolling market beta.

        Returns:
            Tuple of (raw_scores, metadata)
        """
        dates = data.dates
        tickers = data.tickers
        close = data.close.reindex(index=dates, columns=tickers).astype(float)

        # Get beta-specific params
        beta_params = BetaParams(**params.custom) if params.custom else BetaParams()
        lookback = beta_params.lookback_days
        min_obs = beta_params.min_observations

        # Calculate daily returns
        daily_returns = close.pct_change()

        # Calculate market return (benchmark)
        if beta_params.market_proxy == "cap_weight" and data.market_cap is not None:
            mcap = data.market_cap.reindex(index=dates, columns=tickers)
            weights = mcap.div(mcap.sum(axis=1), axis=0)
            market_return = (daily_returns * weights).sum(axis=1)
        else:
            # Equal-weighted market return
            market_return = daily_returns.mean(axis=1)

        # Calculate market variance (rolling)
        market_var = market_return.rolling(lookback, min_periods=min_obs).var()

        # Calculate beta for each stock
        beta_panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)

        for ticker in tickers:
            stock_ret = daily_returns[ticker]
            # Rolling covariance with market
            cov = stock_ret.rolling(lookback, min_periods=min_obs).cov(market_return)
            beta = cov / market_var.replace(0.0, np.nan)
            beta_panel[ticker] = beta

        # Clip extreme values
        raw_scores = beta_panel.clip(
            lower=beta_params.clip_beta[0],
            upper=beta_params.clip_beta[1],
        )

        metadata = {
            "lookback_days": lookback,
            "min_observations": min_obs,
            "market_proxy": beta_params.market_proxy,
            "beta_clip": beta_params.clip_beta,
            "avg_beta": float(raw_scores.iloc[-1].mean()) if len(raw_scores) > 0 else np.nan,
            "coverage_pct": float(raw_scores.notna().sum().sum()) / max(raw_scores.size, 1) * 100,
        }

        return raw_scores, metadata

    def get_default_params(self) -> FactorParams:
        """Return beta-specific default parameters."""
        return FactorParams(
            lookback_days=252,
            lag_days=0,
            winsorize_pct=1.0,
            custom={
                "lookback_days": 252,
                "min_observations": 126,
                "market_proxy": "equal_weight",
                "clip_beta": (-2.0, 5.0),
            },
        )


class LowBetaSignal(FactorSignal):
    """
    Low Beta factor: favors defensive stocks.

    This is the inverse of beta - higher scores indicate lower beta (defensive).
    """

    @property
    def name(self) -> str:
        return "low_beta"

    @property
    def description(self) -> str:
        return (
            "Low beta (defensive) factor. Stocks with lower market sensitivity "
            "receive higher scores. Based on the low-volatility anomaly - "
            "defensive stocks tend to outperform on a risk-adjusted basis."
        )

    @property
    def category(self) -> str:
        return "risk"

    @property
    def higher_is_better(self) -> bool:
        return True  # Higher score = lower beta = more defensive

    def compute_raw_signal(
        self,
        data: FactorData,
        params: FactorParams,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Compute inverted beta (low beta = high score)."""
        beta_signal = BetaSignal()
        beta_scores, metadata = beta_signal.compute_raw_signal(data, params)

        # Invert: low beta becomes high score
        raw_scores = -beta_scores

        metadata["inverted"] = True
        metadata["signal_type"] = "low_beta"

        return raw_scores, metadata


class VolatilitySignal(FactorSignal):
    """
    Realized volatility factor.

    Measures idiosyncratic volatility of returns.
    """

    @property
    def name(self) -> str:
        return "volatility"

    @property
    def description(self) -> str:
        return (
            "Realized volatility factor. Measures rolling standard deviation of returns. "
            "Can be used for low-volatility strategies or volatility timing."
        )

    @property
    def category(self) -> str:
        return "risk"

    @property
    def higher_is_better(self) -> bool:
        return False  # Typically low volatility is preferred

    def compute_raw_signal(
        self,
        data: FactorData,
        params: FactorParams,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Compute rolling realized volatility (annualized)."""
        dates = data.dates
        tickers = data.tickers
        close = data.close.reindex(index=dates, columns=tickers).astype(float)

        lookback = params.lookback_days if params.lookback_days > 1 else 63
        min_obs = max(int(lookback * 0.5), 21)

        daily_returns = close.pct_change()
        volatility = daily_returns.rolling(lookback, min_periods=min_obs).std() * np.sqrt(252)

        raw_scores = volatility.reindex(index=dates, columns=tickers)

        metadata = {
            "lookback_days": lookback,
            "annualized": True,
            "avg_vol": float(raw_scores.iloc[-1].mean()) if len(raw_scores) > 0 else np.nan,
        }

        return raw_scores, metadata

    def get_default_params(self) -> FactorParams:
        return FactorParams(
            lookback_days=63,
            lag_days=0,
        )


# =============================================================================
# POTENTIAL IMPROVEMENTS
# =============================================================================
"""
1. Downside Beta:
   - Beta calculated only on down days
   - Captures asymmetric risk exposure

2. Idiosyncratic Volatility:
   - Residual volatility after removing market factor
   - Pure stock-specific risk

3. CAPM Alpha:
   - Rolling Jensen's alpha
   - Risk-adjusted outperformance

4. Regime-Conditional Beta:
   - Calculate beta separately in high-vol and low-vol periods
   - Beta can change with market conditions

5. Industry-Adjusted Beta:
   - Beta relative to industry rather than market
   - Captures within-industry cyclicality

6. Fundamental Beta:
   - Based on accounting metrics (leverage, operating leverage)
   - More stable than market beta
"""
