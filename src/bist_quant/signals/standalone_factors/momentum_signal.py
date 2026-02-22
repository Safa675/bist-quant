"""
Momentum Factor Signal

Economic Intuition:
------------------
Price momentum is one of the most robust anomalies in finance. Stocks that have
performed well in the recent past tend to continue outperforming, and vice versa.
This is attributed to:
1. Underreaction to information - gradual price adjustment to news
2. Herding behavior - investors follow trends
3. Confirmation bias - winners attract more attention

The standard momentum measure skips the most recent month to avoid short-term
reversal effects.

Mathematical Construction:
-------------------------
Momentum = (P_t-skip / P_t-lookback-skip) - 1

Where:
- lookback = 126 days (6 months) or 252 days (12 months)
- skip = 21 days (1 month) to avoid reversal

Input Data Requirements:
-----------------------
- Daily close prices

Normalization:
-------------
Cross-sectional z-score of momentum returns.
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
class MomentumParams:
    """Momentum-specific parameters."""
    lookback_days: int = 126       # Main lookback window
    skip_days: int = 21            # Skip recent days (avoid reversal)
    use_log_returns: bool = False  # Use log returns instead of simple
    volatility_adjust: bool = False # Divide by realized volatility


class MomentumSignal(FactorSignal):
    """
    Momentum factor: favors recent winners.

    Higher scores indicate stocks with stronger recent performance
    (excluding the most recent month to avoid short-term reversal).
    """

    @property
    def name(self) -> str:
        return "momentum"

    @property
    def description(self) -> str:
        return (
            "Price momentum factor. Stocks with strong recent returns (past 6-12 months) "
            "receive higher scores based on the empirical finding that 'winners' tend to "
            "continue outperforming and 'losers' tend to continue underperforming "
            "(Jegadeesh-Titman momentum)."
        )

    @property
    def category(self) -> str:
        return "momentum"

    @property
    def higher_is_better(self) -> bool:
        return True  # Higher momentum = expected outperformance

    def compute_raw_signal(
        self,
        data: FactorData,
        params: FactorParams,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Compute raw momentum signal.

        Returns:
            Tuple of (raw_scores, metadata)
        """
        dates = data.dates
        tickers = data.tickers
        close = data.close.reindex(index=dates, columns=tickers).astype(float)

        # Get momentum-specific params
        mom_params = MomentumParams(**params.custom) if params.custom else MomentumParams()
        lookback = mom_params.lookback_days
        skip = mom_params.skip_days

        # Calculate momentum
        # Price at t-skip / Price at t-lookback-skip - 1
        if mom_params.use_log_returns:
            log_price = np.log(close)
            momentum = log_price.shift(skip) - log_price.shift(lookback + skip)
        else:
            momentum = close.shift(skip) / close.shift(lookback + skip) - 1.0

        # Optional volatility adjustment (risk-adjusted momentum)
        if mom_params.volatility_adjust:
            daily_ret = close.pct_change()
            vol = daily_ret.rolling(lookback, min_periods=lookback // 2).std()
            momentum = momentum / vol.replace(0, np.nan)

        raw_scores = momentum.reindex(index=dates, columns=tickers)

        metadata = {
            "lookback_days": lookback,
            "skip_days": skip,
            "volatility_adjusted": mom_params.volatility_adjust,
            "coverage_pct": float(raw_scores.notna().sum().sum()) / max(raw_scores.size, 1) * 100,
        }

        return raw_scores, metadata

    def get_default_params(self) -> FactorParams:
        """Return momentum-specific default parameters."""
        return FactorParams(
            lookback_days=126,
            lag_days=0,
            winsorize_pct=1.0,
            custom={
                "lookback_days": 126,
                "skip_days": 21,
                "volatility_adjust": False,
            },
        )


# =============================================================================
# ALTERNATIVE MOMENTUM CONSTRUCTIONS
# =============================================================================
"""
1. Risk-Adjusted Momentum (Sharpe Momentum):
   Momentum / Volatility
   - Reduces exposure to volatile "lottery" stocks
   - More stable performance across market regimes

2. Information Discreteness:
   Count positive days / total days in lookback
   - Captures whether momentum is driven by few large moves or steady gains
   - Steady momentum is more persistent

3. 52-Week High Momentum:
   Current Price / 52-Week High
   - Anchoring effect: investors use 52w high as reference
   - Near-high stocks tend to break out

4. Residual Momentum:
   Orthogonalized to market and industry factors
   - Captures stock-specific momentum
   - More alpha, less beta exposure

5. Fundamental Momentum:
   Based on earnings revisions rather than price
   - Leading indicator of price momentum
   - Less prone to reversal

6. Time-Series Momentum:
   Whether return > 0 over lookback (binary)
   - Absolute momentum, not relative
   - Works well in market timing
"""


class VolatilityAdjustedMomentumSignal(MomentumSignal):
    """Momentum divided by realized volatility (Sharpe momentum)."""

    @property
    def name(self) -> str:
        return "momentum_vol_adj"

    @property
    def description(self) -> str:
        return (
            "Volatility-adjusted momentum (Sharpe momentum). Similar to standard momentum "
            "but divided by realized volatility. Reduces exposure to high-volatility 'lottery' "
            "stocks and provides more risk-adjusted signal."
        )

    def get_default_params(self) -> FactorParams:
        base = super().get_default_params()
        base.custom["volatility_adjust"] = True
        return base


class MultiHorizonMomentumSignal(FactorSignal):
    """
    Ensemble momentum across multiple horizons.

    Combines 1-month, 3-month, 6-month, and 12-month momentum signals.
    """

    @property
    def name(self) -> str:
        return "momentum_ensemble"

    @property
    def description(self) -> str:
        return (
            "Multi-horizon momentum ensemble. Combines momentum signals across "
            "1-month, 3-month, 6-month, and 12-month lookback windows. "
            "More robust than single-horizon momentum."
        )

    @property
    def category(self) -> str:
        return "momentum"

    @property
    def higher_is_better(self) -> bool:
        return True

    def compute_raw_signal(
        self,
        data: FactorData,
        params: FactorParams,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Compute ensemble momentum across multiple horizons."""
        dates = data.dates
        tickers = data.tickers
        close = data.close.reindex(index=dates, columns=tickers).astype(float)

        # Horizons: 1mo, 3mo, 6mo, 12mo (all skip 1 week)
        horizons = [21, 63, 126, 252]
        skip = 5  # 1 week skip

        components = []
        for h in horizons:
            mom = close.shift(skip) / close.shift(h + skip) - 1.0
            # Z-score each horizon
            row_mean = mom.mean(axis=1)
            row_std = mom.std(axis=1).replace(0, np.nan)
            mom_z = mom.sub(row_mean, axis=0).div(row_std, axis=0)
            components.append(mom_z)

        # Equal-weighted average
        stacked = np.stack([c.values for c in components], axis=0)
        raw_scores = pd.DataFrame(
            np.nanmean(stacked, axis=0),
            index=dates,
            columns=tickers,
        )

        metadata = {
            "horizons": horizons,
            "skip_days": skip,
            "n_horizons": len(horizons),
        }

        return raw_scores, metadata
