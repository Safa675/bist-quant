"""
Liquidity Factor Signal

Economic Intuition:
------------------
Liquidity measures the ease of trading without significant price impact.
Illiquid stocks tend to earn a premium because:
1. Higher transaction costs reduce returns
2. Difficult to exit positions quickly
3. Less efficient pricing
4. More concentrated ownership

However, liquid stocks are preferred for portfolio construction due to:
1. Lower execution costs
2. Easier position sizing
3. Better price discovery

Liquidity Components:
1. Amihud Illiquidity - |return| / dollar volume (lower = more liquid)
2. Turnover - Volume / Shares Outstanding (higher = more liquid)
3. Spread Proxy - High-Low range / Close (lower = more liquid)

Mathematical Construction:
-------------------------
Liquidity Score = mean(-z(Amihud), z(Turnover), -z(Spread))

Note: Amihud and Spread are inverted (low = good).

Input Data Requirements:
-----------------------
- Daily close prices
- Daily volume
- Shares outstanding

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
class LiquidityParams:
    """Liquidity-specific parameters."""
    amihud_lookback_days: int = 21
    turnover_lookback_days: int = 63
    min_observations: int = 10
    log_transform_amihud: bool = True  # Log transform reduces skewness


class LiquiditySignal(FactorSignal):
    """
    Liquidity factor: measures ease of trading.

    Higher scores indicate more liquid stocks (easier to trade).
    Combines Amihud illiquidity and turnover metrics.
    """

    @property
    def name(self) -> str:
        return "liquidity"

    @property
    def description(self) -> str:
        return (
            "Liquidity factor. Measures ease of trading without price impact. "
            "Combines Amihud illiquidity ratio (inverted) and share turnover. "
            "Higher scores indicate more liquid stocks."
        )

    @property
    def category(self) -> str:
        return "liquidity"

    @property
    def higher_is_better(self) -> bool:
        return True  # Higher liquidity = easier to trade

    def compute_raw_signal(
        self,
        data: FactorData,
        params: FactorParams,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Compute composite liquidity signal.

        Returns:
            Tuple of (raw_scores, metadata)
        """
        dates = data.dates
        tickers = data.tickers
        close = data.close.reindex(index=dates, columns=tickers).astype(float)

        liq_params = LiquidityParams(**params.custom) if params.custom else LiquidityParams()

        # Get volume
        if data.volume is not None:
            volume = data.volume.reindex(index=dates, columns=tickers)
        else:
            return pd.DataFrame(np.nan, index=dates, columns=tickers), {"warning": "No volume data"}

        # Build component panels
        panels = {}

        # Amihud Illiquidity = |return| / dollar volume
        daily_returns = close.pct_change().abs()
        dollar_volume = close * volume
        amihud_daily = daily_returns / dollar_volume.replace(0, np.nan)
        amihud = amihud_daily.rolling(liq_params.amihud_lookback_days, min_periods=liq_params.min_observations).mean()

        if liq_params.log_transform_amihud:
            amihud = np.log1p(amihud * 1e6).replace([np.inf, -np.inf], np.nan)

        if amihud.notna().sum().sum() > 100:
            panels["amihud"] = amihud

        # Turnover = Volume / Shares Outstanding
        if data.shares_outstanding is not None:
            shares = data.shares_outstanding.reindex(index=dates, columns=tickers).ffill()
            turnover = volume / shares.replace(0, np.nan)
            turnover_smooth = turnover.rolling(liq_params.turnover_lookback_days, min_periods=21).mean()
            turnover_smooth = turnover_smooth.replace([np.inf, -np.inf], np.nan)

            if turnover_smooth.notna().sum().sum() > 100:
                panels["turnover"] = turnover_smooth

        # Combine components
        components = []

        if "amihud" in panels:
            # Invert Amihud (low = more liquid = good)
            components.append(("amihud", -cross_sectional_zscore(panels["amihud"])))

        if "turnover" in panels:
            # High turnover = more liquid = good
            components.append(("turnover", cross_sectional_zscore(panels["turnover"])))

        if not components:
            return pd.DataFrame(np.nan, index=dates, columns=tickers), {"warning": "No liquidity components"}

        raw_scores = combine_components_zscore(components, dates, tickers)

        metadata = {
            "components": [c[0] for c in components],
            "amihud_lookback": liq_params.amihud_lookback_days,
            "turnover_lookback": liq_params.turnover_lookback_days,
            "coverage_pct": float(raw_scores.notna().sum().sum()) / max(raw_scores.size, 1) * 100,
        }

        return raw_scores, metadata

    def get_default_params(self) -> FactorParams:
        """Return liquidity-specific default parameters."""
        return FactorParams(
            lookback_days=21,
            lag_days=0,
            winsorize_pct=2.0,
            custom={
                "amihud_lookback_days": 21,
                "turnover_lookback_days": 63,
                "log_transform_amihud": True,
            },
        )


class IlliquidityPremiumSignal(FactorSignal):
    """
    Illiquidity premium factor: favors illiquid stocks.

    Based on the liquidity premium - illiquid stocks tend to earn
    higher returns as compensation for difficulty in trading.
    """

    @property
    def name(self) -> str:
        return "illiquidity_premium"

    @property
    def description(self) -> str:
        return (
            "Illiquidity premium factor. Illiquid stocks (high Amihud, low turnover) "
            "receive higher scores. Based on the empirical finding that illiquid stocks "
            "earn a premium for bearing liquidity risk."
        )

    @property
    def category(self) -> str:
        return "liquidity"

    @property
    def higher_is_better(self) -> bool:
        return True  # Higher illiquidity = higher expected premium

    def compute_raw_signal(
        self,
        data: FactorData,
        params: FactorParams,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Compute inverted liquidity signal."""
        liq_signal = LiquiditySignal()
        liq_scores, metadata = liq_signal.compute_raw_signal(data, params)

        # Invert: illiquid = high score
        raw_scores = -liq_scores

        metadata["inverted"] = True
        metadata["signal_type"] = "illiquidity_premium"

        return raw_scores, metadata


# =============================================================================
# POTENTIAL IMPROVEMENTS
# =============================================================================
"""
1. Dollar Volume Based Liquidity:
   - Average daily dollar volume
   - More intuitive measure for institutional traders

2. Bid-Ask Spread:
   - Actual spread data if available
   - Most direct measure of transaction costs

3. Price Impact:
   - Kyle's lambda
   - Measures how much price moves per unit volume

4. Liquidity Beta:
   - Covariance with market liquidity
   - Stocks with high liquidity beta are riskier in crises

5. Liquidity Trend:
   - Change in liquidity over time
   - Deteriorating liquidity may predict problems

6. Free Float Adjusted:
   - Consider only tradable shares
   - More accurate for stocks with large blocks
"""
