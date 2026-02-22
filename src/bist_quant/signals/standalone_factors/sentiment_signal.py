"""
Sentiment / Price Action Factor Signal

Economic Intuition:
------------------
Technical price action patterns capture market sentiment and psychology.
Unlike fundamental factors, these are purely price-derived signals that
capture behavioral effects:

1. 52-Week High Proximity: Anchoring bias - investors anchor to reference points
   - Near-high stocks often break out (momentum)
   - Far-from-high stocks may be undervalued or in distress

2. Price Acceleration: Second derivative of price
   - Accelerating prices indicate strengthening trend
   - Decelerating may signal exhaustion

3. Short-term Reversal: Mean reversion effect
   - Very short-term moves tend to reverse
   - Overreaction to news creates opportunities

Sentiment Components:
1. 52-Week High Proximity - Price / 52-Week High
2. Price Acceleration - Short-term momentum - Long-term momentum
3. Reversal - Negative of very short-term return

Mathematical Construction:
-------------------------
Sentiment Score = mean(z(52wHighPct), z(Acceleration), z(Reversal))

Input Data Requirements:
-----------------------
- Daily close prices

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
class SentimentParams:
    """Sentiment-specific parameters."""
    high_lookback_days: int = 252          # 52-week high
    high_min_observations: int = 126       # Minimum for 52w calculation
    acceleration_fast_days: int = 21       # Fast momentum
    acceleration_slow_days: int = 63       # Slow momentum
    reversal_lookback_days: int = 5        # Short-term reversal
    include_reversal: bool = True          # Include reversal component


class SentimentSignal(FactorSignal):
    """
    Sentiment/Price Action factor: captures technical patterns.

    Higher scores indicate stronger bullish price action
    (near highs, accelerating, etc.).
    """

    @property
    def name(self) -> str:
        return "sentiment"

    @property
    def description(self) -> str:
        return (
            "Sentiment/Price Action factor. Captures technical patterns including "
            "52-week high proximity, price acceleration, and short-term reversal. "
            "Higher scores indicate stronger bullish price action."
        )

    @property
    def category(self) -> str:
        return "sentiment"

    @property
    def higher_is_better(self) -> bool:
        return True  # Higher sentiment = bullish price action

    def compute_raw_signal(
        self,
        data: FactorData,
        params: FactorParams,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Compute sentiment signal from price patterns.

        Returns:
            Tuple of (raw_scores, metadata)
        """
        dates = data.dates
        tickers = data.tickers
        close = data.close.reindex(index=dates, columns=tickers).astype(float)

        sent_params = SentimentParams(**params.custom) if params.custom else SentimentParams()

        panels = {}

        # 1. 52-Week High Proximity
        rolling_high = close.rolling(
            sent_params.high_lookback_days,
            min_periods=sent_params.high_min_observations
        ).max()
        high_proximity = close / rolling_high.replace(0, np.nan)
        high_proximity = high_proximity.replace([np.inf, -np.inf], np.nan)

        if high_proximity.notna().sum().sum() > 100:
            panels["52w_high_pct"] = high_proximity

        # 2. Price Acceleration
        mom_fast = close.pct_change(sent_params.acceleration_fast_days)
        mom_slow = close.pct_change(sent_params.acceleration_slow_days)
        acceleration = mom_fast - mom_slow

        if acceleration.notna().sum().sum() > 100:
            panels["price_acceleration"] = acceleration

        # 3. Short-term Reversal (inverted short-term return)
        if sent_params.include_reversal:
            reversal = -close.pct_change(sent_params.reversal_lookback_days)

            if reversal.notna().sum().sum() > 100:
                panels["reversal"] = reversal

        # Combine components
        components = []
        for name, panel in panels.items():
            components.append((name, cross_sectional_zscore(panel)))

        if not components:
            return pd.DataFrame(np.nan, index=dates, columns=tickers), {"warning": "No components"}

        raw_scores = combine_components_zscore(components, dates, tickers)

        metadata = {
            "components": [c[0] for c in components],
            "high_lookback": sent_params.high_lookback_days,
            "acceleration_windows": (sent_params.acceleration_fast_days, sent_params.acceleration_slow_days),
            "reversal_lookback": sent_params.reversal_lookback_days,
            "coverage_pct": float(raw_scores.notna().sum().sum()) / max(raw_scores.size, 1) * 100,
        }

        return raw_scores, metadata

    def get_default_params(self) -> FactorParams:
        """Return sentiment-specific default parameters."""
        return FactorParams(
            lookback_days=252,
            lag_days=0,
            winsorize_pct=1.0,
            custom={
                "high_lookback_days": 252,
                "acceleration_fast_days": 21,
                "acceleration_slow_days": 63,
                "reversal_lookback_days": 5,
                "include_reversal": True,
            },
        )


class FiftyTwoWeekHighSignal(FactorSignal):
    """
    52-Week High Signal: standalone proximity to 52-week high.

    Based on anchoring bias - stocks near their 52-week high
    often continue to new highs (momentum effect).
    """

    @property
    def name(self) -> str:
        return "52w_high"

    @property
    def description(self) -> str:
        return (
            "52-week high proximity factor. Stocks trading near their 52-week high "
            "receive higher scores. Based on anchoring bias and breakout patterns."
        )

    @property
    def category(self) -> str:
        return "sentiment"

    @property
    def higher_is_better(self) -> bool:
        return True

    def compute_raw_signal(
        self,
        data: FactorData,
        params: FactorParams,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Compute 52-week high proximity."""
        dates = data.dates
        tickers = data.tickers
        close = data.close.reindex(index=dates, columns=tickers).astype(float)

        rolling_high = close.rolling(252, min_periods=126).max()
        raw_scores = close / rolling_high.replace(0, np.nan)
        raw_scores = raw_scores.replace([np.inf, -np.inf], np.nan)

        metadata = {"lookback_days": 252}

        return raw_scores, metadata


class ShortTermReversalSignal(FactorSignal):
    """
    Short-Term Reversal: contrarian signal based on mean reversion.

    Very short-term moves tend to reverse due to overreaction.
    """

    @property
    def name(self) -> str:
        return "short_term_reversal"

    @property
    def description(self) -> str:
        return (
            "Short-term reversal factor. Stocks with negative recent returns "
            "(5-day losers) receive higher scores. Based on mean reversion "
            "from short-term overreaction."
        )

    @property
    def category(self) -> str:
        return "sentiment"

    @property
    def higher_is_better(self) -> bool:
        return True  # Recent losers expected to bounce

    def compute_raw_signal(
        self,
        data: FactorData,
        params: FactorParams,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Compute short-term reversal signal."""
        dates = data.dates
        tickers = data.tickers
        close = data.close.reindex(index=dates, columns=tickers).astype(float)

        lookback = params.custom.get("lookback_days", 5) if params.custom else 5

        # Negative of recent return (losers get high score)
        raw_scores = -close.pct_change(lookback)

        metadata = {"lookback_days": lookback}

        return raw_scores, metadata

    def get_default_params(self) -> FactorParams:
        return FactorParams(
            lookback_days=5,
            custom={"lookback_days": 5},
        )


# =============================================================================
# POTENTIAL IMPROVEMENTS
# =============================================================================
"""
1. RSI-Based Sentiment:
   - Relative Strength Index
   - Overbought/oversold indicator

2. Distance from Moving Average:
   - Price / 200-day MA
   - Trend strength indicator

3. Bollinger Band Position:
   - Where price is within bands
   - Volatility-adjusted sentiment

4. New High/New Low Ratio:
   - Breadth indicator adapted to individual stocks
   - Number of days making new highs

5. Volume-Confirmed Breakouts:
   - Near 52w high + above average volume
   - More reliable breakout signal

6. Drawdown Measure:
   - Distance from all-time or recent high
   - Recovery potential indicator
"""
