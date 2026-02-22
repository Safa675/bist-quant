"""
Trading Intensity Factor Signal

Economic Intuition:
------------------
Trading intensity measures the level of market attention and trading activity.
This is distinct from liquidity:
- Liquidity = ease of trading without price impact
- Trading Intensity = level of trading activity / attention

High trading intensity may indicate:
1. Institutional interest - funds accumulating/distributing
2. Information events - news, earnings, analyst coverage
3. Momentum building - trend following activity
4. Speculation - retail attention

Trading Intensity Components:
1. Relative Volume - Volume / Historical Average (unusual activity)
2. Volume Trend - Short-term / Long-term volume ratio (momentum)
3. Turnover Velocity - Annualized turnover rate

Mathematical Construction:
-------------------------
Trading Intensity = mean(z(RelVol), z(VolTrend), z(TurnoverVelocity))

Input Data Requirements:
-----------------------
- Daily volume
- Shares outstanding (for turnover velocity)

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
class TradingIntensityParams:
    """Trading intensity-specific parameters."""
    relative_volume_baseline_days: int = 252  # 1 year baseline
    relative_volume_smooth_days: int = 63     # Smooth relative volume
    volume_trend_short_days: int = 21         # Short-term volume
    volume_trend_long_days: int = 126         # Long-term volume
    turnover_velocity_smooth_days: int = 21
    annualize_turnover: bool = True


class TradingIntensitySignal(FactorSignal):
    """
    Trading Intensity factor: measures level of trading activity.

    Higher scores indicate more active trading (higher attention).
    Distinct from liquidity which measures ease of trading.
    """

    @property
    def name(self) -> str:
        return "trading_intensity"

    @property
    def description(self) -> str:
        return (
            "Trading intensity factor. Measures level of trading activity relative "
            "to historical norms. Combines relative volume, volume trend, and "
            "turnover velocity. Higher scores indicate more active/attention."
        )

    @property
    def category(self) -> str:
        return "trading_intensity"

    @property
    def higher_is_better(self) -> bool:
        return True  # Higher activity = more attention (regime dependent)

    def compute_raw_signal(
        self,
        data: FactorData,
        params: FactorParams,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Compute trading intensity signal.

        Returns:
            Tuple of (raw_scores, metadata)
        """
        dates = data.dates
        tickers = data.tickers

        ti_params = TradingIntensityParams(**params.custom) if params.custom else TradingIntensityParams()

        if data.volume is None:
            return pd.DataFrame(np.nan, index=dates, columns=tickers), {"warning": "No volume data"}

        volume = data.volume.reindex(index=dates, columns=tickers)

        panels = {}

        # 1. Relative Volume = Volume / Historical Average
        avg_volume = volume.rolling(ti_params.relative_volume_baseline_days, min_periods=63).mean()
        relative_volume = (volume / avg_volume.replace(0, np.nan)).rolling(
            ti_params.relative_volume_smooth_days, min_periods=21
        ).mean()
        relative_volume = relative_volume.replace([np.inf, -np.inf], np.nan)

        if relative_volume.notna().sum().sum() > 100:
            panels["relative_volume"] = relative_volume

        # 2. Volume Trend = Short-term Avg / Long-term Avg - 1
        short_vol = volume.rolling(ti_params.volume_trend_short_days, min_periods=10).mean()
        long_vol = volume.rolling(ti_params.volume_trend_long_days, min_periods=42).mean()
        volume_trend = (short_vol / long_vol.replace(0, np.nan) - 1.0).replace([np.inf, -np.inf], np.nan)

        if volume_trend.notna().sum().sum() > 100:
            panels["volume_trend"] = volume_trend

        # 3. Turnover Velocity = (Volume / Shares Outstanding) * 252
        if data.shares_outstanding is not None:
            shares = data.shares_outstanding.reindex(index=dates, columns=tickers).ffill()
            daily_turnover = volume / shares.replace(0, np.nan)
            turnover_smooth = daily_turnover.rolling(ti_params.turnover_velocity_smooth_days, min_periods=10).mean()

            if ti_params.annualize_turnover:
                turnover_velocity = turnover_smooth * 252
            else:
                turnover_velocity = turnover_smooth

            turnover_velocity = turnover_velocity.replace([np.inf, -np.inf], np.nan)

            if turnover_velocity.notna().sum().sum() > 100:
                panels["turnover_velocity"] = turnover_velocity

        # Combine components
        components = []
        for name, panel in panels.items():
            components.append((name, cross_sectional_zscore(panel)))

        if not components:
            return pd.DataFrame(np.nan, index=dates, columns=tickers), {"warning": "No components available"}

        raw_scores = combine_components_zscore(components, dates, tickers)

        metadata = {
            "components": [c[0] for c in components],
            "relative_volume_baseline": ti_params.relative_volume_baseline_days,
            "volume_trend_windows": (ti_params.volume_trend_short_days, ti_params.volume_trend_long_days),
            "coverage_pct": float(raw_scores.notna().sum().sum()) / max(raw_scores.size, 1) * 100,
        }

        return raw_scores, metadata

    def get_default_params(self) -> FactorParams:
        """Return trading intensity-specific default parameters."""
        return FactorParams(
            lookback_days=252,
            lag_days=0,
            winsorize_pct=2.0,
            custom={
                "relative_volume_baseline_days": 252,
                "volume_trend_short_days": 21,
                "volume_trend_long_days": 126,
            },
        )


class VolumeBreakoutSignal(FactorSignal):
    """
    Volume Breakout: identifies stocks with unusually high recent volume.

    High volume breakouts often precede or accompany price moves.
    """

    @property
    def name(self) -> str:
        return "volume_breakout"

    @property
    def description(self) -> str:
        return (
            "Volume breakout signal. Identifies stocks where recent volume "
            "significantly exceeds historical norms. Often accompanies or "
            "precedes significant price moves."
        )

    @property
    def category(self) -> str:
        return "trading_intensity"

    @property
    def higher_is_better(self) -> bool:
        return True

    def compute_raw_signal(
        self,
        data: FactorData,
        params: FactorParams,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Compute volume breakout signal."""
        dates = data.dates
        tickers = data.tickers

        if data.volume is None:
            return pd.DataFrame(np.nan, index=dates, columns=tickers), {"warning": "No volume"}

        volume = data.volume.reindex(index=dates, columns=tickers)

        # Compare last 5 days average to 63-day average
        short_vol = volume.rolling(5, min_periods=3).mean()
        long_vol = volume.rolling(63, min_periods=21).mean()
        long_std = volume.rolling(63, min_periods=21).std()

        # Z-score of recent volume relative to history
        vol_zscore = (short_vol - long_vol) / long_std.replace(0, np.nan)
        vol_zscore = vol_zscore.replace([np.inf, -np.inf], np.nan)

        metadata = {
            "short_window": 5,
            "long_window": 63,
        }

        return vol_zscore, metadata


# =============================================================================
# POTENTIAL IMPROVEMENTS
# =============================================================================
"""
1. Abnormal Volume Events:
   - Binary indicator for volume > 2x average
   - Event-based signal

2. Volume-Price Divergence:
   - Compare volume trend to price trend
   - Divergence may signal reversals

3. Block Trade Detection:
   - Identify large institutional trades
   - Big blocks indicate informed trading

4. Intraday Volume Distribution:
   - Volume concentration in opening/closing
   - Unusual patterns may indicate algo activity

5. Options Volume:
   - Unusual options activity
   - Often leads equity moves

6. Dark Pool Volume:
   - Off-exchange trading activity
   - Indicates institutional interest
"""
