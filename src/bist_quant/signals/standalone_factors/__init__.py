"""
Standalone Factor Signals

This module provides independent, modular implementations of 13 factor signals.
Each factor is self-contained and produces standardized output that can be:
1. Used independently for single-factor analysis
2. Combined in a rotation framework
3. Tested in isolation (ablation studies)

All factors implement the same interface:
    compute_signal(data: FactorData, params: FactorParams) -> pd.DataFrame

The output is a cross-sectional score panel (dates x tickers) with values
normalized as z-scores by default.

Factor Categories:
-----------------
ORIGINAL FACTORS (Fama-French inspired):
1. Size         - Small vs Large market cap
2. Value        - Value Level vs Value Growth
3. Profitability - Current Margin vs Margin Growth
4. Investment   - Conservative vs Reinvestment
5. Momentum     - Winners vs Losers (price momentum)
6. Beta         - High Beta vs Low Beta (market sensitivity)

NEW FACTORS:
7. Quality      - High Quality vs Low Quality (ROE, ROA, Accruals, Piotroski)
8. Liquidity    - Liquid vs Illiquid (Amihud, turnover)
9. Trading Intensity - High Activity vs Low Activity (relative volume)
10. Sentiment   - Strong Price Action vs Weak (52w high proximity)
11. Fundamental Momentum - Improving vs Deteriorating fundamentals
12. Carry       - High Yield vs Low Yield (dividend)
13. Defensive   - Stable vs Cyclical (earnings stability, low beta)
"""

from .base import (
    FactorSignal,
    FactorData,
    FactorParams,
    SignalOutput,
    NormalizationMethod,
    SelectionMethod,
)

from .size_signal import SizeSignal
from .value_signal import ValueSignal
from .profitability_signal import ProfitabilitySignal
from .investment_signal import InvestmentSignal
from .momentum_signal import MomentumSignal
from .beta_signal import BetaSignal
from .quality_signal import QualitySignal
from .liquidity_signal import LiquiditySignal
from .trading_intensity_signal import TradingIntensitySignal
from .sentiment_signal import SentimentSignal
from .fundamental_momentum_signal import FundamentalMomentumSignal
from .carry_signal import CarrySignal
from .defensive_signal import DefensiveSignal

__all__ = [
    # Base classes
    "FactorSignal",
    "FactorData",
    "FactorParams",
    "SignalOutput",
    "NormalizationMethod",
    "SelectionMethod",
    # Factor signals
    "SizeSignal",
    "ValueSignal",
    "ProfitabilitySignal",
    "InvestmentSignal",
    "MomentumSignal",
    "BetaSignal",
    "QualitySignal",
    "LiquiditySignal",
    "TradingIntensitySignal",
    "SentimentSignal",
    "FundamentalMomentumSignal",
    "CarrySignal",
    "DefensiveSignal",
]
