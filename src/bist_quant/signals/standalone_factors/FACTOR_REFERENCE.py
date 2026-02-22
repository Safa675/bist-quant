"""
=============================================================================
STANDALONE FACTOR SIGNALS - REFERENCE DOCUMENTATION
=============================================================================

This module provides 13 independent factor signals for quantitative equity
analysis. Each factor is:
- Self-contained and independently testable
- Uses a consistent interface (FactorSignal base class)
- Produces cross-sectional scores compatible with rotation frameworks

=============================================================================
FACTOR SUMMARY TABLE
=============================================================================

| # | Factor              | Category         | Higher Score Means | Key Inputs                |
|---|---------------------|------------------|--------------------|-----------------------------|
| 1 | Size                | size             | Smaller company    | Market cap                  |
| 2 | Value               | value            | Cheaper valuation  | E/P, FCF/P, EBITDA/EV       |
| 3 | Profitability       | profitability    | Higher margins     | Op Margin, Gross Margin     |
| 4 | Investment          | investment       | More reinvestment  | R&D, Payout, Efficiency     |
| 5 | Momentum            | momentum         | Stronger returns   | 6-12 month return           |
| 6 | Beta                | risk             | Higher beta        | Market covariance           |
| 7 | Quality             | quality          | Higher quality     | ROE, ROA, Accruals, F-score |
| 8 | Liquidity           | liquidity        | More liquid        | Amihud, Turnover            |
| 9 | Trading Intensity   | trading_intensity| More activity      | Rel Vol, Vol Trend          |
|10 | Sentiment           | sentiment        | Bullish action     | 52w High, Acceleration      |
|11 | Fund. Momentum      | fundamental_momentum | Improving      | Margin Δ, Sales Accel       |
|12 | Carry               | carry            | Higher yield       | Dividend Yield              |
|13 | Defensive           | defensive        | More stable        | Earnings Stability, Low β   |

=============================================================================
USAGE EXAMPLE
=============================================================================

```python
from bist_quant.signals.standalone_factors import (
    FactorData,
    FactorParams,
    MomentumSignal,
    ValueSignal,
    QualitySignal,
    NormalizationMethod,
    SelectionMethod,
)

# Prepare data
data = FactorData(
    close=close_df,
    volume=volume_df,
    shares_outstanding=shares_df,
    data_loader=data_loader,
)

# Compute momentum signal
momentum = MomentumSignal()
params = FactorParams(
    lookback_days=126,
    normalization=NormalizationMethod.ZSCORE,
    winsorize_pct=1.0,
)
output = momentum.compute_signal(data, params)

# Get normalized scores
scores = output.scores  # DataFrame (dates x tickers)

# Select top 20 stocks
selection = momentum.select_stocks(
    scores,
    method=SelectionMethod.TOP_N,
    n=20,
)
```

=============================================================================
FACTOR DETAILS
=============================================================================

1. SIZE SIGNAL (size_signal.py)
-------------------------------
Economic Intuition:
  The size premium (SMB) suggests smaller companies outperform larger ones
  due to higher risk premium, information asymmetry, and growth potential.

Mathematical Construction:
  Size Score = -log(Market Cap)
  Negative sign inverts so smaller = higher score.

Key Parameters:
  - log_transform: bool (default True)

Potential Improvements:
  - Liquidity-adjusted size (size / volume percentile)
  - Sector-neutral size scoring
  - Size-momentum interaction


2. VALUE SIGNAL (value_signal.py)
---------------------------------
Economic Intuition:
  Value investing buys "cheap" stocks expecting mean reversion to fair value.
  Based on Fama-French HML factor.

Mathematical Construction:
  Value = mean(z(E/P), z(FCF/P), z(EBITDA/EV), z(S/P))
  Each ratio z-scored cross-sectionally before averaging.

Key Parameters:
  - enabled_metrics: list of ratios to include
  - metric_weights: dict of ratio weights
  - reporting_lag_days: int (default 45)

Potential Improvements:
  - Quality-adjusted value (cheap + profitable)
  - Value momentum (improving valuation)
  - Sector-relative value


3. PROFITABILITY SIGNAL (profitability_signal.py)
-------------------------------------------------
Economic Intuition:
  Profitable firms have competitive advantages and reinvestment capacity.
  Based on Fama-French RMW factor.

Mathematical Construction:
  Profitability = 0.6 × z(Op Margin) + 0.4 × z(Gross Margin)

Key Parameters:
  - operating_margin_weight: float
  - gross_margin_weight: float

Potential Improvements:
  - Cash flow based profitability (CFO/Assets)
  - DuPont decomposition
  - Margin stability adjustment


4. INVESTMENT SIGNAL (investment_signal.py)
-------------------------------------------
Economic Intuition:
  Captures reinvestment vs conservative capital allocation.
  Based on Fama-French CMA factor.

Mathematical Construction:
  Investment = mean(z(R&D Intensity), z(Payout), z(Investment Efficiency))

Key Parameters:
  - reporting_lag_days: int

Potential Improvements:
  - Asset growth factor (simpler)
  - CapEx intensity
  - Share issuance tracking


5. MOMENTUM SIGNAL (momentum_signal.py)
---------------------------------------
Economic Intuition:
  Price momentum persists due to underreaction to information and herding.
  Based on Jegadeesh-Titman momentum.

Mathematical Construction:
  Momentum = Price(t-skip) / Price(t-lookback-skip) - 1
  Default: 126-day lookback, 21-day skip

Key Parameters:
  - lookback_days: int (default 126)
  - skip_days: int (default 21)
  - volatility_adjust: bool

Alternative Constructions:
  - Risk-adjusted momentum (Sharpe momentum)
  - Multi-horizon ensemble (1mo, 3mo, 6mo, 12mo)
  - 52-week high momentum


6. BETA SIGNAL (beta_signal.py)
-------------------------------
Economic Intuition:
  Market beta measures cyclicality and market sensitivity.
  Used for rotating between aggressive and defensive exposures.

Mathematical Construction:
  Beta = Cov(r_stock, r_market) / Var(r_market)
  Rolling 252-day window.

Key Parameters:
  - lookback_days: int (default 252)
  - market_proxy: "equal_weight" or "cap_weight"
  - clip_beta: tuple

Alternative Constructions:
  - Downside beta (only down days)
  - Fundamental beta (from accounting metrics)


7. QUALITY SIGNAL (quality_signal.py)
-------------------------------------
Economic Intuition:
  High-quality companies have sustainable advantages and reliable earnings.

Mathematical Construction:
  Quality = mean(z(ROE), z(ROA), -z(Accruals), z(Piotroski))
  Accruals inverted (low = high quality).

Key Parameters:
  - enabled_components: list
  - component_weights: dict

Potential Improvements:
  - Gross profitability (Novy-Marx)
  - Earnings persistence (AR coefficient)
  - Beneish M-score (manipulation detection)


8. LIQUIDITY SIGNAL (liquidity_signal.py)
-----------------------------------------
Economic Intuition:
  Liquidity measures ease of trading without price impact.
  Distinct from trading intensity.

Mathematical Construction:
  Liquidity = mean(-z(Amihud), z(Turnover))
  Amihud inverted (low illiquidity = good).

Key Parameters:
  - amihud_lookback_days: int (default 21)
  - turnover_lookback_days: int (default 63)

Potential Improvements:
  - Bid-ask spread (if available)
  - Price impact (Kyle's lambda)
  - Liquidity beta


9. TRADING INTENSITY SIGNAL (trading_intensity_signal.py)
---------------------------------------------------------
Economic Intuition:
  Measures level of market attention and activity.
  High intensity may indicate institutional interest or information events.

Mathematical Construction:
  Trading Intensity = mean(z(RelVol), z(VolTrend), z(TurnoverVelocity))

Key Parameters:
  - relative_volume_baseline_days: int
  - volume_trend windows: (short, long)

Potential Improvements:
  - Volume-price divergence
  - Abnormal volume events
  - Options volume integration


10. SENTIMENT SIGNAL (sentiment_signal.py)
------------------------------------------
Economic Intuition:
  Technical patterns capture behavioral biases and market psychology.

Mathematical Construction:
  Sentiment = mean(z(52wHighPct), z(Acceleration), z(Reversal))

Key Parameters:
  - high_lookback_days: int (default 252)
  - acceleration_fast_days, acceleration_slow_days
  - reversal_lookback_days: int (default 5)

Potential Improvements:
  - RSI integration
  - Bollinger Band position
  - Volume-confirmed breakouts


11. FUNDAMENTAL MOMENTUM SIGNAL (fundamental_momentum_signal.py)
----------------------------------------------------------------
Economic Intuition:
  Improving fundamentals often lead price as markets slowly incorporate changes.

Mathematical Construction:
  FundMom = mean(z(MarginChange), z(SalesAcceleration))
  YoY changes in operating margin and revenue growth rate.

Key Parameters:
  - margin_change_quarters: int (default 4)
  - sales_accel_quarters: int (default 4)

Potential Improvements:
  - Earnings surprise (if estimate data available)
  - Analyst revision momentum
  - SUE (Standardized Unexpected Earnings)


12. CARRY SIGNAL (carry_signal.py)
----------------------------------
Economic Intuition:
  Dividend yield provides expected return assuming no price change.
  High yield often indicates undervaluation and management confidence.

Mathematical Construction:
  Carry = Dividend Yield = Dividends TTM / Market Cap

Key Parameters:
  - clip_yield_upper: float (default 0.50)
  - clip_yield_lower: float (default 0.0)

Potential Improvements:
  - Shareholder yield (dividends + buybacks)
  - Net payout yield (accounting for issuance)
  - Dividend growth


13. DEFENSIVE SIGNAL (defensive_signal.py)
------------------------------------------
Economic Intuition:
  Defensive stocks provide downside protection with stable fundamentals.
  Low-volatility anomaly suggests risk-adjusted outperformance.

Mathematical Construction:
  Defensive = mean(z(EarningsStability), -z(Beta))
  Earnings Stability = 1 / CV(earnings)
  Beta inverted (low = defensive).

Key Parameters:
  - earnings_stability_quarters: int (default 8)
  - beta_lookback_days: int (default 252)

Potential Improvements:
  - Downside volatility
  - Maximum drawdown
  - Sector-based defensiveness

=============================================================================
COMBINING FACTORS FOR ROTATION
=============================================================================

Factors can be combined in several ways:

1. EQUAL WEIGHT:
   combined = (factor1 + factor2 + ... + factorN) / N

2. RANK-BASED WEIGHTING:
   Weight factors by their recent spread performance.

3. REGIME CONDITIONING:
   Adjust weights based on market regime (volatility, trend).

4. ORTHOGONALIZATION:
   Remove correlations between factors for pure exposures.

Example (equal weight combination):
```python
factors = [momentum, value, quality]
scores_list = [f.compute_signal(data).scores for f in factors]
combined = sum(scores_list) / len(scores_list)
```

=============================================================================
TESTING AND VALIDATION
=============================================================================

Each factor should be tested for:

1. Coverage - % of universe with valid scores
2. Spread Return - Q5 return - Q1 return
3. Information Coefficient - Correlation with forward returns
4. Turnover - % of portfolio that changes each period
5. Drawdowns - Maximum peak-to-trough decline of factor

Example backtest structure:
```python
for factor in all_factors:
    output = factor.compute_signal(data)
    quintiles = factor.select_stocks(output.scores, method=SelectionMethod.QUINTILE)

    # Compute quintile returns
    q5_ret = returns[quintiles == 5].mean(axis=1)
    q1_ret = returns[quintiles == 1].mean(axis=1)
    spread = q5_ret - q1_ret

    print(f"{factor.name}: Spread Sharpe = {spread.mean() / spread.std() * np.sqrt(252):.2f}")
```

=============================================================================
"""

# This file is for documentation only - no executable code
if __name__ == "__main__":
    print("This file contains documentation only. Import from standalone_factors instead.")
