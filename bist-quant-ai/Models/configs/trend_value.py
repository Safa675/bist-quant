"""
Trend + Value Composite Signal Configuration

Combines SMA trend filter with Value fundamentals.
Only buys value stocks that are in uptrends.

Rationale:
- Value alone can be a "value trap" (cheap stocks getting cheaper)
- Adding trend filter ensures we only buy when price confirms
- Should reduce drawdowns while capturing value premium
"""

SIGNAL_CONFIG = {
    'name': 'trend_value',
    'enabled': True,
    'rebalance_frequency': 'monthly',  # Monthly to capture trend changes
    'timeline': {
        'start_date': '2017-01-01',  # Needs fundamental data
        'end_date': '2026-12-31',
    },
    'description': 'Value stocks filtered by SMA uptrend - avoids value traps',

    # Portfolio engineering options
    'portfolio_options': {
        # Regime filter - switches to gold in Bear/Stress regimes
        'use_regime_filter': True,

        # Volatility targeting - scales leverage to target constant vol
        'use_vol_targeting': False,
        'target_downside_vol': 0.20,
        'vol_lookback': 63,
        'vol_floor': 0.10,
        'vol_cap': 1.0,

        # Inverse volatility position sizing - weights positions by inverse downside vol
        'use_inverse_vol_sizing': False,
        'inverse_vol_lookback': 60,
        'max_position_weight': 0.25,

        # Position stop loss
        'use_stop_loss': False,
        'stop_loss_threshold': 0.15,

        # Liquidity filter - removes bottom quartile by volume
        'use_liquidity_filter': True,
        'liquidity_quantile': 0.25,

        # Transaction costs
        'use_slippage': True,
        'slippage_bps': 5.0,

        # Portfolio size
        'top_n': 20,
    },
}
