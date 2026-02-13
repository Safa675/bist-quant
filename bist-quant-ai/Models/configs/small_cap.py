"""
Small Cap Factor Configuration

Favors smaller market cap stocks (the "size premium").
Smaller companies tend to outperform larger ones over long periods,
but with higher volatility and drawdowns.

Note: This is a pure size factor - for small cap + momentum, use small_cap_momentum.
"""

SIGNAL_CONFIG = {
    'name': 'small_cap',
    'enabled': True,
    'rebalance_frequency': 'quarterly',
    'timeline': {
        'start_date': '2017-01-01',  # Fundamental signals start from 2017
        'end_date': '2026-12-31',
    },
    'description': 'Pure small cap factor - favors smaller market cap stocks',

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
