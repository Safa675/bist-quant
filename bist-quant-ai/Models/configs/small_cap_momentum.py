"""
Small Cap Momentum Signal Configuration

Combines small market cap with momentum to identify early-stage breakouts.
Captures "discovery" plays - small caps that are starting to trend.

Note: Uses fundamental data for market cap, so starts at 2017.
"""

SIGNAL_CONFIG = {
    'name': 'small_cap_momentum',
    'enabled': True,
    'rebalance_frequency': 'monthly',
    'timeline': {
        'start_date': '2017-01-01',  # Needs market cap data (fundamental)
        'end_date': '2026-12-31',
    },
    'description': 'Small cap + Momentum composite - captures discovery plays',

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
        'slippage_bps': 8.0,  # Higher slippage for small caps

        # Portfolio size
        'top_n': 15,  # Fewer positions due to higher risk
    },
}
