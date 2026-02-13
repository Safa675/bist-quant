"""
Supertrend Direction Signal Configuration

Supertrend(10, 3.0) trend-following strategy:
- Buy stocks in uptrend (+1 direction)
- Avoid stocks in downtrend (-1 direction)
"""

SIGNAL_CONFIG = {
    'name': 'supertrend',
    'enabled': True,
    'rebalance_frequency': 'monthly',
    'timeline': {
        'start_date': '2014-01-01',
        'end_date': '2026-12-31',
    },
    'description': 'Supertrend(10,3) direction - buy uptrend, avoid downtrend',
    'parameters': {
        'period': 10,
        'multiplier': 3.0,
    },

    'portfolio_options': {
        'use_regime_filter': True,
        'use_vol_targeting': False,
        'target_downside_vol': 0.20,
        'vol_lookback': 63,
        'vol_floor': 0.10,
        'vol_cap': 1.0,
        'use_inverse_vol_sizing': False,
        'inverse_vol_lookback': 60,
        'max_position_weight': 0.25,
        'use_stop_loss': False,
        'stop_loss_threshold': 0.15,
        'use_liquidity_filter': True,
        'liquidity_quantile': 0.25,
        'use_slippage': True,
        'slippage_bps': 5.0,
        'top_n': 20,
    },
}
