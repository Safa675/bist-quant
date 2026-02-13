"""
Residual Momentum Factor Configuration

Market-adjusted momentum that captures stock-specific performance.
"""

SIGNAL_CONFIG = {
    'name': 'residual_momentum',
    'enabled': True,
    'rebalance_frequency': 'monthly',
    'timeline': {
        'start_date': '2014-01-01',
        'end_date': '2026-12-31',
    },
    'description': 'Residual momentum: market-adjusted stock-specific momentum',

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
