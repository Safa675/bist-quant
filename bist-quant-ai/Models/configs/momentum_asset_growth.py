"""
Momentum + Asset Growth Factor Configuration

Momentum within high-growth stocks universe.
Exploits the interaction between growth and momentum anomalies.
"""

SIGNAL_CONFIG = {
    'name': 'momentum_asset_growth',
    'enabled': True,
    'rebalance_frequency': 'monthly',
    'timeline': {
        'start_date': '2018-01-01',  # Needs fundamental data
        'end_date': '2026-12-31',
    },
    'description': 'Momentum + Asset Growth: momentum winners within high-growth stocks',

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
