"""
Short-Term Reversal Factor Configuration

Weekly reversal anomaly: losers outperform winners in the short term.
"""

SIGNAL_CONFIG = {
    'name': 'short_term_reversal',
    'enabled': True,
    'rebalance_frequency': 'weekly',  # Weekly rebalancing for short-term reversal
    'timeline': {
        'start_date': '2014-01-01',
        'end_date': '2026-12-31',
    },
    'description': 'Short-term reversal: long weekly losers, avoid weekly winners',

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
        'slippage_bps': 10.0,  # Higher slippage for weekly rebalancing
        'top_n': 20,
    },
}
