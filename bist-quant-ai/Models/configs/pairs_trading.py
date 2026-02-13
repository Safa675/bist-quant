"""
Pairs Trading Factor Configuration

Statistical arbitrage based on price cointegration.
Trade mean reversion when correlated pairs diverge.

Note: This signal works best for hedged/market-neutral portfolios.
For long-only, the signal identifies relative value opportunities.
"""

SIGNAL_CONFIG = {
    'name': 'pairs_trading',
    'enabled': True,
    'rebalance_frequency': 'weekly',  # More frequent for mean reversion
    'timeline': {
        'start_date': '2014-01-01',
        'end_date': '2026-12-31',
    },
    'description': 'Pairs trading: mean reversion on correlated stock pairs',

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
        'slippage_bps': 10.0,  # Higher for frequent trading
        'top_n': 20,
    },
}
