"""
Asset Growth Effect Configuration

Low asset growth = Conservative management = Better performance
"""

SIGNAL_CONFIG = {
    'name': 'asset_growth',
    'enabled': True,
    'rebalance_frequency': 'quarterly',  # Asset growth is calculated from quarterly data
    'timeline': {
        'start_date': '2017-01-01',  # Fundamental signals start from 2017
        'end_date': '2026-12-31',
    },
    'description': 'Asset growth effect - companies with low asset growth (conservative management) outperform',

    # Portfolio engineering options
    'portfolio_options': {
        # Regime filter
        'use_regime_filter': True,

        # Volatility targeting
        'use_vol_targeting': False,
        'target_downside_vol': 0.20,
        'vol_lookback': 63,
        'vol_floor': 0.10,
        'vol_cap': 1.0,

        # Inverse volatility position sizing
        'use_inverse_vol_sizing': False,
        'inverse_vol_lookback': 60,
        'max_position_weight': 0.25,

        # Position stop loss
        'use_stop_loss': False,
        'stop_loss_threshold': 0.15,

        # Liquidity filter
        'use_liquidity_filter': True,
        'liquidity_quantile': 0.25,

        # Transaction costs
        'use_slippage': True,
        'slippage_bps': 5.0,

        # Portfolio size
        'top_n': 20,
    },
}
