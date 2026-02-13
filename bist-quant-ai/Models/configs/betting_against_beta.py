"""
Betting Against Beta Configuration

Low beta stocks outperform high beta stocks on a risk-adjusted basis
"""

SIGNAL_CONFIG = {
    'name': 'betting_against_beta',
    'enabled': True,
    'rebalance_frequency': 'monthly',  # Beta changes gradually, monthly rebalancing is sufficient
    'timeline': {
        'start_date': '2014-01-01',  # Price-based signal, can start earlier
        'end_date': '2026-12-31',
    },
    'description': 'Betting against beta - low beta stocks outperform high beta stocks',

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
