"""
Size Rotation Momentum Factor Configuration

Combines size rotation with pure momentum.

When small caps are outperforming -> Only buy highest momentum small caps
When large caps are outperforming -> Only buy highest momentum large caps

This is more aggressive than base size_rotation:
- Base size_rotation: Tilts towards winning segment
- This signal: ONLY buys from winning segment
"""

SIGNAL_CONFIG = {
    'name': 'size_rotation_momentum',
    'enabled': True,
    'rebalance_frequency': 'monthly',
    'timeline': {
        'start_date': '2014-01-01',  # Technical signal
        'end_date': '2026-12-31',
    },
    'description': 'Pure momentum within winning size segment (small vs large caps)',

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

        # Position stop loss - enable for protection
        'use_stop_loss': True,
        'stop_loss_threshold': 0.15,

        # Liquidity filter
        'use_liquidity_filter': True,
        'liquidity_quantile': 0.25,

        # Transaction costs
        'use_slippage': True,
        'slippage_bps': 5.0,
        'small_cap_slippage_bps': 20.0,

        # Portfolio size
        'top_n': 20,
    },
}
