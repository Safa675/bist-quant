"""
Size Rotation Factor Configuration

Dynamically switches between small caps and large caps based on relative performance.

When small caps are outperforming -> Tilt towards small cap momentum
When large caps are outperforming -> Tilt towards large cap momentum

This is an adaptive strategy that doesn't fight the tape:
- 2024-2025: Small caps leading -> Small cap tilt
- 2026: Large caps leading -> Large cap tilt (avoids the small cap crash)
"""

SIGNAL_CONFIG = {
    'name': 'size_rotation',
    'enabled': True,
    'rebalance_frequency': 'monthly',
    'timeline': {
        'start_date': '2014-01-01',  # Technical signal, price data starts 2013
        'end_date': '2026-12-31',
    },
    'description': 'Adaptive size rotation - tilts to winning size segment (small vs large caps)',

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
        'use_stop_loss': True,  # Enable stop loss for protection
        'stop_loss_threshold': 0.15,

        # Liquidity filter - removes bottom quartile by volume
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
