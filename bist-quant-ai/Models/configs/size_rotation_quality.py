"""
Size Rotation Quality Factor Configuration

The most sophisticated size rotation signal combining:
1. Size regime detection (small vs large cap leadership)
2. Momentum (6-month price return)
3. Profitability filter (ROA, ROE, Operating Margin)

Only stocks that pass BOTH momentum AND profitability thresholds get selected.

This signal should:
- Adapt to market regime (large vs small cap)
- Capture momentum (trend following)
- Avoid junk rallies (quality filter)
"""

SIGNAL_CONFIG = {
    'name': 'size_rotation_quality',
    'enabled': True,
    'rebalance_frequency': 'monthly',
    'timeline': {
        'start_date': '2017-01-01',  # Needs fundamental data for profitability
        'end_date': '2026-12-31',
    },
    'description': 'Size rotation + momentum + profitability (most sophisticated)',

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
