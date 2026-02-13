"""
Five-Factor Rotation Configuration

Rotates between factor sides based on trailing side performance:
- Size: Small vs Big (from small_cap composite)
- Value Style: Value Level vs Value Growth (from value ratio composite)
- Profitability Style: Margin Level vs Margin Growth
- Investment Style: Conservative vs Reinvestment (from investment composite)
- Momentum: Winner vs Loser
"""

SIGNAL_CONFIG = {
    'name': 'five_factor_rotation',
    'enabled': True,
    'rebalance_frequency': 'monthly',
    'timeline': {
        'start_date': '2017-01-01',
        'end_date': '2026-12-31',
    },
    'description': '5-axis side rotation using standalone factor fundamentals for each axis',

    # Cache for heavy axis construction (size/value/profitability/investment side panels)
    'construction_cache': {
        'enabled': True,
        'force_rebuild': False,
        # None -> defaults to data/multi_factor_axis_construction.parquet
        'path': 'data/five_factor_axis_construction.parquet',
    },

    # Optional verbose trace for step-by-step debugging.
    # Can also be enabled via env: FIVE_FACTOR_DEBUG=1
    'debug': {
        'enabled': False,
    },

    # Yearly walk-forward for MWU factor-weight training:
    # train 2017-2019 -> test 2020, ... , train 2023-2025 -> test 2026
    'walk_forward': {
        'enabled': True,
        'train_years': 3,
        'first_test_year': 2020,
        'last_test_year': 2026,
    },

    # Make axes capture distinct cross-sectional information by residualizing
    # each later axis against earlier ones date-by-date.
    'axis_orthogonalization': {
        'enabled': True,
        'min_overlap': 20,
        'epsilon': 1e-8,
        # Optional explicit priority order.
        # If omitted, uses axis definition order in the signal builder.
        # 'order': ['size', 'value', 'profitability', 'investment', 'momentum', 'risk',
        #           'quality', 'liquidity', 'trading_intensity', 'sentiment',
        #           'fundmom', 'carry', 'defensive'],
    },

    'portfolio_options': {
        # Regime overlay
        'use_regime_filter': True,

        # Vol targeting
        'use_vol_targeting': False,
        'target_downside_vol': 0.20,
        'vol_lookback': 63,
        'vol_floor': 0.10,
        'vol_cap': 1.0,

        # Position sizing
        'use_inverse_vol_sizing': False,
        'inverse_vol_lookback': 60,
        'max_position_weight': 0.20,

        # Risk controls
        'use_stop_loss': True,
        'stop_loss_threshold': 0.15,

        # Liquidity / costs
        'use_liquidity_filter': True,
        'liquidity_quantile': 0.25,
        'use_slippage': True,
        'slippage_bps': 5.0,

        # Portfolio breadth
        'top_n': 25,
    },
}
