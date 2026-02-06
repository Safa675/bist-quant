"""
Currency Rotation Factor Configuration

Exploits USD/TRY mean reversion to identify sector rotation opportunities.
"""

SIGNAL_CONFIG = {
    'name': 'currency_rotation',
    'enabled': True,
    'rebalance_frequency': 'monthly',
    'timeline': {
        'start_date': '2014-01-01',  # Technical signals start from 2014
        'end_date': '2026-12-31',    # Today
    },
    'description': 'USD/TRY mean reversion with sector rotation (exporters vs domestic)',

    # Portfolio engineering options
    'portfolio_options': {
        # Regime filter - switches to gold in Bear/Stress regimes
        'use_regime_filter': True,  # Enable regime filter for macro-aware signal

        # Volatility targeting - scales leverage to target constant vol
        'use_vol_targeting': False,
        'target_downside_vol': 0.20,
        'vol_lookback': 63,
        'vol_floor': 0.10,
        'vol_cap': 1.0,

        # Inverse volatility position sizing - weights positions by inverse downside vol
        'use_inverse_vol_sizing': True,  # Enable for better risk management
        'inverse_vol_lookback': 60,
        'max_position_weight': 0.25,

        # Position stop loss
        'use_stop_loss': False,
        'stop_loss_threshold': 0.15,

        # Liquidity filter - removes bottom quartile by volume
        'use_liquidity_filter': True,
        'liquidity_quantile': 0.25,

        # Transaction costs
        'use_slippage': True,
        'slippage_bps': 5.0,

        # Portfolio size
        'top_n': 20,
    },
}
