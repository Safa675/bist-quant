"""
XU100 Benchmark Configuration

This configuration runs XU100 through the portfolio engine to compare
regime-aware XU100 vs buy-and-hold XU100.

The portfolio engine will:
- Follow regime signals (exit to gold in Bear/Stress regimes)
- Apply volatility targeting
- Rebalance monthly

This shows how much value the regime filter adds vs passive holding.
"""

SIGNAL_CONFIG = {
    'name': 'xu100',
    'enabled': True,
    'rebalance_frequency': 'monthly',
    'timeline': {
        'start_date': '2014-01-01',  # Match momentum timeline for comparison
        'end_date': '2026-12-31',
    },
    'description': 'XU100 index with regime awareness - benchmark for risk management value',

    # Portfolio engineering options - set all to False for raw signal performance
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
