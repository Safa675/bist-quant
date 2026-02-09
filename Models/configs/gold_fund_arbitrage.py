"""
Gold Fund Arbitrage Configuration

Mean-reversion strategy on Turkish gold funds (TEFAS) versus XAU/TRY.
"""

SIGNAL_CONFIG = {
    'name': 'gold_fund_arbitrage',
    'enabled': True,
    'rebalance_frequency': 'daily',
    'timeline': {
        'start_date': '2021-02-11',
        'end_date': '2026-12-31',
    },
    'description': 'Pair-trade gold funds vs XAU/TRY on spread z-score mean reversion',

    # Signal construction options
    'signal_params': {
        'fund_prices_file': 'gold_funds_daily_prices.csv',
        'xau_try_file': 'xau_try_2013_2026.csv',
        'lookback': 126,
        'min_periods': 63,
        'entry_z': 2.0,
        'exit_z': 0.5,
        'gold_lag_days': 1,
        'top_tracking_count': 12,
        'min_obs_metrics': 60,
    },

    # Portfolio/backtest options
    'portfolio_options': {
        'use_regime_filter': False,
        'use_vol_targeting': False,
        'use_inverse_vol_sizing': False,
        'use_stop_loss': False,
        'use_liquidity_filter': False,

        # Used by TEFAS custom backtest path
        'use_slippage': True,
        'slippage_bps': 5.0,
        'tefas_transaction_cost_bps': 5.0,
        'tefas_execution_lag_days': 1,
        # Pair-trade default: stay flat (cash) when no dislocation.
        'hold_xau_when_flat': False,
        'pair_use_beta_hedge': True,
        'pair_beta_floor': 0.1,
        'pair_beta_cap': 2.0,
        # Gross exposure across fund legs + gold hedge.
        'pair_target_gross_exposure': 1.0,

        # Not used directly in this strategy, kept for consistency
        'top_n': 20,
    },
}
