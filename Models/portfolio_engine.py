#!/usr/bin/env python3
"""
Config-Based Portfolio Engine

Orchestrates all factor models with:
- Centralized data loading (load once, use multiple times)
- Config-based signal integration
- Comprehensive reporting

Usage:
    python portfolio_engine.py --factor profitability
    python portfolio_engine.py --factor momentum
    python portfolio_engine.py --factor all
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import sys
import time
import warnings
import importlib.util
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from common.data_loader import DataLoader
from signals.profitability_signals import build_profitability_signals
from signals.value_signals import build_value_signals
from signals.small_cap_signals import build_small_cap_signals
from signals.investment_signals import build_investment_signals
from signals.momentum_signals import build_momentum_signals
from signals.sma_signals import build_sma_signals
from signals.donchian_signals import build_donchian_signals
from signals.xu100_signals import build_xu100_signals
from signals.trend_value_signals import build_trend_value_signals
from signals.breakout_value_signals import build_breakout_value_signals
from signals.currency_rotation_signals import build_currency_rotation_signals
from signals.dividend_rotation_signals import build_dividend_rotation_signals
from signals.macro_hedge_signals import build_macro_hedge_signals
from signals.quality_momentum_signals import build_quality_momentum_signals
from signals.quality_value_signals import build_quality_value_signals
from signals.small_cap_momentum_signals import build_small_cap_momentum_signals
from signals.size_rotation_signals import build_size_rotation_signals
from signals.size_rotation_momentum_signals import build_size_rotation_momentum_signals
from signals.size_rotation_quality_signals import build_size_rotation_quality_signals



# ============================================================================
# CONFIGURATION
# ============================================================================

# ============================================================================
# DEFAULT CONFIGURATION VALUES
# These can be overridden per-signal via config files
# ============================================================================

REGIME_ALLOCATIONS = {
    'Bull': 1.0,
    'Recovery': 1.0,
    'Choppy': 0.5,
    'Stress': 0.0,
    'Bear': 0.0
}

# Default portfolio options (can be overridden in config files)
DEFAULT_PORTFOLIO_OPTIONS = {
    # Regime filter - switches to gold in Bear/Stress regimes
    'use_regime_filter': True,

    # Volatility targeting - scales leverage to target constant vol
    'use_vol_targeting': True,
    'target_downside_vol': 0.20,
    'vol_lookback': 63,
    'vol_floor': 0.10,
    'vol_cap': 1.0,

    # Inverse volatility position sizing - weights positions by inverse downside vol
    'use_inverse_vol_sizing': True,
    'inverse_vol_lookback': 60,
    'max_position_weight': 0.25,

    # Position stop loss
    'use_stop_loss': True,
    'stop_loss_threshold': 0.15,

    # Liquidity filter - removes bottom quartile by volume
    'use_liquidity_filter': True,
    'liquidity_quantile': 0.25,

    # Transaction costs
    'use_slippage': True,
    'slippage_bps': 5.0,

    # Portfolio size
    'top_n': 20,
}

# Legacy constants (for backward compatibility)
TOP_N = 20
LIQUIDITY_QUANTILE = 0.25
POSITION_STOP_LOSS = 0.15
SLIPPAGE_BPS = 5.0

# Volatility targeting parameters
TARGET_DOWNSIDE_VOL = 0.20
VOL_LOOKBACK = 63
VOL_FLOOR = 0.10
VOL_CAP = 1.0

# Inverse volatility weighting parameters
INVERSE_VOL_LOOKBACK = 60
MAX_POSITION_WEIGHT = 0.25


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def identify_monthly_rebalance_days(trading_days: pd.DatetimeIndex) -> set:
    """
    Identify monthly rebalancing days: first trading day of each month.
    
    Monthly rebalancing is optimal for momentum strategies since the signal
    doesn't change significantly week-to-week.
    
    Returns:
        set: Set of pd.Timestamp representing rebalance days
    """
    df = pd.DataFrame({'date': trading_days})
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # First trading day of each month
    first_of_month = df.groupby(['year', 'month'])['date'].first()
    return set(first_of_month.values)


def inverse_downside_vol_weights(close_df, selected, date, lookback=INVERSE_VOL_LOOKBACK, max_weight=MAX_POSITION_WEIGHT):
    """
    Compute inverse-downside-volatility weights for the selected tickers.
    
    Allocates more capital to lower-risk stocks, improving risk-adjusted returns.
    
    Args:
        close_df: DataFrame of close prices
        selected: List of selected tickers
        date: Current date
        lookback: Days to look back for volatility calculation
        max_weight: Maximum weight per position (prevents concentration)
    
    Returns:
        pd.Series: Weights for each ticker (sum to 1.0)
    """
    if date not in close_df.index:
        return pd.Series(1.0 / len(selected), index=selected)
    
    idx = close_df.index.get_loc(date)
    if idx < lookback:
        return pd.Series(1.0 / len(selected), index=selected)
    
    # Exclude current day (idx) since at rebalance time we don't have today's close yet
    window_data = close_df.iloc[idx-lookback:idx][selected]
    returns = window_data.pct_change(fill_method=None).dropna()
    
    downside_vols = []
    for ticker in selected:
        if ticker in returns.columns:
            ticker_rets = returns[ticker].dropna()
            downside_rets = ticker_rets[ticker_rets < 0]
            if len(downside_rets) > 2:
                downside_vol = downside_rets.std()
            else:
                downside_vol = np.nan
        else:
            downside_vol = np.nan
        downside_vols.append(downside_vol)
    
    downside_vol_series = pd.Series(downside_vols, index=selected)
    
    # Inverse weighting: lower vol = higher weight
    inv = 1.0 / downside_vol_series.replace(0, np.nan)
    median_inv = inv.median()
    if pd.isna(median_inv) or median_inv == 0:
        return pd.Series(1.0 / len(selected), index=selected)
    
    inv = inv.fillna(median_inv)
    weights = inv / inv.sum()
    
    # Cap at max_weight per position
    weights = weights.clip(upper=max_weight)
    weights = weights / weights.sum()  # Renormalize
    
    return weights


def apply_downside_vol_targeting(
    returns: pd.Series,
    target_vol: float = TARGET_DOWNSIDE_VOL,
    lookback: int = VOL_LOOKBACK,
    vol_floor: float = VOL_FLOOR,
    vol_cap: float = VOL_CAP,
) -> pd.Series:
    """
    Apply downside volatility targeting to scale returns.
    
    Scales position sizes to target a constant annualized downside volatility.
    When realized vol is low, increase exposure; when high, reduce it.
    
    Args:
        returns: Daily portfolio returns
        target_vol: Target annualized downside volatility (default 20%)
        lookback: Days to look back for realized vol calculation
        vol_floor: Minimum scaling factor (default 0.10 = 10% leverage min)
        vol_cap: Maximum scaling factor (default 1.0 = 100% leverage max)
    
    Returns:
        pd.Series: Volatility-targeted returns
    """
    if len(returns) < lookback:
        return returns
    
    # Calculate rolling downside volatility
    def calc_rolling_downside_vol(window):
        negative_rets = window[window < 0]
        if len(negative_rets) > 2:
            return negative_rets.std() * np.sqrt(252)  # Annualize
        return np.nan
    
    rolling_downside_vol = returns.rolling(lookback, min_periods=lookback//2).apply(
        calc_rolling_downside_vol, raw=False
    )
    
    # Calculate leverage factor: target_vol / realized_vol
    # Shift by 1 to avoid lookahead bias (use yesterday's vol for today's sizing)
    leverage = target_vol / rolling_downside_vol.shift(1)
    
    # Clip leverage to reasonable bounds
    leverage = leverage.clip(lower=vol_floor, upper=vol_cap)
    
    # Fill NaN (early period) with 1.0 (no scaling)
    leverage = leverage.fillna(1.0)
    
    # Apply leverage to returns
    targeted_returns = returns * leverage
    
    return targeted_returns


def compute_yearly_metrics(returns, benchmark_returns=None, xautry_returns=None):
    """
    Compute yearly Sharpe, Sortino, return, and excess return.
    
    Args:
        returns: Daily returns series
        benchmark_returns: Optional benchmark (XU100) returns
        xautry_returns: Optional XAU/TRY returns
    
    Returns:
        pd.DataFrame: Yearly metrics
    """
    df = pd.DataFrame({"ret": returns})
    if benchmark_returns is not None:
        df["bench"] = benchmark_returns
    if xautry_returns is not None:
        df["xautry"] = xautry_returns

    df = df.dropna(subset=["ret"])
    df["year"] = df.index.year

    yearly_rows = []
    for year, group in df.groupby("year"):
        if group.empty:
            continue
        daily_ret = group["ret"]
        ann_return = (1.0 + daily_ret).prod() - 1.0

        mean_ret = daily_ret.mean()
        std_ret = daily_ret.std()
        sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret and std_ret > 0 else 0.0

        downside = daily_ret[daily_ret < 0]
        downside_std = downside.std()
        sortino = (mean_ret / downside_std * np.sqrt(252)) if downside_std and downside_std > 0 else 0.0

        # XU100 benchmark
        bench_return = np.nan
        if "bench" in group.columns:
            common = group.dropna(subset=["bench"])
            if not common.empty:
                bench_return = (1.0 + common["bench"]).prod() - 1.0

        # XAU/TRY benchmark
        xautry_return = np.nan
        if "xautry" in group.columns:
            common = group.dropna(subset=["xautry"])
            if not common.empty:
                xautry_return = (1.0 + common["xautry"]).prod() - 1.0

        yearly_rows.append({
            "Year": int(year),
            "Return": ann_return,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "XU100_Return": bench_return,
            "Excess_vs_XU100": ann_return - bench_return if pd.notna(bench_return) else np.nan,
            "XAUTRY_Return": xautry_return,
            "Excess_vs_XAUTRY": ann_return - xautry_return if pd.notna(xautry_return) else np.nan,
        })

    return pd.DataFrame(yearly_rows).sort_values("Year")


# ============================================================================
# CONFIG LOADING
# ============================================================================

def load_signal_configs():
    """
    Load all signal configurations from configs/ directory.
    
    Returns:
        dict: Dictionary mapping signal names to their configs
    """
    configs = {}
    config_dir = Path(__file__).parent / 'configs'
    
    if not config_dir.exists():
        print(f"⚠️  Configs directory not found: {config_dir}")
        return configs
    
    for config_file in config_dir.glob('*.py'):
        if config_file.name == '__init__.py':
            continue
        
        try:
            module_name = config_file.stem
            spec = importlib.util.spec_from_file_location(module_name, config_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if hasattr(module, 'SIGNAL_CONFIG'):
                config = module.SIGNAL_CONFIG
                configs[config['name']] = config
        except Exception as e:
            print(f"⚠️  Failed to load config {config_file.name}: {e}")
    
    return configs


# ============================================================================
# PORTFOLIO ENGINE
# ============================================================================

class PortfolioEngine:
    """Config-based portfolio engine"""
    
    def __init__(self, data_dir: Path, regime_model_dir: Path, start_date: str, end_date: str):
        self.data_dir = Path(data_dir)
        self.regime_model_dir = Path(regime_model_dir)
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        
        # Load signal configurations
        self.signal_configs = load_signal_configs()
        print(f"\n📋 Loaded {len(self.signal_configs)} signal configurations:")
        for name, config in self.signal_configs.items():
            status = "✅ Enabled" if config.get('enabled', True) else "⚠️  Disabled"
            rebal = config.get('rebalance_frequency', 'quarterly')
            print(f"   {name}: {status} ({rebal})")
        
        # Initialize data loader
        self.loader = DataLoader(data_dir, regime_model_dir)
        
        # Cached data
        self.prices = None
        self.close_df = None
        self.open_df = None
        self.volume_df = None
        self.regime_series = None
        self.xautry_prices = None
        self.xu100_prices = None
        self.fundamentals = None
        
        # Store factor returns for correlation analysis
        self.factor_returns = {}
        
    def load_all_data(self):
        """Load all data once"""
        print("\n" + "="*70)
        print("LOADING ALL DATA")
        print("="*70)
        
        start_time = time.time()
        
        # Load prices
        prices_file = self.data_dir / "bist_prices_full.csv"
        self.prices = self.loader.load_prices(prices_file)
        
        # Build panels
        self.close_df = self.loader.build_close_panel(self.prices)
        self.open_df = self.loader.build_open_panel(self.prices)
        self.volume_df = self.loader.build_volume_panel(self.prices)
        
        # Load fundamentals
        self.fundamentals = self.loader.load_fundamentals()
        
        
        # Load regime predictions
        regime_outputs_dir = Path(__file__).parent.parent / "Regime Filter" / "outputs"
        features_file = regime_outputs_dir / "regime_features.csv"
        features = pd.read_csv(features_file, index_col=0, parse_dates=True)
        self.regime_series = self.loader.load_regime_predictions(features)
        
        # Load XAU/TRY
        xautry_file = self.data_dir / "xau_try_2013_2026.csv"
        self.xautry_prices = self.loader.load_xautry_prices(xautry_file)
        
        # Load XU100
        xu100_file = self.data_dir / "xu100_prices.csv"
        self.xu100_prices = self.loader.load_xu100_prices(xu100_file)
        
        elapsed = time.time() - start_time
        print(f"\n✅ Data loading completed in {elapsed:.1f} seconds")
        
    def run_factor(self, factor_name: str, override_config: dict = None):
        """Run backtest for a single factor using its config"""
        print("\n" + "="*70)
        print(f"RUNNING {factor_name.upper()} FACTOR")
        print("="*70)
        
        # Load config
        if override_config:
            config = override_config
        else:
            config = self.signal_configs.get(factor_name)
            if not config:
                raise ValueError(f"No config found for factor: {factor_name}")
        
        # Check if enabled
        if not config.get('enabled', True):
            print(f"⚠️  {factor_name.upper()} is disabled in config")
            return None
        
        # Get rebalancing frequency from config
        rebalance_freq = config.get('rebalance_frequency', 'quarterly')
        print(f"Rebalancing frequency: {rebalance_freq}")
        
        # Get custom timeline from config (if specified)
        timeline = config.get('timeline', {})
        custom_start = timeline.get('start_date')
        custom_end = timeline.get('end_date')
        
        # Use custom dates if specified, otherwise use engine defaults
        factor_start_date = pd.Timestamp(custom_start) if custom_start else self.start_date
        factor_end_date = pd.Timestamp(custom_end) if custom_end else self.end_date
        
        # Display timeline
        if custom_start or custom_end:
            print(f"Custom timeline: {factor_start_date.date()} to {factor_end_date.date()}")
        
        start_time = time.time()
        
        # Build signals
        dates = self.close_df.index
        
        if factor_name == "profitability":
            signals = build_profitability_signals(self.fundamentals, dates, self.loader)
        elif factor_name == "value":
            signals = build_value_signals(self.fundamentals, self.close_df, dates, self.loader)
        elif factor_name == "small_cap":
            signals = build_small_cap_signals(self.fundamentals, self.close_df, self.volume_df, dates, self.loader)
        elif factor_name == "investment":
            signals = build_investment_signals(self.fundamentals, self.close_df, dates, self.loader)
        elif factor_name == "momentum":
            signals = build_momentum_signals(self.close_df, dates, self.loader)
        elif factor_name == "sma":
            # SMA uses close prices only
            signals = build_sma_signals(self.close_df, dates, self.loader)
        elif factor_name == "donchian":
            # Donchian needs high/low prices
            high_df = self.prices.pivot_table(index='Date', columns='Ticker', values='High').sort_index()
            high_df.columns = [c.split('.')[0].upper() for c in high_df.columns]
            low_df = self.prices.pivot_table(index='Date', columns='Ticker', values='Low').sort_index()
            low_df.columns = [c.split('.')[0].upper() for c in low_df.columns]
            signals = build_donchian_signals(self.close_df, high_df, low_df, dates, self.loader)
        elif factor_name == "xu100":
            # XU100 benchmark signal - need to add XU100 to close_df
            signals = build_xu100_signals(self.close_df, dates, self.loader)
            # Add XU100 prices to close_df for backtesting
            if 'XU100' not in self.close_df.columns and self.xu100_prices is not None:
                self.close_df['XU100'] = self.xu100_prices.reindex(self.close_df.index)
        elif factor_name == "trend_value":
            # Trend + Value composite: value stocks in uptrends only
            signals = build_trend_value_signals(self.close_df, dates, self.loader)
        elif factor_name == "breakout_value":
            # Breakout + Value composite: value stocks breaking out
            high_df = self.prices.pivot_table(index='Date', columns='Ticker', values='High').sort_index()
            high_df.columns = [c.split('.')[0].upper() for c in high_df.columns]
            low_df = self.prices.pivot_table(index='Date', columns='Ticker', values='Low').sort_index()
            low_df.columns = [c.split('.')[0].upper() for c in low_df.columns]
            signals = build_breakout_value_signals(self.close_df, high_df, low_df, dates, self.loader)
        elif factor_name == "currency_rotation":
            # Currency rotation: USD/TRY mean reversion with sector rotation
            signals = build_currency_rotation_signals(self.close_df, dates, self.loader)
        elif factor_name == "dividend_rotation":
            # Dividend rotation: High-quality dividend stocks during rate normalization
            signals = build_dividend_rotation_signals(self.close_df, dates, self.loader)
        elif factor_name == "macro_hedge":
            # Macro hedge: Fortress balance sheet stocks for macro protection
            signals = build_macro_hedge_signals(self.close_df, dates, self.loader)
        elif factor_name == "quality_momentum":
            # Quality Momentum: Momentum + Profitability composite
            signals = build_quality_momentum_signals(self.close_df, self.fundamentals, dates, self.loader)
        elif factor_name == "quality_value":
            # Quality Value: Value + Profitability composite
            signals = build_quality_value_signals(self.close_df, self.fundamentals, dates, self.loader)
        elif factor_name == "small_cap_momentum":
            # Small Cap Momentum: Size + Momentum composite
            signals = build_small_cap_momentum_signals(self.close_df, self.fundamentals, dates, self.loader)
        elif factor_name == "size_rotation":
            # Size Rotation: Dynamically switches between small and large caps
            signals = build_size_rotation_signals(self.close_df, dates, self.loader)
        elif factor_name == "size_rotation_momentum":
            # Size Rotation Momentum: Pure momentum within winning size segment
            signals = build_size_rotation_momentum_signals(self.close_df, dates, self.loader)
        elif factor_name == "size_rotation_quality":
            # Size Rotation Quality: Momentum + Profitability within winning size segment
            signals = build_size_rotation_quality_signals(self.close_df, self.fundamentals, dates, self.loader)
        else:
            raise ValueError(f"Unknown factor: {factor_name}")

        # Get portfolio options from config
        portfolio_options = config.get('portfolio_options', {})

        # Run backtest with custom timeline and portfolio options
        results = self._run_backtest(signals, factor_name, rebalance_freq, factor_start_date, factor_end_date, portfolio_options)
        
        # Save results
        self.save_results(results, factor_name)
        
        # Store returns for correlation analysis
        self.factor_returns[factor_name] = results['returns']
        
        elapsed = time.time() - start_time
        print(f"\n✅ {factor_name.upper()} completed in {elapsed:.1f} seconds")
        
        return results
    
    def _run_backtest(self, signals: pd.DataFrame, factor_name: str, rebalance_freq: str = 'quarterly',
                     start_date: pd.Timestamp = None, end_date: pd.Timestamp = None,
                     portfolio_options: dict = None):
        """Run backtest with regime awareness, risk management, and configurable rebalancing

        Args:
            signals: DataFrame of signals (date x ticker)
            factor_name: Name of the factor being tested
            rebalance_freq: 'monthly' or 'quarterly'
            start_date: Backtest start date
            end_date: Backtest end date
            portfolio_options: Dict of portfolio engineering toggles (from config)
        """

        # Merge default options with config-provided options
        opts = DEFAULT_PORTFOLIO_OPTIONS.copy()
        if portfolio_options:
            opts.update(portfolio_options)

        # Print active portfolio engineering features
        print(f"\n🔧 Portfolio Engineering Settings:")
        print(f"   Regime Filter: {'ON' if opts['use_regime_filter'] else 'OFF'}")
        print(f"   Vol Targeting: {'ON (' + str(int(opts['target_downside_vol']*100)) + '%)' if opts['use_vol_targeting'] else 'OFF'}")
        print(f"   Inverse Vol Sizing: {'ON' if opts['use_inverse_vol_sizing'] else 'OFF'}")
        print(f"   Stop Loss: {'ON (' + str(int(opts['stop_loss_threshold']*100)) + '%)' if opts['use_stop_loss'] else 'OFF'}")
        print(f"   Liquidity Filter: {'ON' if opts['use_liquidity_filter'] else 'OFF'}")
        print(f"   Slippage: {'ON (' + str(opts['slippage_bps']) + ' bps)' if opts['use_slippage'] else 'OFF'}")
        print(f"   Top N Stocks: {opts['top_n']}")

        # Use provided dates or fall back to engine defaults
        backtest_start = start_date if start_date is not None else self.start_date
        backtest_end = end_date if end_date is not None else self.end_date
        
        # Filter dates using custom timeline
        prices_filtered = self.prices[(self.prices['Date'] >= backtest_start) & 
                                      (self.prices['Date'] <= backtest_end)].copy()
        
        open_df = prices_filtered.pivot_table(index='Date', columns='Ticker', values='Open').sort_index()
        open_df.columns = [c.split('.')[0].upper() for c in open_df.columns]

        # Add XU100 to open_df if available (for XU100 benchmark backtest)
        if self.xu100_prices is not None and 'XU100' not in open_df.columns:
            open_df['XU100'] = self.xu100_prices.reindex(open_df.index)

        # For XU100 factor: filter out dates where XU100 data is missing
        # This ensures fair comparison with benchmark (avoids 0-return days from missing data)
        if factor_name == "xu100":
            valid_xu100_mask = open_df['XU100'].notna()
            n_filtered = (~valid_xu100_mask).sum()
            if n_filtered > 0:
                print(f"   Filtered {n_filtered} dates with missing XU100 data")
                open_df = open_df[valid_xu100_mask]
        
        # Align regime series
        # NOTE: Regime model was trained with a train_end_date cutoff.
        # For dates BEFORE the cutoff, predictions may be in-sample (overfit).
        # For dates AFTER the cutoff, predictions are genuine out-of-sample.
        # The 1-day lag below prevents intraday look-ahead but doesn't address
        # the in-sample training issue for historical backtest periods.
        regime_series = self.regime_series.reindex(open_df.index).ffill()
        regime_series_lagged = regime_series.shift(1).ffill()  # Lag by 1 day to avoid look-ahead
        
        # Align XAU/TRY (avoid cross-run contamination from cached slicing)
        xautry_series = self.loader.load_xautry_prices(
            self.data_dir / "xau_try_2013_2026.csv",
            start_date=backtest_start,
            end_date=backtest_end,
        )
        xautry_prices = xautry_series.reindex(open_df.index).ffill()
        
        # Calculate returns
        open_fwd_ret = open_df.shift(-1) / open_df - 1.0
        xautry_fwd_ret = xautry_prices.shift(-1) / xautry_prices - 1.0
        xautry_fwd_ret = xautry_fwd_ret.fillna(0.0)
        
        # Neutralize splits
        split_mask = (open_fwd_ret < -0.50) | (open_fwd_ret > 1.00)
        n_neutralised = split_mask.sum().sum()
        if n_neutralised > 0:
            open_fwd_ret = open_fwd_ret.where(~split_mask, 0.0)
            print(f"   Neutralised {n_neutralised} split/corporate-action returns")
        
        trading_days = open_df.index
        
        # Determine rebalancing days based on frequency
        if rebalance_freq == 'monthly':
            rebalance_days = identify_monthly_rebalance_days(trading_days)
        else:
            rebalance_days = self._identify_quarterly_rebalance_days(trading_days)
        
        print(f"Period: {trading_days[0].date()} to {trading_days[-1].date()}")
        print(f"Trading days: {len(trading_days)}")
        print(f"Rebalance days: {len(rebalance_days)}")
        
        # Run backtest loop
        portfolio_returns = []
        holdings_history = []  # Track daily holdings with weights
        current_holdings = []
        entry_prices = {}
        stopped_out = set()
        prev_selected = set()
        trade_count = 0
        rebalance_count = 0
        
        # Track regime-specific performance
        regime_returns_tracker = {regime: [] for regime in ['Bull', 'Recovery', 'Choppy', 'Stress', 'Bear']}

        # Get options for this backtest
        slippage_factor = opts['slippage_bps'] / 10000.0 if opts['use_slippage'] else 0.0
        top_n = opts['top_n']
        stop_loss_threshold = opts['stop_loss_threshold']

        for i, date in enumerate(trading_days[:-1]):
            regime = regime_series_lagged.get(date, 'Choppy')
            if pd.isna(regime):
                regime = 'Choppy'

            # Regime filter: if disabled, always use 100% allocation
            if opts['use_regime_filter']:
                allocation = REGIME_ALLOCATIONS.get(regime, 0.5)
            else:
                allocation = 1.0  # Always fully invested
            
            is_rebalance_day = date in rebalance_days
            
            if is_rebalance_day:
                stopped_out.clear()
                rebalance_count += 1
                
                # Capture old holdings BEFORE updating (for robust trade tracking)
                old_selected = prev_selected.copy() if prev_selected else set()
                
                if allocation > 0 and date in signals.index:
                    # Get signals for this date
                    day_signals = signals.loc[date].dropna()
                    
                    # Special handling for XU100 benchmark (skip liquidity filter, allow single holding)
                    if factor_name == "xu100":
                        available = [t for t in day_signals.index if t in open_df.columns 
                                    and pd.notna(open_df.loc[date, t])]
                        if available:
                            current_holdings = available
                            entry_prices = {t: open_df.loc[date, t] for t in available}
                            new_positions = set(current_holdings) - old_selected
                            trade_count += len(new_positions)
                            prev_selected = set(current_holdings)
                    else:
                        # Standard stock selection path
                        available = [t for t in day_signals.index if t in open_df.columns
                                    and pd.notna(open_df.loc[date, t])]

                        # Liquidity filter (optional)
                        if opts['use_liquidity_filter']:
                            available = self._filter_by_liquidity(available, date, opts['liquidity_quantile'])
                        day_signals = day_signals[available]

                        if len(day_signals) >= top_n:
                            # Select top N
                            top_stocks = day_signals.nlargest(top_n).index.tolist()
                            current_holdings = top_stocks
                            entry_prices = {t: open_df.loc[date, t] for t in top_stocks
                                          if t in open_df.columns and pd.notna(open_df.loc[date, t])}

                            # Track new positions
                            new_positions = set(current_holdings) - old_selected
                            trade_count += len(new_positions)
                            prev_selected = set(current_holdings)
            else:
                old_selected = set()  # No rebalance, no turnover

            # Daily stop-loss check (optional)
            if opts['use_stop_loss']:
                # Use open prices for consistency: entry at open, check drawdown vs current open
                holdings_to_keep = []
                for ticker in current_holdings:
                    if ticker in stopped_out:
                        continue
                    if ticker not in entry_prices:
                        holdings_to_keep.append(ticker)
                        continue

                    entry = entry_prices[ticker]
                    # Use OPEN price for stop-loss check (consistent with entry)
                    current_price = open_df.loc[date, ticker] if date in open_df.index and ticker in open_df.columns else np.nan

                    if pd.notna(current_price) and pd.notna(entry) and entry > 0:
                        drawdown = (current_price / entry) - 1.0
                        if drawdown < -stop_loss_threshold:
                            stopped_out.add(ticker)
                            continue

                    holdings_to_keep.append(ticker)

                active_holdings = holdings_to_keep
            else:
                # No stop loss - keep all current holdings
                active_holdings = current_holdings
            
            # Calculate portfolio return
            if active_holdings and allocation > 0:
                # Position weighting: inverse downside vol or equal weight
                if opts['use_inverse_vol_sizing']:
                    weights = inverse_downside_vol_weights(
                        self.close_df, active_holdings, date,
                        lookback=opts['inverse_vol_lookback'],
                        max_weight=opts['max_position_weight']
                    )
                else:
                    # Equal weight
                    weights = pd.Series(1.0 / len(active_holdings), index=active_holdings)

                stock_return = 0.0
                for ticker in active_holdings:
                    if ticker in open_fwd_ret.columns:
                        ret = open_fwd_ret.loc[date, ticker]
                        if pd.notna(ret):
                            stock_return += ret * weights[ticker]

                # Apply slippage (optional)
                if opts['use_slippage'] and is_rebalance_day and old_selected:
                    turnover = len(set(active_holdings) - old_selected) / max(len(active_holdings), 1)
                    stock_return -= turnover * slippage_factor * 2

                # Blend with XAU/TRY (only if regime filter is active)
                xautry_ret = xautry_fwd_ret.loc[date] if date in xautry_fwd_ret.index else 0.0
                if opts['use_regime_filter']:
                    port_ret = allocation * stock_return + (1 - allocation) * xautry_ret
                else:
                    port_ret = stock_return  # No gold blending
            else:
                xautry_ret = xautry_fwd_ret.loc[date] if date in xautry_fwd_ret.index else 0.0
                if opts['use_regime_filter']:
                    port_ret = xautry_ret  # Full gold allocation
                else:
                    port_ret = 0.0  # No holdings, no return
            
            # Track regime-specific returns
            regime_returns_tracker[regime].append(port_ret)
            
            portfolio_returns.append({
                'date': date,
                'return': port_ret,
                'xautry_return': xautry_fwd_ret.loc[date] if date in xautry_fwd_ret.index else 0.0,
                'regime': regime,
                'n_stocks': len(active_holdings),
                'allocation': allocation,
            })
            
            # Track holdings with weights for this day
            if active_holdings and allocation > 0:
                for ticker in active_holdings:
                    holdings_history.append({
                        'date': date,
                        'ticker': ticker,
                        'weight': weights.get(ticker, 0) * allocation,  # Stock weight * allocation
                        'regime': regime,
                        'allocation': allocation,
                    })
            else:
                # Record gold-only position
                holdings_history.append({
                    'date': date,
                    'ticker': 'XAU/TRY',
                    'weight': 1.0,
                    'regime': regime,
                    'allocation': 0.0,
                })
        
        # Build results
        returns_df = pd.DataFrame(portfolio_returns).set_index('date')
        raw_returns = returns_df['return']

        # Apply volatility targeting (optional)
        if opts['use_vol_targeting']:
            print(f"\n📈 Applying {opts['target_downside_vol']*100:.0f}% downside volatility targeting...")
            returns = apply_downside_vol_targeting(
                raw_returns,
                target_vol=opts['target_downside_vol'],
                lookback=opts['vol_lookback'],
                vol_floor=opts['vol_floor'],
                vol_cap=opts['vol_cap']
            )

            # Calculate realized downside vol after targeting
            neg_rets = returns[returns < 0]
            realized_downside_vol = neg_rets.std() * np.sqrt(252) if len(neg_rets) > 2 else 0
            print(f"   Realized downside volatility: {realized_downside_vol*100:.1f}%")
        else:
            print(f"\n📈 Volatility targeting: OFF (using raw returns)")
            returns = raw_returns
        
        # Calculate metrics on vol-targeted returns
        equity = (1 + returns).cumprod()
        total_return = equity.iloc[-1] - 1
        n_years = len(returns) / 252
        cagr = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else 0
        
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        downside = returns[returns < 0]
        sortino = returns.mean() / downside.std() * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0
        
        cummax = equity.cummax()
        drawdown = equity / cummax - 1
        max_dd = drawdown.min()
        
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
        
        # Calculate regime performance
        regime_perf = {}
        for regime in ['Bull', 'Recovery', 'Choppy', 'Stress', 'Bear']:
            mask = returns_df['regime'] == regime
            if mask.sum() > 0:
                r = returns[mask]
                regime_perf[regime] = {
                    'count': mask.sum(),
                    'mean_return': r.mean() * 252,
                    'total_return': (1 + r).prod() - 1,
                    'win_rate': (r > 0).sum() / len(r) if len(r) > 0 else 0,
                }
        
        print(f"\n📊 Results:")
        print(f"   Total Return: {total_return*100:.1f}%")
        print(f"   CAGR: {cagr*100:.2f}%")
        print(f"   Sharpe: {sharpe:.2f}")
        print(f"   Sortino: {sortino:.2f}")
        print(f"   Max Drawdown: {max_dd*100:.2f}%")
        print(f"   Win Rate: {win_rate*100:.1f}%")
        print(f"   Rebalances: {rebalance_count}")
        print(f"   Total Trades: {trade_count}")
        
        return {
            'returns': returns,
            'equity': equity,
            'total_return': total_return,
            'cagr': cagr,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'xautry_returns': returns_df['xautry_return'],
            'regime_performance': regime_perf,
            'rebalance_count': rebalance_count,
            'trade_count': trade_count,
            'returns_df': returns_df,
            'holdings_history': holdings_history,
        }
    
    def _identify_quarterly_rebalance_days(self, trading_days: pd.DatetimeIndex) -> set:
        """Identify quarterly rebalancing days"""
        rebalance_days = set()
        
        for year in range(trading_days.min().year, trading_days.max().year + 1):
            for month, day in [(3, 15), (5, 15), (8, 15), (11, 15)]:
                target = pd.Timestamp(year=year, month=month, day=day)
                valid = trading_days[trading_days >= target]
                if len(valid) > 0:
                    rebalance_days.add(valid[0])
        
        return rebalance_days
    
    def _filter_by_liquidity(self, tickers, date, liquidity_quantile=LIQUIDITY_QUANTILE):
        """Remove bottom quartile by liquidity"""
        if date not in self.volume_df.index:
            candidates = self.volume_df.index[self.volume_df.index <= date]
            if candidates.empty:
                return tickers
            date = candidates.max()

        adv = self.volume_df.loc[date, [t for t in tickers if t in self.volume_df.columns]].dropna()
        if adv.empty:
            return tickers

        threshold = adv.quantile(liquidity_quantile)
        liquid = set(adv[adv >= threshold].index)
        return [t for t in tickers if t in liquid]
    
    def save_results(self, results, factor_name, output_dir=None):
        """Save backtest results with comprehensive metrics"""
        if output_dir is None:
            # Default layout: Models/results/<signal_name>/
            output_dir = Path(__file__).parent / "results" / factor_name
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        returns = results['returns']
        xautry_returns = results['xautry_returns']
        
        # Align XU100 benchmark
        xu100_returns = None
        if self.xu100_prices is not None:
            xu100_returns = self.xu100_prices.shift(-1) / self.xu100_prices - 1.0
            xu100_returns = xu100_returns.reindex(returns.index)
        
        # Save equity curve
        pd.DataFrame({'Equity': results['equity']}).to_csv(output_dir / 'equity_curve.csv')
        
        # Save returns
        returns_df = pd.DataFrame({
            'Return': returns,
            'XAU_TRY_Return': xautry_returns.squeeze()
        })
        if xu100_returns is not None:
            returns_df['XU100_Return'] = xu100_returns.squeeze()
        returns_df.to_csv(output_dir / 'returns.csv')
        
        # Save yearly metrics
        yearly_metrics = compute_yearly_metrics(returns, xu100_returns, xautry_returns)
        yearly_metrics.to_csv(output_dir / 'yearly_metrics.csv', index=False)
        
        # Save regime performance
        regime_perf = pd.DataFrame(results['regime_performance']).T
        regime_perf.to_csv(output_dir / 'regime_performance.csv')
        
        # Save daily holdings with weights
        if results.get('holdings_history'):
            holdings_df = pd.DataFrame(results['holdings_history'])
            holdings_df.to_csv(output_dir / 'holdings.csv', index=False)
            
            # Also create a pivot table for easier analysis (date x ticker)
            holdings_pivot = holdings_df.pivot_table(
                index='date', columns='ticker', values='weight', aggfunc='first'
            ).fillna(0)
            holdings_pivot.to_csv(output_dir / 'holdings_matrix.csv')
        
        # Save summary
        with open(output_dir / 'summary.txt', 'w') as f:
            f.write("="*60 + "\n")
            f.write(f"{factor_name.upper()} FACTOR MODEL\n")
            f.write("="*60 + "\n\n")
            f.write(f"Total Return: {results['total_return'] * 100:.2f}%\n")
            f.write(f"CAGR: {results['cagr'] * 100:.2f}%\n")
            f.write(f"Max Drawdown: {results['max_drawdown'] * 100:.2f}%\n")
            f.write(f"Sharpe Ratio: {results['sharpe']:.2f}\n")
            f.write(f"Sortino Ratio: {results['sortino']:.2f}\n")
            f.write(f"Win Rate: {results['win_rate'] * 100:.2f}%\n")
            f.write(f"Trading Days: {len(returns)}\n")
            f.write(f"Rebalance Days: {results['rebalance_count']}\n")
            f.write(f"Total Trades: {results['trade_count']}\n")
            
            if xu100_returns is not None:
                bench_aligned = xu100_returns.dropna()
                if len(bench_aligned) > 0:
                    bench_total = (1 + bench_aligned).prod() - 1
                    f.write(f"\nBenchmark (XU100) Return: {bench_total * 100:.2f}%\n")
                    f.write(f"Excess vs XU100: {(results['total_return'] - bench_total) * 100:.2f}%\n")
            
            xautry_aligned = xautry_returns.dropna()
            if len(xautry_aligned) > 0:
                xautry_total = (1 + xautry_aligned).prod() - 1
                f.write(f"\nBenchmark (XAU/TRY) Return: {xautry_total * 100:.2f}%\n")
                f.write(f"Excess vs XAU/TRY: {(results['total_return'] - xautry_total) * 100:.2f}%\n")
        
        print(f"\n💾 Results saved to: {output_dir}")
        
        # Print yearly summary
        print("\n" + "="*70)
        print("YEARLY RESULTS")
        print("="*70)
        print(f"{'Year':<6} {'Model':>10} {'XU100':>10} {'Excess':>10}")
        print("-"*40)
        for _, row in yearly_metrics.iterrows():
            xu_ret = row['XU100_Return'] if pd.notna(row['XU100_Return']) else 0
            excess = row['Excess_vs_XU100'] if pd.notna(row['Excess_vs_XU100']) else row['Return']
            print(f"{int(row['Year']):<6} {row['Return']*100:>9.1f}% {xu_ret*100:>9.1f}% {excess*100:>9.1f}%")
    
    def save_correlation_matrix(self, output_dir=None):
        """Calculate and save full return correlation matrix across all strategies and benchmarks"""
        if not self.factor_returns:
            print("⚠️  No factor returns stored - run factors first")
            return

        if output_dir is None:
            output_dir = Path(__file__).parent / "results"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Build returns DataFrame
        all_returns = {}
        for factor_name, returns in self.factor_returns.items():
            all_returns[factor_name] = returns

        # Add benchmarks - use CLOSE prices for correlation comparison
        # (self.xu100_prices uses Open for trading, but Close is standard for benchmarks)
        xu100_file = self.data_dir / "xu100_prices.csv"
        if xu100_file.exists():
            xu100_df = pd.read_csv(xu100_file)
            xu100_df['Date'] = pd.to_datetime(xu100_df['Date'])
            xu100_df = xu100_df.set_index('Date').sort_index()
            if 'Close' in xu100_df.columns:
                xu100_returns = xu100_df['Close'].pct_change().dropna()
                all_returns['XU100'] = xu100_returns

        if self.xautry_prices is not None:
            xautry_returns = self.loader.load_xautry_prices(
                self.data_dir / "xau_try_2013_2026.csv"
            ).pct_change().dropna()
            all_returns['XAUTRY'] = xautry_returns

        # Create DataFrame and align dates
        returns_df = pd.DataFrame(all_returns)
        returns_df = returns_df.dropna(how='all')

        # Calculate full correlation matrix
        corr_matrix = returns_df.corr()

        # Print full correlation matrix
        labels = list(corr_matrix.columns)
        col_width = max(max(len(l) for l in labels), 6) + 2

        print("\n" + "=" * 70)
        print("RETURN CORRELATION MATRIX")
        print("=" * 70)

        # Header row
        header = " " * col_width + "".join(f"{l:>{col_width}}" for l in labels)
        print(header)
        print("-" * len(header))

        # Data rows
        for row_label in labels:
            row_str = f"{row_label:<{col_width}}"
            for col_label in labels:
                val = corr_matrix.loc[row_label, col_label]
                row_str += f"{val:>{col_width}.4f}"
            print(row_str)

        # Save to CSV
        full_corr_file = output_dir / "factor_correlation_matrix.csv"
        corr_matrix.to_csv(full_corr_file)

        print(f"\n💾 Correlation matrix saved to: {full_corr_file}")

        return corr_matrix
    
    def run_all_factors(self):
        """Run all enabled factors"""
        results = {}
        
        for factor_name, config in self.signal_configs.items():
            if config.get('enabled', True):
                results[factor_name] = self.run_factor(factor_name)
            else:
                print(f"\n⚠️  Skipping {factor_name} (disabled in config)")
        
        # Save correlation matrix after all factors complete
        if self.factor_returns:
            self.save_correlation_matrix()
        
        return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Load available signals dynamically from configs
    available_signals = load_signal_configs()
    signal_names = list(available_signals.keys())
    
    parser = argparse.ArgumentParser(
        description='Config-Based Portfolio Engine - Automatically detects signals from configs/',
        epilog=f'Available signals: {", ".join(signal_names)}'
    )
    
    # Support both positional and --factor argument
    parser.add_argument('signal', nargs='?', type=str, default=None,
                       help=f'Signal to run: {", ".join(signal_names)}, or "all"')
    parser.add_argument('--factor', type=str, default=None,
                       help='Alternative way to specify signal (deprecated, use positional arg)')
    parser.add_argument('--start-date', type=str, default='2018-01-01',
                       help='Start date (default: 2018-01-01)')
    parser.add_argument('--end-date', type=str, default='2024-12-31',
                       help='End date (default: 2024-12-31)')
    
    args = parser.parse_args()
    
    # Determine which signal to run (positional takes precedence)
    signal_to_run = args.signal or args.factor or 'all'
    
    # Validate signal name
    if signal_to_run != 'all' and signal_to_run not in signal_names:
        print(f"❌ Unknown signal: {signal_to_run}")
        print(f"Available signals: {', '.join(signal_names)}, all")
        sys.exit(1)
    
    # Setup paths
    script_dir = Path(__file__).parent  # Models/
    bist_root = script_dir.parent        # BIST/
    data_dir = bist_root / "data"
    regime_model_dir = bist_root / "Regime Filter" / "outputs" / "ensemble_model"



    
    # Initialize engine
    engine = PortfolioEngine(data_dir, regime_model_dir, args.start_date, args.end_date)
    engine.load_all_data()
    
    # Run signal(s)
    if signal_to_run == 'all':
        engine.run_all_factors()
    else:
        engine.run_factor(signal_to_run)


if __name__ == "__main__":
    total_start = time.time()
    main()
    total_elapsed = time.time() - total_start
    print("\n" + "="*70)
    print(f"✅ TOTAL RUNTIME: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")
    print("="*70)
