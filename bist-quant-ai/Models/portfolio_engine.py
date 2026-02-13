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
import os
import sys
import time
import warnings
import importlib.util
import json
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
from signals.dividend_rotation_signals import build_dividend_rotation_signals
from signals.macro_hedge_signals import build_macro_hedge_signals
from signals.quality_momentum_signals import build_quality_momentum_signals
from signals.quality_value_signals import build_quality_value_signals
from signals.small_cap_momentum_signals import build_small_cap_momentum_signals
from signals.accrual_signals import build_accrual_signals
from signals.asset_growth_signals import build_asset_growth_signals
from signals.betting_against_beta_signals import build_betting_against_beta_signals
from signals.roa_signals import build_roa_signals
from signals.size_rotation_signals import (
    build_size_rotation_signals,
    build_market_cap_panel as build_size_market_cap_panel,
    get_size_buckets_for_date,
    SIZE_LIQUIDITY_QUANTILE,
)
from signals.size_rotation_momentum_signals import build_size_rotation_momentum_signals
from signals.size_rotation_quality_signals import build_size_rotation_quality_signals
from signals.five_factor_rotation_signals import build_five_factor_rotation_signals

# New imported signals from Quantpedia strategies
from signals.short_term_reversal_signals import build_short_term_reversal_signals
from signals.consistent_momentum_signals import build_consistent_momentum_signals
from signals.residual_momentum_signals import build_residual_momentum_signals
from signals.momentum_reversal_volatility_signals import build_momentum_reversal_volatility_signals
from signals.low_volatility_signals import build_low_volatility_signals
from signals.trend_following_signals import build_trend_following_signals
from signals.sector_rotation_signals import build_sector_rotation_signals
from signals.earnings_quality_signals import build_earnings_quality_signals
from signals.fscore_reversal_signals import build_fscore_reversal_signals
from signals.momentum_asset_growth_signals import build_momentum_asset_growth_signals
from signals.pairs_trading_signals import build_pairs_trading_signals


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

    # Transaction costs - market-cap-based slippage
    'use_slippage': True,
    'slippage_bps': 5.0,  # Base slippage for large caps
    # Market-cap-based slippage (applies to ALL factors now)
    'use_mcap_slippage': True,  # Enable market-cap-based differentiated slippage
    'small_cap_slippage_bps': 20.0,  # Higher slippage for small caps (illiquid)
    'mid_cap_slippage_bps': 10.0,  # Medium slippage for mid caps

    # Portfolio size
    'top_n': 20,
}

SIZE_ROTATION_FACTORS = {"size_rotation", "size_rotation_momentum", "size_rotation_quality"}

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


def compute_capm_metrics(
    strategy_returns: pd.Series,
    market_returns: pd.Series,
    risk_free_daily: float = 0.0,
) -> dict:
    """
    Compute CAPM metrics using OLS:
        (R_i - R_f) = alpha + beta * (R_m - R_f) + epsilon

    Args:
        strategy_returns: Daily strategy returns
        market_returns: Daily market benchmark returns (e.g., XU100)
        risk_free_daily: Daily risk-free rate (default 0.0)

    Returns:
        dict with CAPM metrics (beta, alpha, r_squared, correlation, n_obs)
    """
    df = pd.DataFrame({
        "strategy": strategy_returns,
        "market": market_returns,
    }).dropna()

    n_obs = len(df)
    if n_obs < 30:
        return {
            "n_obs": n_obs,
            "alpha_daily": np.nan,
            "alpha_annual": np.nan,
            "beta": np.nan,
            "r_squared": np.nan,
            "correlation": np.nan,
            "residual_vol_annual": np.nan,
        }

    y = (df["strategy"] - risk_free_daily).values.astype(float)
    x = (df["market"] - risk_free_daily).values.astype(float)

    x_var = np.var(x, ddof=1)
    if not np.isfinite(x_var) or x_var <= 0:
        return {
            "n_obs": n_obs,
            "alpha_daily": np.nan,
            "alpha_annual": np.nan,
            "beta": np.nan,
            "r_squared": np.nan,
            "correlation": np.nan,
            "residual_vol_annual": np.nan,
        }

    # OLS with intercept
    X = np.column_stack([np.ones_like(x), x])
    alpha_daily, beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = alpha_daily + beta * x
    resid = y - y_hat

    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    correlation = np.corrcoef(y, x)[0, 1] if n_obs > 1 else np.nan

    alpha_annual = (1.0 + alpha_daily) ** 252 - 1.0 if np.isfinite(alpha_daily) else np.nan
    residual_vol_annual = np.std(resid, ddof=1) * np.sqrt(252) if n_obs > 1 else np.nan

    return {
        "n_obs": int(n_obs),
        "alpha_daily": float(alpha_daily),
        "alpha_annual": float(alpha_annual),
        "beta": float(beta),
        "r_squared": float(r_squared) if np.isfinite(r_squared) else np.nan,
        "correlation": float(correlation) if np.isfinite(correlation) else np.nan,
        "residual_vol_annual": float(residual_vol_annual) if np.isfinite(residual_vol_annual) else np.nan,
    }


def compute_rolling_beta_series(
    strategy_returns: pd.Series,
    market_returns: pd.Series,
    window: int = 252,
    min_periods: int = 126,
    risk_free_daily: float = 0.0,
) -> pd.Series:
    """
    Compute rolling CAPM beta series:
        beta_t = Cov(R_i - R_f, R_m - R_f)_t / Var(R_m - R_f)_t

    Args:
        strategy_returns: Daily strategy returns
        market_returns: Daily market returns
        window: Rolling window length in trading days
        min_periods: Minimum observations required to compute beta
        risk_free_daily: Daily risk-free rate

    Returns:
        pd.Series of rolling beta indexed by date
    """
    df = pd.DataFrame({
        "strategy": strategy_returns,
        "market": market_returns,
    }).dropna()

    if df.empty:
        return pd.Series(dtype=float)

    x = df["market"] - risk_free_daily
    y = df["strategy"] - risk_free_daily

    cov_xy = y.rolling(window=window, min_periods=min_periods).cov(x)
    var_x = x.rolling(window=window, min_periods=min_periods).var()
    beta = cov_xy / var_x.replace(0.0, np.nan)
    beta = beta.replace([np.inf, -np.inf], np.nan)
    beta.name = "Rolling_Beta"
    return beta


def compute_yearly_rolling_beta_metrics(rolling_beta: pd.Series) -> pd.DataFrame:
    """
    Summarize rolling beta by calendar year.

    Args:
        rolling_beta: Daily rolling beta series

    Returns:
        DataFrame with yearly rolling-beta statistics
    """
    if rolling_beta is None or rolling_beta.empty:
        return pd.DataFrame(
            columns=[
                "Year",
                "Observations",
                "Beta_Start",
                "Beta_End",
                "Beta_Mean",
                "Beta_Median",
                "Beta_Min",
                "Beta_Max",
                "Beta_Std",
                "Beta_Change",
            ]
        )

    s = rolling_beta.dropna()
    if s.empty:
        return pd.DataFrame(
            columns=[
                "Year",
                "Observations",
                "Beta_Start",
                "Beta_End",
                "Beta_Mean",
                "Beta_Median",
                "Beta_Min",
                "Beta_Max",
                "Beta_Std",
                "Beta_Change",
            ]
        )

    s = s.sort_index()
    rows = []
    for year, grp in s.groupby(s.index.year):
        if grp.empty:
            continue
        beta_start = grp.iloc[0]
        beta_end = grp.iloc[-1]
        rows.append({
            "Year": int(year),
            "Observations": int(len(grp)),
            "Beta_Start": float(beta_start),
            "Beta_End": float(beta_end),
            "Beta_Mean": float(grp.mean()),
            "Beta_Median": float(grp.median()),
            "Beta_Min": float(grp.min()),
            "Beta_Max": float(grp.max()),
            "Beta_Std": float(grp.std(ddof=1)) if len(grp) > 1 else np.nan,
            "Beta_Change": float(beta_end - beta_start),
        })

    return pd.DataFrame(rows).sort_values("Year")


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
        self.regime_allocations = REGIME_ALLOCATIONS.copy()
        self.xautry_prices = None
        self.xu100_prices = None
        self.fundamentals = None
        
        # Store factor returns for correlation analysis
        self.factor_returns = {}
        self.factor_capm = {}
        self.factor_yearly_rolling_beta = {}
        
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
        
        
        # Load regime predictions (Simple Regime Filter outputs)
        self.regime_series = self.loader.load_regime_predictions()
        loaded_allocations = self.loader.load_regime_allocations()
        if loaded_allocations:
            self.regime_allocations = REGIME_ALLOCATIONS.copy()
            self.regime_allocations.update(loaded_allocations)
            print("  ✅ Using regime allocations from regime_labels.json:")
            for regime, alloc in sorted(self.regime_allocations.items()):
                print(f"    {regime}: {alloc:.2f}")
        else:
            self.regime_allocations = REGIME_ALLOCATIONS.copy()
            print("  ℹ️  Using fallback regime allocations from portfolio_engine constants.")
        
        # Load XAU/TRY
        xautry_file = self.data_dir / "xau_try_2013_2026.csv"
        self.xautry_prices = self.loader.load_xautry_prices(xautry_file)
        
        # Load XU100
        xu100_file = self.data_dir / "xu100_prices.csv"
        self.xu100_prices = self.loader.load_xu100_prices(xu100_file)
        
        elapsed = time.time() - start_time
        print(f"\n✅ Data loading completed in {elapsed:.1f} seconds")

    def _build_signals_for_factor(self, factor_name: str, dates: pd.DatetimeIndex, config: dict):
        """Build factor signal panel with optional config-based signal parameter overrides."""
        factor_details = {}
        signal_params = config.get("signal_params", {}) if isinstance(config.get("signal_params"), dict) else {}

        if factor_name == "profitability":
            operating_income_weight = float(signal_params.get("operating_income_weight", 0.5))
            gross_profit_weight = float(signal_params.get("gross_profit_weight", 0.5))
            signals = build_profitability_signals(
                self.fundamentals,
                dates,
                self.loader,
                operating_income_weight=operating_income_weight,
                gross_profit_weight=gross_profit_weight,
            )
        elif factor_name == "value":
            metric_weights = signal_params.get("metric_weights")
            if not isinstance(metric_weights, dict):
                metric_weights = None
            enabled_metrics = signal_params.get("enabled_metrics")
            if isinstance(enabled_metrics, list):
                enabled_metrics = [m for m in enabled_metrics if isinstance(m, str)]
            else:
                enabled_metrics = None
            signals = build_value_signals(
                self.fundamentals,
                self.close_df,
                dates,
                self.loader,
                metric_weights=metric_weights,
                enabled_metrics=enabled_metrics,
            )
        elif factor_name == "small_cap":
            signals = build_small_cap_signals(self.fundamentals, self.close_df, self.volume_df, dates, self.loader)
        elif factor_name == "investment":
            signals = build_investment_signals(self.fundamentals, self.close_df, dates, self.loader)
        elif factor_name == "momentum":
            signals = build_momentum_signals(
                self.close_df,
                dates,
                self.loader,
                lookback=int(signal_params.get("lookback", 252)),
                skip=int(signal_params.get("skip", 21)),
                vol_lookback=int(signal_params.get("vol_lookback", 252)),
            )
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
            signals = build_small_cap_momentum_signals(self.close_df, dates, self.loader)
        elif factor_name == "size_rotation":
            # Size Rotation: Dynamically switches between small and large caps
            signals = build_size_rotation_signals(self.close_df, dates, self.loader)
        elif factor_name == "size_rotation_momentum":
            # Size Rotation Momentum: Pure momentum within winning size segment
            signals = build_size_rotation_momentum_signals(self.close_df, dates, self.loader)
        elif factor_name == "size_rotation_quality":
            # Size Rotation Quality: Momentum + Profitability within winning size segment
            signals = build_size_rotation_quality_signals(self.close_df, self.fundamentals, dates, self.loader)
        elif factor_name == "five_factor_rotation":
            # Five-factor side rotation: select winning side across 5 classic axes
            cache_cfg = config.get("construction_cache", {})
            debug_cfg = config.get("debug", {})
            orth_cfg = config.get("axis_orthogonalization", {})
            walk_forward_cfg = config.get("walk_forward", {}) if isinstance(config.get("walk_forward", {}), dict) else {}
            debug_env = os.getenv("FIVE_FACTOR_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
            debug_enabled = bool(debug_cfg.get("enabled", False) or debug_env)
            signals, factor_details = build_five_factor_rotation_signals(
                self.close_df,
                dates,
                self.loader,
                fundamentals=self.fundamentals,
                volume_df=self.volume_df,
                use_construction_cache=cache_cfg.get("enabled", True),
                force_rebuild_construction_cache=cache_cfg.get("force_rebuild", False),
                construction_cache_path=cache_cfg.get("path"),
                mwu_walkforward_config=walk_forward_cfg,
                axis_orthogonalization_config=orth_cfg,
                return_details=True,
                debug=debug_enabled,
            )
        elif factor_name == "accrual":
            # Accrual Anomaly: Low accruals = High quality earnings
            signals = build_accrual_signals(self.fundamentals, dates, self.loader)
        elif factor_name == "asset_growth":
            # Asset Growth Effect: Low asset growth = Conservative management
            signals = build_asset_growth_signals(self.fundamentals, dates, self.loader)
        elif factor_name == "betting_against_beta":
            # Betting Against Beta: Low beta stocks outperform
            signals = build_betting_against_beta_signals(self.close_df, dates, self.loader)
        elif factor_name == "roa":
            # ROA Effect: High profitability relative to assets
            signals = build_roa_signals(self.fundamentals, dates, self.loader)
        # === NEW QUANTPEDIA-IMPORTED SIGNALS ===
        elif factor_name == "short_term_reversal":
            # Short-Term Reversal: Weekly losers outperform
            signals = build_short_term_reversal_signals(self.close_df, dates, self.loader)
        elif factor_name == "consistent_momentum":
            # Consistent Momentum: Winners in multiple timeframes
            signals = build_consistent_momentum_signals(self.close_df, dates, self.loader)
        elif factor_name == "residual_momentum":
            # Residual Momentum: Market-adjusted stock-specific momentum
            signals = build_residual_momentum_signals(self.close_df, dates, self.loader)
        elif factor_name == "momentum_reversal_volatility":
            # Combined: Momentum + Reversal + Low Volatility
            signals = build_momentum_reversal_volatility_signals(self.close_df, dates, self.loader)
        elif factor_name == "low_volatility":
            # Low Volatility: Low-vol stocks outperform high-vol
            signals = build_low_volatility_signals(self.close_df, dates, self.loader)
        elif factor_name == "trend_following":
            # Trend Following: Stocks at/near all-time highs
            signals = build_trend_following_signals(self.close_df, dates, self.loader)
        elif factor_name == "sector_rotation":
            # Sector Rotation: Overweight top-performing BIST sectors
            signals = build_sector_rotation_signals(self.close_df, dates, self.loader)
        elif factor_name == "earnings_quality":
            # Earnings Quality: Multi-factor quality (accruals, ROE, CF/A, D/A)
            signals = build_earnings_quality_signals(self.fundamentals, self.close_df, dates, self.loader)
        elif factor_name == "fscore_reversal":
            # F-Score + Reversal: Quality losers outperform junk winners
            signals = build_fscore_reversal_signals(self.fundamentals, self.close_df, dates, self.loader)
        elif factor_name == "momentum_asset_growth":
            # Momentum + Asset Growth: Momentum within high-growth stocks
            signals = build_momentum_asset_growth_signals(self.fundamentals, self.close_df, dates, self.loader)
        elif factor_name == "pairs_trading":
            # Pairs Trading: Mean reversion on correlated pairs
            signals = build_pairs_trading_signals(self.close_df, dates, self.loader)
        else:
            raise ValueError(f"Unknown factor: {factor_name}")

        return signals, factor_details
        
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

        # Optional walk-forward timeline clamp (used by five_factor_rotation)
        walk_forward_cfg = config.get("walk_forward", {}) if isinstance(config.get("walk_forward", {}), dict) else {}
        if factor_name == "five_factor_rotation" and walk_forward_cfg.get("enabled", False):
            first_test_year = walk_forward_cfg.get("first_test_year")
            last_test_year = walk_forward_cfg.get("last_test_year")
            if first_test_year is not None:
                wf_start = pd.Timestamp(year=int(first_test_year), month=1, day=1)
                if factor_start_date < wf_start:
                    print(f"Walk-forward start clamp: {factor_start_date.date()} -> {wf_start.date()}")
                    factor_start_date = wf_start
            if last_test_year is not None:
                wf_end = pd.Timestamp(year=int(last_test_year), month=12, day=31)
                if factor_end_date > wf_end:
                    print(f"Walk-forward end clamp: {factor_end_date.date()} -> {wf_end.date()}")
                    factor_end_date = wf_end
        
        # Display timeline
        if custom_start or custom_end:
            print(f"Custom timeline: {factor_start_date.date()} to {factor_end_date.date()}")
        
        start_time = time.time()
        dates = self.close_df.index
        signals, factor_details = self._build_signals_for_factor(factor_name, dates, config)

        # Get portfolio options from config
        portfolio_options = config.get('portfolio_options', {})

        # Run backtest with custom timeline and portfolio options
        results = self._run_backtest(
            signals,
            factor_name,
            rebalance_freq,
            factor_start_date,
            factor_end_date,
            portfolio_options,
        )
        if factor_details:
            results.update(factor_details)
            if 'yearly_axis_winners' in results and isinstance(results['yearly_axis_winners'], pd.DataFrame):
                yearly_axis = results['yearly_axis_winners']
                if not yearly_axis.empty and 'Year' in yearly_axis.columns:
                    start_year = int(factor_start_date.year)
                    end_year = int(factor_end_date.year)
                    results['yearly_axis_winners'] = yearly_axis[
                        (yearly_axis['Year'] >= start_year) & (yearly_axis['Year'] <= end_year)
                    ].copy()
        
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
        if opts['use_slippage']:
            if opts.get('use_mcap_slippage', True):
                print(f"   Slippage: ON (Large: {opts['slippage_bps']} bps, Mid: {opts.get('mid_cap_slippage_bps', 10.0)} bps, Small: {opts.get('small_cap_slippage_bps', 20.0)} bps)")
            else:
                print(f"   Slippage: ON ({opts['slippage_bps']} bps flat)")
        else:
            print(f"   Slippage: OFF")
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
        regime_allocations = self.regime_allocations or REGIME_ALLOCATIONS
        fallback_regime = 'Bear' if 'Bear' in regime_allocations else next(iter(regime_allocations), 'Bear')
        regime_returns_tracker = {regime: [] for regime in regime_allocations}

        # Get options for this backtest
        slippage_factor = opts['slippage_bps'] / 10000.0 if opts['use_slippage'] else 0.0
        top_n = opts['top_n']
        stop_loss_threshold = opts['stop_loss_threshold']
        small_cap_slippage_bps = max(
            float(opts.get('small_cap_slippage_bps', opts['slippage_bps'])),
            float(opts['slippage_bps']),
        )

        # Market-cap-based slippage for ALL factors (more realistic transaction costs)
        mcap_slippage_panel = None
        mcap_slippage_liquidity = None
        use_mcap_slippage = opts.get('use_mcap_slippage', True) and opts['use_slippage']
        mid_cap_slippage_bps = opts.get('mid_cap_slippage_bps', 10.0)

        if use_mcap_slippage:
            print("   Preparing market-cap panel for size-based slippage...")
            mcap_slippage_panel = build_size_market_cap_panel(self.close_df, trading_days, self.loader)
            mcap_slippage_liquidity = self.volume_df.reindex(index=trading_days, columns=self.close_df.columns)

        for i, date in enumerate(trading_days[:-1]):
            regime = regime_series_lagged.get(date, fallback_regime)
            if pd.isna(regime) or regime not in regime_allocations:
                regime = fallback_regime

            # Regime filter: if disabled, always use 100% allocation
            if opts['use_regime_filter']:
                allocation = regime_allocations.get(regime, 0.0)
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

                # Apply slippage (optional) - market-cap-based for all factors
                if opts['use_slippage'] and is_rebalance_day and old_selected:
                    new_positions = list(set(active_holdings) - old_selected)
                    if new_positions:
                        turnover = len(new_positions) / max(len(active_holdings), 1)

                        if (
                            use_mcap_slippage
                            and mcap_slippage_panel is not None
                            and date in mcap_slippage_panel.index
                        ):
                            mcaps = mcap_slippage_panel.loc[date]
                            liq = (
                                mcap_slippage_liquidity.loc[date]
                                if mcap_slippage_liquidity is not None and date in mcap_slippage_liquidity.index
                                else pd.Series(dtype=float)
                            )
                            _, small_caps, _ = get_size_buckets_for_date(
                                mcaps,
                                liq,
                                liquidity_quantile=SIZE_LIQUIDITY_QUANTILE,
                            )

                            # Determine slippage per stock based on market cap
                            def get_stock_slippage(ticker):
                                if ticker in small_caps:
                                    return small_cap_slippage_bps
                                # Check if mid-cap (between small and large)
                                if ticker in mcaps.index and not mcaps[ticker] != mcaps[ticker]:
                                    mcap_pct = mcaps.rank(pct=True).get(ticker, 0.5)
                                    if mcap_pct < 0.7:  # Bottom 70% but not small cap
                                        return mid_cap_slippage_bps
                                return opts['slippage_bps']

                            avg_bps = np.mean([get_stock_slippage(t) for t in new_positions])
                            stock_return -= turnover * (avg_bps / 10000.0) * 2
                        else:
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
        for regime in regime_allocations:
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

        # Factor-specific primary benchmark selection
        benchmark_name = "XU100"
        benchmark_returns = xu100_returns
        yearly_bench_col = "XU100_Return"
        yearly_excess_col = "Excess_vs_XU100"
        yearly_table_label = "XU100"
        
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

        # Save CAPM metrics (strategy vs primary benchmark)
        capm_metrics = compute_capm_metrics(returns, benchmark_returns) if benchmark_returns is not None else {
            "n_obs": 0,
            "alpha_daily": np.nan,
            "alpha_annual": np.nan,
            "beta": np.nan,
            "r_squared": np.nan,
            "correlation": np.nan,
            "residual_vol_annual": np.nan,
        }
        capm_metrics["benchmark"] = benchmark_name
        capm_df = pd.DataFrame([capm_metrics])
        capm_df.to_csv(output_dir / 'capm_metrics.csv', index=False)
        self.factor_capm[factor_name] = capm_metrics

        # Save rolling beta (daily) + yearly rolling beta summary
        yearly_rolling_beta = compute_yearly_rolling_beta_metrics(pd.Series(dtype=float))
        if benchmark_returns is not None:
            rolling_beta = compute_rolling_beta_series(
                strategy_returns=returns,
                market_returns=benchmark_returns,
                window=252,
                min_periods=126,
                risk_free_daily=0.0,
            )
            rolling_beta_df = rolling_beta.to_frame().reset_index()
            rolling_beta_df.columns = ['date', 'Rolling_Beta']
            rolling_beta_df.to_csv(output_dir / 'rolling_beta.csv', index=False)

            yearly_rolling_beta = compute_yearly_rolling_beta_metrics(rolling_beta)
            yearly_rolling_beta.to_csv(output_dir / 'yearly_rolling_beta.csv', index=False)
        else:
            pd.DataFrame(columns=['date', 'Rolling_Beta']).to_csv(output_dir / 'rolling_beta.csv', index=False)
            yearly_rolling_beta.to_csv(output_dir / 'yearly_rolling_beta.csv', index=False)

        self.factor_yearly_rolling_beta[factor_name] = yearly_rolling_beta.copy()

        # Save factor-specific yearly winner report (if provided)
        if 'yearly_axis_winners' in results:
            yearly_axis = results['yearly_axis_winners']
            if isinstance(yearly_axis, pd.DataFrame) and not yearly_axis.empty:
                yearly_axis.to_csv(output_dir / 'yearly_axis_winners.csv', index=False)

                with open(output_dir / 'yearly_axis_winners.txt', 'w') as f:
                    f.write("="*70 + "\n")
                    f.write("FIVE-FACTOR YEARLY AXIS WINNERS\n")
                    f.write("="*70 + "\n\n")
                    for year in sorted(yearly_axis['Year'].unique()):
                        f.write(f"{int(year)}\n")
                        f.write("-"*70 + "\n")
                        year_rows = yearly_axis[yearly_axis['Year'] == year].sort_values('Axis')
                        for _, row in year_rows.iterrows():
                            f.write(
                                f"{row['Axis']:<14} Winner: {row['Winner']:<12} | "
                                f"{row['High_Side']}: {row['High_Side_Return']:+.2%} | "
                                f"{row['Low_Side']}: {row['Low_Side_Return']:+.2%} | "
                                f"Spread: {row['Spread_Winner_Minus_Loser']:+.2%}\n"
                            )
                        f.write("\n")
        
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

        # Save detailed per-day diagnostics when provided
        if isinstance(results.get('returns_df'), pd.DataFrame) and not results['returns_df'].empty:
            results['returns_df'].to_csv(output_dir / 'returns_detailed.csv')

        # Save gold-arbitrage specific analytics when available
        if isinstance(results.get('gold_tracking_metrics'), pd.DataFrame) and not results['gold_tracking_metrics'].empty:
            results['gold_tracking_metrics'].to_csv(output_dir / 'gold_tracking_metrics.csv', index=False)

        if isinstance(results.get('gold_spread_zscores'), pd.DataFrame) and not results['gold_spread_zscores'].empty:
            results['gold_spread_zscores'].to_csv(output_dir / 'gold_spread_zscores.csv')

        if 'gold_selected_codes' in results:
            selected_codes = results.get('gold_selected_codes') or []
            with open(output_dir / 'gold_selected_codes.json', 'w', encoding='utf-8') as f:
                json.dump(selected_codes, f, ensure_ascii=False, indent=2)
        
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
            
            if benchmark_returns is not None:
                bench_aligned = benchmark_returns.dropna()
                if len(bench_aligned) > 0:
                    bench_total = (1 + bench_aligned).prod() - 1
                    f.write(f"\nBenchmark ({benchmark_name}) Return: {bench_total * 100:.2f}%\n")
                    f.write(f"Excess vs {benchmark_name}: {(results['total_return'] - bench_total) * 100:.2f}%\n")

            f.write(f"\nCAPM vs {benchmark_name}\n")
            f.write(f"Observations: {capm_metrics['n_obs']}\n")
            f.write(f"Beta: {capm_metrics['beta']:.4f}\n" if pd.notna(capm_metrics['beta']) else "Beta: NaN\n")
            f.write(
                f"Alpha (annualized): {capm_metrics['alpha_annual'] * 100:.2f}%\n"
                if pd.notna(capm_metrics['alpha_annual']) else "Alpha (annualized): NaN\n"
            )
            f.write(
                f"R-squared: {capm_metrics['r_squared']:.4f}\n"
                if pd.notna(capm_metrics['r_squared']) else "R-squared: NaN\n"
            )
            if benchmark_returns is not None:
                rb = compute_rolling_beta_series(returns, benchmark_returns, window=252, min_periods=126, risk_free_daily=0.0)
                rb_valid = rb.dropna()
                if not rb_valid.empty:
                    f.write(f"Latest Rolling Beta (252d): {rb_valid.iloc[-1]:.4f}\n")
            
            if benchmark_name != "XAU/TRY":
                xautry_aligned = xautry_returns.dropna()
                if len(xautry_aligned) > 0:
                    xautry_total = (1 + xautry_aligned).prod() - 1
                    f.write(f"\nBenchmark (XAU/TRY) Return: {xautry_total * 100:.2f}%\n")
                    f.write(f"Excess vs XAU/TRY: {(results['total_return'] - xautry_total) * 100:.2f}%\n")
        
        print(f"\n💾 Results saved to: {output_dir}")
        if pd.notna(capm_metrics.get("beta", np.nan)):
            print(
                f"   CAPM (vs {benchmark_name}) -> Beta: {capm_metrics['beta']:.3f}, "
                f"Alpha: {capm_metrics['alpha_annual']*100:.2f}%, "
                f"R²: {capm_metrics['r_squared']:.3f}"
            )
        
        # Print yearly summary
        print("\n" + "="*70)
        print("YEARLY RESULTS")
        print("="*70)
        print(f"{'Year':<6} {'Model':>10} {yearly_table_label:>10} {'Excess':>10}")
        print("-"*40)
        for _, row in yearly_metrics.iterrows():
            bench_ret = row[yearly_bench_col] if pd.notna(row[yearly_bench_col]) else 0
            excess = row[yearly_excess_col] if pd.notna(row[yearly_excess_col]) else row['Return']
            print(f"{int(row['Year']):<6} {row['Return']*100:>9.1f}% {bench_ret*100:>9.1f}% {excess*100:>9.1f}%")
    
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
        if self.factor_capm:
            self.save_capm_summary()
        if self.factor_yearly_rolling_beta:
            self.save_yearly_rolling_beta_summary()
        
        return results

    def save_capm_summary(self, output_dir=None):
        """Save CAPM summary across all factors."""
        if not self.factor_capm:
            print("⚠️  No CAPM results available")
            return

        if output_dir is None:
            output_dir = Path(__file__).parent / "results"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for factor_name, m in self.factor_capm.items():
            rows.append({
                "Factor": factor_name,
                "Benchmark": m.get("benchmark", "XU100"),
                "Observations": m.get("n_obs"),
                "Beta": m.get("beta"),
                "Alpha_Annual": m.get("alpha_annual"),
                "R_squared": m.get("r_squared"),
                "Correlation": m.get("correlation"),
                "ResidualVol_Annual": m.get("residual_vol_annual"),
            })

        df = pd.DataFrame(rows).sort_values("Factor")
        out_file = output_dir / "capm_summary.csv"
        df.to_csv(out_file, index=False)

        print("\n" + "=" * 70)
        print("CAPM SUMMARY (factor-specific benchmark)")
        print("=" * 70)
        for _, row in df.iterrows():
            beta = row["Beta"]
            alpha = row["Alpha_Annual"]
            r2 = row["R_squared"]
            bench = row["Benchmark"]
            print(
                f"{row['Factor']:<24} vs {bench:<7} beta={beta:>6.3f}  alpha={alpha*100:>7.2f}%  R²={r2:>6.3f}"
                if pd.notna(beta) and pd.notna(alpha) and pd.notna(r2)
                else f"{row['Factor']:<24} vs {bench:<7} beta=  NaN  alpha=   NaN  R²=  NaN"
            )
        print(f"\n💾 CAPM summary saved to: {out_file}")

    def save_yearly_rolling_beta_summary(self, output_dir=None):
        """Save consolidated yearly rolling-beta metrics across factors."""
        if not self.factor_yearly_rolling_beta:
            print("⚠️  No yearly rolling-beta results available")
            return

        if output_dir is None:
            output_dir = Path(__file__).parent / "results"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for factor_name, yearly_df in self.factor_yearly_rolling_beta.items():
            if yearly_df is None or yearly_df.empty:
                continue
            df = yearly_df.copy()
            df.insert(0, "Factor", factor_name)
            rows.append(df)

        if rows:
            summary_df = pd.concat(rows, ignore_index=True)
            summary_df = summary_df.sort_values(["Factor", "Year"]).reset_index(drop=True)
        else:
            summary_df = pd.DataFrame(columns=[
                "Factor",
                "Year",
                "Observations",
                "Beta_Start",
                "Beta_End",
                "Beta_Mean",
                "Beta_Median",
                "Beta_Min",
                "Beta_Max",
                "Beta_Std",
                "Beta_Change",
            ])

        out_file = output_dir / "yearly_rolling_beta_summary.csv"
        summary_df.to_csv(out_file, index=False)
        print(f"💾 Yearly rolling beta summary saved to: {out_file}")


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
    regime_model_dir_candidates = [
        bist_root / "Simple Regime Filter" / "outputs",
        bist_root / "Regime Filter" / "outputs",
    ]
    regime_model_dir = next((p for p in regime_model_dir_candidates if p.exists()), regime_model_dir_candidates[0])



    
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
