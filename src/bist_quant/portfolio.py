#!/usr/bin/env python3
"""
Portfolio Engine - BIST Quant

Comprehensive portfolio construction and backtesting engine.

Architecture:
- Uses BIST modular infrastructure (Backtester, RiskManager, DataLoader, DataManager)
- Preserves legacy factor workflow (`run_factor`, `run_all_factors`)
- Adds Phase 4 portfolio APIs consolidated from bist-quant-ai
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from bist_quant.common.backtester import (
    Backtester,
    identify_monthly_rebalance_days,
    identify_quarterly_rebalance_days,
)
from bist_quant.common.config_manager import (
    DEFAULT_PORTFOLIO_OPTIONS as CONFIG_MANAGER_DEFAULT_PORTFOLIO_OPTIONS,
    LIQUIDITY_QUANTILE,
    REGIME_ALLOCATIONS,
    TARGET_DOWNSIDE_VOL,
    VOL_CAP,
    VOL_FLOOR,
    VOL_LOOKBACK,
    ConfigManager,
)
from bist_quant.common.config_manager import (
    load_signal_configs as load_configs_from_manager,
)
from bist_quant.common.data_loader import DataLoader
from bist_quant.common.data_manager import DataManager
from bist_quant.common.enums import RegimeLabel
from bist_quant.common.panel_cache import PanelCache
from bist_quant.common.report_generator import ReportGenerator
from bist_quant.common.risk_manager import RiskManager
from bist_quant.signals.factory import build_signal, get_available_signals
from bist_quant.signals.size_rotation_signals import (
    build_market_cap_panel as build_size_market_cap_panel,
)

logger = logging.getLogger(__name__)

try:
    from bist_quant.regime.simple_regime import RegimeClassifier as _RegimeClassifier
except Exception:
    _RegimeClassifier = None

try:
    from bist_quant.regime.simple_regime import SimpleRegimeClassifier as _SimpleRegimeClassifier
except Exception:
    _SimpleRegimeClassifier = None

HAS_REGIME_FILTER = _RegimeClassifier is not None or _SimpleRegimeClassifier is not None


def _as_regime_name(value: Any) -> str:
    if isinstance(value, RegimeLabel):
        return value.value
    if isinstance(value, str):
        text = value.strip()
        if text:
            for canonical in ("Bull", "Recovery", "Stress", "Bear"):
                if text.lower() == canonical.lower():
                    return canonical
            return text
    return "Bull"


def _deep_merge_dict(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(base)
    if not override:
        return merged
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


_BASE_REGIME_ALLOCATIONS = {
    _as_regime_name(regime): float(allocation) for regime, allocation in REGIME_ALLOCATIONS.items()
}


# =============================================================================
# DEFAULT CONFIGURATION (Phase 4 consolidation)
# =============================================================================
DEFAULT_PORTFOLIO_OPTIONS: Dict[str, Any] = _deep_merge_dict(
    CONFIG_MANAGER_DEFAULT_PORTFOLIO_OPTIONS,
    {
        # Lookback and rebalancing
        "lookback_days": 252,
        "rebalance_frequency": "monthly",  # daily, weekly, monthly, quarterly
        # Risk management
        "volatility_target": float(
            CONFIG_MANAGER_DEFAULT_PORTFOLIO_OPTIONS.get("target_downside_vol", TARGET_DOWNSIDE_VOL)
        ),
        "use_inverse_vol_sizing": True,
        "stop_loss_threshold": 0.15,
        "max_drawdown_exit": -0.25,
        # Position limits
        "max_position_size": 0.10,
        "min_position_size": 0.01,
        "max_sector_exposure": 0.30,
        # Transaction costs
        "transaction_cost_bps": 10.0,
        "slippage_model": "market_cap_based",  # fixed, market_cap_based, volume_based
        # Liquidity filters
        "min_market_cap": 1e8,  # 100M TRY
        "min_avg_volume": 1e6,  # 1M TRY daily volume
        # Regime-based allocation
        "use_regime_filter": True,
        "regime_allocations": _BASE_REGIME_ALLOCATIONS.copy(),
        "regime_hedge_asset": "XAU",
        # Factor weights (default equal weight)
        "factor_weights": {},
    },
)


# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class PortfolioResult:
    """Backtest output container returned by portfolio execution APIs.

    Attributes:
        returns: Daily strategy return series indexed by trading date.
        positions: Daily position weights (Date x Ticker).
        turnover: Daily turnover ratios.
        transaction_costs: Daily estimated transaction costs.
        regime_history: Optional inferred regime series aligned to returns.
        metrics: Summary statistics such as Sharpe ratio or max drawdown.
    """

    returns: pd.Series
    positions: pd.DataFrame
    turnover: pd.Series
    transaction_costs: pd.Series
    regime_history: Optional[pd.Series] = None
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class SignalResult:
    """Container for generated factor signals and metadata.

    Attributes:
        signals: Signal matrix indexed by date with ticker columns.
        metadata: Optional strategy metadata attached during signal creation.
    """

    signals: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)


def apply_downside_vol_targeting(
    returns: pd.Series,
    target_vol: float = TARGET_DOWNSIDE_VOL,
    lookback: int = VOL_LOOKBACK,
    vol_floor: float = VOL_FLOOR,
    vol_cap: float = VOL_CAP,
) -> pd.Series:
    """
    Apply downside volatility targeting to scale returns.
    """
    if len(returns) < lookback:
        return returns

    min_periods = lookback // 2
    negative_only = returns.where(returns < 0.0)
    total_counts = returns.rolling(lookback, min_periods=min_periods).count()
    negative_counts = negative_only.rolling(lookback, min_periods=1).count()
    rolling_downside_vol = negative_only.rolling(lookback, min_periods=1).std() * np.sqrt(252)
    rolling_downside_vol = rolling_downside_vol.where(
        (total_counts >= min_periods) & (negative_counts > 2)
    )

    leverage = target_vol / rolling_downside_vol.shift(1)
    leverage = leverage.clip(lower=vol_floor, upper=vol_cap).fillna(1.0)
    return returns * leverage


# ============================================================================
# CONFIG LOADING
# ============================================================================
def load_signal_configs(prefer_yaml: bool = True) -> dict:
    """
    Backward-compatible signal config loader shim.

    Delegates to common.config_manager.ConfigManager.
    """
    return load_configs_from_manager(prefer_yaml=prefer_yaml)


# ============================================================================
# PORTFOLIO ENGINE
# ============================================================================
class PortfolioEngine:
    """Coordinate data loading, signal building, and backtest execution.

    Args:
        data_dir: Optional market data root directory.
        regime_model_dir: Optional regime model/output directory.
        start_date: Backtest start date (inclusive).
        end_date: Backtest end date (inclusive).
        options: Portfolio and risk control options.
        data_loader: Optional preconfigured :class:`DataLoader`.
        backtester: Optional preconfigured :class:`Backtester`.
        risk_manager: Optional preconfigured :class:`RiskManager`.

    Attributes:
        options: Resolved runtime portfolio options.
        signal_configs: Loaded strategy configuration map.
        data_loader: Shared loader used by engine workflows.
        backtester: Backtesting service used for factor simulations.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        regime_model_dir: Optional[Path] = None,
        start_date: str = "2018-01-01",
        end_date: str = "2024-12-31",
        options: Optional[Dict[str, Any]] = None,
        data_loader: Optional[DataLoader] = None,
        backtester: Optional[Backtester] = None,
        risk_manager: Optional[RiskManager] = None,
    ):
        # Positional compatibility: PortfolioEngine(options={...})
        if isinstance(data_dir, dict) and options is None and regime_model_dir is None:
            options = data_dir
            data_dir = None

        script_dir = Path(__file__).parent
        bist_root = script_dir.parent.parent.parent
        default_data_dir = bist_root / "data"
        regime_candidates = [
            bist_root / "outputs" / "regime" / "simple_regime",
            bist_root / "outputs" / "regime",
            bist_root / "regime_filter" / "outputs",
            bist_root / "Simple Regime Filter" / "outputs",
            bist_root / "Regime Filter" / "outputs",
        ]

        resolved_data_dir = Path(data_dir) if data_dir is not None else default_data_dir
        if regime_model_dir is None:
            resolved_regime_model_dir = next(
                (p for p in regime_candidates if p.exists()),
                regime_candidates[0],
            )
        else:
            resolved_regime_model_dir = Path(regime_model_dir)

        self.data_dir = resolved_data_dir
        self.regime_model_dir = resolved_regime_model_dir
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.options = _deep_merge_dict(DEFAULT_PORTFOLIO_OPTIONS, options)

        # Load signal configurations
        self.signal_configs = load_signal_configs()
        logger.info(f"\nLoaded {len(self.signal_configs)} signal configurations")

        # Initialize data loader
        self.loader = data_loader or DataLoader(self.data_dir, self.regime_model_dir)
        self.data_loader = self.loader
        if not hasattr(self.loader, "panel_cache") or self.loader.panel_cache is None:
            self.loader.panel_cache = PanelCache()
        self.panel_cache = self.loader.panel_cache

        self.data_manager = DataManager(
            data_loader=self.loader,
            data_dir=self.data_dir,
            base_regime_allocations=REGIME_ALLOCATIONS,
        )

        self.risk_manager = (
            risk_manager or getattr(backtester, "risk_manager", None) or RiskManager()
        )
        self.backtester = backtester or Backtester(
            loader=self.loader,
            data_dir=self.data_dir,
            risk_manager=self.risk_manager,
            build_size_market_cap_panel=build_size_market_cap_panel,
        )
        self.report_generator = ReportGenerator(
            models_dir=Path(__file__).parent,
            data_dir=self.data_dir,
            loader=self.loader,
        )

        # Cached data
        self.prices = None
        self.close_df = None
        self.open_df = None
        self.volume_df = None
        self.regime_series = None
        self.regime_allocations = _BASE_REGIME_ALLOCATIONS.copy()
        self.xautry_prices = None
        self.xu100_prices = None
        self.fundamentals = None

        # Additional caches for Phase 4 APIs
        self._price_cache: Optional[pd.DataFrame] = None
        self._fundamental_cache: Optional[pd.DataFrame] = None

        # Store factor returns for correlation analysis
        self.factor_returns = {}
        self.factor_capm = {}
        self.factor_yearly_rolling_beta = {}

        # Optional regime classifier
        self.regime_classifier = None
        if HAS_REGIME_FILTER and self.options.get("use_regime_filter", True):
            try:
                if _RegimeClassifier is not None:
                    self.regime_classifier = _RegimeClassifier()
                elif _SimpleRegimeClassifier is not None:
                    self.regime_classifier = _SimpleRegimeClassifier()
            except Exception as exc:
                logger.debug(f"Regime classifier initialization skipped: {exc}")

    # -------------------------------------------------------------------------
    # Data Loading
    # -------------------------------------------------------------------------
    def _apply_loaded_data(self, loaded_data) -> None:
        self.prices = loaded_data.prices
        self.close_df = loaded_data.close_df
        self.open_df = loaded_data.open_df
        self.volume_df = loaded_data.volume_df
        self.fundamentals = loaded_data.fundamentals
        self.regime_series = loaded_data.regime_series
        self.regime_allocations = {
            _as_regime_name(k): float(v) for k, v in loaded_data.regime_allocations.items()
        }
        self.xautry_prices = loaded_data.xautry_prices
        self.xu100_prices = loaded_data.xu100_prices

        self.backtester.update_data(
            prices=self.prices,
            close_df=self.close_df,
            volume_df=self.volume_df,
            regime_series=self.regime_series,
            regime_allocations=loaded_data.regime_allocations,
            xu100_prices=self.xu100_prices,
            xautry_prices=self.xautry_prices,
        )

    def load_all_data(self, use_cache: bool = True):
        """Load all required datasets and cache prepared data panels."""
        loaded_data = self.data_manager.load_all(use_cache=use_cache)
        self._apply_loaded_data(loaded_data)

    def _build_market_cap_panel(
        self,
        dates: pd.DatetimeIndex,
        symbols: pd.Index,
    ) -> pd.DataFrame:
        if self.close_df is None or dates.empty:
            return pd.DataFrame(index=dates, columns=symbols, dtype=float)

        try:
            panel = build_size_market_cap_panel(self.close_df, dates, self.loader)
        except Exception as exc:
            logger.warning(f"Market-cap panel unavailable, using NaN fallback: {exc}")
            panel = pd.DataFrame(index=dates, columns=symbols, dtype=float)
        return panel.reindex(index=dates, columns=symbols).apply(pd.to_numeric, errors="coerce")

    def load_data(
        self,
        start_date: str,
        end_date: str,
        symbols: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Load price/fundamental/market-cap panels for backtesting."""
        if self.close_df is None or self.volume_df is None:
            self.load_all_data(use_cache=True)

        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        prices = self.close_df.loc[
            (self.close_df.index >= start_ts) & (self.close_df.index <= end_ts)
        ].copy()
        if symbols:
            selected = [s for s in symbols if s in prices.columns]
            prices = prices[selected]

        volumes = (
            self.volume_df.reindex(index=prices.index, columns=prices.columns).copy()
            if self.volume_df is not None
            else pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)
        )

        market_caps = self._build_market_cap_panel(prices.index, prices.columns)

        fundamentals = self.fundamentals
        if isinstance(fundamentals, pd.DataFrame):
            fundamentals = fundamentals.reindex(index=prices.index, columns=prices.columns)
        else:
            fundamentals = pd.DataFrame(index=prices.index, columns=prices.columns, dtype=float)

        return {
            "prices": prices,
            "fundamentals": fundamentals,
            "market_caps": market_caps,
            "volumes": volumes,
        }

    # -------------------------------------------------------------------------
    # Internal Signal/Config Helpers
    # -------------------------------------------------------------------------
    def _build_runtime_config(self, config: dict | None) -> dict:
        runtime_config = dict(config) if isinstance(config, dict) else {}
        runtime_config["_runtime_context"] = self.data_manager.build_runtime_context()
        return runtime_config

    def _resolve_factor_config(self, factor_name: str, override_config: dict | None) -> dict:
        if override_config:
            if not isinstance(override_config, dict):
                raise TypeError(
                    f"override_config must be dict or None, got {type(override_config).__name__}"
                )
            return override_config

        config = self.signal_configs.get(factor_name)
        if not isinstance(config, dict):
            raise ValueError(f"No config found for factor: {factor_name}")
        return config

    def _resolve_factor_timeline(
        self,
        factor_name: str,
        config: dict,
    ) -> tuple[pd.Timestamp, pd.Timestamp, bool]:
        timeline = config.get("timeline", {})
        if not isinstance(timeline, dict):
            timeline = {}

        custom_start = timeline.get("start_date")
        custom_end = timeline.get("end_date")
        factor_start_date = pd.Timestamp(custom_start) if custom_start else self.start_date
        factor_end_date = pd.Timestamp(custom_end) if custom_end else self.end_date

        walk_forward_cfg = config.get("walk_forward", {})
        if not isinstance(walk_forward_cfg, dict):
            walk_forward_cfg = {}
        if factor_name == "five_factor_rotation" and walk_forward_cfg.get("enabled", False):
            first_test_year = walk_forward_cfg.get("first_test_year")
            last_test_year = walk_forward_cfg.get("last_test_year")
            if first_test_year is not None:
                wf_start = pd.Timestamp(year=int(first_test_year), month=1, day=1)
                if factor_start_date < wf_start:
                    logger.info(
                        f"Walk-forward start clamp: {factor_start_date.date()} -> {wf_start.date()}"
                    )
                    factor_start_date = wf_start
            if last_test_year is not None:
                wf_end = pd.Timestamp(year=int(last_test_year), month=12, day=31)
                if factor_end_date > wf_end:
                    logger.info(
                        f"Walk-forward end clamp: {factor_end_date.date()} -> {wf_end.date()}"
                    )
                    factor_end_date = wf_end

        has_custom_timeline = bool(custom_start or custom_end)
        return factor_start_date, factor_end_date, has_custom_timeline

    def _to_legacy_portfolio_options(self, portfolio_options: dict | None) -> dict:
        """
        Convert Phase 4 unified options to legacy Backtester option keys.
        """
        merged = _deep_merge_dict(self.options, portfolio_options)

        legacy = CONFIG_MANAGER_DEFAULT_PORTFOLIO_OPTIONS.copy()
        for key in legacy:
            if key in merged:
                legacy[key] = merged[key]

        legacy["target_downside_vol"] = float(
            merged.get(
                "target_downside_vol",
                merged.get("volatility_target", legacy["target_downside_vol"]),
            )
        )
        legacy["vol_lookback"] = int(
            merged.get("vol_lookback", merged.get("lookback_days", legacy["vol_lookback"]))
        )
        legacy["max_position_weight"] = float(
            merged.get(
                "max_position_weight",
                merged.get("max_position_size", legacy["max_position_weight"]),
            )
        )
        legacy["slippage_bps"] = float(
            merged.get("slippage_bps", merged.get("transaction_cost_bps", legacy["slippage_bps"]))
        )
        legacy["stop_loss_threshold"] = abs(
            float(merged.get("stop_loss_threshold", legacy["stop_loss_threshold"]))
        )
        legacy["liquidity_quantile"] = float(
            merged.get("liquidity_quantile", legacy["liquidity_quantile"])
        )
        legacy["inverse_vol_lookback"] = int(
            merged.get(
                "inverse_vol_lookback", merged.get("lookback_days", legacy["inverse_vol_lookback"])
            )
        )
        legacy["top_n"] = int(merged.get("top_n", legacy["top_n"]))
        legacy["signal_lag_days"] = int(merged.get("signal_lag_days", legacy["signal_lag_days"]))
        return legacy

    def _resolve_portfolio_options(self, portfolio_options: dict | None) -> dict:
        return self._to_legacy_portfolio_options(portfolio_options)

    def _print_portfolio_settings(self, opts: dict) -> None:
        self.risk_manager.print_settings(opts)

    def _build_signals_for_factor(self, factor_name: str, dates: pd.DatetimeIndex, config: dict):
        """Build factor signal panel using signals.factory dispatch."""
        from bist_quant.signals.signal_exporter import SignalExporter

        runtime_config = self._build_runtime_config(config)
        signals = build_signal(factor_name, dates, self.loader, runtime_config)
        factor_details = runtime_config.get("_factor_details", {})

        # Intercept and export the raw signals before returning them
        try:
            exporter = SignalExporter(signal_name=factor_name)
            exporter.export_factor_scores(signals, filename="raw_scores.csv")
            
            # If the strategy generated detailed intermediate scores (like value_scores, momentum_scores),
            # we can export them too from the factor_details.
            if isinstance(factor_details, dict):
                # Search for components ending in '_scores' or typical DataFrames that might be raw factor values
                for key, val in factor_details.items():
                    if key.endswith("_scores") and isinstance(val, (pd.DataFrame, pd.Series, dict)):
                        exporter.export_factor_scores(val, filename=f"{key}.csv")
        except Exception as e:
            logger.warning(f"Failed to export raw factor scores for {factor_name}: {e}")

        return signals, factor_details

    # -------------------------------------------------------------------------
    # Legacy Factor Workflow (preserved)
    # -------------------------------------------------------------------------
    def run_factor(self, factor_name: str, override_config: dict = None):
        """Run backtest for a single factor using its config."""
        logger.info("\n" + "=" * 70)
        logger.info(f"RUNNING {factor_name.upper()} FACTOR")
        logger.info("=" * 70)

        config = self._resolve_factor_config(factor_name, override_config)

        if not config.get("enabled", True):
            logger.warning(f"{factor_name.upper()} is disabled in config")
            return None

        rebalance_freq = config.get("rebalance_frequency", "quarterly")
        logger.info(f"Rebalancing frequency: {rebalance_freq}")

        factor_start_date, factor_end_date, has_custom_timeline = self._resolve_factor_timeline(
            factor_name,
            config,
        )
        if has_custom_timeline:
            logger.info(f"Custom timeline: {factor_start_date.date()} to {factor_end_date.date()}")

        if self.close_df is None:
            self.load_all_data()

        start_time = time.time()
        dates = self.close_df.index
        signals, factor_details = self._build_signals_for_factor(factor_name, dates, config)
        portfolio_options = config.get("portfolio_options", {})

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
            if "yearly_axis_winners" in results and isinstance(
                results["yearly_axis_winners"], pd.DataFrame
            ):
                yearly_axis = results["yearly_axis_winners"]
                if not yearly_axis.empty and "Year" in yearly_axis.columns:
                    start_year = int(factor_start_date.year)
                    end_year = int(factor_end_date.year)
                    results["yearly_axis_winners"] = yearly_axis[
                        (yearly_axis["Year"] >= start_year) & (yearly_axis["Year"] <= end_year)
                    ].copy()

        self.save_results(results, factor_name)
        self.factor_returns[factor_name] = results["returns"]

        elapsed = time.time() - start_time
        logger.info(f"\n{factor_name.upper()} completed in {elapsed:.1f} seconds")
        return results

    def _run_backtest(
        self,
        signals: pd.DataFrame,
        factor_name: str,
        rebalance_freq: str = "quarterly",
        start_date: pd.Timestamp = None,
        end_date: pd.Timestamp = None,
        portfolio_options: dict = None,
    ):
        """Run backtest through the modular Backtester service."""
        resolved_options = self._resolve_portfolio_options(portfolio_options)
        return self.backtester.run(
            signals=signals,
            factor_name=factor_name,
            rebalance_freq=rebalance_freq,
            start_date=start_date if start_date is not None else self.start_date,
            end_date=end_date if end_date is not None else self.end_date,
            portfolio_options=resolved_options,
        )

    # -------------------------------------------------------------------------
    # Phase 4 Regime Integration
    # -------------------------------------------------------------------------
    def get_regime(self, date: pd.Timestamp) -> str:
        """Get market regime for a given date."""
        ts = pd.Timestamp(date)

        if isinstance(self.regime_series, pd.Series) and not self.regime_series.empty:
            if ts in self.regime_series.index:
                return _as_regime_name(self.regime_series.loc[ts])
            candidates = self.regime_series.index[self.regime_series.index <= ts]
            if len(candidates) > 0:
                return _as_regime_name(self.regime_series.loc[candidates.max()])

        if self.regime_classifier is not None:
            try:
                if hasattr(self.regime_classifier, "predict"):
                    predicted = self.regime_classifier.predict(ts)
                    if predicted is not None:
                        return _as_regime_name(predicted)
            except Exception:
                pass

        return "Bull"

    def apply_regime_allocation(
        self,
        positions: pd.DataFrame,
        regime: str,
    ) -> pd.DataFrame:
        """Apply regime-based position scaling."""
        allocations = self.options.get("regime_allocations", {})
        regime_key = _as_regime_name(regime)
        allocation = float(allocations.get(regime_key, allocations.get(str(regime), 1.0)))
        return positions * allocation

    # -------------------------------------------------------------------------
    # Phase 4 Risk Management
    # -------------------------------------------------------------------------
    def apply_volatility_targeting(
        self,
        positions: pd.DataFrame,
        returns: pd.DataFrame,
        target_vol: Optional[float] = None,
    ) -> pd.DataFrame:
        """Scale positions to target portfolio volatility."""
        target_vol = float(target_vol or self.options.get("volatility_target", TARGET_DOWNSIDE_VOL))
        lookback = int(self.options.get("lookback_days", 252))
        min_periods = max(20, lookback // 4)

        portfolio_returns = (positions.shift(1).fillna(0.0) * returns.fillna(0.0)).sum(axis=1)
        realized_vol = portfolio_returns.rolling(lookback, min_periods=min_periods).std() * np.sqrt(
            252
        )

        vol_floor = float(self.options.get("vol_floor", 0.10))
        vol_cap = float(self.options.get("vol_cap", 2.0))
        scale = target_vol / realized_vol.shift(1).clip(lower=0.01)
        scale = scale.clip(lower=vol_floor, upper=max(vol_cap, vol_floor)).fillna(1.0)
        return positions.multiply(scale, axis=0)

    def apply_inverse_vol_sizing(
        self,
        signals: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """Weight signals by inverse downside volatility."""
        lookback = int(
            self.options.get(
                "inverse_vol_lookback",
                self.options.get("lookback_days", 60),
            )
        )
        min_periods = max(5, lookback // 3)

        downside_returns = returns.where(returns < 0.0)
        downside_vol = downside_returns.rolling(lookback, min_periods=min_periods).std() * np.sqrt(
            252
        )
        inverse_vol = 1.0 / downside_vol.clip(lower=0.01)
        inverse_vol = inverse_vol.replace([np.inf, -np.inf], np.nan)

        weights = inverse_vol.div(inverse_vol.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
        sized = signals * weights
        gross = sized.abs().sum(axis=1).replace(0.0, np.nan)
        return sized.div(gross, axis=0).fillna(0.0)

    def apply_stop_loss(
        self,
        positions: pd.DataFrame,
        cumulative_returns: pd.Series,
        threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """Reduce positions when drawdown exceeds threshold."""
        threshold = float(
            threshold if threshold is not None else self.options.get("stop_loss_threshold", 0.15)
        )
        trigger = -abs(threshold)

        wealth = cumulative_returns.replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)
        rolling_max = wealth.cummax().replace(0.0, np.nan).ffill().fillna(1.0)
        drawdown = wealth / rolling_max - 1.0
        stop_mask = drawdown < trigger

        adjusted = positions.copy()
        adjusted.loc[stop_mask] = 0.0
        return adjusted

    def apply_position_limits(
        self,
        positions: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply position size limits."""
        max_size = float(self.options.get("max_position_size", 0.10))
        min_size = float(self.options.get("min_position_size", 0.01))

        limited = positions.clip(lower=-max_size, upper=max_size)
        limited = limited.where(limited.abs() >= min_size, 0.0)

        gross = limited.abs().sum(axis=1)
        limited = limited.div(gross.clip(lower=1.0), axis=0).fillna(0.0)
        return limited

    # -------------------------------------------------------------------------
    # Phase 4 Transaction Costs
    # -------------------------------------------------------------------------
    def calculate_transaction_costs(
        self,
        positions: pd.DataFrame,
        market_caps: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Calculate transaction costs including optional market-cap slippage."""
        turnover = positions.diff().abs().sum(axis=1).fillna(0.0)
        if not self.options.get("use_slippage", True):
            return pd.Series(0.0, index=turnover.index)

        base_cost_bps = float(
            self.options.get(
                "transaction_cost_bps",
                self.options.get("slippage_bps", 5.0),
            )
        )
        slippage_model = str(self.options.get("slippage_model", "fixed")).lower()

        if slippage_model == "fixed" or market_caps is None or market_caps.empty:
            total_cost_bps = pd.Series(base_cost_bps, index=turnover.index)
        elif slippage_model == "market_cap_based":
            aligned_caps = market_caps.reindex(index=positions.index, columns=positions.columns)
            abs_positions = positions.abs()
            denom = abs_positions.sum(axis=1).replace(0.0, np.nan)
            weighted_mcap = (aligned_caps * abs_positions).sum(axis=1) / denom
            weighted_mcap = weighted_mcap.fillna(aligned_caps.median(axis=1))

            large_cap_threshold = float(self.options.get("large_cap_threshold", 1e10))
            small_cap_threshold = float(self.options.get("min_market_cap", 1e8))
            if large_cap_threshold <= small_cap_threshold:
                large_cap_threshold = small_cap_threshold + 1.0

            clipped_mcap = weighted_mcap.clip(lower=small_cap_threshold, upper=large_cap_threshold)
            slippage_mult = 3.0 - 2.0 * (
                (clipped_mcap - small_cap_threshold) / (large_cap_threshold - small_cap_threshold)
            )
            total_cost_bps = base_cost_bps * slippage_mult
            total_cost_bps = total_cost_bps.fillna(base_cost_bps)
        elif slippage_model == "volume_based":
            total_cost_bps = pd.Series(base_cost_bps, index=turnover.index)
            total_cost_bps = total_cost_bps + turnover * base_cost_bps
        else:
            total_cost_bps = pd.Series(base_cost_bps, index=turnover.index)

        return (turnover * (total_cost_bps / 10000.0)).fillna(0.0)

    # -------------------------------------------------------------------------
    # Phase 4 Liquidity / Signal Combination
    # -------------------------------------------------------------------------
    def apply_liquidity_filter(
        self,
        signals: pd.DataFrame,
        market_caps: pd.DataFrame,
        volumes: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Filter out illiquid securities."""
        if signals.empty:
            return signals
        if market_caps is None or market_caps.empty:
            return signals

        min_mcap = float(self.options.get("min_market_cap", 1e8))
        min_volume = float(self.options.get("min_avg_volume", 1e6))

        aligned_mcap = market_caps.reindex(index=signals.index, columns=signals.columns)
        mask = aligned_mcap >= min_mcap

        if volumes is None:
            volumes = self.volume_df
        if isinstance(volumes, pd.DataFrame) and not volumes.empty:
            aligned_vol = volumes.reindex(index=signals.index, columns=signals.columns)
            volume_mask = aligned_vol.rolling(20, min_periods=5).mean() >= min_volume
            mask = mask & volume_mask

        return signals.where(mask, 0.0).fillna(0.0)

    def combine_signals(
        self,
        signal_dict: Dict[str, pd.DataFrame],
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """Combine multiple signals with optional weighting."""
        if not signal_dict:
            raise ValueError("signal_dict is empty")

        weights = weights or self.options.get("factor_weights", {})
        if not weights:
            equal = 1.0 / len(signal_dict)
            weights = {name: equal for name in signal_dict}

        total_abs_weight = sum(abs(float(weights.get(name, 0.0))) for name in signal_dict)
        if total_abs_weight <= 0:
            total_abs_weight = float(len(signal_dict))
            weights = {name: 1.0 for name in signal_dict}

        combined = None
        for name, signal in signal_dict.items():
            weight = float(weights.get(name, 0.0)) / total_abs_weight
            weighted = signal * weight
            combined = weighted if combined is None else combined.add(weighted, fill_value=0.0)

        return combined.fillna(0.0)

    # -------------------------------------------------------------------------
    # Phase 4 Main Backtest API
    # -------------------------------------------------------------------------
    def _identify_rebalance_days(
        self,
        trading_days: pd.DatetimeIndex,
        frequency: str,
    ) -> set[pd.Timestamp]:
        freq = str(frequency).lower()
        if freq == "daily":
            return set(trading_days)
        if freq == "weekly":
            iso = trading_days.isocalendar()
            frame = pd.DataFrame(
                {
                    "date": trading_days,
                    "year": iso.year.to_numpy(),
                    "week": iso.week.to_numpy(),
                }
            )
            first_of_week = frame.groupby(["year", "week"])["date"].first()
            return set(first_of_week.values)
        if freq == "monthly":
            return identify_monthly_rebalance_days(trading_days)
        return identify_quarterly_rebalance_days(trading_days)

    def _apply_rebalance_schedule(
        self,
        positions: pd.DataFrame,
        frequency: str,
    ) -> pd.DataFrame:
        if positions.empty:
            return positions

        rebalance_days = self._identify_rebalance_days(positions.index, frequency)
        out = pd.DataFrame(0.0, index=positions.index, columns=positions.columns)
        current = pd.Series(0.0, index=positions.columns)

        for date in positions.index:
            if date in rebalance_days:
                current = positions.loc[date].fillna(0.0)
            out.loc[date] = current
        return out

    def run_backtest(
        self,
        signals: Union[List[str], Dict[str, pd.DataFrame], pd.DataFrame],
        start_date: str,
        end_date: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> PortfolioResult:
        """
        Run portfolio backtest with given signals.
        """
        runtime_options = _deep_merge_dict(self.options, options)
        original_options = self.options
        self.options = runtime_options

        try:
            data = self.load_data(start_date, end_date)
            prices = data["prices"]
            fundamentals = data["fundamentals"]
            market_caps = data["market_caps"]
            volumes = data["volumes"]

            if prices.empty:
                raise ValueError("No price data available for requested backtest window")

            returns = (
                prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            )

            if isinstance(signals, pd.DataFrame):
                combined_signals = signals.copy()
            elif isinstance(signals, dict):
                combined_signals = self.combine_signals(signals)
            else:
                signal_dict = {}
                for signal_name in signals:
                    signal_dict[signal_name] = self._load_signal(signal_name, prices, fundamentals)
                combined_signals = self.combine_signals(signal_dict)

            combined_signals = combined_signals.reindex(
                index=prices.index, columns=prices.columns
            ).fillna(0.0)

            if runtime_options.get("use_liquidity_filter", True):
                combined_signals = self.apply_liquidity_filter(
                    combined_signals, market_caps, volumes
                )

            positions = self._signals_to_positions(combined_signals)
            positions = self._apply_rebalance_schedule(
                positions,
                str(runtime_options.get("rebalance_frequency", "monthly")),
            )

            if runtime_options.get("use_inverse_vol_sizing", True):
                positions = self.apply_inverse_vol_sizing(positions, returns)

            positions = self.apply_position_limits(positions)

            gross_returns_pre = (positions.shift(1).fillna(0.0) * returns).sum(axis=1)
            cumulative_returns = (1.0 + gross_returns_pre).cumprod()

            if runtime_options.get("use_stop_loss", True):
                positions = self.apply_stop_loss(
                    positions,
                    cumulative_returns,
                    threshold=runtime_options.get("stop_loss_threshold"),
                )

            regime_history = None
            if runtime_options.get("use_regime_filter", True):
                regime_history = pd.Series(index=positions.index, dtype=object)
                for date in positions.index:
                    regime = self.get_regime(date)
                    regime_history.at[date] = regime
                    positions.loc[date] = self.apply_regime_allocation(
                        positions.loc[[date]],
                        regime,
                    ).iloc[0]

            if runtime_options.get("use_vol_targeting", True):
                target_vol = float(
                    runtime_options.get(
                        "volatility_target",
                        runtime_options.get("target_downside_vol", TARGET_DOWNSIDE_VOL),
                    )
                )
                positions = self.apply_volatility_targeting(
                    positions, returns, target_vol=target_vol
                )

            positions = self.apply_position_limits(positions)
            gross_returns = (positions.shift(1).fillna(0.0) * returns).sum(axis=1)
            transaction_costs = self.calculate_transaction_costs(positions, market_caps)
            net_returns = (gross_returns - transaction_costs).fillna(0.0)
            turnover = positions.diff().abs().sum(axis=1).fillna(0.0)

            metrics = self._calculate_metrics(net_returns, gross_returns, turnover)

            return PortfolioResult(
                returns=net_returns,
                positions=positions,
                turnover=turnover,
                transaction_costs=transaction_costs,
                regime_history=regime_history,
                metrics=metrics,
            )
        finally:
            self.options = original_options

    # -------------------------------------------------------------------------
    # Phase 4 Helper Methods
    # -------------------------------------------------------------------------
    def _signals_to_positions(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Convert raw signals to long-only normalized positions."""
        cleaned = signals.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        long_only = cleaned.clip(lower=0.0)
        row_sums = long_only.sum(axis=1).replace(0.0, np.nan)
        return long_only.div(row_sums, axis=0).fillna(0.0)

    def _load_signal(
        self,
        signal_name: str,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame,
    ) -> pd.DataFrame:
        """Load a signal by name."""
        try:
            return self._load_signal_from_config(signal_name, prices, fundamentals)
        except Exception:
            pass

        if self.close_df is None:
            self.load_all_data(use_cache=True)

        runtime_config = self._build_runtime_config({})
        signal_df = build_signal(signal_name, prices.index, self.loader, runtime_config)
        return signal_df.reindex(index=prices.index, columns=prices.columns).fillna(0.0)

    def _load_signal_from_config(
        self,
        signal_name: str,
        prices: pd.DataFrame,
        fundamentals: pd.DataFrame,
    ) -> pd.DataFrame:
        """Load signal using config manager + signals factory."""
        config = self.signal_configs.get(signal_name)
        if not isinstance(config, dict):
            manager = ConfigManager.from_default_paths()
            config = manager.load_signal_configs().get(signal_name)
        if not isinstance(config, dict):
            raise ValueError(f"Signal {signal_name} not found in loaded configs")

        runtime_config = self._build_runtime_config(config)
        signal_df = build_signal(signal_name, prices.index, self.loader, runtime_config)
        return signal_df.reindex(index=prices.index, columns=prices.columns).fillna(0.0)

    def _calculate_metrics(
        self,
        net_returns: pd.Series,
        gross_returns: pd.Series,
        turnover: pd.Series,
    ) -> Dict[str, float]:
        """Calculate performance metrics."""
        clean_net = net_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        clean_gross = gross_returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        clean_turnover = turnover.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if len(clean_net) == 0:
            return {
                "total_return": 0.0,
                "annual_return": 0.0,
                "annual_volatility": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "avg_annual_turnover": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "gross_total_return": 0.0,
            }

        total_return = float((1.0 + clean_net).prod() - 1.0)
        gross_total_return = float((1.0 + clean_gross).prod() - 1.0)

        periods = max(len(clean_net), 1)
        annual_return = float((1.0 + total_return) ** (252 / periods) - 1.0)
        annual_vol = float(clean_net.std() * np.sqrt(252))
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0

        downside = clean_net[clean_net < 0.0]
        downside_vol = float(downside.std() * np.sqrt(252)) if len(downside) > 1 else 0.0
        sortino = annual_return / downside_vol if downside_vol > 0 else 0.0

        cumulative = (1.0 + clean_net).cumprod()
        rolling_max = cumulative.cummax().replace(0.0, np.nan).ffill().fillna(1.0)
        drawdown = cumulative / rolling_max - 1.0
        max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

        avg_turnover = float(clean_turnover.mean() * 252)
        win_rate = float((clean_net > 0).mean()) if len(clean_net) > 0 else 0.0

        gains = float(clean_net[clean_net > 0].sum())
        losses = float(clean_net[clean_net < 0].sum())
        profit_factor = gains / abs(losses) if losses < 0 else float("inf")

        return {
            "total_return": total_return,
            "gross_total_return": gross_total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_vol,
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": max_drawdown,
            "avg_annual_turnover": avg_turnover,
            "win_rate": win_rate,
            "profit_factor": float(profit_factor),
        }

    # -------------------------------------------------------------------------
    # Legacy Wrappers and Reporting
    # -------------------------------------------------------------------------
    def _identify_quarterly_rebalance_days(self, trading_days: pd.DatetimeIndex) -> set:
        """Backward-compatible wrapper around modular quarterly rebalance calendar."""
        return identify_quarterly_rebalance_days(trading_days)

    def _filter_by_liquidity(self, tickers, date, liquidity_quantile=LIQUIDITY_QUANTILE):
        """Backward-compatible wrapper around RiskManager liquidity filter."""
        return self.risk_manager.filter_by_liquidity(tickers, date, liquidity_quantile)

    def save_results(self, results, factor_name, output_dir=None):
        """Save backtest results via modular ReportGenerator."""
        self.report_generator.save_results(
            results=results,
            factor_name=factor_name,
            xu100_prices=self.xu100_prices,
            xautry_prices=self.xautry_prices,
            factor_capm_store=self.factor_capm,
            factor_yearly_rolling_beta_store=self.factor_yearly_rolling_beta,
            output_dir=output_dir,
        )

    def save_correlation_matrix(self, output_dir=None):
        """Save full return-correlation matrix via modular ReportGenerator."""
        return self.report_generator.save_correlation_matrix(
            factor_returns=self.factor_returns,
            xautry_prices=self.xautry_prices,
            output_dir=output_dir,
        )

    def run_all_factors(self):
        """Run all enabled factors."""
        results = {}

        for factor_name, config in self.signal_configs.items():
            if config.get("enabled", True):
                results[factor_name] = self.run_factor(factor_name)
            else:
                logger.warning(f"Skipping {factor_name} (disabled in config)")

        if self.factor_returns:
            self.save_correlation_matrix()
        if self.factor_capm:
            self.save_capm_summary()
        if self.factor_yearly_rolling_beta:
            self.save_yearly_rolling_beta_summary()

        return results

    def save_capm_summary(self, output_dir=None):
        """Save CAPM summary across all factors via modular ReportGenerator."""
        self.report_generator.save_capm_summary(
            factor_capm=self.factor_capm,
            output_dir=output_dir,
            models_dir=Path(__file__).parent,
        )

    def save_yearly_rolling_beta_summary(self, output_dir=None):
        """Save yearly rolling-beta summary via modular ReportGenerator."""
        self.report_generator.save_yearly_rolling_beta_summary(
            factor_yearly_rolling_beta=self.factor_yearly_rolling_beta,
            output_dir=output_dir,
            models_dir=Path(__file__).parent,
        )

    # -------------------------------------------------------------------------
    # Available Signals Property
    # -------------------------------------------------------------------------
    @property
    def available_signals(self) -> List[str]:
        """List available signal names."""
        config_signals = set(self.signal_configs.keys())
        factory_signals = set()
        try:
            factory_signals = set(get_available_signals())
        except Exception:
            pass

        signals = config_signals | factory_signals
        if not signals:
            signals_dir = Path(__file__).parent / "signals"
            for path in signals_dir.glob("*_signals.py"):
                name = path.stem.replace("_signals", "")
                if not name.startswith("test_"):
                    signals.add(name)
        return sorted(signals)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================
def run_backtest(
    signals: Union[List[str], Dict[str, pd.DataFrame], pd.DataFrame],
    start_date: str,
    end_date: str,
    **options,
) -> PortfolioResult:
    """Run a backtest using a short functional API.

    Args:
        signals: Signal input accepted by :meth:`PortfolioEngine.run_backtest`.
        start_date: Backtest start date (inclusive).
        end_date: Backtest end date (inclusive).
        **options: Portfolio options forwarded to :class:`PortfolioEngine`.

    Returns:
        PortfolioResult: Backtest results for the requested window.
    """
    engine = PortfolioEngine(options=options)
    return engine.run_backtest(signals, start_date, end_date)


def get_default_options() -> Dict[str, Any]:
    """Return a defensive copy of default portfolio options."""
    return _deep_merge_dict(DEFAULT_PORTFOLIO_OPTIONS, {})


# ============================================================================
# MAIN
# ============================================================================
def main():
    available_signals = load_signal_configs()
    signal_names = list(available_signals.keys())

    parser = argparse.ArgumentParser(
        description="Config-Based Portfolio Engine - Automatically detects signals from configs/",
        epilog=f"Available signals: {', '.join(signal_names)}",
    )
    parser.add_argument(
        "signal",
        nargs="?",
        type=str,
        default=None,
        help=f"Signal to run: {', '.join(signal_names)}, or 'all'",
    )
    parser.add_argument(
        "--factor",
        type=str,
        default=None,
        help="Alternative way to specify signal (deprecated, use positional arg)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2018-01-01",
        help="Start date (default: 2018-01-01)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-12-31",
        help="End date (default: 2024-12-31)",
    )

    args = parser.parse_args()
    signal_to_run = args.signal or args.factor or "all"

    if signal_to_run != "all" and signal_to_run not in signal_names:
        logger.error(f"Unknown signal: {signal_to_run}")
        logger.info(f"Available signals: {', '.join(signal_names)}, all")
        sys.exit(1)

    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    data_dir = project_root / "data"
    regime_model_dir_candidates = [
        project_root / "outputs" / "regime" / "simple_regime",
        project_root / "outputs" / "regime",
        project_root / "regime_filter" / "outputs",
        project_root / "Simple Regime Filter" / "outputs",
    ]
    regime_model_dir = next(
        (p for p in regime_model_dir_candidates if p.exists()),
        regime_model_dir_candidates[0],
    )

    engine = PortfolioEngine(data_dir, regime_model_dir, args.start_date, args.end_date)
    engine.load_all_data()

    if signal_to_run == "all":
        engine.run_all_factors()
    else:
        engine.run_factor(signal_to_run)


if __name__ == "__main__":
    total_start = time.time()
    main()
    total_elapsed = time.time() - total_start
    logger.info("\n" + "=" * 70)
    logger.info(f"TOTAL RUNTIME: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")
    logger.info("=" * 70)
