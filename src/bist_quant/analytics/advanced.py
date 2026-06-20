"""Advanced quantitative analytics.

This module is now a thin facade that re-exports the public API previously
defined inline here. The implementation lives in focused domain sub-modules:

- ``volatility``            GARCH/EWMA volatility forecasting + proxy series
- ``position_sizing``       Kelly / fixed-fractional / optimal-f / correlation-adjusted sizing / hedges
- ``parameter_optimization`` MA-crossover heatmap + walk-forward optimization
- ``risk_metrics``          Monte Carlo robustness + advanced risk metrics
- ``portfolio_construction`` MPT / risk-parity / min-variance / ERC / factor-based
- ``regime_backtest``       bull/bear/sideways regime classification
- ``charting``              volume profile / Renko / Point & Figure / timeframes
- ``signals``               indicator combination signal
- ``attribution``           performance attribution + significance testing

All existing ``from bist_quant.analytics.advanced import X`` call sites
continue to work unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from .core_metrics import (
    MonteCarloSummary,
    PerformanceMetrics,
    SeriesPoint,
    apply_transaction_costs,
    compute_performance_metrics,
    returns_to_equity,
    run_monte_carlo_bootstrap,
)
from ._shared import PricePoint
from .attribution import (
    PerformanceAttributionBreakdown,
    SignificanceResult,
    compute_performance_attribution_breakdown,
    estimate_strategy_significance,
)
from .charting import (
    PointFigureColumn,
    RenkoBrick,
    VolumeProfileBucket,
    build_point_figure_columns,
    build_renko_bricks,
    build_synchronized_timeframes,
    build_volume_profile,
)
from .parameter_optimization import (
    ParameterHeatmapCell,
    ParameterHeatmapResult,
    WalkForwardOptimizationSplit,
    WalkForwardOptimizationResult,
    build_parameter_sensitivity_heatmap,
    run_walk_forward_parameter_optimization,
)
from .portfolio_construction import (
    PortfolioConstructionResult,
    construct_portfolio_weights,
)
from .position_sizing import (
    CorrelationAdjustedSizingResult,
    HedgeSuggestion,
    compute_correlation_adjusted_sizing,
    compute_fixed_fractional_notional,
    compute_kelly_fraction_percent,
    compute_optimal_f,
    suggest_cross_asset_hedges,
)
from .regime_backtest import (
    RegimeBacktestResult,
    RegimeStatRow,
    run_regime_aware_backtest,
)
from .risk_metrics import (
    AdvancedRiskMetrics,
    MonteCarloRobustnessSummary,
    compute_advanced_risk_metrics,
    summarize_monte_carlo_robustness,
)
from .signals import (
    IndicatorCombinationSignal,
    build_indicator_combination_signal,
)
from .volatility import (
    VolatilityForecastPoint,
    VolatilityForecastResult,
    build_garch_volatility_forecast,
    build_proxy_asset_return_series,
)


# ---------------------------------------------------------------------------
# Integration diagnostics — depends on multiple domain modules, so it stays here.
# ---------------------------------------------------------------------------


@dataclass
class BacktestIntegrationDiagnostics:
    adjusted_metrics: PerformanceMetrics
    monte_carlo: MonteCarloSummary
    robustness: MonteCarloRobustnessSummary
    regime: RegimeBacktestResult
    significance: SignificanceResult
    total_cost_impact_pct: float


def run_backtest_integration_diagnostics(
    returns: list[SeriesPoint],
    benchmark_returns: list[SeriesPoint] | None,
    *,
    slippage_bps: float,
    spread_bps: float,
    impact_bps: float,
    tax_pct: float,
    rebalance_days: int,
) -> BacktestIntegrationDiagnostics:
    """Full diagnostic pipeline: cost-adjust → metrics → MC → regime → significance."""
    adjusted = apply_transaction_costs(
        returns,
        slippage_bps=slippage_bps,
        spread_bps=spread_bps,
        market_impact_bps=impact_bps,
        tax_rate_pct=tax_pct,
        rebalance_every_days=rebalance_days,
    )
    eq = returns_to_equity(adjusted.adjusted, 1)
    adj_metrics = compute_performance_metrics(adjusted.adjusted, equity_curve=eq,
                                               benchmark_returns=benchmark_returns)
    mc = run_monte_carlo_bootstrap(adjusted.adjusted, iterations=700, horizon_days=252, seed=49)
    robust = summarize_monte_carlo_robustness(mc)
    regime = run_regime_aware_backtest(adjusted.adjusted)
    sig = estimate_strategy_significance(adjusted.adjusted, benchmark_returns)

    return BacktestIntegrationDiagnostics(
        adjusted_metrics=adj_metrics,
        monte_carlo=mc,
        robustness=robust,
        regime=regime,
        significance=sig,
        total_cost_impact_pct=round(adjusted.total_cost_impact_pct, 2),
    )


__all__ = [
    # Data classes
    "PricePoint",
    "VolatilityForecastPoint",
    "VolatilityForecastResult",
    "CorrelationAdjustedSizingResult",
    "HedgeSuggestion",
    "ParameterHeatmapCell",
    "ParameterHeatmapResult",
    "WalkForwardOptimizationSplit",
    "WalkForwardOptimizationResult",
    "MonteCarloRobustnessSummary",
    "AdvancedRiskMetrics",
    "PortfolioConstructionResult",
    "RegimeStatRow",
    "RegimeBacktestResult",
    "SignificanceResult",
    "VolumeProfileBucket",
    "RenkoBrick",
    "PointFigureColumn",
    "IndicatorCombinationSignal",
    "PerformanceAttributionBreakdown",
    "BacktestIntegrationDiagnostics",
    # Functions
    "build_garch_volatility_forecast",
    "build_proxy_asset_return_series",
    "compute_kelly_fraction_percent",
    "compute_fixed_fractional_notional",
    "compute_optimal_f",
    "compute_correlation_adjusted_sizing",
    "suggest_cross_asset_hedges",
    "build_parameter_sensitivity_heatmap",
    "run_walk_forward_parameter_optimization",
    "summarize_monte_carlo_robustness",
    "compute_advanced_risk_metrics",
    "construct_portfolio_weights",
    "run_regime_aware_backtest",
    "estimate_strategy_significance",
    "build_volume_profile",
    "build_renko_bricks",
    "build_point_figure_columns",
    "build_synchronized_timeframes",
    "build_indicator_combination_signal",
    "compute_performance_attribution_breakdown",
    "run_backtest_integration_diagnostics",
]
