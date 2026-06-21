"""
Professional trading analytics — public facade.

Re-exports the public API of the domain sub-modules so every existing
``from bist_quant.analytics.professional import X`` call site continues to
work unchanged. The implementation lives in:

- :mod:`.trading`     — crypto/forex/options/futures sizing, fund screening, commodity spreads
- :mod:`.risk`        — strategy risk limits, stress testing, counterparty/liquidity/concentration
                        risk, strategy builder, factor exposure, constrained optimization
- :mod:`.reporting`   — sentiment, risk-adjusted ratios, benchmarks, attribution, tax, reporting,
                        market intelligence
- :mod:`.execution`   — iceberg/TWAP/VWAP scheduling, bracket orders, slippage simulation
- :mod:`.compliance`  — compliance rule engine, position limits, anomaly detection, alerting

``run_performance_snapshot`` stays inline here because it composes results
from multiple domain modules (mirroring the pattern ``advanced.py`` uses for
``run_backtest_integration_diagnostics``).
"""

from __future__ import annotations

from typing import Any

from ..core_metrics import (
    SeriesPoint,
    compute_performance_metrics,
    curve_to_returns,
    returns_to_equity,
)

from .trading import (  # noqa: F401
    CommoditySpreadResult,
    CryptoTradeInput,
    CryptoTradePlan,
    ForexPipResult,
    FundCandidate,
    FundScreenResult,
    FuturesMarginResult,
    OptionGreeksInput,
    OptionGreeksResult,
    build_crypto_trade_plan,
    compute_commodity_spread,
    compute_forex_pip_value,
    compute_futures_margin,
    compute_option_greeks,
    screen_funds,
)
from .risk import (  # noqa: F401
    ConcentrationAlert,
    ConcentrationPosition,
    ConstrainedOptimizationResult,
    CounterpartyAssessment,
    CounterpartyExposure,
    FactorExposureResult,
    LiquidityPosition,
    LiquidityRiskResult,
    OptimizationConstraints,
    StrategyBlock,
    StrategyCondition,
    StrategyDefinition,
    StrategyDiff,
    StrategyHealth,
    StrategyRiskBreach,
    StrategyRiskLimit,
    StrategyRiskState,
    StressFactorShock,
    StressTestResult,
    assess_counterparty_risk,
    build_factor_exposure_model,
    build_strategy_health,
    detect_concentration_risk,
    diff_strategy_versions,
    evaluate_strategy_definition,
    evaluate_strategy_risk,
    monitor_liquidity_risk,
    optimize_constrained_portfolio,
    run_portfolio_stress_test,
)
from .reporting import (  # noqa: F401
    AttributionInput,
    AttributionResult,
    BenchmarkComparison,
    EconomicEvent,
    EarningsSurprise,
    InsiderActivity,
    RiskFactorContribution,
    SentimentItem,
    SentimentSummary,
    ShortInterest,
    TaxReport,
    TaxTrade,
    analyze_sentiment,
    build_client_report_template,
    build_market_intelligence_snapshot,
    build_monthly_quarterly_returns,
    build_tax_report,
    compare_benchmarks,
    compute_attribution,
    compute_risk_adjusted_ratios,
    decompose_portfolio_risk_factors,
    schedule_report_runs,
)
from .execution import (  # noqa: F401
    BracketOrder,
    ExecutionSimulationResult,
    ExecutionSlice,
    IcebergSlice,
    OrderBookLevel,
    build_bracket_order,
    build_twap_schedule,
    build_vwap_schedule,
    create_iceberg_slices,
    simulate_execution_with_slippage,
)
from .compliance import (  # noqa: F401
    AlertCondition,
    ComplianceHit,
    ComplianceRule,
    NotificationAlert,
    TransactionRecord,
    build_escalation_plan,
    detect_user_activity_anomalies,
    evaluate_alert_conditions,
    group_alerts,
    monitor_position_limits,
    run_compliance_rule_engine,
)


# ---------------------------------------------------------------------------
# Integration — depends on multiple domain modules, so it stays here
# ---------------------------------------------------------------------------

def run_performance_snapshot(curve: list[SeriesPoint], benchmarks: dict[str, list[SeriesPoint]]) -> dict[str, Any]:
    rets = curve_to_returns(curve)
    eq = returns_to_equity(rets, 1)
    return {"metrics": compute_performance_metrics(rets, equity_curve=eq),
            "risk_adjusted": compute_risk_adjusted_ratios(rets),
            "monthly_quarterly": build_monthly_quarterly_returns(rets),
            "benchmark": compare_benchmarks(rets, benchmarks)}


__all__ = [
    # trading
    "CryptoTradeInput", "CryptoTradePlan", "ForexPipResult",
    "OptionGreeksInput", "OptionGreeksResult", "FuturesMarginResult",
    "FundCandidate", "FundScreenResult", "CommoditySpreadResult",
    "build_crypto_trade_plan", "compute_forex_pip_value",
    "compute_option_greeks", "compute_futures_margin",
    "screen_funds", "compute_commodity_spread",
    # risk
    "StrategyRiskLimit", "StrategyRiskState", "StrategyRiskBreach",
    "StressFactorShock", "StressTestResult",
    "CounterpartyExposure", "CounterpartyAssessment",
    "LiquidityPosition", "LiquidityRiskResult",
    "ConcentrationPosition", "ConcentrationAlert",
    "StrategyCondition", "StrategyBlock", "StrategyDefinition",
    "StrategyDiff", "StrategyHealth",
    "FactorExposureResult",
    "OptimizationConstraints", "ConstrainedOptimizationResult",
    "evaluate_strategy_risk", "run_portfolio_stress_test",
    "assess_counterparty_risk", "monitor_liquidity_risk",
    "detect_concentration_risk",
    "evaluate_strategy_definition", "diff_strategy_versions",
    "build_strategy_health",
    "build_factor_exposure_model", "optimize_constrained_portfolio",
    # reporting
    "SentimentItem", "SentimentSummary",
    "BenchmarkComparison", "AttributionInput", "AttributionResult",
    "TaxTrade", "TaxReport",
    "EconomicEvent", "EarningsSurprise", "InsiderActivity", "ShortInterest",
    "RiskFactorContribution",
    "analyze_sentiment",
    "compute_risk_adjusted_ratios", "build_monthly_quarterly_returns",
    "compare_benchmarks", "compute_attribution",
    "decompose_portfolio_risk_factors",
    "build_tax_report",
    "schedule_report_runs", "build_client_report_template",
    "build_market_intelligence_snapshot",
    # execution
    "IcebergSlice", "ExecutionSlice", "OrderBookLevel",
    "ExecutionSimulationResult", "BracketOrder",
    "create_iceberg_slices", "build_twap_schedule", "build_vwap_schedule",
    "build_bracket_order", "simulate_execution_with_slippage",
    # compliance
    "TransactionRecord", "ComplianceRule", "ComplianceHit",
    "AlertCondition", "NotificationAlert",
    "run_compliance_rule_engine", "monitor_position_limits",
    "detect_user_activity_anomalies",
    "evaluate_alert_conditions", "group_alerts", "build_escalation_plan",
    # integration
    "run_performance_snapshot",
]
