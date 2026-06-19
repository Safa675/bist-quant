"""Pydantic request/response schemas for the BIST Quant API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Factors
# ---------------------------------------------------------------------------

class FactorSnapshotRequest(BaseModel):
    """Cross-sectional signal scores request."""

    indicators: list[str] = Field(default_factory=list)
    universe: str = Field(default="XU100")
    period: str = Field(default="1y")
    interval: str = Field(default="1d")
    max_symbols: int = Field(default=100, ge=1, le=500)
    custom_symbols: list[str] = Field(default_factory=list)
    indicator_params: dict[str, dict[str, Any]] = Field(default_factory=dict)
    buy_threshold: float = Field(default=0.3)
    sell_threshold: float = Field(default=-0.3)
    top_n: int = Field(default=50, ge=1, le=500)
    focus_symbol: str = Field(default="")


class FactorCombineRequest(BaseModel):
    """Combine multiple factors with optional timing."""

    signals: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="List of {name, weight, ...params} factor specs",
    )
    weights: dict[str, float] | None = None
    method: str = Field(
        default="custom",
        description="Weighting scheme: custom, equal, risk_parity, mean_variance, min_variance",
    )
    start_date: str = Field(default="2020-01-01")
    end_date: str = Field(default="2025-12-31")
    timing_enabled: bool = False
    timing_lookback: int = Field(default=63, ge=5)
    timing_threshold: float = Field(default=0.0)
    benchmark: str | None = "XU100"
    rebalance_frequency: str | None = None
    top_n: int | None = None
    max_position_weight: float | None = None


# ---------------------------------------------------------------------------
# Screener
# ---------------------------------------------------------------------------

class ScreenerRunRequest(BaseModel):
    """Stock screener filter payload."""

    data_source: str = Field(default="local")
    template: str = Field(default="")
    sector: str = Field(default="")
    index: str = Field(default="")
    recommendation: str = Field(default="")
    symbols: list[str] = Field(default_factory=list)
    filters: dict[str, dict[str, float]] = Field(default_factory=dict)
    percentile_filters: dict[str, dict[str, float]] = Field(default_factory=dict)
    sort_by: str = Field(default="upside_potential")
    sort_desc: bool = True
    limit: int = Field(default=100, ge=1, le=2000)
    page: int = Field(default=1, ge=1)
    offset: int | None = None
    fields: list[str] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    chart_symbol: str = Field(default="")
    chart_points: int = Field(default=252, ge=30, le=756)
    refresh_cache: bool = False
    technical_scan: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

class AnalyticsRunRequest(BaseModel):
    """Run analytics on an equity curve."""

    equity_curve: list[dict[str, Any]] = Field(
        ...,
        min_length=2,
        description="List of {date, value} points",
    )
    benchmark_curve: list[dict[str, Any]] = Field(default_factory=list)
    include_benchmark: bool = Field(default=True)
    benchmark_symbol: str = Field(default="XU100")
    methods: list[
        Literal[
            "performance",
            "drawdown",
            "rolling",
            "walk_forward",
            "monte_carlo",
            "attribution",
            "stress",
            "risk",
            "cost",
            "transaction_costs",
        ]
    ] = Field(
        default_factory=lambda: [
            "performance",
            "rolling",
            "walk_forward",
            "monte_carlo",
            "attribution",
            "risk",
            "stress",
            "transaction_costs",
        ]
    )
    train_ratio: float = Field(default=0.7, ge=0.3, le=0.9)
    walk_forward_splits: int = Field(default=5, ge=0, le=20)
    slippage_bps: float = Field(default=5.0, ge=0)
    spread_bps: float = Field(default=5.0, ge=0)
    market_impact_bps: float = Field(default=2.0, ge=0)
    tax_rate_pct: float = Field(default=0.0, ge=0)
    rebalance_every_days: int = Field(default=21, ge=1)
    monte_carlo_iterations: int = Field(default=750, ge=100, le=10000)
    monte_carlo_horizon: int = Field(default=252, ge=30, le=2520)
    rolling_window: int = Field(default=63, ge=10, le=504)


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

class OptimizationRunRequest(BaseModel):
    """Run strategy parameter optimization."""

    signal: str = Field(..., min_length=1, description="Base factor name")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Base backtest parameters (start_date, end_date, etc.)",
    )
    method: str = Field(default="grid", description="grid or random")
    parameter_space: list[dict[str, Any]] | None = None
    max_trials: int = Field(default=50, ge=1, le=500)
    random_seed: int | None = None
    train_ratio: float = Field(default=0.7, ge=0.3, le=0.9)
    walk_forward_splits: int = Field(default=0, ge=0, le=20)
    constraints: dict[str, Any] | None = None
    objective: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Professional
# ---------------------------------------------------------------------------

class GreeksRequest(BaseModel):
    """Black-Scholes option Greeks calculation."""

    option_type: Literal["call", "put"]
    spot: float = Field(..., gt=0)
    strike: float = Field(..., gt=0)
    time_years: float = Field(..., gt=0)
    volatility: float = Field(..., gt=0)
    risk_free_rate: float


class StressTestRequest(BaseModel):
    """Portfolio stress test with factor shocks."""

    portfolio_value: float = Field(..., gt=0)
    shocks: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="List of {factor, shock_pct, beta} shocks",
    )


class CryptoSizingRequest(BaseModel):
    """Crypto position sizing calculator."""

    pair: str = Field(..., min_length=1)
    side: Literal["long", "short"]
    entry_price: float = Field(..., gt=0)
    equity: float = Field(..., gt=0)
    risk_pct: float = Field(..., gt=0, le=100)
    leverage: float = Field(default=1.0, ge=1)
    stop_distance_pct: float = Field(..., gt=0)
    taker_fee_bps: float = Field(default=5.0, ge=0)


class PipValueRequest(BaseModel):
    """Compute forex pip value."""

    pair: str = Field(..., min_length=1)
    lot_size: float = Field(..., gt=0)
    account_conversion_rate: float = Field(default=1.0, gt=0)


# ---------------------------------------------------------------------------
# Compliance
# ---------------------------------------------------------------------------

class ComplianceTransactionRequest(BaseModel):
    """Check a transaction against compliance rules."""

    transaction: dict[str, Any] = Field(
        ...,
        description="Transaction record: {id, timestamp, user_id, order_id, symbol, side, quantity, price, venue, strategy_id}",
    )
    rules: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Optional rule overrides. If empty, default rules are used.",
    )


class PositionLimitsRequest(BaseModel):
    """Check position limit breaches."""

    positions: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="List of {name, value, limit, ...} position entries",
    )


# ---------------------------------------------------------------------------
# Signal Construction
# ---------------------------------------------------------------------------

class SignalConstructionSnapshotRequest(BaseModel):
    """Run signal-construction snapshot."""

    universe: str = Field(default="XU100")
    period: str = Field(default="6mo")
    interval: str = Field(default="1d")
    top_n: int = Field(default=20, ge=1, le=200)
    max_symbols: int = Field(default=100, ge=5, le=1000)
    buy_threshold: float = Field(default=0.2, ge=-1.0, le=1.0)
    sell_threshold: float = Field(default=-0.2, ge=-1.0, le=1.0)
    indicators: dict[str, dict[str, Any]] = Field(default_factory=dict)


class SignalConstructionBacktestRequest(SignalConstructionSnapshotRequest):
    """Run signal-construction backtest."""


class SignalConstructionFiveFactorRequest(BaseModel):
    """Run a five-factor rotation backtest via core service."""

    factor_name: str = Field(default="five_factor_rotation")
    start_date: str = Field(default="2020-01-01")
    end_date: str = Field(default="2025-12-31")
    rebalance_frequency: str = Field(default="monthly")
    top_n: int = Field(default=20, ge=5, le=100)
    max_position_weight: float = Field(default=0.25, ge=0.05, le=1.0)


class SignalConstructionOrthogonalizationRequest(BaseModel):
    """Orthogonalization diagnostics request."""

    axes: list[str] = Field(default_factory=list)
    min_overlap: int = Field(default=20, ge=2, le=1000)
    epsilon: float = Field(default=1e-8, gt=0)
    enabled: bool = Field(default=True)
