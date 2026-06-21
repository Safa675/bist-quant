"""Risk evaluation, strategy building, factor exposure, and constrained optimization.

Split out of the former professional.py monolith. Covers strategy risk-limit
breaches, portfolio stress testing, counterparty/liquidity/concentration risk,
strategy definition evaluation, factor exposure regression, and Sharpe-aware
constrained portfolio optimization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from ..core_metrics import (
    SeriesPoint,
    align_series_by_date,
    compute_performance_metrics,
    covariance,
    curve_to_returns,
    mean,
    returns_to_equity,
    sample_std_dev,
)
from .._shared import _clamp, _compare, _to_fixed

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seeded_random(seed: int):
    state = [seed & 0xFFFFFFFF or 123456789]
    def _next():
        state[0] = (1664525 * state[0] + 1013904223) & 0xFFFFFFFF
        return state[0] / 4294967296
    return _next

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class StrategyRiskLimit:
    strategy_id: str; max_notional: float; max_drawdown_pct: float
    max_var95_pct: float; max_leverage: float

@dataclass
class StrategyRiskState:
    strategy_id: str; notional: float; drawdown_pct: float
    var95_pct: float; leverage: float

@dataclass
class StrategyRiskBreach:
    strategy_id: str; rule: str; limit: float; observed: float
    severity: Literal["warning","critical"]

@dataclass
class StressFactorShock:
    factor: str; shock_pct: float; beta: float

@dataclass
class StressTestResult:
    scenario_loss_pct: float; scenario_loss_value: float
    by_factor: list[dict[str, object]]

@dataclass
class CounterpartyExposure:
    counterparty: str; exposure: float; collateral: float
    credit_score: float; cds_bps: float = 120

@dataclass
class CounterpartyAssessment:
    counterparty: str; unsecured_exposure: float
    risk_score: float; rating: Literal["low","moderate","high"]

@dataclass
class LiquidityPosition:
    symbol: str; position_value: float; avg_daily_dollar_volume: float

@dataclass
class LiquidityRiskResult:
    symbol: str; days_to_liquidate: float
    participation_rate_pct: float; risk_level: Literal["low","moderate","high"]

@dataclass
class ConcentrationPosition:
    symbol: str; asset_class: str; sector: str; region: str; weight_pct: float

@dataclass
class ConcentrationAlert:
    dimension: str; key: str; weight_pct: float; limit_pct: float

@dataclass
class StrategyCondition:
    id: str; field: str; comparator: str; value: float

@dataclass
class StrategyBlock:
    id: str; operator: Literal["AND","OR"]; conditions: list[StrategyCondition]

@dataclass
class StrategyDefinition:
    id: str; name: str; version: int; blocks: list[StrategyBlock]
    shared_with: list[str]; updated_at: str

@dataclass
class StrategyDiff:
    added_conditions: int; removed_conditions: int; changed_conditions: int
    old_version: int; new_version: int

@dataclass
class StrategyHealth:
    sharpe: float; sortino: float; calmar: float; max_drawdown: float
    alert: Literal["ok","warning","critical"]

@dataclass
class FactorExposureResult:
    factor_betas: dict[str, float]; factor_contribution_pct: dict[str, float]
    residual_volatility_pct: float; r_squared: float

@dataclass
class OptimizationConstraints:
    min_weight_pct: float; max_weight_pct: float
    target_volatility_pct: float | None = None; iterations: int = 2200
    risk_free_rate_pct: float = 4; previous_weights: dict[str, float] | None = None
    max_turnover_pct: float | None = None

@dataclass
class ConstrainedOptimizationResult:
    weights: dict[str, float]; expected_return_pct: float
    expected_volatility_pct: float; expected_sharpe: float; turnover_pct: float

# ---------------------------------------------------------------------------
# Risk Evaluation
# ---------------------------------------------------------------------------

def evaluate_strategy_risk(limits: list[StrategyRiskLimit],
                           states: list[StrategyRiskState]) -> list[StrategyRiskBreach]:
    sm = {s.strategy_id: s for s in states}
    breaches: list[StrategyRiskBreach] = []
    for lim in limits:
        st = sm.get(lim.strategy_id)
        if not st: continue
        if st.notional > lim.max_notional:
            breaches.append(StrategyRiskBreach(lim.strategy_id,"max_notional",lim.max_notional,st.notional,"critical"))
        if st.drawdown_pct < lim.max_drawdown_pct:
            breaches.append(StrategyRiskBreach(lim.strategy_id,"max_drawdown_pct",lim.max_drawdown_pct,st.drawdown_pct,"critical"))
        if st.var95_pct < lim.max_var95_pct:
            breaches.append(StrategyRiskBreach(lim.strategy_id,"max_var95_pct",lim.max_var95_pct,st.var95_pct,"warning"))
        if st.leverage > lim.max_leverage:
            breaches.append(StrategyRiskBreach(lim.strategy_id,"max_leverage",lim.max_leverage,st.leverage,"critical"))
    return breaches

def run_portfolio_stress_test(portfolio_value: float, shocks: list[StressFactorShock]) -> StressTestResult:
    rows = [{"factor": s.factor, "loss_pct": _to_fixed(s.shock_pct*s.beta,4)} for s in shocks]
    total = sum(r["loss_pct"] for r in rows)
    return StressTestResult(_to_fixed(total,4), _to_fixed(portfolio_value*total/100,2), rows)

def assess_counterparty_risk(exposures: list[CounterpartyExposure]) -> list[CounterpartyAssessment]:
    out = []
    for e in exposures:
        unsec = max(0, e.exposure - e.collateral)
        cr = 100 - _clamp(e.credit_score, 0, 100)
        cds = _clamp(e.cds_bps/8, 0, 100)
        sz = min(100, math.log1p(unsec)*7)
        sc = _to_fixed(cr*0.45 + cds*0.25 + sz*0.3, 2)
        rating: Literal["low","moderate","high"] = "high" if sc >= 70 else "moderate" if sc >= 40 else "low"
        out.append(CounterpartyAssessment(e.counterparty, _to_fixed(unsec,2), sc, rating))
    return sorted(out, key=lambda a: a.risk_score, reverse=True)

def monitor_liquidity_risk(positions: list[LiquidityPosition],
                           participation_rate_pct: float = 12) -> list[LiquidityRiskResult]:
    p = _clamp(participation_rate_pct, 1, 50) / 100
    out = []
    for r in positions:
        adv = max(1, r.avg_daily_dollar_volume)
        days = r.position_value / (adv * p)
        lvl: Literal["low","moderate","high"] = "high" if days > 5 else "moderate" if days > 2 else "low"
        out.append(LiquidityRiskResult(r.symbol, _to_fixed(days,2), _to_fixed(p*100,2), lvl))
    return out

def detect_concentration_risk(positions: list[ConcentrationPosition],
                              single_name_limit_pct: float = 12,
                              bucket_limit_pct: float = 35) -> list[ConcentrationAlert]:
    alerts: list[ConcentrationAlert] = []
    for r in positions:
        if r.weight_pct > single_name_limit_pct:
            alerts.append(ConcentrationAlert("single_name", r.symbol, _to_fixed(r.weight_pct,2), single_name_limit_pct))
    for dim, key_attr in [("asset_class","asset_class"),("sector","sector"),("region","region")]:
        grouped: dict[str, float] = {}
        for r in positions:
            k = getattr(r, key_attr)
            grouped[k] = grouped.get(k, 0) + r.weight_pct
        for k, v in grouped.items():
            if v > bucket_limit_pct:
                alerts.append(ConcentrationAlert(dim, k, _to_fixed(v,2), bucket_limit_pct))
    return sorted(alerts, key=lambda a: a.weight_pct, reverse=True)

# ---------------------------------------------------------------------------
# Strategy Builder
# ---------------------------------------------------------------------------

def evaluate_strategy_definition(strategy: StrategyDefinition,
                                 context: dict[str, float]) -> bool:
    if not strategy.blocks: return False
    for block in strategy.blocks:
        outcomes = [_compare(context.get(c.field, float('nan')), c.comparator, c.value) for c in block.conditions]
        if not outcomes: return False
        if block.operator == "AND" and not all(outcomes): return False
        if block.operator == "OR" and not any(outcomes): return False
    return True

def diff_strategy_versions(prev: StrategyDefinition, nxt: StrategyDefinition) -> StrategyDiff:
    pm: dict[str, StrategyCondition] = {}
    for b in prev.blocks:
        for c in b.conditions: pm[c.id] = c
    nm: dict[str, StrategyCondition] = {}
    for b in nxt.blocks:
        for c in b.conditions: nm[c.id] = c
    added = sum(1 for i in nm if i not in pm)
    removed = sum(1 for i in pm if i not in nm)
    changed = 0
    for i, c in nm.items():
        if i in pm:
            o = pm[i]
            if o.field != c.field or o.comparator != c.comparator or o.value != c.value:
                changed += 1
    return StrategyDiff(added, removed, changed, prev.version, nxt.version)

def build_strategy_health(curve: list[SeriesPoint]) -> StrategyHealth:
    rets = curve_to_returns(curve)
    eq = returns_to_equity(rets, 1)
    m = compute_performance_metrics(rets, equity_curve=eq)
    alert: Literal["ok","warning","critical"] = (
        "critical" if m.max_drawdown <= -25 or m.sharpe < 0
        else "warning" if m.max_drawdown <= -15 or m.sharpe < 0.8
        else "ok"
    )
    return StrategyHealth(_to_fixed(m.sharpe,4), _to_fixed(m.sortino,4), _to_fixed(m.calmar,4), _to_fixed(m.max_drawdown,4), alert)

# ---------------------------------------------------------------------------
# Factor Exposure Model
# ---------------------------------------------------------------------------

def build_factor_exposure_model(strategy_returns: list[SeriesPoint],
                                factor_series_map: dict[str, list[SeriesPoint]]) -> FactorExposureResult:
    aligned = align_series_by_date({"strategy": strategy_returns, **factor_series_map})
    y = aligned["values"].get("strategy", [])
    if len(y) < 20:
        return FactorExposureResult({}, {}, 0, 0)
    betas: dict[str, float] = {}
    contribs: dict[str, float] = {}
    predicted = [0.0] * len(y)
    for factor in factor_series_map:
        x = aligned["values"].get(factor, [])
        if len(x) != len(y) or len(x) < 20: continue
        vx = sample_std_dev(x) ** 2
        if vx <= 0: continue
        b = covariance(x, y) / vx
        betas[factor] = _to_fixed(b, 6)
        contribs[factor] = _to_fixed(b * mean(x) * 252 * 100, 4)
        predicted = [predicted[i] + b * x[i] for i in range(len(y))]
    residuals = [y[i] - predicted[i] for i in range(len(y))]
    vy = sample_std_dev(y) ** 2
    vr = sample_std_dev(residuals) ** 2
    r2 = 1 - vr / vy if vy > 0 else 0
    return FactorExposureResult(betas, contribs, _to_fixed(sample_std_dev(residuals)*math.sqrt(252)*100,4), _to_fixed(_clamp(r2,-1,1),4))

# ---------------------------------------------------------------------------
# Constrained Portfolio Optimization
# ---------------------------------------------------------------------------

def _normalize_weights_bounded(weights: list[float], mn: float, mx: float) -> list[float]:
    if not weights: return []
    w = [_clamp(v, mn, mx) for v in weights]
    for _ in range(25):
        t = sum(w)
        if t <= 0: w = [mn]*len(w); continue
        w = [_clamp(v/t, mn, mx) for v in w]
        diff = 1 - sum(w)
        if abs(diff) < 1e-6: break
        if diff > 0:
            room = [mx - v for v in w]
            rt = sum(max(0, r) for r in room)
            if rt > 0: w = [v + max(0, room[i])/rt*diff for i, v in enumerate(w)]
        else:
            room = [v - mn for v in w]
            rt = sum(max(0, r) for r in room)
            if rt > 0: w = [v - max(0, room[i])/rt*abs(diff) for i, v in enumerate(w)]
    return [_clamp(v, mn, mx) for v in w]

def optimize_constrained_portfolio(series_by_asset: dict[str, list[SeriesPoint]],
                                   constraints: OptimizationConstraints) -> ConstrainedOptimizationResult:
    assets = list(series_by_asset.keys())
    aligned = align_series_by_date(series_by_asset)
    vals = [aligned["values"].get(a, []) for a in assets]
    n = len(assets)
    length = len(vals[0]) if vals else 0
    if not n or length < 20:
        return ConstrainedOptimizationResult({}, 0, 0, 0, 0)
    means_v = [mean(v) for v in vals]
    cov_m = [[covariance(vals[i], vals[j]) for j in range(n)] for i in range(n)]
    mn_w = _clamp(constraints.min_weight_pct/100, 0, 1)
    mx_w = _clamp(constraints.max_weight_pct/100, mn_w, 1)
    fmn, fmx = min(mn_w, 1/n), max(mx_w, 1/n)
    iters = max(300, constraints.iterations)
    rf = (constraints.risk_free_rate_pct) / 100
    tv = constraints.target_volatility_pct / 100 if constraints.target_volatility_pct else None
    mt = constraints.max_turnover_pct / 100 if constraints.max_turnover_pct else None
    prev = constraints.previous_weights or {}
    rng = _seeded_random(42)
    best_sc, best_w, best_r, best_v, best_sh, best_to = float('-inf'), [1/n]*n, 0.0, 0.0, 0.0, 0.0
    for _ in range(iters):
        raw = [rng() + 0.0001 for _ in range(n)]
        w = _normalize_weights_bounded(raw, fmn, fmx)
        er = sum(w[i]*means_v[i] for i in range(n))
        vp = sum(w[i]*w[j]*cov_m[i][j] for i in range(n) for j in range(n))
        vol = math.sqrt(max(vp, 0))
        if vol <= 0: continue
        ar, av = er*252, vol*math.sqrt(252)
        sh = (ar - rf) / av
        to = sum(abs(w[i] - prev.get(assets[i], 0)) for i in range(n)) / 2
        pen = 0.0
        if tv is not None: pen += abs(av - tv) * 1.2
        if mt is not None and to > mt: pen += (to - mt) * 2.5
        sc = sh - pen
        if sc > best_sc:
            best_sc, best_w, best_r, best_v, best_sh, best_to = sc, w, ar, av, sh, to
    return ConstrainedOptimizationResult(
        {assets[i]: _to_fixed(best_w[i],6) for i in range(n)},
        _to_fixed(best_r*100,4), _to_fixed(best_v*100,4), _to_fixed(best_sh,4), _to_fixed(best_to*100,4))


__all__ = [
    "StrategyRiskLimit",
    "StrategyRiskState",
    "StrategyRiskBreach",
    "StressFactorShock",
    "StressTestResult",
    "CounterpartyExposure",
    "CounterpartyAssessment",
    "LiquidityPosition",
    "LiquidityRiskResult",
    "ConcentrationPosition",
    "ConcentrationAlert",
    "StrategyCondition",
    "StrategyBlock",
    "StrategyDefinition",
    "StrategyDiff",
    "StrategyHealth",
    "FactorExposureResult",
    "OptimizationConstraints",
    "ConstrainedOptimizationResult",
    "evaluate_strategy_risk",
    "run_portfolio_stress_test",
    "assess_counterparty_risk",
    "monitor_liquidity_risk",
    "detect_concentration_risk",
    "evaluate_strategy_definition",
    "diff_strategy_versions",
    "build_strategy_health",
    "build_factor_exposure_model",
    "optimize_constrained_portfolio",
]
