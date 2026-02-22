"""
Professional trading analytics.

Ported from bist_quant.ai/src/lib/professional.ts.
Covers crypto/forex/options/futures sizing, fund screening, risk evaluation,
strategy builder, factor exposure, constrained optimization, sentiment,
benchmark comparison, attribution, tax, execution algorithms, compliance,
alerts, and reporting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from .core_metrics import (
    SeriesPoint,
    align_series_by_date,
    compute_performance_metrics,
    correlation,
    covariance,
    curve_to_returns,
    mean,
    returns_to_equity,
    sample_std_dev,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    if not math.isfinite(v): return lo
    return max(lo, min(hi, v))

def _rd(v: float, d: int = 4) -> float:
    if not math.isfinite(v): return 0.0
    return round(v, d)

def _variance(vals: list[float]) -> float:
    if len(vals) < 2: return 0.0
    avg = mean(vals)
    return sum((x - avg) ** 2 for x in vals) / (len(vals) - 1)

def _normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def _normal_cdf(x: float) -> float:
    sign = -1 if x < 0 else 1
    ax = abs(x) / math.sqrt(2)
    t = 1 / (1 + 0.3275911 * ax)
    erf = 1 - (((((1.061405429*t - 1.453152027)*t + 1.421413741)*t - 0.284496736)*t + 0.254829592)*t * math.exp(-ax*ax))
    return 0.5 * (1 + sign * erf)

def _seeded_random(seed: int):
    state = [seed & 0xFFFFFFFF or 123456789]
    def _next():
        state[0] = (1664525 * state[0] + 1013904223) & 0xFFFFFFFF
        return state[0] / 4294967296
    return _next

def _compare(left: float, comp: str, right: float) -> bool:
    if not math.isfinite(left) or not math.isfinite(right): return False
    if comp == ">": return left > right
    if comp == ">=": return left >= right
    if comp == "<": return left < right
    if comp == "<=": return left <= right
    if comp == "==": return left == right
    return left != right

def _parse_date(s: str):
    import datetime
    try: return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except: return datetime.datetime(1970, 1, 1)

# ---------------------------------------------------------------------------
# Data Classes — Trading
# ---------------------------------------------------------------------------

@dataclass
class CryptoTradeInput:
    pair: str; side: Literal["long","short"]; entry_price: float; equity: float
    risk_pct: float; leverage: float; stop_distance_pct: float; taker_fee_bps: float = 5

@dataclass
class CryptoTradePlan:
    pair: str; side: Literal["long","short"]; notional: float; margin_required: float
    quantity: float; liquidation_price: float; max_loss: float; estimated_fees: float

@dataclass
class ForexPipResult:
    pair: str; pip_size: float; pip_value_quote: float; pip_value_account: float

@dataclass
class OptionGreeksInput:
    option_type: Literal["call","put"]; spot: float; strike: float
    time_years: float; volatility: float; risk_free_rate: float

@dataclass
class OptionGreeksResult:
    delta: float; gamma: float; theta_per_day: float
    vega_per_1pct: float; rho_per_1pct: float; theoretical_price: float

@dataclass
class FuturesMarginResult:
    symbol: str; notional: float; initial_margin: float
    maintenance_margin: float; tick_value_total: float

@dataclass
class FundCandidate:
    symbol: str; expense_ratio: float; tracking_error: float
    aum_mn: float; spread_bps: float; one_year_return: float; three_year_sharpe: float

@dataclass
class FundScreenResult:
    symbol: str; score: float; passed: bool

@dataclass
class CommoditySpreadResult:
    spread_value: float; spread_pct_of_front: float; dollar_exposure: float

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
    by_factor: list[dict[str, Any]]

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

@dataclass
class SentimentItem:
    symbol: str; source: Literal["news","social"]; text: str; engagement: float = 0

@dataclass
class SentimentSummary:
    overall_score: float; by_symbol: dict[str, float]
    bullish_symbols: list[str]; bearish_symbols: list[str]

@dataclass
class BenchmarkComparison:
    benchmark: str; correlation: float; beta: float; alpha_pct: float
    tracking_error_pct: float; information_ratio: float

@dataclass
class AttributionInput:
    driver: str; contribution_pct: float

@dataclass
class AttributionResult:
    total_alpha_pct: float; normalized: list[dict[str, float]]

@dataclass
class TaxTrade:
    symbol: str; pnl: float; holding_days: int

@dataclass
class TaxReport:
    jurisdiction: Literal["TR","US","EU"]; short_term_tax: float
    long_term_tax: float; net_after_tax: float

@dataclass
class IcebergSlice:
    sequence: int; quantity: int

@dataclass
class ExecutionSlice:
    timestamp: str; quantity: float

@dataclass
class OrderBookLevel:
    price: float; size: float

@dataclass
class ExecutionSimulationResult:
    filled_qty: float; average_fill_price: float
    slippage_bps: float; residual_qty: float

@dataclass
class BracketOrder:
    entry_price: float; stop_price: float; take_profit_price: float; quantity: float

@dataclass
class TransactionRecord:
    id: str; timestamp: str; user_id: str; order_id: str; symbol: str
    side: Literal["buy","sell"]; quantity: float; price: float
    venue: str; strategy_id: str

@dataclass
class ComplianceRule:
    id: str; field: str; comparator: str; threshold: float
    message: str; severity: Literal["warning","critical"]

@dataclass
class ComplianceHit:
    rule_id: str; message: str; severity: Literal["warning","critical"]

@dataclass
class AlertCondition:
    id: str; name: str; metric: str; comparator: str; threshold: float
    severity: Literal["info","warning","critical"]
    channels: list[str]; group_key: str = ""; cooldown_sec: int = 0

@dataclass
class NotificationAlert:
    condition_id: str; name: str; severity: Literal["info","warning","critical"]
    channels: list[str]; message: str; metric_value: float
    triggered_at: str; group_key: str

@dataclass
class EconomicEvent:
    id: str; name: str; region: str; category: str; date: str
    impact: Literal["low","medium","high"]
    consensus: float | None = None; actual: float | None = None

@dataclass
class EarningsSurprise:
    symbol: str; expected_eps: float; actual_eps: float

@dataclass
class InsiderActivity:
    symbol: str; net_buy_value: float

@dataclass
class ShortInterest:
    symbol: str; short_float_pct: float; borrow_fee_pct: float

@dataclass
class RiskFactorContribution:
    factor: str; weighted_exposure: float; contribution_pct: float

# ---------------------------------------------------------------------------
# Crypto / Forex / Options / Futures
# ---------------------------------------------------------------------------

def build_crypto_trade_plan(inp: CryptoTradeInput) -> CryptoTradePlan:
    entry = max(1e-6, inp.entry_price)
    risk_cap = max(0, inp.equity) * _clamp(inp.risk_pct, 0.05, 100) / 100
    stop_dist = entry * _clamp(inp.stop_distance_pct, 0.05, 95) / 100
    qty = risk_cap / stop_dist if stop_dist > 0 else 0
    notional = qty * entry
    lev = _clamp(inp.leverage, 1, 125)
    margin = notional / lev
    liq_buf = 0.92 / lev
    liq = entry * (1 - liq_buf) if inp.side == "long" else entry * (1 + liq_buf)
    fee_bps = _clamp(inp.taker_fee_bps, 0, 200)
    fees = notional * fee_bps / 10000
    return CryptoTradePlan(inp.pair.upper(), inp.side, _rd(notional,2), _rd(margin,2),
                           _rd(qty,6), _rd(liq,4), _rd(risk_cap,2), _rd(fees,2))

def compute_forex_pip_value(pair: str, lot_size: float, account_conversion_rate: float = 1.0) -> ForexPipResult:
    p = pair.upper()
    pip_size = 0.01 if p.endswith("JPY") else 0.0001
    pv = lot_size * pip_size
    conv = max(0, account_conversion_rate) if math.isfinite(account_conversion_rate) else 1
    return ForexPipResult(p, pip_size, _rd(pv,6), _rd(pv*conv,6))

def compute_option_greeks(inp: OptionGreeksInput) -> OptionGreeksResult:
    s, k = max(1e-4, inp.spot), max(1e-4, inp.strike)
    t = max(1/365, inp.time_years)
    sig = _clamp(inp.volatility, 1e-4, 5)
    r = _clamp(inp.risk_free_rate, -1, 1)
    st = math.sqrt(t)
    d1 = (math.log(s/k) + (r + 0.5*sig*sig)*t) / (sig*st)
    d2 = d1 - sig*st
    nd1, nd2 = _normal_cdf(d1), _normal_cdf(d2)
    pdf1 = _normal_pdf(d1)
    disc = math.exp(-r*t)
    call_p = s*nd1 - k*disc*nd2
    put_p = k*disc*_normal_cdf(-d2) - s*_normal_cdf(-d1)
    delta = nd1 if inp.option_type == "call" else nd1 - 1
    gamma = pdf1 / (s*sig*st)
    vega = s*pdf1*st / 100
    tc = (-s*pdf1*sig/(2*st) - r*k*disc*nd2) / 365
    tp = (-s*pdf1*sig/(2*st) + r*k*disc*_normal_cdf(-d2)) / 365
    rc = k*t*disc*nd2 / 100
    rp = -k*t*disc*_normal_cdf(-d2) / 100
    is_call = inp.option_type == "call"
    return OptionGreeksResult(_rd(delta,6), _rd(gamma,6), _rd(tc if is_call else tp,6),
                              _rd(vega,6), _rd(rc if is_call else rp,6),
                              _rd(call_p if is_call else put_p,6))

def compute_futures_margin(symbol: str, contracts: int, price: float,
                           contract_size: float, initial_margin_rate: float,
                           maintenance_margin_rate: float, tick_size: float,
                           tick_value: float) -> FuturesMarginResult:
    c = max(1, round(contracts))
    n = max(0, price) * max(0, contract_size) * c
    return FuturesMarginResult(symbol.upper(), _rd(n,2),
                               _rd(n*_clamp(initial_margin_rate,0,1),2),
                               _rd(n*_clamp(maintenance_margin_rate,0,1),2),
                               _rd(max(0,tick_value)*c,4))

# ---------------------------------------------------------------------------
# Fund Screening
# ---------------------------------------------------------------------------

def _norm_higher(vals: list[float]) -> list[float]:
    if not vals: return []
    mn, mx = min(vals), max(vals)
    if mx == mn: return [0.5]*len(vals)
    return [(v-mn)/(mx-mn) for v in vals]

def _norm_lower(vals: list[float]) -> list[float]:
    return [1-v for v in _norm_higher(vals)]

def screen_funds(candidates: list[FundCandidate]) -> list[FundScreenResult]:
    if not candidates: return []
    exp = _norm_lower([c.expense_ratio for c in candidates])
    trk = _norm_lower([c.tracking_error for c in candidates])
    aum = _norm_higher([c.aum_mn for c in candidates])
    spr = _norm_lower([c.spread_bps for c in candidates])
    ret = _norm_higher([c.one_year_return for c in candidates])
    shp = _norm_higher([c.three_year_sharpe for c in candidates])
    results = []
    for i, c in enumerate(candidates):
        sc = exp[i]*0.18 + trk[i]*0.17 + aum[i]*0.12 + spr[i]*0.13 + ret[i]*0.2 + shp[i]*0.2
        results.append(FundScreenResult(c.symbol, _rd(sc*100,2), sc >= 0.62))
    return sorted(results, key=lambda r: r.score, reverse=True)

# ---------------------------------------------------------------------------
# Commodity Spread
# ---------------------------------------------------------------------------

def compute_commodity_spread(front_price: float, back_price: float,
                             multiplier: float, contracts: int) -> CommoditySpreadResult:
    sp = front_price - back_price
    return CommoditySpreadResult(_rd(sp,4),
                                 _rd(sp/front_price*100,4) if front_price > 0 else 0,
                                 _rd(sp*multiplier*contracts,2))

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
    rows = [{"factor": s.factor, "loss_pct": _rd(s.shock_pct*s.beta,4)} for s in shocks]
    total = sum(r["loss_pct"] for r in rows)
    return StressTestResult(_rd(total,4), _rd(portfolio_value*total/100,2), rows)

def assess_counterparty_risk(exposures: list[CounterpartyExposure]) -> list[CounterpartyAssessment]:
    out = []
    for e in exposures:
        unsec = max(0, e.exposure - e.collateral)
        cr = 100 - _clamp(e.credit_score, 0, 100)
        cds = _clamp(e.cds_bps/8, 0, 100)
        sz = min(100, math.log1p(unsec)*7)
        sc = _rd(cr*0.45 + cds*0.25 + sz*0.3, 2)
        rating: Literal["low","moderate","high"] = "high" if sc >= 70 else "moderate" if sc >= 40 else "low"
        out.append(CounterpartyAssessment(e.counterparty, _rd(unsec,2), sc, rating))
    return sorted(out, key=lambda a: a.risk_score, reverse=True)

def monitor_liquidity_risk(positions: list[LiquidityPosition],
                           participation_rate_pct: float = 12) -> list[LiquidityRiskResult]:
    p = _clamp(participation_rate_pct, 1, 50) / 100
    out = []
    for r in positions:
        adv = max(1, r.avg_daily_dollar_volume)
        days = r.position_value / (adv * p)
        lvl: Literal["low","moderate","high"] = "high" if days > 5 else "moderate" if days > 2 else "low"
        out.append(LiquidityRiskResult(r.symbol, _rd(days,2), _rd(p*100,2), lvl))
    return out

def detect_concentration_risk(positions: list[ConcentrationPosition],
                              single_name_limit_pct: float = 12,
                              bucket_limit_pct: float = 35) -> list[ConcentrationAlert]:
    alerts: list[ConcentrationAlert] = []
    for r in positions:
        if r.weight_pct > single_name_limit_pct:
            alerts.append(ConcentrationAlert("single_name", r.symbol, _rd(r.weight_pct,2), single_name_limit_pct))
    for dim, key_attr in [("asset_class","asset_class"),("sector","sector"),("region","region")]:
        grouped: dict[str, float] = {}
        for r in positions:
            k = getattr(r, key_attr)
            grouped[k] = grouped.get(k, 0) + r.weight_pct
        for k, v in grouped.items():
            if v > bucket_limit_pct:
                alerts.append(ConcentrationAlert(dim, k, _rd(v,2), bucket_limit_pct))
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
    return StrategyHealth(_rd(m.sharpe,4), _rd(m.sortino,4), _rd(m.calmar,4), _rd(m.max_drawdown,4), alert)

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
        vx = _variance(x)
        if vx <= 0: continue
        b = covariance(x, y) / vx
        betas[factor] = _rd(b, 6)
        contribs[factor] = _rd(b * mean(x) * 252 * 100, 4)
        predicted = [predicted[i] + b * x[i] for i in range(len(y))]
    residuals = [y[i] - predicted[i] for i in range(len(y))]
    vy = _variance(y)
    vr = _variance(residuals)
    r2 = 1 - vr / vy if vy > 0 else 0
    return FactorExposureResult(betas, contribs, _rd(sample_std_dev(residuals)*math.sqrt(252)*100,4), _rd(_clamp(r2,-1,1),4))

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
        {assets[i]: _rd(best_w[i],6) for i in range(n)},
        _rd(best_r*100,4), _rd(best_v*100,4), _rd(best_sh,4), _rd(best_to*100,4))

# ---------------------------------------------------------------------------
# Sentiment
# ---------------------------------------------------------------------------

_POS = ["beat","upgrade","growth","strong","outperform","bullish","record","surge","improve","positive"]
_NEG = ["miss","downgrade","weak","decline","underperform","bearish","drop","risk","negative","lawsuit"]

def analyze_sentiment(items: list[SentimentItem]) -> SentimentSummary:
    if not items: return SentimentSummary(0, {}, [], [])
    by_sym: dict[str, list[float]] = {}
    all_sc: list[float] = []
    for it in items:
        txt = it.text.lower()
        pos = sum(1 for w in _POS if w in txt)
        neg = sum(1 for w in _NEG if w in txt)
        raw = (pos - neg) / max(1, pos + neg)
        sw = 1.2 if it.source == "news" else 1
        eng = math.log1p(max(0, it.engagement))
        sc = raw * sw * (1 + eng / 8)
        by_sym.setdefault(it.symbol, []).append(sc)
        all_sc.append(sc)
    sym_scores = {s: _rd(mean(v),4) for s, v in by_sym.items()}
    srt = sorted(sym_scores.items(), key=lambda x: x[1], reverse=True)
    return SentimentSummary(_rd(mean(all_sc),4), sym_scores,
                            [s for s, v in srt if v > 0.18][:8],
                            [s for s, v in srt if v < -0.18][-8:])

# ---------------------------------------------------------------------------
# Risk-Adjusted Ratios / Monthly Returns / Benchmarks / Attribution
# ---------------------------------------------------------------------------

def compute_risk_adjusted_ratios(returns: list[SeriesPoint], risk_free_rate_pct: float = 4):
    if not returns: return {"sharpe": 0, "sortino": 0, "calmar": 0}
    drf = risk_free_rate_pct / 100 / 252
    vals = [r.value for r in returns]
    ae = mean(vals) - drf
    vol = sample_std_dev(vals)
    ds = sample_std_dev([v for v in vals if v < 0])
    eq = returns_to_equity(returns, 1)
    peak = eq[0].value if eq else 1
    md = 0.0
    for r in eq:
        if r.value > peak: peak = r.value
        md = min(md, r.value / peak - 1)
    av, ad = vol*math.sqrt(252), ds*math.sqrt(252)
    ar = mean(vals)*252
    return {"sharpe": _rd(ae*252/av,4) if av > 0 else 0,
            "sortino": _rd(ae*252/ad,4) if ad > 0 else 0,
            "calmar": _rd(ar/abs(md),4) if md < 0 else 0}

def build_monthly_quarterly_returns(returns: list[SeriesPoint]):
    mmap: dict[str, list[float]] = {}
    qmap: dict[str, list[float]] = {}
    for r in returns:
        try:
            parts = r.date[:10].split("-")
            y, m = int(parts[0]), int(parts[1])
        except: continue
        mk = f"{y}-{m:02d}"
        qk = f"{y}-Q{(m-1)//3+1}"
        mmap.setdefault(mk, []).append(r.value)
        qmap.setdefault(qk, []).append(r.value)
    def to_rows(d):
        out = []
        for p in sorted(d):
            cum = 1.0
            for v in d[p]: cum *= 1+v
            out.append({"period": p, "return_pct": _rd((cum-1)*100, 4)})
        return out
    return {"monthly": to_rows(mmap), "quarterly": to_rows(qmap)}

def compare_benchmarks(strategy_returns: list[SeriesPoint],
                       benchmark_map: dict[str, list[SeriesPoint]]) -> list[BenchmarkComparison]:
    aligned = align_series_by_date({"strategy": strategy_returns, **benchmark_map})
    strat = aligned["values"].get("strategy", [])
    if len(strat) < 20: return []
    out = []
    for name in benchmark_map:
        bench = aligned["values"].get(name, [])
        if len(bench) != len(strat) or len(bench) < 20:
            out.append(BenchmarkComparison(name, 0, 0, 0, 0, 0)); continue
        c = correlation(strat, bench)
        vb = _variance(bench)
        b = covariance(strat, bench)/vb if vb > 0 else 0
        a = (mean(strat) - b*mean(bench))*252*100
        active = [strat[i]-bench[i] for i in range(len(strat))]
        te = sample_std_dev(active)*math.sqrt(252)*100
        ir = mean(active)*252*100/te if te > 0 else 0
        out.append(BenchmarkComparison(name, _rd(c,4), _rd(b,4), _rd(a,4), _rd(te,4), _rd(ir,4)))
    return sorted(out, key=lambda x: x.alpha_pct, reverse=True)

def compute_attribution(drivers: list[AttributionInput]) -> AttributionResult:
    total = sum(d.contribution_pct for d in drivers)
    base = abs(total) if abs(total) > 1e-5 else 1
    norm = sorted([{"driver": d.driver, "share_pct": _rd(d.contribution_pct/base*100,4)} for d in drivers],
                  key=lambda x: abs(x["share_pct"]), reverse=True)
    return AttributionResult(_rd(total,4), norm)

def decompose_portfolio_risk_factors(weights: dict[str, float],
                                     exposures_by_asset: dict[str, dict[str, float]]) -> list[RiskFactorContribution]:
    ft: dict[str, float] = {}
    for asset, w in weights.items():
        exp = exposures_by_asset.get(asset, {})
        for f, v in exp.items():
            ft[f] = ft.get(f, 0) + w * v
    at = sum(abs(v) for v in ft.values()) or 1
    return sorted([RiskFactorContribution(f, _rd(we,6), _rd(abs(we)/at*100,4)) for f, we in ft.items()],
                  key=lambda x: x.contribution_pct, reverse=True)

# ---------------------------------------------------------------------------
# Tax
# ---------------------------------------------------------------------------

def build_tax_report(trades: list[TaxTrade], jurisdiction: Literal["TR","US","EU"]) -> TaxReport:
    rates = {"TR": (0.15,0.1,365), "US": (0.35,0.2,365), "EU": (0.3,0.18,365)}[jurisdiction]
    st_tax = lt_tax = gross = 0.0
    for t in trades:
        gross += t.pnl
        if t.pnl <= 0: continue
        if t.holding_days >= rates[2]: lt_tax += t.pnl * rates[1]
        else: st_tax += t.pnl * rates[0]
    return TaxReport(jurisdiction, _rd(st_tax,2), _rd(lt_tax,2), _rd(gross-st_tax-lt_tax,2))

# ---------------------------------------------------------------------------
# Execution Algorithms
# ---------------------------------------------------------------------------

def create_iceberg_slices(total_quantity: int, visible_quantity: int) -> list[IcebergSlice]:
    total, vis = max(0, round(total_quantity)), max(1, round(visible_quantity))
    if total <= 0: return []
    out, rem, seq = [], total, 1
    while rem > 0:
        q = min(vis, rem); out.append(IcebergSlice(seq, q)); rem -= q; seq += 1
    return out

def build_twap_schedule(total_quantity: float, start_iso: str, end_iso: str, slices: int) -> list[ExecutionSlice]:
    total = max(0, total_quantity); sc = max(1, round(slices))
    s = _parse_date(start_iso); e = _parse_date(end_iso)
    span = max(1000, int((e - s).total_seconds() * 1000))
    base = total / sc; rows = []; alloc = 0.0
    import datetime
    for i in range(sc):
        q = total - alloc if i == sc - 1 else int(base)
        alloc += q
        at = s + datetime.timedelta(milliseconds=span * i / max(1, sc - 1)) if sc > 1 else s
        rows.append(ExecutionSlice(at.isoformat(), max(0, q)))
    return rows

def build_vwap_schedule(total_quantity: float, projected_volume_curve: list[float]) -> list[ExecutionSlice]:
    total = max(0, total_quantity)
    if not projected_volume_curve or total <= 0: return []
    pos = [max(0, v) for v in projected_volume_curve]
    cs = sum(pos)
    base = pos if cs > 0 else [1.0]*len(projected_volume_curve)
    denom = sum(base)
    import datetime
    now = datetime.datetime.utcnow()
    alloc = 0.0
    out = []
    for i, w in enumerate(base):
        tgt = total - alloc if i == len(base)-1 else round(total * w / denom)
        alloc += tgt
        out.append(ExecutionSlice((now + datetime.timedelta(minutes=i)).isoformat(), tgt))
    return out

def build_bracket_order(entry: float, stop_pct: float, tp_pct: float, quantity: float) -> BracketOrder:
    b = max(1e-6, entry)
    return BracketOrder(_rd(b,4), _rd(b*(1-_clamp(stop_pct,0.05,95)/100),4),
                        _rd(b*(1+_clamp(tp_pct,0.05,300)/100),4), _rd(max(0,quantity),4))

def simulate_execution_with_slippage(order_quantity: float, side: Literal["buy","sell"],
                                     levels: list[OrderBookLevel], benchmark_price: float) -> ExecutionSimulationResult:
    qty = max(0, order_quantity)
    if not qty or not levels or benchmark_price <= 0:
        return ExecutionSimulationResult(0, 0, 0, qty)
    srt = sorted(levels, key=lambda l: l.price if side == "buy" else -l.price)
    rem, cost = qty, 0.0
    for lv in srt:
        if rem <= 0: break
        fill = min(rem, max(0, lv.size)); rem -= fill; cost += fill * lv.price
    filled = qty - rem
    avg = cost / filled if filled > 0 else 0
    slip = ((avg/benchmark_price - 1)*10000 if side == "buy" else (benchmark_price/avg - 1)*10000) if filled > 0 else 0
    return ExecutionSimulationResult(_rd(filled,4), _rd(avg,6), _rd(slip,4), _rd(rem,4))

# ---------------------------------------------------------------------------
# Compliance & Alerts
# ---------------------------------------------------------------------------

def run_compliance_rule_engine(record: TransactionRecord, rules: list[ComplianceRule]) -> list[ComplianceHit]:
    hits = []
    for rule in rules:
        raw = getattr(record, rule.field, None)
        obs = raw if isinstance(raw, (int, float)) else float(raw) if raw is not None else float('nan')
        if _compare(obs, rule.comparator, rule.threshold):
            hits.append(ComplianceHit(rule.id, rule.message, rule.severity))
    return hits

def monitor_position_limits(limits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted([l for l in limits if l["value"] > l["limit"]],
                  key=lambda l: l["value"] - l["limit"], reverse=True)

def detect_user_activity_anomalies(events: list[dict[str, str]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for e in events: counts[e["user_id"]] = counts.get(e["user_id"], 0) + 1
    vals = list(counts.values())
    avg = mean(vals); sd = sample_std_dev(vals)
    if sd <= 0: return []
    return sorted([{"user_id": u, "actions_per_hour": _rd(c,4), "z_score": _rd((c-avg)/sd,4)}
                   for u, c in counts.items() if (c-avg)/sd >= 2.2],
                  key=lambda x: x["z_score"], reverse=True)

def _get_metric(metrics: dict, path: str) -> float:
    cursor: Any = metrics
    for p in path.split("."):
        if not isinstance(cursor, dict): return float('nan')
        cursor = cursor.get(p)
    return cursor if isinstance(cursor, (int, float)) else float('nan')

def evaluate_alert_conditions(metrics: dict, conditions: list[AlertCondition],
                              state: dict[str, str | None], now_iso: str):
    now = _parse_date(now_iso)
    next_state = dict(state)
    alerts: list[NotificationAlert] = []
    for cond in conditions:
        val = _get_metric(metrics, cond.metric)
        if not _compare(val, cond.comparator, cond.threshold): continue
        last = _parse_date(state.get(cond.id, "") or "1970-01-01")
        cd_ms = max(0, cond.cooldown_sec) * 1000
        if state.get(cond.id) and (now - last).total_seconds()*1000 < cd_ms: continue
        alerts.append(NotificationAlert(
            cond.id, cond.name, cond.severity, cond.channels,
            f"{cond.name}: {cond.metric}={_rd(val,4)} breached {cond.comparator} {cond.threshold}",
            _rd(val,6), now_iso, cond.group_key or cond.id))
        next_state[cond.id] = now_iso
    return {"alerts": alerts, "nextState": next_state}

def group_alerts(alerts: list[NotificationAlert], window_sec: int = 120) -> list[NotificationAlert]:
    if not alerts: return []
    srt = sorted(alerts, key=lambda a: a.triggered_at)
    grouped: list[NotificationAlert] = []
    last_by: dict[str, str] = {}
    for a in srt:
        last = last_by.get(a.group_key)
        if last:
            dt = (_parse_date(a.triggered_at) - _parse_date(last)).total_seconds()
            if dt <= window_sec: continue
        grouped.append(a); last_by[a.group_key] = a.triggered_at
    return grouped

def build_escalation_plan(alert: NotificationAlert):
    if alert.severity == "critical":
        return [{"after_minutes": 0, "channel": "push", "action": "Instant push notification"},
                {"after_minutes": 1, "channel": "sms", "action": "Escalate to on-call desk"},
                {"after_minutes": 3, "channel": "email", "action": "Send incident summary"},
                {"after_minutes": 5, "channel": "webhook", "action": "Open incident ticket"}]
    if alert.severity == "warning":
        return [{"after_minutes": 0, "channel": "push", "action": "Notify portfolio manager"},
                {"after_minutes": 5, "channel": "email", "action": "Send warning digest"}]
    return [{"after_minutes": 0, "channel": "push", "action": "Record informational update"}]

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_tax_report_jurisdiction(trades: list[TaxTrade], jurisdiction: Literal["TR","US","EU"]) -> TaxReport:
    return build_tax_report(trades, jurisdiction)

def schedule_report_runs(schedules: list[dict[str, Any]], now_iso: str) -> list[dict[str, str]]:
    import datetime
    now = _parse_date(now_iso)
    out = []
    for s in schedules:
        nxt = now.replace(hour=s["hour_utc"], minute=s["minute_utc"], second=0, microsecond=0)
        while nxt <= now:
            if s["cadence"] == "daily": nxt += datetime.timedelta(days=1)
            elif s["cadence"] == "weekly": nxt += datetime.timedelta(weeks=1)
            else:
                m = nxt.month + 1
                y = nxt.year + (1 if m > 12 else 0)
                nxt = nxt.replace(year=y, month=(m-1)%12+1)
        out.append({"report_id": s["report_id"], "next_run_at": nxt.isoformat()})
    return out

def build_client_report_template(template: Literal["institutional","family_office","advisor"]):
    if template == "institutional":
        return {"template": template, "sections": ["Executive summary","Portfolio performance","Risk decomposition","Liquidity and counterparty profile","Compliance attestations"], "automation": "Monthly auto-delivery + quarterly board deck"}
    if template == "family_office":
        return {"template": template, "sections": ["Capital preservation dashboard","Drawdown and stress tests","Tax-aware realization report","Manager attribution"], "automation": "Monthly PDF + ad-hoc scenario pack"}
    return {"template": template, "sections": ["Client-level return summary","Benchmark comparison","Attribution highlights","Action items and watchlist"], "automation": "Weekly digest + end-of-month statement"}

def run_performance_snapshot(curve: list[SeriesPoint], benchmarks: dict[str, list[SeriesPoint]]):
    rets = curve_to_returns(curve)
    eq = returns_to_equity(rets, 1)
    return {"metrics": compute_performance_metrics(rets, equity_curve=eq),
            "risk_adjusted": compute_risk_adjusted_ratios(rets),
            "monthly_quarterly": build_monthly_quarterly_returns(rets),
            "benchmark": compare_benchmarks(rets, benchmarks)}

# ---------------------------------------------------------------------------
# Market Intelligence
# ---------------------------------------------------------------------------

def build_market_intelligence_snapshot(alternative_data, earnings, economic_impact_score,
                                       sentiment: SentimentSummary, insider, short_interest):
    by_sym: dict[str, float] = {}
    alerts: list[str] = []
    for r in alternative_data:
        by_sym[r.symbol] = by_sym.get(r.symbol, 0) + r.z_score * 0.8
    for r in earnings:
        if r.expected_eps == 0: continue
        surp = (r.actual_eps - r.expected_eps) / abs(r.expected_eps)
        by_sym[r.symbol] = by_sym.get(r.symbol, 0) + surp * 12
        if abs(surp) >= 0.2: alerts.append(f"{r.symbol} earnings surprise {_rd(surp*100,2)}%")
    for r in insider:
        sig = math.copysign(1, r.net_buy_value) * math.log1p(abs(r.net_buy_value)) / 8
        by_sym[r.symbol] = by_sym.get(r.symbol, 0) + sig
    for r in short_interest:
        pressure = r.short_float_pct * 0.35 + r.borrow_fee_pct * 0.65
        by_sym[r.symbol] = by_sym.get(r.symbol, 0) - pressure / 10
        if r.borrow_fee_pct > 15: alerts.append(f"{r.symbol} elevated borrow fee {r.borrow_fee_pct:.2f}%")
    for sym, sc in sentiment.by_symbol.items():
        by_sym[sym] = by_sym.get(sym, 0) + sc * 10
    sym_sc = {s: _rd(v,4) for s, v in by_sym.items()}
    conv = mean(list(sym_sc.values())) + economic_impact_score * 0.02 + sentiment.overall_score * 8
    return {"conviction_score": _rd(conv,4), "by_symbol": sym_sc, "alerts": alerts}
