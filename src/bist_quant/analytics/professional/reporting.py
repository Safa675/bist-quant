"""Reporting analytics: sentiment, benchmarks, attribution, tax, reporting, market intelligence.

Split out of the former professional.py monolith. Covers news/social sentiment
scoring, risk-adjusted ratios, monthly/quarterly return aggregation, benchmark
comparison, performance attribution, tax computation, report scheduling, and
market intelligence snapshots.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass
from typing import Any, Literal

from ..core_metrics import (
    SeriesPoint,
    align_series_by_date,
    correlation,
    covariance,
    curve_to_returns,
    mean,
    returns_to_equity,
    sample_std_dev,
)
from .._shared import _parse_date, _to_fixed

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

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
    sym_scores = {s: _to_fixed(mean(v),4) for s, v in by_sym.items()}
    srt = sorted(sym_scores.items(), key=lambda x: x[1], reverse=True)
    return SentimentSummary(_to_fixed(mean(all_sc),4), sym_scores,
                            [s for s, v in srt if v > 0.18][:8],
                            [s for s, v in srt if v < -0.18][-8:])

# ---------------------------------------------------------------------------
# Risk-Adjusted Ratios / Monthly Returns / Benchmarks / Attribution
# ---------------------------------------------------------------------------

def compute_risk_adjusted_ratios(returns: list[SeriesPoint], risk_free_rate_pct: float = 4) -> dict[str, float]:
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
    return {"sharpe": _to_fixed(ae*252/av,4) if av > 0 else 0,
            "sortino": _to_fixed(ae*252/ad,4) if ad > 0 else 0,
            "calmar": _to_fixed(ar/abs(md),4) if md < 0 else 0}

def build_monthly_quarterly_returns(returns: list[SeriesPoint]) -> dict[str, list[dict[str, Any]]]:
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
            out.append({"period": p, "return_pct": _to_fixed((cum-1)*100, 4)})
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
        vb = sample_std_dev(bench) ** 2
        b = covariance(strat, bench)/vb if vb > 0 else 0
        a = (mean(strat) - b*mean(bench))*252*100
        active = [strat[i]-bench[i] for i in range(len(strat))]
        te = sample_std_dev(active)*math.sqrt(252)*100
        ir = mean(active)*252*100/te if te > 0 else 0
        out.append(BenchmarkComparison(name, _to_fixed(c,4), _to_fixed(b,4), _to_fixed(a,4), _to_fixed(te,4), _to_fixed(ir,4)))
    return sorted(out, key=lambda x: x.alpha_pct, reverse=True)

def compute_attribution(drivers: list[AttributionInput]) -> AttributionResult:
    total = sum(d.contribution_pct for d in drivers)
    base = abs(total) if abs(total) > 1e-5 else 1
    norm = sorted([{"driver": d.driver, "share_pct": _to_fixed(d.contribution_pct/base*100,4)} for d in drivers],
                  key=lambda x: abs(x["share_pct"]), reverse=True)
    return AttributionResult(_to_fixed(total,4), norm)

def decompose_portfolio_risk_factors(weights: dict[str, float],
                                     exposures_by_asset: dict[str, dict[str, float]]) -> list[RiskFactorContribution]:
    ft: dict[str, float] = {}
    for asset, w in weights.items():
        exp = exposures_by_asset.get(asset, {})
        for f, v in exp.items():
            ft[f] = ft.get(f, 0) + w * v
    at = sum(abs(v) for v in ft.values()) or 1
    return sorted([RiskFactorContribution(f, _to_fixed(we,6), _to_fixed(abs(we)/at*100,4)) for f, we in ft.items()],
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
    return TaxReport(jurisdiction, _to_fixed(st_tax,2), _to_fixed(lt_tax,2), _to_fixed(gross-st_tax-lt_tax,2))

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def schedule_report_runs(schedules: list[dict[str, Any]], now_iso: str) -> list[dict[str, str]]:
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

def build_client_report_template(template: Literal["institutional","family_office","advisor"]) -> dict[str, Any]:
    if template == "institutional":
        return {"template": template, "sections": ["Executive summary","Portfolio performance","Risk decomposition","Liquidity and counterparty profile","Compliance attestations"], "automation": "Monthly auto-delivery + quarterly board deck"}
    if template == "family_office":
        return {"template": template, "sections": ["Capital preservation dashboard","Drawdown and stress tests","Tax-aware realization report","Manager attribution"], "automation": "Monthly PDF + ad-hoc scenario pack"}
    return {"template": template, "sections": ["Client-level return summary","Benchmark comparison","Attribution highlights","Action items and watchlist"], "automation": "Weekly digest + end-of-month statement"}

# ---------------------------------------------------------------------------
# Market Intelligence
# ---------------------------------------------------------------------------

def build_market_intelligence_snapshot(alternative_data, earnings, economic_impact_score,
                                       sentiment: SentimentSummary, insider, short_interest) -> dict[str, Any]:
    by_sym: dict[str, float] = {}
    alerts: list[str] = []
    for r in alternative_data:
        by_sym[r.symbol] = by_sym.get(r.symbol, 0) + r.z_score * 0.8
    for r in earnings:
        if r.expected_eps == 0: continue
        surp = (r.actual_eps - r.expected_eps) / abs(r.expected_eps)
        by_sym[r.symbol] = by_sym.get(r.symbol, 0) + surp * 12
        if abs(surp) >= 0.2: alerts.append(f"{r.symbol} earnings surprise {_to_fixed(surp*100,2)}%")
    for r in insider:
        sig = math.copysign(1, r.net_buy_value) * math.log1p(abs(r.net_buy_value)) / 8
        by_sym[r.symbol] = by_sym.get(r.symbol, 0) + sig
    for r in short_interest:
        pressure = r.short_float_pct * 0.35 + r.borrow_fee_pct * 0.65
        by_sym[r.symbol] = by_sym.get(r.symbol, 0) - pressure / 10
        if r.borrow_fee_pct > 15: alerts.append(f"{r.symbol} elevated borrow fee {r.borrow_fee_pct:.2f}%")
    for sym, sc in sentiment.by_symbol.items():
        by_sym[sym] = by_sym.get(sym, 0) + sc * 10
    sym_sc = {s: _to_fixed(v,4) for s, v in by_sym.items()}
    conv = mean(list(sym_sc.values())) + economic_impact_score * 0.02 + sentiment.overall_score * 8
    return {"conviction_score": _to_fixed(conv,4), "by_symbol": sym_sc, "alerts": alerts}


__all__ = [
    "SentimentItem",
    "SentimentSummary",
    "BenchmarkComparison",
    "AttributionInput",
    "AttributionResult",
    "TaxTrade",
    "TaxReport",
    "EconomicEvent",
    "EarningsSurprise",
    "InsiderActivity",
    "ShortInterest",
    "RiskFactorContribution",
    "analyze_sentiment",
    "compute_risk_adjusted_ratios",
    "build_monthly_quarterly_returns",
    "compare_benchmarks",
    "compute_attribution",
    "decompose_portfolio_risk_factors",
    "build_tax_report",
    "schedule_report_runs",
    "build_client_report_template",
    "build_market_intelligence_snapshot",
]
