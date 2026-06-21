"""Trading analytics: crypto, forex, options, futures, commodity spreads, fund screening.

Split out of the former professional.py monolith. Covers trade-plan sizing,
pip/Greeks/margin math, cross-commodity spread analysis, and multi-factor
fund/ETF screening.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from ..core_metrics import normal_cdf
from .._shared import _clamp, _to_fixed

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

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
    return CryptoTradePlan(inp.pair.upper(), inp.side, _to_fixed(notional,2), _to_fixed(margin,2),
                           _to_fixed(qty,6), _to_fixed(liq,4), _to_fixed(risk_cap,2), _to_fixed(fees,2))

def compute_forex_pip_value(pair: str, lot_size: float, account_conversion_rate: float = 1.0) -> ForexPipResult:
    p = pair.upper()
    pip_size = 0.01 if p.endswith("JPY") else 0.0001
    pv = lot_size * pip_size
    conv = max(0, account_conversion_rate) if math.isfinite(account_conversion_rate) else 1
    return ForexPipResult(p, pip_size, _to_fixed(pv,6), _to_fixed(pv*conv,6))

def compute_option_greeks(inp: OptionGreeksInput) -> OptionGreeksResult:
    s, k = max(1e-4, inp.spot), max(1e-4, inp.strike)
    t = max(1/365, inp.time_years)
    sig = _clamp(inp.volatility, 1e-4, 5)
    r = _clamp(inp.risk_free_rate, -1, 1)
    st = math.sqrt(t)
    d1 = (math.log(s/k) + (r + 0.5*sig*sig)*t) / (sig*st)
    d2 = d1 - sig*st
    nd1, nd2 = normal_cdf(d1), normal_cdf(d2)
    pdf1 = _normal_pdf(d1)
    disc = math.exp(-r*t)
    call_p = s*nd1 - k*disc*nd2
    put_p = k*disc*normal_cdf(-d2) - s*normal_cdf(-d1)
    delta = nd1 if inp.option_type == "call" else nd1 - 1
    gamma = pdf1 / (s*sig*st)
    vega = s*pdf1*st / 100
    tc = (-s*pdf1*sig/(2*st) - r*k*disc*nd2) / 365
    tp = (-s*pdf1*sig/(2*st) + r*k*disc*normal_cdf(-d2)) / 365
    rc = k*t*disc*nd2 / 100
    rp = -k*t*disc*normal_cdf(-d2) / 100
    is_call = inp.option_type == "call"
    return OptionGreeksResult(_to_fixed(delta,6), _to_fixed(gamma,6), _to_fixed(tc if is_call else tp,6),
                              _to_fixed(vega,6), _to_fixed(rc if is_call else rp,6),
                              _to_fixed(call_p if is_call else put_p,6))

def compute_futures_margin(symbol: str, contracts: int, price: float,
                           contract_size: float, initial_margin_rate: float,
                           maintenance_margin_rate: float, tick_size: float,
                           tick_value: float) -> FuturesMarginResult:
    c = max(1, round(contracts))
    n = max(0, price) * max(0, contract_size) * c
    return FuturesMarginResult(symbol.upper(), _to_fixed(n,2),
                               _to_fixed(n*_clamp(initial_margin_rate,0,1),2),
                               _to_fixed(n*_clamp(maintenance_margin_rate,0,1),2),
                               _to_fixed(max(0,tick_value)*c,4))

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
        results.append(FundScreenResult(c.symbol, _to_fixed(sc*100,2), sc >= 0.62))
    return sorted(results, key=lambda r: r.score, reverse=True)

# ---------------------------------------------------------------------------
# Commodity Spread
# ---------------------------------------------------------------------------

def compute_commodity_spread(front_price: float, back_price: float,
                             multiplier: float, contracts: int) -> CommoditySpreadResult:
    sp = front_price - back_price
    return CommoditySpreadResult(_to_fixed(sp,4),
                                 _to_fixed(sp/front_price*100,4) if front_price > 0 else 0,
                                 _to_fixed(sp*multiplier*contracts,2))


__all__ = [
    "CryptoTradeInput",
    "CryptoTradePlan",
    "ForexPipResult",
    "OptionGreeksInput",
    "OptionGreeksResult",
    "FuturesMarginResult",
    "FundCandidate",
    "FundScreenResult",
    "CommoditySpreadResult",
    "build_crypto_trade_plan",
    "compute_forex_pip_value",
    "compute_option_greeks",
    "compute_futures_margin",
    "screen_funds",
    "compute_commodity_spread",
]
