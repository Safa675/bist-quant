"""Execution algorithms: iceberg, TWAP, VWAP, bracket orders, slippage simulation.

Split out of the former professional.py monolith. Covers order slicing for
minimizing market impact and benchmark execution analysis.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Literal

from .._shared import _clamp, _parse_date, _to_fixed

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

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
    now = datetime.datetime.now(datetime.timezone.utc)
    alloc = 0.0
    out = []
    for i, w in enumerate(base):
        tgt = total - alloc if i == len(base)-1 else round(total * w / denom)
        alloc += tgt
        out.append(ExecutionSlice((now + datetime.timedelta(minutes=i)).isoformat(), tgt))
    return out

def build_bracket_order(entry: float, stop_pct: float, tp_pct: float, quantity: float) -> BracketOrder:
    b = max(1e-6, entry)
    return BracketOrder(_to_fixed(b,4), _to_fixed(b*(1-_clamp(stop_pct,0.05,95)/100),4),
                        _to_fixed(b*(1+_clamp(tp_pct,0.05,300)/100),4), _to_fixed(max(0,quantity),4))

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
    return ExecutionSimulationResult(_to_fixed(filled,4), _to_fixed(avg,6), _to_fixed(slip,4), _to_fixed(rem,4))


__all__ = [
    "IcebergSlice",
    "ExecutionSlice",
    "OrderBookLevel",
    "ExecutionSimulationResult",
    "BracketOrder",
    "create_iceberg_slices",
    "build_twap_schedule",
    "build_vwap_schedule",
    "build_bracket_order",
    "simulate_execution_with_slippage",
]
