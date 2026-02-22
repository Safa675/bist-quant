"""
Realtime portfolio valuation from live ticks.

Migrated from bist_quant.ai/api/realtime_api.py (get_portfolio function).
"""

from __future__ import annotations

from typing import Any

from .quotes import _quote_payload, _utc_iso_now, normalize_symbol, to_float


def get_portfolio(payload: dict[str, Any]) -> dict[str, Any]:
    """Compute portfolio valuation from holdings and optional cost basis.

    Args:
        payload: dict with ``holdings`` (symbol → quantity) and optional
                 ``cost_basis`` (symbol → unit cost).

    Returns:
        Portfolio summary with positions, total value, and optional PnL.
    """
    holdings_raw = payload.get("holdings", payload)
    cost_basis_raw = payload.get("cost_basis", {})

    if not isinstance(holdings_raw, dict):
        return {"error": "Holdings must be an object"}

    holdings: dict[str, float] = {}
    for symbol, qty in holdings_raw.items():
        norm = normalize_symbol(symbol)
        if not norm:
            continue
        quantity = to_float(qty)
        if quantity is None:
            continue
        holdings[norm] = quantity

    cost_basis: dict[str, float] = {}
    if isinstance(cost_basis_raw, dict):
        for symbol, cost in cost_basis_raw.items():
            norm = normalize_symbol(symbol)
            if not norm:
                continue
            parsed_cost = to_float(cost)
            if parsed_cost is not None:
                cost_basis[norm] = parsed_cost

    if not holdings:
        return {"error": "No valid holdings provided"}

    quotes = {symbol: _quote_payload(symbol) for symbol in holdings.keys()}
    positions: list[dict[str, Any]] = []
    total_value = 0.0
    total_cost = 0.0
    total_value_with_cost = 0.0
    positions_with_cost_basis = 0

    for symbol, quantity in holdings.items():
        quote = quotes.get(symbol, {})
        price = to_float(quote.get("last_price"))
        if price is None:
            positions.append(
                {
                    "symbol": symbol,
                    "quantity": quantity,
                    "error": quote.get("error", "No price data"),
                }
            )
            continue

        market_value = price * quantity
        total_value += market_value
        row: dict[str, Any] = {
            "symbol": symbol,
            "quantity": quantity,
            "price": price,
            "market_value": market_value,
            "change_pct": quote.get("change_pct"),
        }

        if symbol in cost_basis:
            unit_cost = cost_basis[symbol]
            cost = unit_cost * quantity
            pnl = market_value - cost
            pnl_pct = (pnl / cost * 100.0) if cost > 0 else 0.0
            total_cost += cost
            total_value_with_cost += market_value
            positions_with_cost_basis += 1
            row.update(
                {
                    "cost_basis": unit_cost,
                    "cost": cost,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                }
            )
        positions.append(row)

    result: dict[str, Any] = {
        "total_value": total_value,
        "positions": sorted(positions, key=lambda x: x.get("market_value", 0.0), reverse=True),
        "timestamp": _utc_iso_now(),
    }
    if positions_with_cost_basis > 0:
        total_pnl = total_value_with_cost - total_cost
        result.update(
            {
                "total_cost": total_cost,
                "total_pnl": total_pnl,
                "total_pnl_pct": (total_pnl / total_cost * 100.0) if total_cost > 0 else 0.0,
                "priced_with_cost_basis": positions_with_cost_basis,
            }
        )
    return result
