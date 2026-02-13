#!/usr/bin/env python3
"""
Real-Time Quote API Script

Provides JSON output for real-time quotes to be consumed by the
Next.js dashboard API endpoint.

Usage:
    # Get single quote
    python realtime_api.py quote THYAO

    # Get batch quotes
    python realtime_api.py quotes THYAO,AKBNK,GARAN

    # Get index quotes
    python realtime_api.py index XU030

    # Get portfolio snapshot
    python realtime_api.py portfolio '{"THYAO": 100, "AKBNK": 200}'

    # Get market summary
    python realtime_api.py market
"""

import sys
import json
import argparse
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "data" / "Fetcher-Scrapper"))


def get_quote(symbol: str) -> dict:
    """Get quote for a single symbol."""
    from realtime_stream import RealtimeQuoteService

    try:
        service = RealtimeQuoteService()
        return service.get_quote(symbol)
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}


def get_quotes(symbols: list[str]) -> dict:
    """Get quotes for multiple symbols."""
    from realtime_stream import RealtimeQuoteService

    try:
        service = RealtimeQuoteService()
        quotes = service.get_quotes_batch(symbols)
        return {
            "quotes": quotes,
            "count": len(quotes),
            "timestamp": quotes.get(symbols[0], {}).get("timestamp"),
        }
    except Exception as e:
        return {"error": str(e)}


def get_index_quotes(index: str) -> dict:
    """Get quotes for all stocks in an index."""
    from realtime_stream import RealtimeQuoteService

    try:
        service = RealtimeQuoteService()
        quotes = service.get_index_quotes(index)

        if "error" in quotes:
            return quotes

        # Calculate summary stats
        prices = [q.get("last_price") for q in quotes.values() if q.get("last_price")]
        changes = [q.get("change_pct") for q in quotes.values() if q.get("change_pct")]

        gainers = sorted(
            [(s, q.get("change_pct", 0)) for s, q in quotes.items() if q.get("change_pct")],
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        losers = sorted(
            [(s, q.get("change_pct", 0)) for s, q in quotes.items() if q.get("change_pct")],
            key=lambda x: x[1],
        )[:5]

        return {
            "index": index,
            "count": len(quotes),
            "quotes": quotes,
            "summary": {
                "avg_change_pct": sum(changes) / len(changes) if changes else 0,
                "gainers_count": sum(1 for c in changes if c > 0),
                "losers_count": sum(1 for c in changes if c < 0),
                "unchanged_count": sum(1 for c in changes if c == 0),
            },
            "top_gainers": [{"symbol": s, "change_pct": c} for s, c in gainers],
            "top_losers": [{"symbol": s, "change_pct": c} for s, c in losers],
        }
    except Exception as e:
        return {"error": str(e)}


def get_portfolio_snapshot(holdings: dict, cost_basis: dict = None) -> dict:
    """Get portfolio snapshot."""
    from realtime_stream import RealtimeQuoteService

    try:
        service = RealtimeQuoteService()
        return service.get_portfolio_snapshot(holdings, cost_basis)
    except Exception as e:
        return {"error": str(e)}


def get_market_summary() -> dict:
    """Get market summary."""
    from realtime_stream import RealtimeQuoteService

    try:
        service = RealtimeQuoteService()
        return service.get_market_summary()
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Real-time quote API")
    parser.add_argument(
        "command",
        choices=["quote", "quotes", "index", "portfolio", "market"],
        help="Command to execute",
    )
    parser.add_argument(
        "args",
        nargs="?",
        default="",
        help="Command arguments (symbol, symbols, index name, or JSON)",
    )

    args = parser.parse_args()

    result = {}

    if args.command == "quote":
        if not args.args:
            result = {"error": "Symbol required"}
        else:
            result = get_quote(args.args)

    elif args.command == "quotes":
        if not args.args:
            result = {"error": "Symbols required (comma-separated)"}
        else:
            symbols = [s.strip() for s in args.args.split(",")]
            result = get_quotes(symbols)

    elif args.command == "index":
        index = args.args or "XU100"
        result = get_index_quotes(index)

    elif args.command == "portfolio":
        if not args.args:
            result = {"error": "Holdings JSON required"}
        else:
            try:
                holdings = json.loads(args.args)
                result = get_portfolio_snapshot(holdings)
            except json.JSONDecodeError:
                result = {"error": "Invalid JSON"}

    elif args.command == "market":
        result = get_market_summary()

    # Output JSON
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
