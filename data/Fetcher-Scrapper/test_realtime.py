#!/usr/bin/env python3
"""
Test script for real-time streaming functionality.

Run from project root:
    python data/Fetcher-Scrapper/test_realtime.py
"""

import sys
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "data" / "Fetcher-Scrapper"))


def test_quote_service():
    """Test RealtimeQuoteService."""
    print("=" * 60)
    print("TEST 1: RealtimeQuoteService")
    print("=" * 60)

    from realtime_stream import RealtimeQuoteService

    service = RealtimeQuoteService(cache_ttl=60)

    # Test single quote
    print("\n1. Getting single quote (THYAO)...")
    quote = service.get_quote("THYAO")
    if "error" not in quote:
        print(f"   Symbol: {quote.get('symbol')}")
        print(f"   Price: {quote.get('last_price')}")
        print(f"   Change: {quote.get('change_pct')}%")
        print(f"   Volume: {quote.get('volume')}")
    else:
        print(f"   Error: {quote.get('error')}")

    # Test batch quotes
    print("\n2. Getting batch quotes...")
    symbols = ["AKBNK", "GARAN", "EREGL", "TUPRS", "SISE"]
    quotes = service.get_quotes_batch(symbols)
    print(f"   Fetched {len(quotes)} quotes")
    for symbol, q in quotes.items():
        if "error" not in q:
            change_pct = q.get("change_pct")
            change_str = f"{change_pct:+.2f}%" if isinstance(change_pct, (int, float)) else "N/A"
            print(f"   {symbol}: {q.get('last_price')} ({change_str})")
        else:
            print(f"   {symbol}: Error - {q.get('error')}")

    # Test cache
    print("\n3. Testing cache (should be instant)...")
    import time
    start = time.time()
    quote2 = service.get_quote("THYAO")
    elapsed = time.time() - start
    print(f"   Cached quote retrieved in {elapsed*1000:.1f}ms")

    # Test portfolio snapshot
    print("\n4. Getting portfolio snapshot...")
    holdings = {"THYAO": 100, "AKBNK": 200, "GARAN": 150}
    cost_basis = {"THYAO": 250.0, "AKBNK": 45.0, "GARAN": 60.0}
    snapshot = service.get_portfolio_snapshot(holdings, cost_basis)

    if "error" not in snapshot:
        print(f"   Total Value: {snapshot.get('total_value'):,.2f} TRY")
        print(f"   Total Cost: {snapshot.get('total_cost'):,.2f} TRY")
        print(f"   Total P&L: {snapshot.get('total_pnl'):,.2f} TRY ({snapshot.get('total_pnl_pct', 0):+.2f}%)")
        print(f"   Positions: {len(snapshot.get('positions', []))}")
    else:
        print(f"   Error: {snapshot.get('error')}")

    print("\n" + "=" * 60)
    print("RealtimeQuoteService tests completed!")
    print("=" * 60)


def test_index_quotes():
    """Test getting all quotes for an index."""
    print("\n" + "=" * 60)
    print("TEST 2: Index Quotes (XU030)")
    print("=" * 60)

    from realtime_stream import RealtimeQuoteService

    service = RealtimeQuoteService()

    print("\n1. Fetching XU030 quotes...")
    quotes = service.get_index_quotes("XU030", max_symbols=10)

    if "error" in quotes:
        print(f"   Error: {quotes.get('error')}")
        return

    print(f"   Fetched {len(quotes)} quotes")

    # Calculate stats
    valid_quotes = {s: q for s, q in quotes.items() if "error" not in q and q.get("change_pct") is not None}
    if valid_quotes:
        changes = [q["change_pct"] for q in valid_quotes.values()]
        avg_change = sum(changes) / len(changes)
        gainers = sum(1 for c in changes if c > 0)
        losers = sum(1 for c in changes if c < 0)

        print(f"\n   Average change: {avg_change:+.2f}%")
        print(f"   Gainers: {gainers}")
        print(f"   Losers: {losers}")

        # Top gainers
        sorted_quotes = sorted(valid_quotes.items(), key=lambda x: x[1].get("change_pct", 0), reverse=True)
        print("\n   Top 3 gainers:")
        for symbol, q in sorted_quotes[:3]:
            print(f"     {symbol}: {q.get('change_pct', 0):+.2f}%")

        print("\n   Top 3 losers:")
        for symbol, q in sorted_quotes[-3:]:
            print(f"     {symbol}: {q.get('change_pct', 0):+.2f}%")

    print("\n" + "=" * 60)
    print("Index quotes test completed!")
    print("=" * 60)


def test_market_summary():
    """Test market summary."""
    print("\n" + "=" * 60)
    print("TEST 3: Market Summary")
    print("=" * 60)

    from realtime_stream import RealtimeQuoteService

    service = RealtimeQuoteService()

    print("\n1. Getting market summary...")
    summary = service.get_market_summary()

    print(f"   Timestamp: {summary.get('timestamp')}")

    xu100 = summary.get("xu100", {})
    if "error" not in xu100:
        change_pct = xu100.get("change_pct")
        change_str = f"{change_pct:+.2f}%" if isinstance(change_pct, (int, float)) else "N/A"
        print(f"   XU100: {xu100.get('value')} ({change_str})")
    else:
        print(f"   XU100: {xu100.get('error', 'N/A')}")

    xu030 = summary.get("xu030", {})
    if "error" not in xu030:
        change_pct = xu030.get("change_pct")
        change_str = f"{change_pct:+.2f}%" if isinstance(change_pct, (int, float)) else "N/A"
        print(f"   XU030: {xu030.get('value')} ({change_str})")
    else:
        print(f"   XU030: {xu030.get('error', 'N/A')}")

    usdtry = summary.get("usdtry", {})
    if "error" not in usdtry:
        change_pct = usdtry.get("change_pct")
        change_str = f"{change_pct:+.2f}%" if isinstance(change_pct, (int, float)) else "N/A"
        print(f"   USD/TRY: {usdtry.get('rate')} ({change_str})")
    else:
        print(f"   USD/TRY: {usdtry.get('error', 'N/A')}")

    print("\n" + "=" * 60)
    print("Market summary test completed!")
    print("=" * 60)


def test_realtime_api():
    """Test the realtime_api.py CLI."""
    print("\n" + "=" * 60)
    print("TEST 4: Realtime API CLI")
    print("=" * 60)

    import subprocess
    import json

    script_path = PROJECT_ROOT / "data" / "Fetcher-Scrapper" / "realtime_api.py"

    # Test quote command
    print("\n1. Testing 'quote' command...")
    result = subprocess.run(
        ["python3", str(script_path), "quote", "THYAO"],
        capture_output=True,
        text=True,
        cwd=str(script_path.parent),
    )
    if result.returncode == 0:
        data = json.loads(result.stdout)
        print(f"   THYAO: {data.get('last_price')} TRY")
    else:
        print(f"   Error: {result.stderr}")

    # Test quotes command
    print("\n2. Testing 'quotes' command...")
    result = subprocess.run(
        ["python3", str(script_path), "quotes", "AKBNK,GARAN,EREGL"],
        capture_output=True,
        text=True,
        cwd=str(script_path.parent),
    )
    if result.returncode == 0:
        data = json.loads(result.stdout)
        print(f"   Fetched {data.get('count', 0)} quotes")
    else:
        print(f"   Error: {result.stderr}")

    # Test market command
    print("\n3. Testing 'market' command...")
    result = subprocess.run(
        ["python3", str(script_path), "market"],
        capture_output=True,
        text=True,
        cwd=str(script_path.parent),
    )
    if result.returncode == 0:
        data = json.loads(result.stdout)
        print(f"   Timestamp: {data.get('timestamp')}")
    else:
        print(f"   Error: {result.stderr}")

    print("\n" + "=" * 60)
    print("Realtime API CLI test completed!")
    print("=" * 60)


def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# REAL-TIME STREAMING TEST SUITE")
    print("#" * 60)

    try:
        test_quote_service()
    except Exception as e:
        print(f"\n❌ Quote service test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_index_quotes()
    except Exception as e:
        print(f"\n❌ Index quotes test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_market_summary()
    except Exception as e:
        print(f"\n❌ Market summary test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_realtime_api()
    except Exception as e:
        print(f"\n❌ Realtime API CLI test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "#" * 60)
    print("# ALL TESTS COMPLETED")
    print("#" * 60)


if __name__ == "__main__":
    main()
