#!/usr/bin/env python3
"""
Test script for real-time streaming functionality.

Run from project root:
    python data/Fetcher-Scrapper/test_realtime.py
"""

import logging
import sys
from pathlib import Path
logger = logging.getLogger(__name__)

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "data" / "Fetcher-Scrapper"))


def test_quote_service():
    """Test RealtimeQuoteService."""
    logger.info("=" * 60)
    logger.info("TEST 1: RealtimeQuoteService")
    logger.info("=" * 60)

    from realtime_stream import RealtimeQuoteService

    service = RealtimeQuoteService(cache_ttl=60)

    # Test single quote
    logger.info("\n1. Getting single quote (THYAO)...")
    quote = service.get_quote("THYAO")
    if "error" not in quote:
        logger.info(f"   Symbol: {quote.get('symbol')}")
        logger.info(f"   Price: {quote.get('last_price')}")
        logger.info(f"   Change: {quote.get('change_pct')}%")
        logger.info(f"   Volume: {quote.get('volume')}")
    else:
        logger.info(f"   Error: {quote.get('error')}")

    # Test batch quotes
    logger.info("\n2. Getting batch quotes...")
    symbols = ["AKBNK", "GARAN", "EREGL", "TUPRS", "SISE"]
    quotes = service.get_quotes_batch(symbols)
    logger.info(f"   Fetched {len(quotes)} quotes")
    for symbol, q in quotes.items():
        if "error" not in q:
            change_pct = q.get("change_pct")
            change_str = f"{change_pct:+.2f}%" if isinstance(change_pct, (int, float)) else "N/A"
            logger.info(f"   {symbol}: {q.get('last_price')} ({change_str})")
        else:
            logger.info(f"   {symbol}: Error - {q.get('error')}")

    # Test cache
    logger.info("\n3. Testing cache (should be instant)...")
    import time
    start = time.time()
    quote2 = service.get_quote("THYAO")
    elapsed = time.time() - start
    logger.info(f"   Cached quote retrieved in {elapsed*1000:.1f}ms")

    # Test portfolio snapshot
    logger.info("\n4. Getting portfolio snapshot...")
    holdings = {"THYAO": 100, "AKBNK": 200, "GARAN": 150}
    cost_basis = {"THYAO": 250.0, "AKBNK": 45.0, "GARAN": 60.0}
    snapshot = service.get_portfolio_snapshot(holdings, cost_basis)

    if "error" not in snapshot:
        logger.info(f"   Total Value: {snapshot.get('total_value'):,.2f} TRY")
        logger.info(f"   Total Cost: {snapshot.get('total_cost'):,.2f} TRY")
        logger.info(f"   Total P&L: {snapshot.get('total_pnl'):,.2f} TRY ({snapshot.get('total_pnl_pct', 0):+.2f}%)")
        logger.info(f"   Positions: {len(snapshot.get('positions', []))}")
    else:
        logger.info(f"   Error: {snapshot.get('error')}")

    logger.info("\n" + "=" * 60)
    logger.info("RealtimeQuoteService tests completed!")
    logger.info("=" * 60)


def test_index_quotes():
    """Test getting all quotes for an index."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Index Quotes (XU030)")
    logger.info("=" * 60)

    from realtime_stream import RealtimeQuoteService

    service = RealtimeQuoteService()

    logger.info("\n1. Fetching XU030 quotes...")
    quotes = service.get_index_quotes("XU030", max_symbols=10)

    if "error" in quotes:
        logger.info(f"   Error: {quotes.get('error')}")
        return

    logger.info(f"   Fetched {len(quotes)} quotes")

    # Calculate stats
    valid_quotes = {s: q for s, q in quotes.items() if "error" not in q and q.get("change_pct") is not None}
    if valid_quotes:
        changes = [q["change_pct"] for q in valid_quotes.values()]
        avg_change = sum(changes) / len(changes)
        gainers = sum(1 for c in changes if c > 0)
        losers = sum(1 for c in changes if c < 0)

        logger.info(f"\n   Average change: {avg_change:+.2f}%")
        logger.info(f"   Gainers: {gainers}")
        logger.info(f"   Losers: {losers}")

        # Top gainers
        sorted_quotes = sorted(valid_quotes.items(), key=lambda x: x[1].get("change_pct", 0), reverse=True)
        logger.info("\n   Top 3 gainers:")
        for symbol, q in sorted_quotes[:3]:
            logger.info(f"     {symbol}: {q.get('change_pct', 0):+.2f}%")

        logger.info("\n   Top 3 losers:")
        for symbol, q in sorted_quotes[-3:]:
            logger.info(f"     {symbol}: {q.get('change_pct', 0):+.2f}%")

    logger.info("\n" + "=" * 60)
    logger.info("Index quotes test completed!")
    logger.info("=" * 60)


def test_market_summary():
    """Test market summary."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Market Summary")
    logger.info("=" * 60)

    from realtime_stream import RealtimeQuoteService

    service = RealtimeQuoteService()

    logger.info("\n1. Getting market summary...")
    summary = service.get_market_summary()

    logger.info(f"   Timestamp: {summary.get('timestamp')}")

    xu100 = summary.get("xu100", {})
    if "error" not in xu100:
        change_pct = xu100.get("change_pct")
        change_str = f"{change_pct:+.2f}%" if isinstance(change_pct, (int, float)) else "N/A"
        logger.info(f"   XU100: {xu100.get('value')} ({change_str})")
    else:
        logger.info(f"   XU100: {xu100.get('error', 'N/A')}")

    xu030 = summary.get("xu030", {})
    if "error" not in xu030:
        change_pct = xu030.get("change_pct")
        change_str = f"{change_pct:+.2f}%" if isinstance(change_pct, (int, float)) else "N/A"
        logger.info(f"   XU030: {xu030.get('value')} ({change_str})")
    else:
        logger.info(f"   XU030: {xu030.get('error', 'N/A')}")

    usdtry = summary.get("usdtry", {})
    if "error" not in usdtry:
        change_pct = usdtry.get("change_pct")
        change_str = f"{change_pct:+.2f}%" if isinstance(change_pct, (int, float)) else "N/A"
        logger.info(f"   USD/TRY: {usdtry.get('rate')} ({change_str})")
    else:
        logger.info(f"   USD/TRY: {usdtry.get('error', 'N/A')}")

    logger.info("\n" + "=" * 60)
    logger.info("Market summary test completed!")
    logger.info("=" * 60)


def test_realtime_api():
    """Test the realtime_api.py CLI."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Realtime API CLI")
    logger.info("=" * 60)

    import subprocess
    import json

    script_path = PROJECT_ROOT / "data" / "Fetcher-Scrapper" / "realtime_api.py"

    # Test quote command
    logger.info("\n1. Testing 'quote' command...")
    result = subprocess.run(
        ["python3", str(script_path), "quote", "THYAO"],
        capture_output=True,
        text=True,
        cwd=str(script_path.parent),
    )
    if result.returncode == 0:
        data = json.loads(result.stdout)
        logger.info(f"   THYAO: {data.get('last_price')} TRY")
    else:
        logger.info(f"   Error: {result.stderr}")

    # Test quotes command
    logger.info("\n2. Testing 'quotes' command...")
    result = subprocess.run(
        ["python3", str(script_path), "quotes", "AKBNK,GARAN,EREGL"],
        capture_output=True,
        text=True,
        cwd=str(script_path.parent),
    )
    if result.returncode == 0:
        data = json.loads(result.stdout)
        logger.info(f"   Fetched {data.get('count', 0)} quotes")
    else:
        logger.info(f"   Error: {result.stderr}")

    # Test market command
    logger.info("\n3. Testing 'market' command...")
    result = subprocess.run(
        ["python3", str(script_path), "market"],
        capture_output=True,
        text=True,
        cwd=str(script_path.parent),
    )
    if result.returncode == 0:
        data = json.loads(result.stdout)
        logger.info(f"   Timestamp: {data.get('timestamp')}")
    else:
        logger.info(f"   Error: {result.stderr}")

    logger.info("\n" + "=" * 60)
    logger.info("Realtime API CLI test completed!")
    logger.info("=" * 60)


def main():
    """Run all tests."""
    logger.info("\n" + "#" * 60)
    logger.info("# REAL-TIME STREAMING TEST SUITE")
    logger.info("#" * 60)

    try:
        test_quote_service()
    except Exception as e:
        logger.error(f"\n❌ Quote service test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_index_quotes()
    except Exception as e:
        logger.error(f"\n❌ Index quotes test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_market_summary()
    except Exception as e:
        logger.error(f"\n❌ Market summary test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_realtime_api()
    except Exception as e:
        logger.error(f"\n❌ Realtime API CLI test failed: {e}")
        import traceback
        traceback.print_exc()

    logger.info("\n" + "#" * 60)
    logger.info("# ALL TESTS COMPLETED")
    logger.info("#" * 60)


if __name__ == "__main__":
    main()
