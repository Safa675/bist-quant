#!/usr/bin/env python3
"""
Test script for macro events module.

Run from project root:
    python data/Fetcher-Scrapper/test_macro_events.py
"""

import logging
import sys
from pathlib import Path
logger = logging.getLogger(__name__)

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "data" / "Fetcher-Scrapper"))
sys.path.insert(0, str(PROJECT_ROOT / "Models" / "common"))


def test_economic_calendar():
    """Test economic calendar functionality."""
    logger.info("=" * 60)
    logger.info("TEST 1: Economic Calendar")
    logger.info("=" * 60)

    from macro_events import MacroEventsClient

    client = MacroEventsClient()

    # Test full calendar
    logger.info("\n1. Getting economic calendar (next 7 days, TR + US)...")
    calendar = client.get_economic_calendar(days_ahead=7, countries=["TR", "US"])

    if not calendar.empty:
        logger.info(f"   Found {len(calendar)} events")
        logger.info(f"   Columns: {list(calendar.columns)}")
        logger.info("\n   First 5 events:")
        logger.info(calendar.head().to_string())
    else:
        logger.info("   No events found (may be API limitation)")

    # Test high-impact events
    logger.info("\n2. Getting high-impact events...")
    high_impact = client.get_upcoming_high_impact_events(days_ahead=14)

    if not high_impact.empty:
        logger.info(f"   Found {len(high_impact)} high-impact events")
    else:
        logger.info("   No high-impact events found")

    logger.info("\n" + "=" * 60)
    logger.info("Economic calendar test completed!")
    logger.info("=" * 60)


def test_inflation_data():
    """Test inflation data functionality."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Inflation Data")
    logger.info("=" * 60)

    from macro_events import MacroEventsClient

    client = MacroEventsClient()

    # Test inflation history
    logger.info("\n1. Getting inflation data (last 12 months)...")
    inflation = client.get_inflation_data(periods=12)

    if not inflation.empty:
        logger.info(f"   Found {len(inflation)} periods")
        logger.info(f"   Columns: {list(inflation.columns)}")
        logger.info("\n   Recent data:")
        logger.info(inflation.tail().to_string())
    else:
        logger.info("   No inflation data found")

    # Test latest inflation
    logger.info("\n2. Getting latest inflation reading...")
    latest = client.get_latest_inflation()

    if "error" not in latest:
        logger.info(f"   Latest data:")
        for key, value in latest.items():
            logger.info(f"     {key}: {value}")
    else:
        logger.info(f"   Error: {latest.get('error')}")

    logger.info("\n" + "=" * 60)
    logger.info("Inflation data test completed!")
    logger.info("=" * 60)


def test_bond_yields():
    """Test bond yields functionality."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Bond Yields")
    logger.info("=" * 60)

    from macro_events import MacroEventsClient

    client = MacroEventsClient()

    # Test bond yields
    logger.info("\n1. Getting bond yields...")
    yields = client.get_bond_yields()

    if "error" not in yields:
        logger.info(f"   Bond yields:")
        for key, value in yields.items():
            if key != "timestamp":
                logger.info(f"     {key}: {value}")
    else:
        logger.info(f"   Error: {yields.get('error')}")

    # Test yield curve
    logger.info("\n2. Getting yield curve...")
    curve = client.get_yield_curve()

    if not curve.empty:
        logger.info(f"   Yield curve:")
        logger.info(curve.to_string())
    else:
        logger.info("   No yield curve data")

    logger.info("\n" + "=" * 60)
    logger.info("Bond yields test completed!")
    logger.info("=" * 60)


def test_stock_news():
    """Test stock news functionality."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Stock News & Analyst Data")
    logger.info("=" * 60)

    from macro_events import MacroEventsClient

    client = MacroEventsClient()

    # Test stock news
    logger.info("\n1. Getting THYAO news...")
    news = client.get_stock_news("THYAO", limit=5)

    if news:
        logger.info(f"   Found {len(news)} news items")
        for i, item in enumerate(news[:3], 1):
            if isinstance(item, dict):
                title = item.get("title", item.get("headline", "N/A"))
                logger.info(f"   {i}. {title[:60]}...")
            else:
                logger.info(f"   {i}. {str(item)[:60]}...")
    else:
        logger.info("   No news found")

    # Test earnings calendar
    logger.info("\n2. Getting AKBNK earnings calendar...")
    earnings = client.get_earnings_calendar("AKBNK")

    if not earnings.empty:
        logger.info(f"   Found {len(earnings)} earnings dates")
        logger.info(earnings.head().to_string())
    else:
        logger.info("   No earnings data found")

    # Test analyst recommendations
    logger.info("\n3. Getting GARAN analyst recommendations...")
    analyst = client.get_analyst_recommendations("GARAN")

    if "error" not in analyst:
        logger.info(f"   Analyst data for {analyst.get('symbol')}:")
        if "price_targets" in analyst:
            logger.info(f"     Price targets: {analyst.get('price_targets')}")
        if "recommendations" in analyst:
            logger.info(f"     Recommendations: {analyst.get('recommendations')}")
    else:
        logger.info(f"   Error: {analyst.get('error')}")

    logger.info("\n" + "=" * 60)
    logger.info("Stock news test completed!")
    logger.info("=" * 60)


def test_eurobonds():
    """Test eurobonds functionality."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: Eurobonds")
    logger.info("=" * 60)

    from macro_events import MacroEventsClient

    client = MacroEventsClient()

    logger.info("\n1. Getting eurobonds...")
    eurobonds = client.get_eurobonds()

    if not eurobonds.empty:
        logger.info(f"   Found {len(eurobonds)} eurobonds")
        logger.info(f"   Columns: {list(eurobonds.columns)}")
        logger.info("\n   First 5 bonds:")
        logger.info(eurobonds.head().to_string())
    else:
        logger.info("   No eurobond data found")

    logger.info("\n" + "=" * 60)
    logger.info("Eurobonds test completed!")
    logger.info("=" * 60)


def test_macro_summary():
    """Test comprehensive macro summary."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 6: Macro Summary")
    logger.info("=" * 60)

    from macro_events import MacroEventsClient

    client = MacroEventsClient()

    logger.info("\n1. Getting comprehensive macro summary...")
    summary = client.get_macro_summary()

    logger.info(f"\n   Timestamp: {summary.get('timestamp')}")

    # Inflation
    inflation = summary.get("inflation", {})
    if "error" not in inflation:
        logger.info(f"\n   Inflation:")
        for key, value in inflation.items():
            if key not in ["timestamp", "note"]:
                logger.info(f"     {key}: {value}")
    else:
        logger.info(f"   Inflation: {inflation.get('error', 'N/A')}")

    # Bond yields
    yields = summary.get("bond_yields", {})
    if "error" not in yields:
        logger.info(f"\n   Bond Yields:")
        for key, value in yields.items():
            if key != "timestamp":
                logger.info(f"     {key}: {value}")
    else:
        logger.info(f"   Bond Yields: {yields.get('error', 'N/A')}")

    # Sentiment
    sentiment = summary.get("sentiment", {})
    logger.info(f"\n   Market Sentiment:")
    if "xu100" in sentiment:
        xu100 = sentiment["xu100"]
        if "error" not in xu100:
            logger.info(f"     XU100: {xu100.get('value')} ({xu100.get('change_pct', 'N/A')}%)")
        else:
            logger.info(f"     XU100: {xu100.get('error')}")
    if "usdtry" in sentiment:
        usdtry = sentiment["usdtry"]
        if "error" not in usdtry:
            logger.info(f"     USD/TRY: {usdtry.get('rate')}")
        else:
            logger.info(f"     USD/TRY: {usdtry.get('error')}")

    # Upcoming events
    events = summary.get("upcoming_events", [])
    logger.info(f"\n   Upcoming High-Impact Events: {len(events)}")

    logger.info("\n" + "=" * 60)
    logger.info("Macro summary test completed!")
    logger.info("=" * 60)


def test_dataloader_integration():
    """Test DataLoader integration."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 7: DataLoader Integration")
    logger.info("=" * 60)

    try:
        from data_loader import DataLoader

        # Initialize loader
        data_dir = PROJECT_ROOT / "data"
        regime_dir = PROJECT_ROOT / "regime_filter"
        loader = DataLoader(data_dir=data_dir, regime_model_dir=regime_dir)

        # Test macro property
        logger.info("\n1. Accessing macro client via DataLoader...")
        macro = loader.macro
        if macro is not None:
            logger.info("   ✅ Macro client accessible")
        else:
            logger.error("   ❌ Macro client not available")
            return

        # Test get_economic_calendar
        logger.info("\n2. Getting economic calendar via DataLoader...")
        calendar = loader.get_economic_calendar(days_ahead=7)
        logger.info(f"   Found {len(calendar)} events")

        # Test get_inflation_data
        logger.info("\n3. Getting inflation data via DataLoader...")
        inflation = loader.get_inflation_data(periods=6)
        logger.info(f"   Found {len(inflation)} periods")

        # Test get_bond_yields
        logger.info("\n4. Getting bond yields via DataLoader...")
        yields = loader.get_bond_yields()
        if yields:
            logger.info(f"   Yields: {yields}")

        # Test get_stock_news
        logger.info("\n5. Getting stock news via DataLoader...")
        news = loader.get_stock_news("EREGL", limit=3)
        logger.info(f"   Found {len(news)} news items")

        # Test get_macro_summary
        logger.info("\n6. Getting macro summary via DataLoader...")
        summary = loader.get_macro_summary()
        logger.info(f"   Summary keys: {list(summary.keys())}")

        logger.info("\n" + "=" * 60)
        logger.info("DataLoader integration test completed!")
        logger.info("=" * 60)

    except Exception as e:
        logger.warning(f"  ⚠️  DataLoader integration test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    logger.info("\n" + "#" * 60)
    logger.info("# MACRO EVENTS TEST SUITE")
    logger.info("#" * 60)

    try:
        test_economic_calendar()
    except Exception as e:
        logger.error(f"\n❌ Economic calendar test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_inflation_data()
    except Exception as e:
        logger.error(f"\n❌ Inflation data test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_bond_yields()
    except Exception as e:
        logger.error(f"\n❌ Bond yields test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_stock_news()
    except Exception as e:
        logger.error(f"\n❌ Stock news test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_eurobonds()
    except Exception as e:
        logger.error(f"\n❌ Eurobonds test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_macro_summary()
    except Exception as e:
        logger.error(f"\n❌ Macro summary test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_dataloader_integration()
    except Exception as e:
        logger.error(f"\n❌ DataLoader integration test failed: {e}")
        import traceback
        traceback.print_exc()

    logger.info("\n" + "#" * 60)
    logger.info("# ALL TESTS COMPLETED")
    logger.info("#" * 60)


if __name__ == "__main__":
    main()
