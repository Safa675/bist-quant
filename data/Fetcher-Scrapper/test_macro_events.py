#!/usr/bin/env python3
"""
Test script for macro events module.

Run from project root:
    python data/Fetcher-Scrapper/test_macro_events.py
"""

import sys
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "data" / "Fetcher-Scrapper"))
sys.path.insert(0, str(PROJECT_ROOT / "Models" / "common"))


def test_economic_calendar():
    """Test economic calendar functionality."""
    print("=" * 60)
    print("TEST 1: Economic Calendar")
    print("=" * 60)

    from macro_events import MacroEventsClient

    client = MacroEventsClient()

    # Test full calendar
    print("\n1. Getting economic calendar (next 7 days, TR + US)...")
    calendar = client.get_economic_calendar(days_ahead=7, countries=["TR", "US"])

    if not calendar.empty:
        print(f"   Found {len(calendar)} events")
        print(f"   Columns: {list(calendar.columns)}")
        print("\n   First 5 events:")
        print(calendar.head().to_string())
    else:
        print("   No events found (may be API limitation)")

    # Test high-impact events
    print("\n2. Getting high-impact events...")
    high_impact = client.get_upcoming_high_impact_events(days_ahead=14)

    if not high_impact.empty:
        print(f"   Found {len(high_impact)} high-impact events")
    else:
        print("   No high-impact events found")

    print("\n" + "=" * 60)
    print("Economic calendar test completed!")
    print("=" * 60)


def test_inflation_data():
    """Test inflation data functionality."""
    print("\n" + "=" * 60)
    print("TEST 2: Inflation Data")
    print("=" * 60)

    from macro_events import MacroEventsClient

    client = MacroEventsClient()

    # Test inflation history
    print("\n1. Getting inflation data (last 12 months)...")
    inflation = client.get_inflation_data(periods=12)

    if not inflation.empty:
        print(f"   Found {len(inflation)} periods")
        print(f"   Columns: {list(inflation.columns)}")
        print("\n   Recent data:")
        print(inflation.tail().to_string())
    else:
        print("   No inflation data found")

    # Test latest inflation
    print("\n2. Getting latest inflation reading...")
    latest = client.get_latest_inflation()

    if "error" not in latest:
        print(f"   Latest data:")
        for key, value in latest.items():
            print(f"     {key}: {value}")
    else:
        print(f"   Error: {latest.get('error')}")

    print("\n" + "=" * 60)
    print("Inflation data test completed!")
    print("=" * 60)


def test_bond_yields():
    """Test bond yields functionality."""
    print("\n" + "=" * 60)
    print("TEST 3: Bond Yields")
    print("=" * 60)

    from macro_events import MacroEventsClient

    client = MacroEventsClient()

    # Test bond yields
    print("\n1. Getting bond yields...")
    yields = client.get_bond_yields()

    if "error" not in yields:
        print(f"   Bond yields:")
        for key, value in yields.items():
            if key != "timestamp":
                print(f"     {key}: {value}")
    else:
        print(f"   Error: {yields.get('error')}")

    # Test yield curve
    print("\n2. Getting yield curve...")
    curve = client.get_yield_curve()

    if not curve.empty:
        print(f"   Yield curve:")
        print(curve.to_string())
    else:
        print("   No yield curve data")

    print("\n" + "=" * 60)
    print("Bond yields test completed!")
    print("=" * 60)


def test_stock_news():
    """Test stock news functionality."""
    print("\n" + "=" * 60)
    print("TEST 4: Stock News & Analyst Data")
    print("=" * 60)

    from macro_events import MacroEventsClient

    client = MacroEventsClient()

    # Test stock news
    print("\n1. Getting THYAO news...")
    news = client.get_stock_news("THYAO", limit=5)

    if news:
        print(f"   Found {len(news)} news items")
        for i, item in enumerate(news[:3], 1):
            if isinstance(item, dict):
                title = item.get("title", item.get("headline", "N/A"))
                print(f"   {i}. {title[:60]}...")
            else:
                print(f"   {i}. {str(item)[:60]}...")
    else:
        print("   No news found")

    # Test earnings calendar
    print("\n2. Getting AKBNK earnings calendar...")
    earnings = client.get_earnings_calendar("AKBNK")

    if not earnings.empty:
        print(f"   Found {len(earnings)} earnings dates")
        print(earnings.head().to_string())
    else:
        print("   No earnings data found")

    # Test analyst recommendations
    print("\n3. Getting GARAN analyst recommendations...")
    analyst = client.get_analyst_recommendations("GARAN")

    if "error" not in analyst:
        print(f"   Analyst data for {analyst.get('symbol')}:")
        if "price_targets" in analyst:
            print(f"     Price targets: {analyst.get('price_targets')}")
        if "recommendations" in analyst:
            print(f"     Recommendations: {analyst.get('recommendations')}")
    else:
        print(f"   Error: {analyst.get('error')}")

    print("\n" + "=" * 60)
    print("Stock news test completed!")
    print("=" * 60)


def test_eurobonds():
    """Test eurobonds functionality."""
    print("\n" + "=" * 60)
    print("TEST 5: Eurobonds")
    print("=" * 60)

    from macro_events import MacroEventsClient

    client = MacroEventsClient()

    print("\n1. Getting eurobonds...")
    eurobonds = client.get_eurobonds()

    if not eurobonds.empty:
        print(f"   Found {len(eurobonds)} eurobonds")
        print(f"   Columns: {list(eurobonds.columns)}")
        print("\n   First 5 bonds:")
        print(eurobonds.head().to_string())
    else:
        print("   No eurobond data found")

    print("\n" + "=" * 60)
    print("Eurobonds test completed!")
    print("=" * 60)


def test_macro_summary():
    """Test comprehensive macro summary."""
    print("\n" + "=" * 60)
    print("TEST 6: Macro Summary")
    print("=" * 60)

    from macro_events import MacroEventsClient

    client = MacroEventsClient()

    print("\n1. Getting comprehensive macro summary...")
    summary = client.get_macro_summary()

    print(f"\n   Timestamp: {summary.get('timestamp')}")

    # Inflation
    inflation = summary.get("inflation", {})
    if "error" not in inflation:
        print(f"\n   Inflation:")
        for key, value in inflation.items():
            if key not in ["timestamp", "note"]:
                print(f"     {key}: {value}")
    else:
        print(f"   Inflation: {inflation.get('error', 'N/A')}")

    # Bond yields
    yields = summary.get("bond_yields", {})
    if "error" not in yields:
        print(f"\n   Bond Yields:")
        for key, value in yields.items():
            if key != "timestamp":
                print(f"     {key}: {value}")
    else:
        print(f"   Bond Yields: {yields.get('error', 'N/A')}")

    # Sentiment
    sentiment = summary.get("sentiment", {})
    print(f"\n   Market Sentiment:")
    if "xu100" in sentiment:
        xu100 = sentiment["xu100"]
        if "error" not in xu100:
            print(f"     XU100: {xu100.get('value')} ({xu100.get('change_pct', 'N/A')}%)")
        else:
            print(f"     XU100: {xu100.get('error')}")
    if "usdtry" in sentiment:
        usdtry = sentiment["usdtry"]
        if "error" not in usdtry:
            print(f"     USD/TRY: {usdtry.get('rate')}")
        else:
            print(f"     USD/TRY: {usdtry.get('error')}")

    # Upcoming events
    events = summary.get("upcoming_events", [])
    print(f"\n   Upcoming High-Impact Events: {len(events)}")

    print("\n" + "=" * 60)
    print("Macro summary test completed!")
    print("=" * 60)


def test_dataloader_integration():
    """Test DataLoader integration."""
    print("\n" + "=" * 60)
    print("TEST 7: DataLoader Integration")
    print("=" * 60)

    try:
        from data_loader import DataLoader

        # Initialize loader
        data_dir = PROJECT_ROOT / "data"
        regime_dir = PROJECT_ROOT / "Regime Filter"
        loader = DataLoader(data_dir=data_dir, regime_model_dir=regime_dir)

        # Test macro property
        print("\n1. Accessing macro client via DataLoader...")
        macro = loader.macro
        if macro is not None:
            print("   ✅ Macro client accessible")
        else:
            print("   ❌ Macro client not available")
            return

        # Test get_economic_calendar
        print("\n2. Getting economic calendar via DataLoader...")
        calendar = loader.get_economic_calendar(days_ahead=7)
        print(f"   Found {len(calendar)} events")

        # Test get_inflation_data
        print("\n3. Getting inflation data via DataLoader...")
        inflation = loader.get_inflation_data(periods=6)
        print(f"   Found {len(inflation)} periods")

        # Test get_bond_yields
        print("\n4. Getting bond yields via DataLoader...")
        yields = loader.get_bond_yields()
        if yields:
            print(f"   Yields: {yields}")

        # Test get_stock_news
        print("\n5. Getting stock news via DataLoader...")
        news = loader.get_stock_news("EREGL", limit=3)
        print(f"   Found {len(news)} news items")

        # Test get_macro_summary
        print("\n6. Getting macro summary via DataLoader...")
        summary = loader.get_macro_summary()
        print(f"   Summary keys: {list(summary.keys())}")

        print("\n" + "=" * 60)
        print("DataLoader integration test completed!")
        print("=" * 60)

    except Exception as e:
        print(f"  ⚠️  DataLoader integration test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# MACRO EVENTS TEST SUITE")
    print("#" * 60)

    try:
        test_economic_calendar()
    except Exception as e:
        print(f"\n❌ Economic calendar test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_inflation_data()
    except Exception as e:
        print(f"\n❌ Inflation data test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_bond_yields()
    except Exception as e:
        print(f"\n❌ Bond yields test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_stock_news()
    except Exception as e:
        print(f"\n❌ Stock news test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_eurobonds()
    except Exception as e:
        print(f"\n❌ Eurobonds test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_macro_summary()
    except Exception as e:
        print(f"\n❌ Macro summary test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_dataloader_integration()
    except Exception as e:
        print(f"\n❌ DataLoader integration test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "#" * 60)
    print("# ALL TESTS COMPLETED")
    print("#" * 60)


if __name__ == "__main__":
    main()
