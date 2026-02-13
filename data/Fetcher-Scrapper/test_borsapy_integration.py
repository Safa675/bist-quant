#!/usr/bin/env python3
"""
Test script for borsapy integration.

Run from project root:
    python data/Fetcher-Scrapper/test_borsapy_integration.py
"""

import sys
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "Models" / "common"))
sys.path.insert(0, str(PROJECT_ROOT / "data" / "Fetcher-Scrapper"))


def test_borsapy_client():
    """Test BorsapyClient directly."""
    print("=" * 60)
    print("TEST 1: BorsapyClient Direct Access")
    print("=" * 60)

    from borsapy_client import BorsapyClient

    client = BorsapyClient()

    # Test 1: Get XU100 components
    print("\n1. Getting XU100 index components...")
    components = client.get_index_components("XU100")
    print(f"   Found {len(components)} stocks in XU100")
    print(f"   First 10: {components[:10]}")

    # Test 2: Get single ticker info
    print("\n2. Getting THYAO fast info...")
    info = client.get_fast_info("THYAO")
    if info:
        print(f"   Last price: {info.get('last_price', 'N/A')}")
        print(f"   Volume: {info.get('volume', 'N/A')}")
        print(f"   Market cap: {info.get('market_cap', 'N/A')}")
    else:
        print("   No data returned")

    # Test 3: Get price history
    print("\n3. Getting AKBNK price history (1 month)...")
    history = client.get_history("AKBNK", period="1ay", interval="1d")
    if not history.empty:
        print(f"   Rows: {len(history)}")
        print(f"   Columns: {list(history.columns)}")
        print(f"   Date range: {history.index.min()} to {history.index.max()}")
    else:
        print("   No history data returned")

    # Test 4: Get financials
    print("\n4. Getting EREGL financials...")
    financials = client.get_financials("EREGL")
    for name, df in financials.items():
        if df is not None and not df.empty:
            print(f"   {name}: {df.shape[0]} rows x {df.shape[1]} cols")
        else:
            print(f"   {name}: No data")

    # Test 5: Get dividends
    print("\n5. Getting TUPRS dividends...")
    dividends = client.get_dividends("TUPRS")
    if not dividends.empty:
        print(f"   Found {len(dividends)} dividend records")
        print(dividends.tail(3))
    else:
        print("   No dividend data")

    # Test 6: Technical indicators
    print("\n6. Getting GARAN history with RSI indicator...")
    df_with_ta = client.get_history_with_indicators(
        "GARAN", indicators=["rsi"], period="3ay"
    )
    if not df_with_ta.empty:
        print(f"   Columns: {list(df_with_ta.columns)}")
        print(f"   Rows: {len(df_with_ta)}")
    else:
        print("   No data returned")

    # Test 7: Stock screener
    print("\n7. Running stock screener (high ROE)...")
    screened = client.screen_stocks(roe_min=20)
    if not screened.empty:
        print(f"   Found {len(screened)} stocks with ROE > 20%")
        print(f"   Columns: {list(screened.columns)[:5]}...")
    else:
        print("   No stocks matched criteria")

    # Test 8: All indices
    print("\n8. Getting all BIST indices...")
    indices = client.get_all_indices()
    print(f"   Found {len(indices)} indices")
    print(f"   Sample: {indices[:5]}")

    print("\n" + "=" * 60)
    print("BorsapyClient tests completed!")
    print("=" * 60)


def test_data_loader_integration():
    """Test DataLoader with borsapy integration."""
    print("\n" + "=" * 60)
    print("TEST 2: DataLoader Borsapy Integration")
    print("=" * 60)

    from data_loader import DataLoader

    # Initialize DataLoader
    data_dir = PROJECT_ROOT / "data"
    regime_dir = PROJECT_ROOT / "Regime Filter"

    loader = DataLoader(data_dir=data_dir, regime_model_dir=regime_dir)

    # Test borsapy property
    print("\n1. Accessing borsapy client via DataLoader...")
    bp_client = loader.borsapy
    if bp_client is not None:
        print("   ✅ Borsapy client accessible")
    else:
        print("   ❌ Borsapy client not available")
        return

    # Test index components
    print("\n2. Getting XU030 components via DataLoader...")
    components = loader.get_index_components_borsapy("XU030")
    print(f"   Found {len(components)} stocks in XU030")

    # Test price loading
    print("\n3. Loading prices via borsapy (5 stocks, 1 month)...")
    test_symbols = ["THYAO", "AKBNK", "GARAN", "EREGL", "TUPRS"]
    prices = loader.load_prices_borsapy(symbols=test_symbols, period="1ay")
    if not prices.empty:
        print(f"   Loaded {len(prices)} price records")
        print(f"   Tickers: {prices['Ticker'].unique().tolist()}")
    else:
        print("   No price data loaded")

    # Test financials
    print("\n4. Getting SISE financials via DataLoader...")
    financials = loader.get_financials_borsapy("SISE")
    for name, df in financials.items():
        status = f"{df.shape}" if df is not None and not df.empty else "No data"
        print(f"   {name}: {status}")

    # Test screener
    print("\n5. Screening stocks via DataLoader...")
    screened = loader.screen_stocks_borsapy(pe_max=10, market_cap_min=10_000_000_000)
    if not screened.empty:
        print(f"   Found {len(screened)} stocks matching criteria")
    else:
        print("   No stocks matched")

    # Test technical indicators
    print("\n6. Getting KCHOL with MACD indicator...")
    df_ta = loader.get_history_with_indicators_borsapy(
        "KCHOL", indicators=["macd"], period="3ay"
    )
    if not df_ta.empty:
        print(f"   Got {len(df_ta)} rows with columns: {list(df_ta.columns)[:6]}...")
    else:
        print("   No data returned")

    print("\n" + "=" * 60)
    print("DataLoader integration tests completed!")
    print("=" * 60)


def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# BORSAPY INTEGRATION TEST SUITE")
    print("#" * 60)

    try:
        test_borsapy_client()
    except Exception as e:
        print(f"\n❌ BorsapyClient test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_data_loader_integration()
    except Exception as e:
        print(f"\n❌ DataLoader integration test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "#" * 60)
    print("# ALL TESTS COMPLETED")
    print("#" * 60)


if __name__ == "__main__":
    main()
