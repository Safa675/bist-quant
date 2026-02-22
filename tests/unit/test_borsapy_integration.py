#!/usr/bin/env python3
"""
Test script for borsapy integration.

Run from project root:
    python data/Fetcher-Scrapper/test_borsapy_integration.py
"""

import logging
import sys
from pathlib import Path
logger = logging.getLogger(__name__)

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "Models" / "common"))
sys.path.insert(0, str(PROJECT_ROOT / "data" / "Fetcher-Scrapper"))


def test_borsapy_client():
    """Test BorsapyClient directly."""
    logger.info("=" * 60)
    logger.info("TEST 1: BorsapyClient Direct Access")
    logger.info("=" * 60)

    from borsapy_client import BorsapyClient

    client = BorsapyClient()

    # Test 1: Get XU100 components
    logger.info("\n1. Getting XU100 index components...")
    components = client.get_index_components("XU100")
    logger.info(f"   Found {len(components)} stocks in XU100")
    logger.info(f"   First 10: {components[:10]}")

    # Test 2: Get single ticker info
    logger.info("\n2. Getting THYAO fast info...")
    info = client.get_fast_info("THYAO")
    if info:
        logger.info(f"   Last price: {info.get('last_price', 'N/A')}")
        logger.info(f"   Volume: {info.get('volume', 'N/A')}")
        logger.info(f"   Market cap: {info.get('market_cap', 'N/A')}")
    else:
        logger.info("   No data returned")

    # Test 3: Get price history
    logger.info("\n3. Getting AKBNK price history (1 month)...")
    history = client.get_history("AKBNK", period="1ay", interval="1d")
    if not history.empty:
        logger.info(f"   Rows: {len(history)}")
        logger.info(f"   Columns: {list(history.columns)}")
        logger.info(f"   Date range: {history.index.min()} to {history.index.max()}")
    else:
        logger.info("   No history data returned")

    # Test 4: Get financials
    logger.info("\n4. Getting EREGL financials...")
    financials = client.get_financials("EREGL")
    for name, df in financials.items():
        if df is not None and not df.empty:
            logger.info(f"   {name}: {df.shape[0]} rows x {df.shape[1]} cols")
        else:
            logger.info(f"   {name}: No data")

    # Test 5: Get dividends
    logger.info("\n5. Getting TUPRS dividends...")
    dividends = client.get_dividends("TUPRS")
    if not dividends.empty:
        logger.info(f"   Found {len(dividends)} dividend records")
        logger.info(dividends.tail(3))
    else:
        logger.info("   No dividend data")

    # Test 6: Technical indicators
    logger.info("\n6. Getting GARAN history with RSI indicator...")
    df_with_ta = client.get_history_with_indicators(
        "GARAN", indicators=["rsi"], period="3ay"
    )
    if not df_with_ta.empty:
        logger.info(f"   Columns: {list(df_with_ta.columns)}")
        logger.info(f"   Rows: {len(df_with_ta)}")
    else:
        logger.info("   No data returned")

    # Test 7: Stock screener
    logger.info("\n7. Running stock screener (high ROE)...")
    screened = client.screen_stocks(roe_min=20)
    if not screened.empty:
        logger.info(f"   Found {len(screened)} stocks with ROE > 20%")
        logger.info(f"   Columns: {list(screened.columns)[:5]}...")
    else:
        logger.info("   No stocks matched criteria")

    # Test 8: All indices
    logger.info("\n8. Getting all BIST indices...")
    indices = client.get_all_indices()
    logger.info(f"   Found {len(indices)} indices")
    logger.info(f"   Sample: {indices[:5]}")

    logger.info("\n" + "=" * 60)
    logger.info("BorsapyClient tests completed!")
    logger.info("=" * 60)


def test_data_loader_integration():
    """Test DataLoader with borsapy integration."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: DataLoader Borsapy Integration")
    logger.info("=" * 60)

    from data_loader import DataLoader

    # Initialize DataLoader
    data_dir = PROJECT_ROOT / "data"
    regime_dir = PROJECT_ROOT / "regime_filter"

    loader = DataLoader(data_dir=data_dir, regime_model_dir=regime_dir)

    # Test borsapy property
    logger.info("\n1. Accessing borsapy client via DataLoader...")
    bp_client = loader.borsapy
    if bp_client is not None:
        logger.info("   ✅ Borsapy client accessible")
    else:
        logger.error("   ❌ Borsapy client not available")
        return

    # Test index components
    logger.info("\n2. Getting XU030 components via DataLoader...")
    components = loader.get_index_components_borsapy("XU030")
    logger.info(f"   Found {len(components)} stocks in XU030")

    # Test price loading
    logger.info("\n3. Loading prices via borsapy (5 stocks, 1 month)...")
    test_symbols = ["THYAO", "AKBNK", "GARAN", "EREGL", "TUPRS"]
    prices = loader.load_prices_borsapy(symbols=test_symbols, period="1ay")
    if not prices.empty:
        logger.info(f"   Loaded {len(prices)} price records")
        logger.info(f"   Tickers: {prices['Ticker'].unique().tolist()}")
    else:
        logger.info("   No price data loaded")

    # Test financials
    logger.info("\n4. Getting SISE financials via DataLoader...")
    financials = loader.get_financials_borsapy("SISE")
    for name, df in financials.items():
        status = f"{df.shape}" if df is not None and not df.empty else "No data"
        logger.info(f"   {name}: {status}")

    # Test screener
    logger.info("\n5. Screening stocks via DataLoader...")
    screened = loader.screen_stocks_borsapy(pe_max=10, market_cap_min=10_000_000_000)
    if not screened.empty:
        logger.info(f"   Found {len(screened)} stocks matching criteria")
    else:
        logger.info("   No stocks matched")

    # Test technical indicators
    logger.info("\n6. Getting KCHOL with MACD indicator...")
    df_ta = loader.get_history_with_indicators_borsapy(
        "KCHOL", indicators=["macd"], period="3ay"
    )
    if not df_ta.empty:
        logger.info(f"   Got {len(df_ta)} rows with columns: {list(df_ta.columns)[:6]}...")
    else:
        logger.info("   No data returned")

    logger.info("\n" + "=" * 60)
    logger.info("DataLoader integration tests completed!")
    logger.info("=" * 60)


def main():
    """Run all tests."""
    logger.info("\n" + "#" * 60)
    logger.info("# BORSAPY INTEGRATION TEST SUITE")
    logger.info("#" * 60)

    try:
        test_borsapy_client()
    except Exception as e:
        logger.error(f"\n❌ BorsapyClient test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_data_loader_integration()
    except Exception as e:
        logger.error(f"\n❌ DataLoader integration test failed: {e}")
        import traceback
        traceback.print_exc()

    logger.info("\n" + "#" * 60)
    logger.info("# ALL TESTS COMPLETED")
    logger.info("#" * 60)


if __name__ == "__main__":
    main()
