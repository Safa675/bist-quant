#!/usr/bin/env python3
"""
Test script for borsapy technical indicators integration.

Run from project root:
    python Models/signals/test_borsapy_indicators.py
"""

import sys
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "Models" / "common"))
sys.path.insert(0, str(PROJECT_ROOT / "Models" / "signals"))


def test_indicator_calculations():
    """Test local indicator calculations."""
    print("=" * 60)
    print("TEST 1: Local Indicator Calculations")
    print("=" * 60)

    import pandas as pd
    import numpy as np
    from borsapy_indicators import BorsapyIndicators

    # Create sample price data
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=252, freq="B")
    tickers = ["THYAO", "AKBNK", "GARAN", "EREGL", "TUPRS"]

    # Generate realistic price data
    close_data = {}
    high_data = {}
    low_data = {}

    for ticker in tickers:
        # Random walk with drift
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        close_data[ticker] = prices
        high_data[ticker] = prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates))))
        low_data[ticker] = prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates))))

    close_df = pd.DataFrame(close_data, index=dates)
    high_df = pd.DataFrame(high_data, index=dates)
    low_df = pd.DataFrame(low_data, index=dates)

    print(f"\nSample data: {close_df.shape[0]} days × {close_df.shape[1]} tickers")

    # Test RSI
    print("\n1. Testing RSI panel...")
    rsi_panel = BorsapyIndicators.build_rsi_panel(close_df, period=14)
    latest_rsi = rsi_panel.iloc[-1]
    print(f"   Latest RSI values: {latest_rsi.round(2).to_dict()}")

    # Test MACD
    print("\n2. Testing MACD panel...")
    macd_panel = BorsapyIndicators.build_macd_panel(close_df, output="histogram")
    latest_macd = macd_panel.iloc[-1]
    print(f"   Latest MACD histogram: {latest_macd.round(4).to_dict()}")

    # Test Bollinger Bands
    print("\n3. Testing Bollinger Bands panel...")
    bb_panel = BorsapyIndicators.build_bollinger_panel(close_df, output="pct_b")
    latest_bb = bb_panel.iloc[-1]
    print(f"   Latest %B values: {latest_bb.round(2).to_dict()}")

    # Test ATR
    print("\n4. Testing ATR panel...")
    atr_panel = BorsapyIndicators.build_atr_panel(high_df, low_df, close_df, period=14)
    latest_atr = atr_panel.iloc[-1]
    print(f"   Latest ATR values: {latest_atr.round(2).to_dict()}")

    # Test Stochastic
    print("\n5. Testing Stochastic panel...")
    stoch_panel = BorsapyIndicators.build_stochastic_panel(
        high_df, low_df, close_df, output="k"
    )
    latest_stoch = stoch_panel.iloc[-1]
    print(f"   Latest %K values: {latest_stoch.round(2).to_dict()}")

    # Test ADX
    print("\n6. Testing ADX panel...")
    adx_panel = BorsapyIndicators.build_adx_panel(high_df, low_df, close_df, period=14)
    latest_adx = adx_panel.iloc[-1]
    print(f"   Latest ADX values: {latest_adx.round(2).to_dict()}")

    # Test Supertrend
    print("\n7. Testing Supertrend panel...")
    st_panel = BorsapyIndicators.build_supertrend_panel(
        high_df, low_df, close_df, period=10, multiplier=3.0, output="direction"
    )
    latest_st = st_panel.iloc[-1]
    print(f"   Latest direction (1=up, -1=down): {latest_st.to_dict()}")

    print("\n" + "=" * 60)
    print("Local indicator tests completed!")
    print("=" * 60)

    return close_df, high_df, low_df


def test_multi_indicator_panel():
    """Test building multiple indicators at once."""
    print("\n" + "=" * 60)
    print("TEST 2: Multi-Indicator Panel Builder")
    print("=" * 60)

    import pandas as pd
    import numpy as np
    from borsapy_indicators import build_multi_indicator_panel

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    tickers = ["THYAO", "AKBNK", "GARAN"]

    close_data = {}
    high_data = {}
    low_data = {}

    for ticker in tickers:
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))
        close_data[ticker] = prices
        high_data[ticker] = prices * 1.01
        low_data[ticker] = prices * 0.99

    close_df = pd.DataFrame(close_data, index=dates)
    high_df = pd.DataFrame(high_data, index=dates)
    low_df = pd.DataFrame(low_data, index=dates)

    # Build multiple indicators
    print("\nBuilding multiple indicators at once...")
    panels = build_multi_indicator_panel(
        close_df, high_df, low_df,
        indicators=["rsi", "macd", "bb", "atr", "stoch"]
    )

    print(f"\nBuilt {len(panels)} indicator panels:")
    for name, panel in panels.items():
        print(f"  - {name}: {panel.shape}")

    print("\n" + "=" * 60)
    print("Multi-indicator test completed!")
    print("=" * 60)


def test_borsapy_api_fetch():
    """Test fetching indicators via borsapy API."""
    print("\n" + "=" * 60)
    print("TEST 3: Borsapy API Indicator Fetch")
    print("=" * 60)

    from borsapy_indicators import BorsapyIndicators, BORSAPY_AVAILABLE

    if not BORSAPY_AVAILABLE:
        print("  ⚠️  Borsapy not available, skipping API tests")
        return

    # Test single ticker
    print("\n1. Fetching THYAO with RSI indicator...")
    df = BorsapyIndicators.fetch_indicators_for_ticker(
        "THYAO", indicators=["rsi"], period="3ay"
    )
    if not df.empty:
        print(f"   Columns: {list(df.columns)}")
        print(f"   Rows: {len(df)}")
        if "RSI_14" in df.columns:
            print(f"   Latest RSI: {df['RSI_14'].iloc[-1]:.2f}")
    else:
        print("   No data returned")

    # Test with multiple indicators
    print("\n2. Fetching GARAN with multiple indicators...")
    df = BorsapyIndicators.fetch_indicators_for_ticker(
        "GARAN", indicators=["rsi", "macd"], period="3ay"
    )
    if not df.empty:
        print(f"   Columns: {list(df.columns)}")
    else:
        print("   No data returned")

    print("\n" + "=" * 60)
    print("Borsapy API tests completed!")
    print("=" * 60)


def test_with_real_data():
    """Test indicators with real data from DataLoader."""
    print("\n" + "=" * 60)
    print("TEST 4: Integration with Real Data")
    print("=" * 60)

    try:
        from data_loader import DataLoader
        from borsapy_indicators import BorsapyIndicators

        # Initialize loader
        data_dir = PROJECT_ROOT / "data"
        regime_dir = PROJECT_ROOT / "Regime Filter"
        loader = DataLoader(data_dir=data_dir, regime_model_dir=regime_dir)

        # Load real prices
        prices_file = data_dir / "bist_prices_full.parquet"
        if not prices_file.exists():
            prices_file = data_dir / "bist_prices_full.csv"

        if not prices_file.exists():
            print("  ⚠️  Price file not found, skipping real data test")
            return

        print("\n1. Loading real price data...")
        prices = loader.load_prices(prices_file)
        close_df = loader.build_close_panel(prices)

        print(f"   Close panel: {close_df.shape[0]} days × {close_df.shape[1]} tickers")

        # Build RSI for real data
        print("\n2. Building RSI panel for real data...")
        rsi_panel = BorsapyIndicators.build_rsi_panel(close_df.iloc[-252:], period=14)

        # Get top RSI stocks
        latest_rsi = rsi_panel.iloc[-1].dropna().sort_values(ascending=False)
        print(f"\n   Top 5 RSI (overbought):")
        for ticker, rsi in latest_rsi.head(5).items():
            print(f"     {ticker}: {rsi:.1f}")

        print(f"\n   Bottom 5 RSI (oversold):")
        for ticker, rsi in latest_rsi.tail(5).items():
            print(f"     {ticker}: {rsi:.1f}")

        print("\n" + "=" * 60)
        print("Real data integration test completed!")
        print("=" * 60)

    except Exception as e:
        print(f"  ⚠️  Real data test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# BORSAPY TECHNICAL INDICATORS TEST SUITE")
    print("#" * 60)

    try:
        test_indicator_calculations()
    except Exception as e:
        print(f"\n❌ Local indicator test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_multi_indicator_panel()
    except Exception as e:
        print(f"\n❌ Multi-indicator test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_borsapy_api_fetch()
    except Exception as e:
        print(f"\n❌ Borsapy API test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_with_real_data()
    except Exception as e:
        print(f"\n❌ Real data test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "#" * 60)
    print("# ALL TESTS COMPLETED")
    print("#" * 60)


if __name__ == "__main__":
    main()
