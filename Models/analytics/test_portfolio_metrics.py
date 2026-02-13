#!/usr/bin/env python3
"""
Test script for portfolio analytics module.

Run from project root:
    python Models/analytics/test_portfolio_metrics.py
"""

import sys
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "Models" / "common"))
sys.path.insert(0, str(PROJECT_ROOT / "Models" / "analytics"))
sys.path.insert(0, str(PROJECT_ROOT / "Models"))


def test_standalone_functions():
    """Test standalone metric functions."""
    print("=" * 60)
    print("TEST 1: Standalone Metric Functions")
    print("=" * 60)

    import numpy as np
    import pandas as pd
    from portfolio_metrics import (
        calculate_sharpe_ratio,
        calculate_sortino_ratio,
        calculate_max_drawdown,
        calculate_beta,
        calculate_alpha,
        calculate_var,
        calculate_cvar,
    )

    # Create sample returns
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")

    # Portfolio with positive drift
    portfolio_returns = pd.Series(
        np.random.normal(0.001, 0.02, len(dates)),
        index=dates,
        name="portfolio",
    )

    # Benchmark with lower return
    benchmark_returns = pd.Series(
        np.random.normal(0.0005, 0.015, len(dates)),
        index=dates,
        name="benchmark",
    )

    print(f"\nSample data: {len(dates)} trading days")
    print(f"Portfolio mean daily return: {portfolio_returns.mean()*100:.4f}%")
    print(f"Benchmark mean daily return: {benchmark_returns.mean()*100:.4f}%")

    # Test Sharpe
    print("\n1. Sharpe Ratio:")
    sharpe = calculate_sharpe_ratio(portfolio_returns, risk_free_rate=0.40)
    print(f"   Portfolio Sharpe: {sharpe:.2f}")

    # Test Sortino
    print("\n2. Sortino Ratio:")
    sortino = calculate_sortino_ratio(portfolio_returns, risk_free_rate=0.40)
    print(f"   Portfolio Sortino: {sortino:.2f}")

    # Test Max Drawdown
    print("\n3. Maximum Drawdown:")
    max_dd, peak, trough = calculate_max_drawdown(returns=portfolio_returns)
    print(f"   Max DD: {max_dd*100:.2f}%")
    print(f"   Peak: {peak}")
    print(f"   Trough: {trough}")

    # Test Beta
    print("\n4. Beta:")
    beta = calculate_beta(portfolio_returns, benchmark_returns)
    print(f"   Portfolio Beta: {beta:.2f}")

    # Test Alpha
    print("\n5. Alpha:")
    alpha = calculate_alpha(portfolio_returns, benchmark_returns, risk_free_rate=0.40)
    print(f"   Portfolio Alpha: {alpha*100:.2f}%")

    # Test VaR
    print("\n6. Value at Risk (95%):")
    var = calculate_var(portfolio_returns, confidence=0.95)
    print(f"   Daily VaR: {var*100:.2f}%")

    # Test CVaR
    print("\n7. Conditional VaR (95%):")
    cvar = calculate_cvar(portfolio_returns, confidence=0.95)
    print(f"   Daily CVaR: {cvar*100:.2f}%")

    print("\n" + "=" * 60)
    print("Standalone function tests completed!")
    print("=" * 60)


def test_portfolio_analytics_class():
    """Test PortfolioAnalytics class."""
    print("\n" + "=" * 60)
    print("TEST 2: PortfolioAnalytics Class")
    print("=" * 60)

    import numpy as np
    import pandas as pd
    from portfolio_metrics import PortfolioAnalytics

    # Create sample returns
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=504, freq="B")  # 2 years

    portfolio_returns = pd.Series(
        np.random.normal(0.0012, 0.018, len(dates)),
        index=dates,
    )

    benchmark_returns = pd.Series(
        np.random.normal(0.0006, 0.015, len(dates)),
        index=dates,
    )

    # Create analytics
    print("\n1. Creating PortfolioAnalytics...")
    analytics = PortfolioAnalytics(
        returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        name="Test Portfolio",
    )

    print(f"   Created: {analytics}")

    # Get all metrics
    print("\n2. All Metrics:")
    metrics = analytics.get_all_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            if "ratio" in key.lower() or key in ["beta"]:
                print(f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value*100:.2f}%")
        else:
            print(f"   {key}: {value}")

    # Get summary
    print("\n3. Summary:")
    print(analytics.summary())

    # Compare to benchmark
    print("\n4. Benchmark Comparison:")
    comparison = analytics.compare_to_benchmark()
    print(f"   Portfolio CAGR: {comparison['portfolio']['cagr']*100:.2f}%")
    print(f"   Benchmark CAGR: {comparison['benchmark']['cagr']*100:.2f}%")
    print(f"   Excess Return: {comparison['relative']['excess_return']*100:.2f}%")
    print(f"   Beta: {comparison['relative']['beta']:.2f}")
    print(f"   Alpha: {comparison['relative']['alpha']*100:.2f}%")

    # Get rolling metrics
    print("\n5. Rolling Metrics (last 5 days):")
    rolling = analytics.get_rolling_metrics(window=63)
    print(rolling.tail())

    print("\n" + "=" * 60)
    print("PortfolioAnalytics class tests completed!")
    print("=" * 60)


def test_with_real_data():
    """Test with real BIST data."""
    print("\n" + "=" * 60)
    print("TEST 3: Real Data Integration")
    print("=" * 60)

    try:
        from data_loader import DataLoader
        from portfolio_metrics import PortfolioAnalytics

        # Initialize loader
        data_dir = PROJECT_ROOT / "data"
        regime_dir = PROJECT_ROOT / "Regime Filter"
        loader = DataLoader(data_dir=data_dir, regime_model_dir=regime_dir)

        # Load prices
        prices_file = data_dir / "bist_prices_full.parquet"
        if not prices_file.exists():
            prices_file = data_dir / "bist_prices_full.csv"

        if not prices_file.exists():
            print("  ⚠️  Price file not found, skipping real data test")
            return

        print("\n1. Loading price data...")
        prices = loader.load_prices(prices_file)
        close_df = loader.build_close_panel(prices)

        # Select some stocks for a sample portfolio
        print("\n2. Creating sample portfolio...")
        holdings = {"THYAO": 100, "AKBNK": 200, "GARAN": 150, "EREGL": 300, "TUPRS": 50}
        available = [s for s in holdings.keys() if s in close_df.columns]

        if len(available) < 3:
            print(f"  ⚠️  Not enough stocks available: {available}")
            return

        print(f"   Holdings: {available}")

        # Create analytics from holdings
        print("\n3. Creating PortfolioAnalytics from holdings...")
        analytics = PortfolioAnalytics.from_holdings(
            holdings={s: holdings[s] for s in available},
            close_df=close_df.iloc[-504:],  # Last 2 years
            name="BIST Portfolio",
        )

        print(f"   Created: {analytics}")

        # Get metrics
        print("\n4. Portfolio Metrics:")
        metrics = analytics.get_all_metrics()
        print(f"   CAGR: {metrics['cagr']*100:.2f}%")
        print(f"   Volatility: {metrics['volatility']*100:.2f}%")
        print(f"   Sharpe: {metrics['sharpe_ratio']:.2f}")
        print(f"   Sortino: {metrics['sortino_ratio']:.2f}")
        print(f"   Max Drawdown: {metrics['max_drawdown']*100:.2f}%")

        # Test DataLoader integration
        print("\n5. Testing DataLoader integration...")
        if "XU100" in close_df.columns:
            analytics2 = loader.create_portfolio_analytics(
                holdings={s: holdings[s] for s in available},
                benchmark="XU100",
                name="BIST Portfolio (with XU100)",
            )
            print(f"   Created via DataLoader: {analytics2}")

            comparison = analytics2.compare_to_benchmark()
            if "error" not in comparison:
                print(f"   Alpha vs XU100: {comparison['relative']['alpha']*100:.2f}%")
                print(f"   Beta vs XU100: {comparison['relative']['beta']:.2f}")

        print("\n" + "=" * 60)
        print("Real data integration test completed!")
        print("=" * 60)

    except Exception as e:
        print(f"  ⚠️  Real data test failed: {e}")
        import traceback
        traceback.print_exc()


def test_correlation_matrix():
    """Test correlation matrix calculation."""
    print("\n" + "=" * 60)
    print("TEST 4: Correlation Matrix")
    print("=" * 60)

    import numpy as np
    import pandas as pd
    from portfolio_metrics import calculate_correlation_matrix

    # Create correlated returns
    np.random.seed(42)
    n = 252

    # Base market factor
    market = np.random.normal(0.001, 0.015, n)

    # Create stocks with different correlations to market
    returns_df = pd.DataFrame({
        "STOCK_A": market + np.random.normal(0, 0.01, n),  # High correlation
        "STOCK_B": market * 0.5 + np.random.normal(0.0005, 0.012, n),  # Medium
        "STOCK_C": np.random.normal(0.0008, 0.02, n),  # Low correlation
        "MARKET": market,
    })

    print("\n1. Sample returns created")
    print(f"   {returns_df.shape[0]} days, {returns_df.shape[1]} assets")

    # Calculate correlation matrix
    corr = calculate_correlation_matrix(returns_df)
    print("\n2. Correlation Matrix:")
    print(corr.round(2))

    print("\n" + "=" * 60)
    print("Correlation matrix test completed!")
    print("=" * 60)


def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# PORTFOLIO ANALYTICS TEST SUITE")
    print("#" * 60)

    try:
        test_standalone_functions()
    except Exception as e:
        print(f"\n❌ Standalone functions test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_portfolio_analytics_class()
    except Exception as e:
        print(f"\n❌ PortfolioAnalytics class test failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_correlation_matrix()
    except Exception as e:
        print(f"\n❌ Correlation matrix test failed: {e}")
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
