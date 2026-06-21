"""
BIST Quant Command Line Interface.

Usage:
    bist-quant --help
    bist-quant info
    bist-quant validate
    bist-quant backtest momentum --start 2020-01-01 --end 2024-01-01
    bist-quant signals list
    bist-quant screener run --index XU100 --template high_dividend --limit 25
    bist-quant scan --template oversold --universe XU100
"""

from __future__ import annotations

import argparse
import sys

from . import __version__


def cmd_info(args) -> None:
    """Show package information."""
    from .common.data_paths import get_data_paths
    from .configs import list_strategies
    from .signals import list_available_signals

    paths = get_data_paths()

    print(f"BIST Quant v{__version__}")
    print(f"\nData Directory: {paths.data_dir}")
    print(f"Data Dir Exists: {paths.data_dir.exists()}")
    print(f"\nAvailable Signals: {len(list_available_signals())}")
    print(f"Available Strategies: {len(list_strategies())}")


def cmd_validate(args) -> None:
    """Validate data paths and files."""
    from .common.data_paths import validate_data_paths

    result = validate_data_paths()

    print(f"Data Directory: {result['data_dir']}")
    print(f"Valid: {result['valid']}")

    print("\nRequired Files:")
    for name, info in result["required"].items():
        status = "[OK]" if info["exists"] else "[X]"
        print(f"  {status} {name}: {info['path']}")

    print("\nOptional Files:")
    for name, info in result["optional"].items():
        status = "[OK]" if info["exists"] else "[ ]"
        print(f"  {status} {name}")

    if result["missing_required"]:
        print(f"\n[WARN] Missing required files: {result['missing_required']}")
        sys.exit(1)


def cmd_signals_list(args) -> None:
    """List available signals."""
    from .signals import list_available_signals

    signals = list_available_signals()
    print(f"Available Signals ({len(signals)}):\n")
    for signal in signals:
        print(f"  - {signal}")


def cmd_strategies_list(args) -> None:
    """List available strategies."""
    from .configs import get_strategy_info, list_strategies

    strategies = list_strategies()
    print(f"Available Strategies ({len(strategies)}):\n")

    for strategy in strategies:
        try:
            info = get_strategy_info(strategy)
            description = str(info.get("description", "No description"))[:60]
            print(f"  - {strategy}: {description}")
        except Exception:
            print(f"  - {strategy}")


def cmd_backtest(args) -> None:
    """Run a backtest."""
    from . import PortfolioEngine

    engine = PortfolioEngine()
    signals = [s.strip() for s in args.signals.split(",") if s.strip()]

    print("Running backtest...")
    print(f"  Signals: {signals}")
    print(f"  Period: {args.start} to {args.end}")

    result = engine.run_backtest(
        signals=signals,
        start_date=args.start,
        end_date=args.end,
    )

    print("\nResults:")
    print(f"  Total Return: {result.metrics['total_return']:.2%}")
    print(f"  Annual Return: {result.metrics['annual_return']:.2%}")
    print(f"  Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {result.metrics['max_drawdown']:.2%}")


def cmd_fundamentals_fetch(args) -> None:
    """Fetch and consolidate fundamental data from İş Yatırım."""
    from .data_pipeline import FundamentalsPipeline

    tickers = None
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    print("Fetching fundamental data from İş Yatırım...")
    if tickers:
        print(f"  Tickers: {len(tickers)} ({', '.join(tickers[:5])}{'...' if len(tickers) > 5 else ''})")
    else:
        print("  Tickers: all BIST")

    pipeline = FundamentalsPipeline()
    result = pipeline.run(
        tickers=tickers,
        force=args.force,
        max_tickers=args.max_tickers,
    )

    print("\n✅ Fundamentals fetched and consolidated.")
    print(f"  Output: {pipeline.paths.consolidated_parquet}")
    if result.merged_bundle is not None:
        print(f"  Tickers with data: {result.merged_bundle.merged_consolidated.index.get_level_values('ticker').nunique()}")
    print(f"  Freshness passed: {result.freshness_passed}")
    print(f"\nNext: run 'bist-quant fundamentals metrics' to derive ratios.")


def cmd_fundamentals_metrics(args) -> None:
    """Derive fundamental metrics (ratios) from consolidated statements."""
    from .data_pipeline.calculate_metrics import compute_fundamental_metrics, write_fundamental_metrics

    print("Computing fundamental metrics...")
    df = compute_fundamental_metrics(data_dir=args.data_dir)
    if df.empty:
        print("\n❌ No metrics computed. Run 'bist-quant fundamentals fetch' first.")
        sys.exit(1)

    output_path = write_fundamental_metrics(df, data_dir=args.data_dir)
    print(f"\n✅ Fundamental metrics written to: {output_path}")
    print(f"   Tickers: {df.index.get_level_values('ticker').nunique()}")
    print(f"   Observations: {len(df)}")


def cmd_fundamentals_status(args) -> None:
    """Show the status of fundamental data files."""
    from .common.data_paths import get_data_paths

    paths = get_data_paths()
    pipeline_output = paths.fundamentals_file
    metrics_file = paths.data_dir / "fundamental_metrics.parquet"
    borsapy_consolidated = paths.borsapy_cache_dir / "financials_consolidated.parquet"

    print("Fundamental Data Status")
    print(f"  Data directory: {paths.data_dir}\n")

    files = [
        ("Pipeline output (consolidated)", pipeline_output),
        ("borsapy_cache (consolidated)", borsapy_consolidated),
        ("Derived metrics", metrics_file),
    ]

    for label, path in files:
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"  [OK]   {label}: {path} ({size_kb:.0f} KB)")
        else:
            print(f"  [----] {label}: {path}")

    print(f"\nTo fetch:  bist-quant fundamentals fetch")
    print(f"To derive: bist-quant fundamentals metrics")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="bist-quant",
        description="BIST Quant - Quantitative Research Library",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    info_parser = subparsers.add_parser("info", help="Show package information")
    info_parser.set_defaults(func=cmd_info)

    validate_parser = subparsers.add_parser("validate", help="Validate data paths")
    validate_parser.set_defaults(func=cmd_validate)

    signals_parser = subparsers.add_parser("signals", help="Signal operations")
    signals_sub = signals_parser.add_subparsers(dest="signals_cmd")
    signals_list = signals_sub.add_parser("list", help="List available signals")
    signals_list.set_defaults(func=cmd_signals_list)

    strategies_parser = subparsers.add_parser("strategies", help="Strategy operations")
    strategies_sub = strategies_parser.add_subparsers(dest="strategies_cmd")
    strategies_list = strategies_sub.add_parser("list", help="List available strategies")
    strategies_list.set_defaults(func=cmd_strategies_list)

    backtest_parser = subparsers.add_parser("backtest", help="Run backtest")
    backtest_parser.add_argument("signals", help="Comma-separated signal names")
    backtest_parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    backtest_parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    backtest_parser.set_defaults(func=cmd_backtest)

    # Fundamentals data pipeline
    fund_parser = subparsers.add_parser("fundamentals", help="Fundamental data pipeline")
    fund_sub = fund_parser.add_subparsers(dest="fund_cmd")

    fund_fetch = fund_sub.add_parser("fetch", help="Fetch & consolidate fundamentals from İş Yatırım")
    fund_fetch.add_argument("--tickers", default=None, help="Comma-separated tickers (default: all BIST)")
    fund_fetch.add_argument("--force", action="store_true", help="Force re-fetch even if cached")
    fund_fetch.add_argument("--max-tickers", type=int, default=None, help="Limit number of tickers")
    fund_fetch.set_defaults(func=cmd_fundamentals_fetch)

    fund_metrics = fund_sub.add_parser("metrics", help="Derive fundamental metrics from statements")
    fund_metrics.add_argument("--data-dir", default=None, help="Data directory override")
    fund_metrics.set_defaults(func=cmd_fundamentals_metrics)

    fund_status = fund_sub.add_parser("status", help="Show fundamentals data status")
    fund_status.set_defaults(func=cmd_fundamentals_status)

    from bist_quant.cli.screener_cli import register_screener_commands

    register_screener_commands(subparsers)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


__all__ = ["main"]


if __name__ == "__main__":
    main()
