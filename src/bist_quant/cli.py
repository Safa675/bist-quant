"""
BIST Quant Command Line Interface.

Usage:
    bist-quant --help
    bist-quant info
    bist-quant validate
    bist-quant backtest momentum --start 2020-01-01 --end 2024-01-01
    bist-quant signals list
    bist-quant api serve --port 8001
"""

from __future__ import annotations

import argparse
import sys


def cmd_info(args) -> None:
    """Show package information."""
    from . import __version__
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


def cmd_api_serve(args) -> None:
    """Start API server."""
    import uvicorn

    from .api.main import app

    print(f"Starting BIST Quant API on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="bist-quant",
        description="BIST Quant - Quantitative Research Library",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.3.0")

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

    api_parser = subparsers.add_parser("api", help="API operations")
    api_sub = api_parser.add_subparsers(dest="api_cmd")
    api_serve = api_sub.add_parser("serve", help="Start API server")
    api_serve.add_argument("--host", default="127.0.0.1", help="Host to bind")
    api_serve.add_argument("--port", type=int, default=8001, help="Port to bind")
    api_serve.set_defaults(func=cmd_api_serve)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
