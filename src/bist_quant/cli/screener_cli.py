"""CLI for stock screener and technical scans."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from bist_quant.clients.technical_scan import PREDEFINED_SCANS, TechnicalScanner
from bist_quant.screening import get_screener_metadata, run_screener
from bist_quant.screening.errors import ScreeningError


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


def cmd_screener_metadata(args: argparse.Namespace) -> None:
    meta = get_screener_metadata()
    if args.json:
        _print_json(meta)
        return

    print("Screener metadata")
    print(f"  Data sources: {', '.join(meta.get('data_sources', []))}")
    print(f"  Indexes: {', '.join(meta.get('indexes', []))}")
    print(f"  Templates ({len(meta.get('templates', []))}): {', '.join(meta.get('templates', [])[:8])}...")
    print(f"  Filter fields: {len(meta.get('filters', []))}")
    print(f"  Technical scans: {', '.join(meta.get('technical_scans', {}).keys())}")


def cmd_screener_run(args: argparse.Namespace) -> None:
    payload: dict[str, Any] = {
        "data_source": args.data_source,
        "limit": args.limit,
        "sort_by": args.sort_by,
        "sort_desc": not args.ascending,
    }
    if args.index:
        payload["index"] = args.index
    if args.template:
        payload["template"] = args.template
    if args.sector:
        payload["sector"] = args.sector
    if args.recommendation:
        payload["recommendation"] = args.recommendation.upper()
    if args.symbols:
        payload["symbols"] = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if args.technical_scan:
        payload["technical_scan"] = args.technical_scan
    if args.refresh_cache:
        payload["refresh_cache"] = True

    try:
        result = run_screener(payload)
    except ScreeningError as exc:
        print(f"Error: {exc.user_message}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        _print_json(result)
        return

    meta = result.get("meta", {})
    rows = result.get("rows", [])
    print(f"As of {meta.get('as_of')} — {meta.get('returned_rows', len(rows))} rows "
          f"(matches: {meta.get('total_matches', len(rows))})")
    if not rows:
        return

    columns = [col["key"] for col in result.get("columns", []) if col["key"] in rows[0]]
    if not columns:
        columns = list(rows[0].keys())

    widths = {col: max(len(col), *(len(str(row.get(col, ""))) for row in rows)) for col in columns}
    header = "  ".join(col.ljust(widths[col]) for col in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns))


def cmd_scan(args: argparse.Namespace) -> None:
    scanner = TechnicalScanner()
    universe = args.universe
    if args.symbols:
        universe = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    try:
        if args.conditions:
            conditions = [c.strip() for c in args.conditions.split(";") if c.strip()]
            frame = scanner.scan_multi(universe=universe, conditions=conditions, interval=args.interval)
        else:
            frame = scanner.scan(
                universe=universe,
                condition=args.condition or "",
                interval=args.interval,
                template=args.template,
            )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        _print_json(frame.to_dict(orient="records"))
        return

    if frame.empty:
        print("No matches.")
        return

    print(frame.to_string(index=False))


def register_screener_commands(subparsers: argparse._SubParsersAction) -> None:
    screener_parser = subparsers.add_parser("screener", help="Multi-dimensional BIST stock screener")
    screener_sub = screener_parser.add_subparsers(dest="screener_cmd")

    meta_parser = screener_sub.add_parser("metadata", help="Show screener field and template metadata")
    meta_parser.add_argument("--json", action="store_true", help="Emit JSON")
    meta_parser.set_defaults(func=cmd_screener_metadata)

    run_parser = screener_sub.add_parser("run", help="Run the stock screener")
    run_parser.add_argument("--index", default="", help="Index universe (XU030, XU050, XU100, XUTUM, CUSTOM)")
    run_parser.add_argument("--template", default="", help="Named template preset")
    run_parser.add_argument("--data-source", default="local", choices=["local", "isyatirim", "hybrid"])
    run_parser.add_argument("--sector", default="", help="Sector filter")
    run_parser.add_argument("--recommendation", default="", help="AL, TUT, or SAT")
    run_parser.add_argument("--symbols", default="", help="Comma-separated symbols (for CUSTOM index)")
    run_parser.add_argument("--technical-scan", default="", help="Technical scan expression or template name")
    run_parser.add_argument("--limit", type=int, default=25, help="Max rows (default: 25)")
    run_parser.add_argument("--sort-by", default="upside_potential", help="Sort column")
    run_parser.add_argument("--ascending", action="store_true", help="Sort ascending (default: descending)")
    run_parser.add_argument("--refresh-cache", action="store_true", help="Bypass screener frame cache")
    run_parser.add_argument("--json", action="store_true", help="Emit JSON")
    run_parser.set_defaults(func=cmd_screener_run)

    scan_parser = subparsers.add_parser("scan", help="Run a technical condition scan")
    scan_parser.add_argument("--universe", default="XU100", help="Index universe (default: XU100)")
    scan_parser.add_argument("--symbols", default="", help="Comma-separated symbols instead of index")
    scan_parser.add_argument("--condition", default="", help="Scan expression (e.g. 'rsi < 30')")
    scan_parser.add_argument(
        "--template",
        default="",
        help=f"Predefined scan name ({', '.join(sorted(PREDEFINED_SCANS.keys())[:5])}, ...)",
    )
    scan_parser.add_argument(
        "--conditions",
        default="",
        help="Multiple expressions combined with AND, separated by ';'",
    )
    scan_parser.add_argument("--interval", default="1d", help="Bar interval (default: 1d)")
    scan_parser.add_argument("--json", action="store_true", help="Emit JSON")
    scan_parser.set_defaults(func=cmd_scan)
