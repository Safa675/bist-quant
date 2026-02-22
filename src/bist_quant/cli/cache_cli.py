"""
Cache management CLI for borsapy disk cache.

Usage:
    python -m bist_quant.cli.cache_cli warm   --index XUTUM --period 5y
    python -m bist_quant.cli.cache_cli warm   --index XU100 --period 5y   (quick)
    python -m bist_quant.cli.cache_cli inspect
    python -m bist_quant.cli.cache_cli clear   --expired
    python -m bist_quant.cli.cache_cli clear   --all
    python -m bist_quant.cli.cache_cli clear   --category prices
    python -m bist_quant.cli.cache_cli consolidate
    python -m bist_quant.cli.cache_cli validate
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Rate limiting defaults (seconds between requests per provider)
PRICE_DELAY = 0.5       # TradingView
FINANCIAL_DELAY = 1.0   # İş Yatırım
FAST_INFO_DELAY = 0.3   # TradingView


def _get_cache(cache_dir: Path | None = None):
    """Build DiskCache from default or explicit cache directory."""
    from bist_quant.common.cache_config import CacheTTL
    from bist_quant.common.disk_cache import DiskCache
    from bist_quant.common.data_paths import get_data_paths

    paths = get_data_paths()
    resolved = cache_dir or paths.borsapy_cache_dir
    return DiskCache(cache_dir=resolved, ttl=CacheTTL.from_env())


def _get_client_and_cache(cache_dir: Path | None = None):
    """Return (BorsapyClient, DiskCache) pair."""
    cache = _get_cache(cache_dir)

    from bist_quant.common.data_paths import get_data_paths
    paths = get_data_paths()
    try:
        from bist_quant.clients.borsapy_client import BorsapyClient
    except ImportError:
        logger.error("Failed to import borsapy_client from bist_quant.clients. Ensure the module is installed.")
        return 1

    client = BorsapyClient(
        cache_dir=paths.data_dir / "borsapy_cache",
        use_mcp_fallback=True,
    )
    return client, cache


def _progress_bar(current: int, total: int, width: int = 30) -> str:
    """Build a simple ASCII progress bar."""
    pct = current / max(total, 1)
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {current}/{total} ({pct:.0%})"


def _eta(elapsed: float, current: int, total: int) -> str:
    """Estimate time remaining."""
    if current == 0:
        return "?"
    rate = elapsed / current
    remaining = rate * (total - current)
    if remaining < 60:
        return f"{remaining:.0f}s"
    if remaining < 3600:
        return f"{remaining/60:.0f}m"
    return f"{remaining/3600:.1f}h"


# =========================================================================
# cmd_inspect
# =========================================================================
def cmd_inspect(args: argparse.Namespace) -> None:
    cache = _get_cache(args.cache_dir)
    info = cache.inspect()
    if not info:
        logger.info("Cache is empty.")
        return
    total_files = 0
    for category, entries in info.items():
        valid = sum(1 for e in entries if not e.get("expired"))
        expired = len(entries) - valid
        logger.info(f"\n📁 {category}  ({len(entries)} entries, {valid} valid, {expired} expired)")
        for entry in entries[:10]:  # Show first 10 per category
            status = "❌ expired" if entry.get("expired") else "✅ valid"
            logger.info(
                f"  {entry.get('key', '?'):30s}  rows={entry.get('row_count', '?'):>6}  "
                f"source={entry.get('source', '?'):10s}  {status}"
            )
            total_files += 1
        if len(entries) > 10:
            logger.info(f"  ... and {len(entries) - 10} more")
            total_files += len(entries) - 10
    logger.info(f"\nTotal: {total_files} cached entries")


# =========================================================================
# cmd_clear
# =========================================================================
def cmd_clear(args: argparse.Namespace) -> None:
    cache = _get_cache(args.cache_dir)
    if args.expired:
        removed = cache.clear_expired()
        logger.info(f"Removed {removed} expired cache files.")
    elif args.category:
        removed = cache.clear(category=args.category)
        logger.info(f"Cleared category '{args.category}': {removed} files removed.")
    elif args.all:
        removed = cache.clear()
        logger.info(f"Cleared entire cache: {removed} files removed.")
    else:
        logger.info("Specify --expired, --category <name>, or --all")


# =========================================================================
# cmd_warm - Pre-populate cache for the full BIST universe
# =========================================================================
def cmd_warm(args: argparse.Namespace) -> None:
    """Pre-populate the disk cache for the full BIST universe."""
    client, cache = _get_client_and_cache(args.cache_dir)

    logger.info("=" * 60)
    logger.info(f"🔥 CACHE WARMING — {args.index} (period={args.period})")
    logger.info("=" * 60)
    t0 = time.time()

    # -------------------------------------------------------------------
    # Step 1: Resolve symbols
    # -------------------------------------------------------------------
    logger.info(f"\n📋 Step 1: Resolving {args.index} components...")
    symbols = client.get_index_components(args.index)
    if not symbols:
        logger.error(f"Could not resolve symbols for {args.index}")
        sys.exit(1)
    logger.info(f"  ✅ {len(symbols)} symbols resolved from {args.index}")

    # -------------------------------------------------------------------
    # Step 2: Warm prices
    # -------------------------------------------------------------------
    if not args.skip_prices:
        logger.info(f"\n📈 Step 2: Downloading prices ({args.period})...")
        logger.info(f"  Rate limit: {PRICE_DELAY}s between requests")
        t1 = time.time()
        p_success, p_cached, p_errors = 0, 0, 0
        p_error_list: list[str] = []

        for i, sym in enumerate(symbols, 1):
            # Check if already cached and valid
            if not args.force and cache.is_valid("prices", f"{sym}_{args.period}_1d"):
                p_cached += 1
                if i % 100 == 0 or i == len(symbols):
                    elapsed = time.time() - t1
                    logger.info(
                        f"  {_progress_bar(i, len(symbols))} "
                        f"✅{p_success} 📦{p_cached} ❌{p_errors} "
                        f"ETA: {_eta(elapsed, i, len(symbols))}"
                    )
                continue

            try:
                df = client.get_history(sym, period=args.period, interval="1d", use_cache=True)
                if df is not None and not df.empty:
                    p_success += 1
                else:
                    p_errors += 1
                    p_error_list.append(sym)
            except Exception as exc:
                p_errors += 1
                p_error_list.append(sym)
                if args.verbose:
                    logger.warning(f"    {sym}: {exc}")

            if i % 50 == 0 or i == len(symbols):
                elapsed = time.time() - t1
                logger.info(
                    f"  {_progress_bar(i, len(symbols))} "
                    f"✅{p_success} 📦{p_cached} ❌{p_errors} "
                    f"ETA: {_eta(elapsed, i, len(symbols))}"
                )

            time.sleep(PRICE_DELAY)

        logger.info(
            f"\n  Price summary: {p_success} fetched, {p_cached} cached, "
            f"{p_errors} errors ({time.time()-t1:.0f}s)"
        )
        if p_error_list and args.verbose:
            logger.info(f"  Failed: {', '.join(p_error_list[:20])}")

    # -------------------------------------------------------------------
    # Step 3: Warm fundamentals
    # -------------------------------------------------------------------
    if not args.skip_fundamentals:
        logger.info(f"\n📊 Step 3: Downloading fundamentals...")
        logger.info(f"  Rate limit: {FINANCIAL_DELAY}s between requests")
        logger.info(f"  Bank/financial tickers use UFRS accounting group")
        t2 = time.time()
        f_success, f_cached, f_errors = 0, 0, 0
        f_error_list: list[str] = []

        for i, sym in enumerate(symbols, 1):
            # Check if already cached and valid
            if (
                not args.force
                and cache.is_valid("financials", f"{sym}/balance_sheet")
            ):
                f_cached += 1
                if i % 100 == 0 or i == len(symbols):
                    elapsed = time.time() - t2
                    logger.info(
                        f"  {_progress_bar(i, len(symbols))} "
                        f"✅{f_success} 📦{f_cached} ❌{f_errors} "
                        f"ETA: {_eta(elapsed, i, len(symbols))}"
                    )
                continue

            try:
                stmts = client.get_financial_statements(sym)
                non_empty = sum(
                    1 for v in stmts.values()
                    if isinstance(v, pd.DataFrame) and not v.empty
                )
                if non_empty > 0:
                    f_success += 1
                else:
                    f_errors += 1
                    f_error_list.append(sym)
            except Exception as exc:
                f_errors += 1
                f_error_list.append(sym)
                if args.verbose:
                    logger.warning(f"    {sym}: {exc}")

            if i % 50 == 0 or i == len(symbols):
                elapsed = time.time() - t2
                logger.info(
                    f"  {_progress_bar(i, len(symbols))} "
                    f"✅{f_success} 📦{f_cached} ❌{f_errors} "
                    f"ETA: {_eta(elapsed, i, len(symbols))}"
                )

            time.sleep(FINANCIAL_DELAY)

        logger.info(
            f"\n  Fundamentals summary: {f_success} fetched, {f_cached} cached, "
            f"{f_errors} errors ({time.time()-t2:.0f}s)"
        )
        if f_error_list and args.verbose:
            logger.info(f"  Failed: {', '.join(f_error_list[:20])}")

    # -------------------------------------------------------------------
    # Step 4: Consolidate into signal-ready panels
    # -------------------------------------------------------------------
    if not args.skip_consolidate:
        logger.info("\n🔨 Step 4: Consolidating into signal-ready panels...")
        _consolidate(cache)

    total_time = time.time() - t0
    logger.info(f"\n{'='*60}")
    logger.info(f"🎉 Cache warming complete in {total_time:.0f}s")
    logger.info(f"   Run 'python -m bist_quant.cli.cache_cli inspect' to see results")
    logger.info(f"   Run 'python -m bist_quant.cli.cache_cli validate' to check coverage")
    logger.info(f"{'='*60}")


# =========================================================================
# cmd_consolidate - Build signal-ready panels from cached data
# =========================================================================
def _consolidate(cache) -> dict[str, Path]:
    """Build consolidated panel files from per-ticker cache."""
    panels_dir = cache._cache_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    # --- Prices panel ---
    logger.info("  Building prices panel...")
    prices_dir = cache._cache_dir / "prices"
    price_frames: list[pd.DataFrame] = []

    if prices_dir.exists():
        for pf in sorted(prices_dir.glob("*.parquet")):
            if pf.name.endswith(".meta.json"):
                continue
            try:
                df = pd.read_parquet(pf)
                if df.empty:
                    continue
                # Ensure Ticker column exists
                if "Ticker" not in df.columns:
                    # Extract ticker from filename (e.g., THYAO_5y_1d.parquet)
                    ticker = pf.stem.split("_")[0]
                    df["Ticker"] = ticker
                # Ensure Date column
                if "Date" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
                    df = df.reset_index()
                    df = df.rename(columns={"index": "Date"})
                elif "Date" not in df.columns and df.index.name == "Date":
                    df = df.reset_index()
                price_frames.append(df)
            except Exception:
                continue

    if price_frames:
        prices_panel = pd.concat(price_frames, ignore_index=True)
        if "Date" in prices_panel.columns:
            prices_panel["Date"] = pd.to_datetime(prices_panel["Date"], errors="coerce")
            prices_panel = prices_panel.dropna(subset=["Date"])
            prices_panel = (
                prices_panel
                .drop_duplicates(subset=["Date", "Ticker"], keep="last")
                .sort_values(["Ticker", "Date"])
                .reset_index(drop=True)
            )

        out_path = panels_dir / "prices_panel.parquet"
        prices_panel.to_parquet(out_path, index=False)
        outputs["prices_panel"] = out_path

        n_tickers = prices_panel["Ticker"].nunique() if "Ticker" in prices_panel.columns else 0
        logger.info(
            f"    ✅ prices_panel.parquet: {len(prices_panel)} rows, "
            f"{n_tickers} tickers"
        )

        # Also build close panel (wide format: Date × Ticker)
        if "Date" in prices_panel.columns and "Ticker" in prices_panel.columns and "Close" in prices_panel.columns:
            close_panel = prices_panel.pivot_table(
                index="Date", columns="Ticker", values="Close", aggfunc="last",
            ).sort_index()
            close_path = panels_dir / "close_panel.parquet"
            close_panel.to_parquet(close_path)
            outputs["close_panel"] = close_path
            logger.info(
                f"    ✅ close_panel.parquet: {close_panel.shape[0]} days × "
                f"{close_panel.shape[1]} tickers"
            )
    else:
        logger.warning("    ⚠️  No price data found in cache")

    # --- Fundamentals panel ---
    logger.info("  Building fundamentals panel...")
    financials_dir = cache._cache_dir / "financials"

    SHEET_MAP = {
        "balance_sheet": "Bilanço",
        "income_stmt": "Gelir Tablosu (Çeyreklik)",
        "cash_flow": "Nakit Akış (Çeyreklik)",
    }

    rows: list[pd.Series] = []
    ticker_count = 0

    if financials_dir.exists():
        for ticker_dir in sorted(financials_dir.iterdir()):
            if not ticker_dir.is_dir():
                continue
            ticker = ticker_dir.name
            has_data = False

            for parquet_file in ticker_dir.glob("*.parquet"):
                sheet_key = parquet_file.stem
                if sheet_key not in SHEET_MAP:
                    continue
                sheet_name = SHEET_MAP[sheet_key]

                try:
                    df = pd.read_parquet(parquet_file)
                    if df.empty:
                        continue
                    if df.index.name != "Item":
                        if "Item" in df.columns:
                            df = df.set_index("Item")
                        else:
                            continue

                    df_reset = df.reset_index()
                    for _, row in df_reset.iterrows():
                        row_name = row["Item"]
                        values = row.drop("Item").to_dict()
                        if not values:
                            continue
                        s = pd.Series(values, name=(ticker, sheet_name, row_name))
                        rows.append(s)
                    has_data = True
                except Exception:
                    continue

            if has_data:
                ticker_count += 1

    if rows:
        panel = pd.DataFrame(rows)
        panel.index = pd.MultiIndex.from_tuples(
            panel.index.tolist(),
            names=["ticker", "sheet_name", "row_name"],
        )
        panel = panel.dropna(axis=1, how="all")

        out_path = panels_dir / "fundamentals_panel.parquet"
        panel.to_parquet(out_path)
        outputs["fundamentals_panel"] = out_path
        logger.info(
            f"    ✅ fundamentals_panel.parquet: {len(panel)} rows, "
            f"{ticker_count} tickers, {panel.shape[1]} periods"
        )
    else:
        logger.warning("    ⚠️  No fundamentals data found in cache")

    # --- Manifest ---
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "outputs": {k: str(v) for k, v in outputs.items()},
        "prices_tickers": n_tickers if price_frames else 0,
        "fundamentals_tickers": ticker_count,
    }
    manifest_path = panels_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    logger.info(f"    ✅ manifest.json written")

    return outputs


def cmd_consolidate(args: argparse.Namespace) -> None:
    """Build signal-ready panels from cached per-ticker data."""
    cache = _get_cache(args.cache_dir)
    logger.info("🔨 Consolidating cached data into signal-ready panels...\n")
    outputs = _consolidate(cache)
    if outputs:
        logger.info(f"\n✅ Consolidation complete. {len(outputs)} panels built.")
    else:
        logger.warning("\n⚠️  No data to consolidate. Run 'warm' first.")


# =========================================================================
# cmd_validate - Check data coverage
# =========================================================================
def cmd_validate(args: argparse.Namespace) -> None:
    """Validate cache coverage and data quality."""
    cache = _get_cache(args.cache_dir)
    panels_dir = cache._cache_dir / "panels"

    logger.info("🔍 Validating data coverage...\n")
    issues: list[str] = []

    # Check panels exist
    for panel_name in ["prices_panel.parquet", "close_panel.parquet", "fundamentals_panel.parquet"]:
        path = panels_dir / panel_name
        if path.exists():
            df = pd.read_parquet(path)
            if panel_name == "prices_panel.parquet":
                n_tickers = df["Ticker"].nunique() if "Ticker" in df.columns else 0
                n_rows = len(df)
                logger.info(f"  ✅ {panel_name}: {n_rows} rows, {n_tickers} tickers")
                if n_tickers < 50:
                    issues.append(f"  Only {n_tickers} tickers in prices (target: ≥450)")
            elif panel_name == "close_panel.parquet":
                logger.info(f"  ✅ {panel_name}: {df.shape[0]} days × {df.shape[1]} tickers")
            elif panel_name == "fundamentals_panel.parquet":
                n_tickers = df.index.get_level_values("ticker").nunique() if isinstance(df.index, pd.MultiIndex) else 0
                logger.info(f"  ✅ {panel_name}: {len(df)} rows, {n_tickers} tickers")
                if n_tickers < 30:
                    issues.append(f"  Only {n_tickers} tickers in fundamentals (target: ≥300)")
        else:
            logger.warning(f"  ❌ {panel_name}: MISSING")
            issues.append(f"  {panel_name} not found — run 'consolidate'")

    # Check per-ticker cache
    prices_dir = cache._cache_dir / "prices"
    financials_dir = cache._cache_dir / "financials"

    if prices_dir.exists():
        price_files = list(prices_dir.glob("*.parquet"))
        logger.info(f"\n  📁 Per-ticker prices: {len(price_files)} files")
    if financials_dir.exists():
        fin_dirs = [d for d in financials_dir.iterdir() if d.is_dir()]
        logger.info(f"  📁 Per-ticker financials: {len(fin_dirs)} tickers")

    # Check manifest
    manifest_path = panels_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        logger.info(f"\n  📋 Last consolidated: {manifest.get('built_at', 'unknown')}")
        logger.info(f"     Prices tickers: {manifest.get('prices_tickers', '?')}")
        logger.info(f"     Fundamentals tickers: {manifest.get('fundamentals_tickers', '?')}")

    if issues:
        logger.info("\n⚠️  Issues found:")
        for issue in issues:
            logger.info(f"  {issue}")
        logger.info("\nRun 'warm --index XUTUM' to populate missing data.")
    else:
        logger.info("\n✅ All validations passed!")


# =========================================================================
# main
# =========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Borsapy disk cache management",
        prog="python -m bist_quant.cli.cache_cli",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Override cache directory path",
    )
    sub = parser.add_subparsers(dest="command")

    # inspect
    sub.add_parser("inspect", help="Show cache contents and status")

    # clear
    clear_p = sub.add_parser("clear", help="Clear cache entries")
    clear_group = clear_p.add_mutually_exclusive_group()
    clear_group.add_argument("--expired", action="store_true", help="Remove only expired entries")
    clear_group.add_argument("--all", action="store_true", help="Remove all entries")
    clear_group.add_argument("--category", type=str, help="Clear a specific category")

    # warm
    warm_p = sub.add_parser("warm", help="Pre-populate cache for stock universe")
    warm_p.add_argument(
        "--index", type=str, default="XUTUM",
        help="Index to warm (default: XUTUM = full BIST universe)",
    )
    warm_p.add_argument("--period", type=str, default="5y", help="History period (default: 5y)")
    warm_p.add_argument("--skip-prices", action="store_true", help="Skip price download")
    warm_p.add_argument("--skip-fundamentals", action="store_true", help="Skip fundamentals download")
    warm_p.add_argument("--skip-consolidate", action="store_true", help="Skip panel consolidation")
    warm_p.add_argument("--force", action="store_true", help="Re-fetch even if cached")
    warm_p.add_argument("-v", "--verbose", action="store_true", help="Show per-ticker errors")

    # consolidate
    sub.add_parser("consolidate", help="Build signal-ready panels from cached data")

    # validate
    sub.add_parser("validate", help="Check data coverage and quality")

    args = parser.parse_args()

    if args.command == "inspect":
        cmd_inspect(args)
    elif args.command == "clear":
        cmd_clear(args)
    elif args.command == "warm":
        cmd_warm(args)
    elif args.command == "consolidate":
        cmd_consolidate(args)
    elif args.command == "validate":
        cmd_validate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
