"""
Data validation script: compare borsapy-fetched data against local files.

Usage:
    python -m bist_quant.cli.validate_data --symbols THYAO AKBNK --period 1y
    python -m bist_quant.cli.validate_data --index XU100 --sample 10
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from typing import Sequence

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _compare_prices(local: pd.DataFrame, borsapy: pd.DataFrame, symbol: str) -> dict:
    """Compare price DataFrames for a single ticker and return mismatch report."""
    report = {
        "symbol": symbol,
        "local_rows": 0,
        "borsapy_rows": 0,
        "overlapping_dates": 0,
        "close_mismatch_pct": 0.0,
        "status": "SKIP",
    }

    # Filter to this symbol
    if "Ticker" in local.columns:
        sym_upper = symbol.upper().split(".")[0]
        local_sym = local[local["Ticker"].astype(str).str.upper().str.split(".").str[0] == sym_upper].copy()
    else:
        local_sym = local.copy()

    if "Ticker" in borsapy.columns:
        borsapy_sym = borsapy[borsapy["Ticker"].astype(str).str.upper().str.split(".").str[0] == sym_upper].copy()
    else:
        borsapy_sym = borsapy.copy()

    report["local_rows"] = len(local_sym)
    report["borsapy_rows"] = len(borsapy_sym)

    if local_sym.empty or borsapy_sym.empty:
        report["status"] = "NO_DATA"
        return report

    # Normalize dates
    for df in (local_sym, borsapy_sym):
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df.set_index("Date", inplace=True)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce")

    # Find overlapping dates
    overlap = local_sym.index.intersection(borsapy_sym.index)
    report["overlapping_dates"] = len(overlap)

    if len(overlap) == 0:
        report["status"] = "NO_OVERLAP"
        return report

    # Compare close prices
    if "Close" in local_sym.columns and "Close" in borsapy_sym.columns:
        local_close = local_sym.loc[overlap, "Close"].astype(float)
        borsapy_close = borsapy_sym.loc[overlap, "Close"].astype(float)
        pct_diff = ((local_close - borsapy_close) / local_close.replace(0, float("nan"))).abs()
        mismatch_pct = (pct_diff > 0.01).mean() * 100  # % of days with >1% diff
        report["close_mismatch_pct"] = round(mismatch_pct, 2)
        report["status"] = "OK" if mismatch_pct < 5 else "MISMATCH"
    else:
        report["status"] = "NO_CLOSE_COL"

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate borsapy data against local files",
        prog="python -m bist_quant.cli.validate_data",
    )
    parser.add_argument("--symbols", nargs="+", help="Specific symbols to validate")
    parser.add_argument("--index", default="XU100", help="Index to use for symbol resolution")
    parser.add_argument("--sample", type=int, default=5, help="Number of symbols to sample if --symbols not given")
    parser.add_argument("--period", default="1y", help="Period for borsapy fetch")

    args = parser.parse_args()

    # Import DataLoader
    import os
    os.environ.setdefault("BIST_DATA_SOURCE", "local")
    from bist_quant.common.data_loader import DataLoader

    logger.info("📊 Data Validation: Borsapy vs Local Files\n")

    # Load local prices
    loader_local = DataLoader(data_source_priority="local")
    try:
        local_prices = loader_local.load_prices()
        logger.info(f"Local prices: {len(local_prices)} records")
    except FileNotFoundError:
        logger.error("❌ Local price file not found. Cannot validate.")
        sys.exit(1)

    # Resolve symbols
    if args.symbols:
        symbols = [s.upper().split(".")[0] for s in args.symbols]
    else:
        # Get available tickers from local data
        if "Ticker" in local_prices.columns:
            all_tickers = local_prices["Ticker"].astype(str).str.upper().str.split(".").str[0].unique().tolist()
        else:
            all_tickers = list(local_prices.columns[:20])
        symbols = random.sample(all_tickers, min(args.sample, len(all_tickers)))

    logger.info(f"Validating {len(symbols)} symbols: {', '.join(symbols)}\n")

    # Load borsapy prices
    loader_borsapy = DataLoader(data_source_priority="borsapy")
    try:
        borsapy_prices = loader_borsapy.load_prices_borsapy(symbols=symbols, period=args.period)
        logger.info(f"Borsapy prices: {len(borsapy_prices)} records\n")
    except Exception as exc:
        logger.error(f"❌ Borsapy fetch failed: {exc}")
        sys.exit(1)

    # Compare
    results: list[dict] = []
    for sym in symbols:
        report = _compare_prices(local_prices, borsapy_prices, sym)
        results.append(report)
        status_icon = {"OK": "✅", "MISMATCH": "❌", "NO_DATA": "⚠️", "NO_OVERLAP": "⚠️"}.get(
            report["status"], "❓"
        )
        logger.info(
            f"  {status_icon} {report['symbol']:8s}  "
            f"local={report['local_rows']:>6} rows  "
            f"borsapy={report['borsapy_rows']:>6} rows  "
            f"overlap={report['overlapping_dates']:>5}  "
            f"mismatch={report['close_mismatch_pct']:>5.1f}%"
        )

    # Summary
    ok_count = sum(1 for r in results if r["status"] == "OK")
    mismatch_count = sum(1 for r in results if r["status"] == "MISMATCH")
    no_data_count = sum(1 for r in results if r["status"] in ("NO_DATA", "NO_OVERLAP", "SKIP"))

    logger.info(f"\n📋 Summary: {ok_count} OK, {mismatch_count} mismatched, {no_data_count} no data")

    if mismatch_count > 0:
        logger.warning(
            "\n⚠️  Some symbols have >5% of trading days with >1% close price difference. "
            "This may be due to adjusted vs unadjusted prices, splits, or data source differences."
        )


if __name__ == "__main__":
    main()
