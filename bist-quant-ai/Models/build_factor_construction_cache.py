#!/usr/bin/env python3
"""
Precompute and persist five-factor axis construction cache to parquet.

This cache stores heavy intermediate panels used by five_factor_rotation:
- size_small_signal
- value_level_signal
- value_growth_signal
- profit_margin_level
- profit_margin_growth
- investment_reinvestment_signal

Usage:
    python Models/build_factor_construction_cache.py
    python Models/build_factor_construction_cache.py --force-rebuild
    python Models/build_factor_construction_cache.py --cache-path data/my_axis_cache.parquet
    python Models/build_factor_construction_cache.py --start-date 2018-01-01 --end-date 2026-12-31
"""

import argparse
import time
from pathlib import Path
import sys

import pandas as pd

# Add Models/ to path
sys.path.insert(0, str(Path(__file__).parent))

from common.data_loader import DataLoader
from signals.five_factor_rotation_signals import (
    build_five_factor_rotation_axis_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build five-factor axis construction cache parquet")
    parser.add_argument("--cache-path", type=str, default=None, help="Output parquet path")
    parser.add_argument("--force-rebuild", action="store_true", help="Ignore existing cache and fully rebuild")
    parser.add_argument("--start-date", type=str, default=None, help="Optional start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="Optional end date (YYYY-MM-DD)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()

    models_dir = Path(__file__).parent
    project_root = models_dir.parent
    data_dir = project_root / "data"
    regime_model_dir_candidates = [
        project_root / "Simple Regime Filter" / "outputs",
        project_root / "Regime Filter" / "outputs",
    ]
    regime_model_dir = next((p for p in regime_model_dir_candidates if p.exists()), regime_model_dir_candidates[0])

    print("=" * 70)
    print("BUILDING FIVE-FACTOR AXIS CONSTRUCTION CACHE")
    print("=" * 70)

    loader = DataLoader(data_dir=data_dir, regime_model_dir=regime_model_dir)

    prices_file = data_dir / "bist_prices_full.csv"
    prices = loader.load_prices(prices_file)
    close_df = loader.build_close_panel(prices)
    volume_df = loader.build_volume_panel(prices)
    fundamentals = loader.load_fundamentals()

    dates = close_df.index
    if args.start_date is not None:
        start_ts = pd.Timestamp(args.start_date)
        dates = dates[dates >= start_ts]
    if args.end_date is not None:
        end_ts = pd.Timestamp(args.end_date)
        dates = dates[dates <= end_ts]

    close_slice = close_df.reindex(dates)
    volume_slice = volume_df.reindex(dates)

    print(f"Date range: {dates.min().date()} -> {dates.max().date()} ({len(dates)} days)")
    print(f"Tickers: {close_slice.shape[1]}")

    cache_path = build_five_factor_rotation_axis_cache(
        close_df=close_slice,
        dates=dates,
        data_loader=loader,
        fundamentals=fundamentals,
        volume_df=volume_slice,
        cache_path=args.cache_path,
        force_rebuild=args.force_rebuild,
    )

    elapsed = time.time() - started
    print(f"\n✅ Axis cache ready: {cache_path}")
    print(f"⏱️  Total runtime: {elapsed:.1f}s ({elapsed/60.0:.1f}m)")


if __name__ == "__main__":
    main()
