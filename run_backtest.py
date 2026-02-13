#!/usr/bin/env python3
"""
Robust backtest entrypoint for the BIST project.

Phase 5 goals:
1) Stable CLI/config loading
2) Working-directory independent path resolution
3) Loud, actionable failures for missing inputs/empty data windows
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "Models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from portfolio_engine import PortfolioEngine, load_signal_configs


def _abs_path(path_str: str | None, default: Path) -> Path:
    if path_str is None:
        return default.resolve()
    return Path(path_str).expanduser().resolve()


def _resolve_regime_outputs_dir(project_root: Path, cli_value: str | None) -> Path:
    if cli_value:
        out_dir = _abs_path(cli_value, project_root / "Regime Filter" / "outputs")
        if not out_dir.exists():
            raise FileNotFoundError(
                f"Regime outputs directory does not exist: {out_dir}\n"
                "Pass a valid --regime-outputs path that contains regime_features.csv."
            )
        return out_dir

    candidates = [
        project_root / "Simple Regime Filter" / "outputs",
        project_root / "Regime Filter" / "outputs",
    ]
    found = next((p for p in candidates if p.exists()), None)
    if found is None:
        cands = "\n".join(f"- {p}" for p in candidates)
        raise FileNotFoundError(
            "No regime outputs directory found.\n"
            f"Checked:\n{cands}\n"
            "Run the regime pipeline first to generate outputs."
        )
    return found


def _validate_required_inputs(data_dir: Path, regime_outputs_dir: Path) -> None:
    price_csv = data_dir / "bist_prices_full.csv"
    price_parquet = data_dir / "bist_prices_full.parquet"
    if not price_csv.exists() and not price_parquet.exists():
        raise FileNotFoundError(
            "No price file found for backtest universe.\n"
            f"Expected one of:\n- {price_parquet}\n- {price_csv}\n"
            "Run data/Fetcher-Scrapper/update_prices.py to build/update prices."
        )

    xautry_file = data_dir / "xau_try_2013_2026.csv"
    if not xautry_file.exists():
        raise FileNotFoundError(
            f"Missing required gold benchmark file: {xautry_file}\n"
            "Backtest requires this for regime allocation blending."
        )

    xu100_file = data_dir / "xu100_prices.csv"
    if not xu100_file.exists():
        raise FileNotFoundError(
            f"Missing required benchmark file: {xu100_file}\n"
            "Backtest requires XU100 for benchmark alignment and reporting."
        )

    regime_features_file = regime_outputs_dir / "regime_features.csv"
    if not regime_features_file.exists():
        raise FileNotFoundError(
            f"Missing regime features file: {regime_features_file}\n"
            "Run the regime pipeline first (Simple Regime Filter / Regime Filter outputs)."
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BIST backtests with robust path and input validation."
    )
    parser.add_argument(
        "signal",
        nargs="?",
        default=None,
        help="Signal/factor name to run, or 'all'.",
    )
    parser.add_argument(
        "--factor",
        type=str,
        default=None,
        help="Alias for positional signal argument.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2018-01-01",
        help="Backtest start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-12-31",
        help="Backtest end date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to data directory (default: <project>/data).",
    )
    parser.add_argument(
        "--regime-outputs",
        type=str,
        default=None,
        help="Path to regime outputs directory (contains regime_features.csv).",
    )
    parser.add_argument(
        "--list-signals",
        action="store_true",
        help="List available signal configs and exit.",
    )
    parser.add_argument(
        "--use-config-timeline",
        action="store_true",
        help=(
            "Respect per-signal timeline in config files. "
            "Default behavior enforces CLI --start-date/--end-date."
        ),
    )
    return parser.parse_args()


def _validate_dates(start_date: str, end_date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    try:
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
    except Exception as exc:
        raise ValueError(f"Invalid date format: {exc}") from exc
    if pd.isna(start_ts) or pd.isna(end_ts):
        raise ValueError("Invalid date provided. Use YYYY-MM-DD.")
    if end_ts <= start_ts:
        raise ValueError(
            f"Invalid date range: start={start_ts.date()} end={end_ts.date()}. "
            "end-date must be after start-date."
        )
    return start_ts, end_ts


def _validate_loaded_universe(engine: PortfolioEngine, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> None:
    if engine.prices is None or engine.prices.empty:
        raise ValueError(
            "No price data loaded (empty price table). "
            "Check data files under data/ and run update_prices.py."
        )
    if engine.open_df is None or engine.open_df.empty:
        raise ValueError("Open-price panel is empty; cannot run backtest.")
    if engine.close_df is None or engine.close_df.empty:
        raise ValueError("Close-price panel is empty; cannot run backtest.")
    if engine.close_df.shape[1] == 0:
        raise ValueError("Universe is empty (0 tickers) after panel construction.")

    window = engine.close_df.loc[(engine.close_df.index >= start_ts) & (engine.close_df.index <= end_ts)]
    if window.empty:
        raise ValueError(
            f"No price rows available in requested date range: {start_ts.date()}..{end_ts.date()}."
        )
    if window.notna().sum().sum() == 0:
        raise ValueError(
            f"Requested date range has only NaN prices: {start_ts.date()}..{end_ts.date()}."
        )


def main() -> int:
    args = _parse_args()
    configs = load_signal_configs()
    signal_names = sorted(configs.keys())

    if args.list_signals:
        print("Available signals:")
        for name in signal_names:
            print(f"- {name}")
        return 0

    if not configs:
        raise RuntimeError("No signal configs loaded from Models/configs.")

    signal_to_run = args.signal or args.factor or "all"
    if signal_to_run != "all" and signal_to_run not in signal_names:
        available_preview = ", ".join(signal_names[:30])
        raise ValueError(
            f"Unknown signal: {signal_to_run}\n"
            f"Available (first 30): {available_preview}\n"
            "Use --list-signals to print all."
        )

    start_ts, end_ts = _validate_dates(args.start_date, args.end_date)
    data_dir = _abs_path(args.data_dir, PROJECT_ROOT / "data")
    regime_outputs_dir = _resolve_regime_outputs_dir(PROJECT_ROOT, args.regime_outputs)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    _validate_required_inputs(data_dir, regime_outputs_dir)

    print("=" * 80)
    print("BIST BACKTEST RUNNER")
    print("=" * 80)
    print(f"project_root={PROJECT_ROOT}")
    print(f"signal={signal_to_run}")
    print(f"date_range={start_ts.date()}..{end_ts.date()}")
    print(f"data_dir={data_dir}")
    print(f"regime_outputs={regime_outputs_dir}")
    print(
        "timeline_mode="
        + ("config" if args.use_config_timeline else "cli_override")
    )

    engine = PortfolioEngine(
        data_dir=data_dir,
        regime_model_dir=regime_outputs_dir,
        start_date=str(start_ts.date()),
        end_date=str(end_ts.date()),
    )
    engine.load_all_data()
    _validate_loaded_universe(engine, start_ts, end_ts)

    if signal_to_run == "all":
        if not args.use_config_timeline:
            for cfg in engine.signal_configs.values():
                tl = dict(cfg.get("timeline", {}))
                tl["start_date"] = str(start_ts.date())
                tl["end_date"] = str(end_ts.date())
                cfg["timeline"] = tl
        engine.run_all_factors()
    else:
        override_cfg = None
        if not args.use_config_timeline:
            base_cfg = engine.signal_configs.get(signal_to_run)
            if base_cfg is None:
                raise ValueError(f"Signal config not found at runtime: {signal_to_run}")
            override_cfg = copy.deepcopy(base_cfg)
            tl = dict(override_cfg.get("timeline", {}))
            tl["start_date"] = str(start_ts.date())
            tl["end_date"] = str(end_ts.date())
            override_cfg["timeline"] = tl
        engine.run_factor(signal_to_run, override_config=override_cfg)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"❌ Backtest failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
