#!/usr/bin/env python3
"""
Phase-4 debug entrypoint for backtesting infrastructure.

Runs a tiny backtest and prints:
1) key performance metrics
2) weight/turnover/return sanity table
3) benchmark alignment diagnostics
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from portfolio_engine import PortfolioEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tiny backtest diagnostics (Phase 4)")
    parser.add_argument("--factor", type=str, default="five_factor_rotation")
    parser.add_argument("--start-date", type=str, default="2025-07-01")
    parser.add_argument("--end-date", type=str, default="2025-09-30")
    parser.add_argument("--rebalance", type=str, default="monthly", choices=["monthly", "quarterly"])
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--signal-lag-days", type=int, default=1)
    parser.add_argument("--disable-regime-filter", action="store_true")
    parser.add_argument("--disable-slippage", action="store_true")
    parser.add_argument("--disable-vol-targeting", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    if end <= start:
        raise ValueError("end-date must be after start-date")

    print("=" * 84)
    print("PHASE 4 DEBUG: BACKTEST INFRASTRUCTURE")
    print("=" * 84)
    print(f"factor={args.factor} period={start.date()}..{end.date()} rebalance={args.rebalance}")

    script_dir = Path(__file__).resolve().parent
    bist_root = script_dir.parent
    data_dir = bist_root / "data"
    regime_model_dir_candidates = [
        bist_root / "Simple Regime Filter" / "outputs",
        bist_root / "Regime Filter" / "outputs",
    ]
    regime_model_dir = next((p for p in regime_model_dir_candidates if p.exists()), regime_model_dir_candidates[0])

    engine = PortfolioEngine(
        data_dir=data_dir,
        regime_model_dir=regime_model_dir,
        start_date=start,
        end_date=end,
    )
    engine.load_all_data()

    cfg = engine.signal_configs.get(args.factor)
    if not cfg:
        available = sorted(engine.signal_configs.keys())
        raise ValueError(f"Unknown factor '{args.factor}'. Available sample: {available[:20]}")

    dates = engine.close_df.index
    signals, _ = engine._build_signals_for_factor(args.factor, dates, cfg)

    portfolio_opts = dict(cfg.get("portfolio_options", {}))
    portfolio_opts["top_n"] = int(args.top_n)
    portfolio_opts["signal_lag_days"] = int(args.signal_lag_days)
    if args.disable_regime_filter:
        portfolio_opts["use_regime_filter"] = False
    if args.disable_slippage:
        portfolio_opts["use_slippage"] = False
    if args.disable_vol_targeting:
        portfolio_opts["use_vol_targeting"] = False

    results = engine._run_backtest(
        signals=signals,
        factor_name=args.factor,
        rebalance_freq=args.rebalance,
        start_date=start,
        end_date=end,
        portfolio_options=portfolio_opts,
    )

    returns = results["returns"]
    sanity = results.get("sanity_checks", pd.DataFrame())

    print("\nKey metrics:")
    print(f"  total_return={results['total_return']:.4%}")
    print(f"  cagr={results['cagr']:.4%}")
    print(f"  sharpe={results['sharpe']:.4f}")
    print(f"  sortino={results['sortino']:.4f}")
    print(f"  max_drawdown={results['max_drawdown']:.4%}")
    print(f"  win_rate={results['win_rate']:.2%}")
    print(f"  rebalance_count={results['rebalance_count']}")
    print(f"  trade_count={results['trade_count']}")
    print(f"  signal_lag_days={results.get('signal_lag_days')}")

    if not sanity.empty:
        show_cols = [
            "regime",
            "allocation",
            "is_rebalance_day",
            "signal_count",
            "n_active_holdings",
            "weight_sum_raw",
            "effective_weight_sum",
            "rebalance_turnover",
            "portfolio_return",
        ]
        show_cols = [c for c in show_cols if c in sanity.columns]
        print("\nSanity table (tail 12 rows):")
        print(sanity[show_cols].tail(12).to_string())

        invested = sanity[(sanity["allocation"] > 0) & (sanity["n_active_holdings"] > 0)]
        max_weight_dev = (
            float((invested["weight_sum_raw"] - 1.0).abs().max()) if not invested.empty else 0.0
        )
        nan_returns = int(sanity["portfolio_return"].isna().sum()) if "portfolio_return" in sanity.columns else 0
        print("\nSanity summary:")
        print(f"  invested_days={len(invested)}")
        print(f"  max_weight_sum_deviation={max_weight_dev:.3e}")
        print(f"  nan_portfolio_returns={nan_returns}")

    if engine.xu100_prices is not None:
        xu100_alignment = (engine.xu100_prices.shift(-1) / engine.xu100_prices - 1.0).reindex(returns.index)
        common = pd.concat([returns.rename("strategy"), xu100_alignment.rename("xu100")], axis=1).dropna()
        print("\nBenchmark alignment:")
        print(f"  xu100_overlap_days={len(common)}/{len(returns)}")
        if not common.empty:
            corr = common["strategy"].corr(common["xu100"])
            print(f"  xu100_correlation={corr:.4f}")
    else:
        print("\nBenchmark alignment:")
        print("  xu100_data=missing")

    xu030_file = engine.data_dir / "xu030_prices.csv"
    print(f"  xu030_file_present={xu030_file.exists()}")
    if xu030_file.exists():
        try:
            xu030_df = pd.read_csv(xu030_file)
            date_col = "Date" if "Date" in xu030_df.columns else xu030_df.columns[0]
            px_col = "Open" if "Open" in xu030_df.columns else ("Close" if "Close" in xu030_df.columns else xu030_df.columns[-1])
            xu030_series = pd.Series(
                xu030_df[px_col].values,
                index=pd.to_datetime(xu030_df[date_col], errors="coerce"),
                dtype=float,
            ).sort_index()
            xu030_ret = (xu030_series.shift(-1) / xu030_series - 1.0).reindex(returns.index)
            xu030_common = pd.concat([returns.rename("strategy"), xu030_ret.rename("xu030")], axis=1).dropna()
            print(f"  xu030_overlap_days={len(xu030_common)}/{len(returns)}")
        except Exception as exc:
            print(f"  xu030_read_error={exc}")

    print("\nDone. Tiny backtest diagnostics completed.")


if __name__ == "__main__":
    main()
