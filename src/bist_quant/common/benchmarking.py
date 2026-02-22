from __future__ import annotations

import json
import platform
import statistics
import time
import tracemalloc
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from bist_quant.common.backtester import Backtester
from bist_quant.common.risk_manager import RiskManager
from bist_quant.data_pipeline.pipeline import (
    FundamentalsPipeline,
    build_default_config,
    build_default_paths,
)
from bist_quant.data_pipeline.types import RawDataBundle


@dataclass(frozen=True)
class BenchmarkConfig:
    repeats: int = 3
    warmup: int = 1
    days: int = 504
    tickers: int = 100
    top_n: int = 20


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _measure_once(fn: Callable[[], dict[str, Any]]) -> tuple[float, float, dict[str, Any]]:
    tracemalloc.start()
    started = time.perf_counter()
    payload = fn()
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)
    return elapsed, peak_mb, payload


class _SyntheticLoader:
    def __init__(self, xautry_prices: pd.Series, data_dir: Path) -> None:
        self._xautry_prices = xautry_prices
        self.data_dir = data_dir

    def load_xautry_prices(
        self,
        _path: Path,
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
    ) -> pd.Series:
        series = self._xautry_prices.copy()
        if start_date is not None:
            series = series[series.index >= pd.Timestamp(start_date)]
        if end_date is not None:
            series = series[series.index <= pd.Timestamp(end_date)]
        return series


def _build_size_market_cap_panel(
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    _loader: Any,
) -> pd.DataFrame:
    n_tickers = max(1, len(close_df.columns))
    caps = np.linspace(10_000_000_000.0, 200_000_000.0, n_tickers, dtype=float)
    panel = pd.DataFrame(
        np.tile(caps, (len(dates), 1)),
        index=dates,
        columns=close_df.columns,
        dtype=float,
    )
    return panel


def _build_synthetic_market_data(
    *,
    days: int,
    tickers: int,
    seed: int = 42,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    days = max(40, int(days))
    tickers = max(4, int(tickers))

    dates = pd.bdate_range("2022-01-03", periods=days)
    ticker_names = [f"T{idx:03d}" for idx in range(tickers)]
    idx = np.arange(days, dtype=float)

    noise = rng.normal(loc=0.0002, scale=0.012, size=(days, tickers))
    trend = np.linspace(0.0, 0.015, tickers, dtype=float)[None, :]
    close_returns = noise + trend / 252.0
    close_values = 100.0 * np.exp(np.cumsum(close_returns, axis=0))

    open_noise = rng.normal(loc=0.0, scale=0.0015, size=(days, tickers))
    open_values = close_values * (1.0 + open_noise)
    volume_values = rng.uniform(600_000.0, 3_500_000.0, size=(days, tickers))

    close_df = pd.DataFrame(close_values, index=dates, columns=ticker_names)
    open_df = pd.DataFrame(open_values, index=dates, columns=ticker_names)
    volume_df = pd.DataFrame(volume_values, index=dates, columns=ticker_names)

    stacked_index = pd.MultiIndex.from_product([dates, ticker_names], names=["Date", "Ticker"])
    prices = pd.DataFrame(
        {
            "Date": stacked_index.get_level_values("Date"),
            "Ticker": [f"{ticker}.IS" for ticker in stacked_index.get_level_values("Ticker")],
            "Open": open_values.reshape(-1),
            "Close": close_values.reshape(-1),
            "Volume": volume_values.reshape(-1),
        }
    )

    momentum = close_df.pct_change(20, fill_method=None)
    signals = momentum.rank(axis=1, pct=True) * 100.0
    signals = signals.ffill().bfill().fillna(50.0)

    regimes = ["Bull", "Bear", "Recovery", "Stress"]
    regime_series = pd.Series(
        [regimes[int(i) % len(regimes)] for i in idx],
        index=dates,
        dtype="object",
    )
    xu100_prices = pd.Series(5_000.0 + np.arange(days, dtype=float) * 8.0, index=dates, name="XU100")
    xautry_prices = pd.Series(1_800.0 + np.arange(days, dtype=float) * 1.5, index=dates, name="XAU_TRY")

    return {
        "prices": prices,
        "close_df": close_df,
        "open_df": open_df,
        "volume_df": volume_df,
        "signals": signals,
        "regime_series": regime_series,
        "xu100_prices": xu100_prices,
        "xautry_prices": xautry_prices,
    }


def _backtester_options(top_n: int) -> dict[str, Any]:
    return {
        "use_regime_filter": True,
        "use_vol_targeting": False,
        "use_inverse_vol_sizing": True,
        "use_stop_loss": True,
        "use_liquidity_filter": True,
        "use_slippage": True,
        "use_mcap_slippage": True,
        "top_n": max(2, top_n),
        "signal_lag_days": 1,
        "slippage_bps": 5.0,
        "small_cap_slippage_bps": 20.0,
        "mid_cap_slippage_bps": 10.0,
        "stop_loss_threshold": 0.15,
        "liquidity_quantile": 0.25,
        "inverse_vol_lookback": 20,
        "max_position_weight": 0.35,
        "target_downside_vol": 0.2,
        "vol_lookback": 30,
        "vol_floor": 0.1,
        "vol_cap": 1.0,
    }


def run_backtester_benchmark(
    *,
    days: int,
    tickers: int,
    top_n: int,
    seed: int = 42,
) -> dict[str, Any]:
    dataset = _build_synthetic_market_data(days=days, tickers=tickers, seed=seed)
    loader = _SyntheticLoader(
        xautry_prices=dataset["xautry_prices"],
        data_dir=Path("."),
    )
    risk_manager = RiskManager(close_df=dataset["close_df"], volume_df=dataset["volume_df"])
    backtester = Backtester(
        loader=loader,
        data_dir=Path("."),
        risk_manager=risk_manager,
        build_size_market_cap_panel=_build_size_market_cap_panel,
    )
    backtester.update_data(
        prices=dataset["prices"],
        close_df=dataset["close_df"],
        volume_df=dataset["volume_df"],
        regime_series=dataset["regime_series"],
        regime_allocations={"Bull": 1.0, "Bear": 0.0, "Recovery": 0.5, "Stress": 0.0},
        xu100_prices=dataset["xu100_prices"],
        xautry_prices=dataset["xautry_prices"],
    )
    result = backtester.run(
        signals=dataset["signals"],
        factor_name="benchmark_signal",
        rebalance_freq="monthly",
        portfolio_options=_backtester_options(top_n=top_n),
    )
    returns = result["returns"]
    return {
        "returns_count": int(len(returns)),
        "trade_events": int(len(result.get("trade_events", []))),
        "sharpe": _safe_float(result.get("sharpe")),
        "max_drawdown": _safe_float(result.get("max_drawdown")),
    }


def run_pipeline_benchmark(
    *,
    raw_payload: dict[str, Any],
    workdir: Path,
) -> dict[str, Any]:
    paths = build_default_paths(base_dir=workdir)
    config = build_default_config(
        enforce_freshness_gate=False,
        allow_stale_override=True,
    )
    pipeline = FundamentalsPipeline(paths=paths, config=config)
    raw_bundle = RawDataBundle(
        raw_by_ticker=raw_payload,
        errors=[],
        source_name="benchmark_fixture",
        fetched_at=datetime.now(timezone.utc),
    )
    result = pipeline.process_raw_bundle(raw_bundle=raw_bundle)
    merged_bundle = result.merged_bundle
    if merged_bundle is None:
        raise RuntimeError("Pipeline benchmark did not produce merged bundle")

    return {
        "merged_rows": int(merged_bundle.merged_consolidated.shape[0]),
        "merged_cols": int(merged_bundle.merged_consolidated.shape[1]),
        "cache_hit": bool(merged_bundle.merge_stats.get("cache_hit", False)),
    }


def _run_target(
    *,
    name: str,
    fn: Callable[[], dict[str, Any]],
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    for _ in range(max(0, warmup)):
        fn()

    elapsed_samples: list[float] = []
    memory_samples: list[float] = []
    metadata: dict[str, Any] = {}

    for _ in range(max(1, repeats)):
        elapsed, peak_mb, payload = _measure_once(fn)
        elapsed_samples.append(elapsed)
        memory_samples.append(peak_mb)
        metadata = payload

    return {
        "name": name,
        "runs": len(elapsed_samples),
        "elapsed_seconds": elapsed_samples,
        "peak_memory_mb": memory_samples,
        "median_elapsed_seconds": float(statistics.median(elapsed_samples)),
        "median_peak_memory_mb": float(statistics.median(memory_samples)),
        "metadata": metadata,
    }


def run_benchmark_suite(
    *,
    raw_payload: dict[str, Any],
    config: BenchmarkConfig | None = None,
    tmp_root: Path | None = None,
) -> dict[str, Any]:
    cfg = config or BenchmarkConfig()
    root = (tmp_root or Path("benchmarks") / "tmp_runs").resolve()
    root.mkdir(parents=True, exist_ok=True)

    pipeline_counter = {"idx": 0}

    def _pipeline_runner() -> dict[str, Any]:
        pipeline_counter["idx"] += 1
        run_dir = root / f"pipeline_run_{pipeline_counter['idx']}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_pipeline_benchmark(raw_payload=raw_payload, workdir=run_dir)

    def _backtester_runner() -> dict[str, Any]:
        return run_backtester_benchmark(
            days=cfg.days,
            tickers=cfg.tickers,
            top_n=cfg.top_n,
            seed=42,
        )

    backtester_result = _run_target(
        name="backtester_run",
        fn=_backtester_runner,
        warmup=cfg.warmup,
        repeats=cfg.repeats,
    )
    pipeline_result = _run_target(
        name="fundamentals_pipeline",
        fn=_pipeline_runner,
        warmup=cfg.warmup,
        repeats=cfg.repeats,
    )

    return {
        "generated_at_utc": _now_utc_iso(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "config": {
            "repeats": cfg.repeats,
            "warmup": cfg.warmup,
            "days": cfg.days,
            "tickers": cfg.tickers,
            "top_n": cfg.top_n,
        },
        "benchmarks": {
            "backtester_run": backtester_result,
            "fundamentals_pipeline": pipeline_result,
        },
    }


def compare_with_baseline(
    *,
    current: dict[str, Any],
    baseline: dict[str, Any],
    max_slowdown_pct: float = 20.0,
    max_memory_regression_pct: float = 20.0,
) -> list[str]:
    issues: list[str] = []
    current_bench = current.get("benchmarks", {})
    baseline_bench = baseline.get("benchmarks", {})

    for name, base_metrics in baseline_bench.items():
        current_metrics = current_bench.get(name)
        if current_metrics is None:
            issues.append(f"Missing benchmark target in current run: {name}")
            continue

        base_elapsed = _safe_float(base_metrics.get("median_elapsed_seconds"), default=np.nan)
        curr_elapsed = _safe_float(current_metrics.get("median_elapsed_seconds"), default=np.nan)
        if np.isfinite(base_elapsed) and np.isfinite(curr_elapsed) and base_elapsed > 0:
            allowed_elapsed = base_elapsed * (1.0 + max_slowdown_pct / 100.0)
            if curr_elapsed > allowed_elapsed:
                issues.append(
                    f"{name}: elapsed regression {curr_elapsed:.4f}s > {allowed_elapsed:.4f}s "
                    f"(baseline {base_elapsed:.4f}s, tolerance {max_slowdown_pct:.1f}%)"
                )

        base_mem = _safe_float(base_metrics.get("median_peak_memory_mb"), default=np.nan)
        curr_mem = _safe_float(current_metrics.get("median_peak_memory_mb"), default=np.nan)
        if np.isfinite(base_mem) and np.isfinite(curr_mem) and base_mem > 0:
            allowed_mem = base_mem * (1.0 + max_memory_regression_pct / 100.0)
            if curr_mem > allowed_mem:
                issues.append(
                    f"{name}: memory regression {curr_mem:.2f}MB > {allowed_mem:.2f}MB "
                    f"(baseline {base_mem:.2f}MB, tolerance {max_memory_regression_pct:.1f}%)"
                )

    return issues


def load_benchmark_report(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_benchmark_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

