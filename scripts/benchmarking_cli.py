from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from bist_quant.common.benchmarking import (
    BenchmarkConfig,
    compare_with_baseline,
    load_benchmark_report,
    run_benchmark_suite,
    save_benchmark_report,
)

logger = logging.getLogger(__name__)


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _parse_args() -> argparse.Namespace:
    repo_root = _default_repo_root()
    parser = argparse.ArgumentParser(
        description="Run Phase 6 integrated performance and memory benchmarks.",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Measured repeats per benchmark target.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per benchmark target.")
    parser.add_argument("--days", type=int, default=504, help="Synthetic trading days for backtester benchmark.")
    parser.add_argument("--tickers", type=int, default=100, help="Synthetic ticker count for backtester benchmark.")
    parser.add_argument("--top-n", type=int, default=20, help="Portfolio top-N holdings in backtester benchmark.")
    parser.add_argument(
        "--raw-payload",
        type=str,
        default=str(repo_root / "tests" / "fixtures" / "raw_fundamentals_payload.json"),
        help="JSON fixture used by fundamentals pipeline benchmark.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(repo_root / "benchmarks" / "phase6_latest.json"),
        help="Output report path.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=str(repo_root / "benchmarks" / "phase6_baseline.json"),
        help="Baseline report path for regression checking.",
    )
    parser.add_argument(
        "--tmp-root",
        type=str,
        default=str(repo_root / "benchmarks" / "tmp_runs"),
        help="Temporary benchmark workspace directory.",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Write current run as baseline and exit 0.",
    )
    parser.add_argument(
        "--skip-regression-check",
        action="store_true",
        help="Skip baseline regression checks even if baseline file exists.",
    )
    parser.add_argument(
        "--max-slowdown-pct",
        type=float,
        default=20.0,
        help="Maximum allowed runtime regression percentage.",
    )
    parser.add_argument(
        "--max-memory-regression-pct",
        type=float,
        default=20.0,
        help="Maximum allowed peak memory regression percentage.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    raw_payload_path = Path(args.raw_payload).expanduser().resolve()
    if not raw_payload_path.exists():
        raise FileNotFoundError(f"Raw payload fixture not found: {raw_payload_path}")

    raw_payload = json.loads(raw_payload_path.read_text(encoding="utf-8"))
    report = run_benchmark_suite(
        raw_payload=raw_payload,
        config=BenchmarkConfig(
            repeats=max(1, int(args.repeats)),
            warmup=max(0, int(args.warmup)),
            days=max(40, int(args.days)),
            tickers=max(4, int(args.tickers)),
            top_n=max(2, int(args.top_n)),
        ),
        tmp_root=Path(args.tmp_root).expanduser().resolve(),
    )

    output_path = Path(args.output).expanduser().resolve()
    save_benchmark_report(report, output_path)
    logger.info(f"Saved benchmark report: {output_path}")

    baseline_path = Path(args.baseline).expanduser().resolve()
    if args.write_baseline:
        save_benchmark_report(report, baseline_path)
        logger.info(f"Wrote baseline report: {baseline_path}")
        return 0

    if args.skip_regression_check:
        logger.info("Skipped regression check (--skip-regression-check).")
        return 0

    if not baseline_path.exists():
        logger.info(f"No baseline found at {baseline_path}; skipping regression check.")
        return 0

    baseline_report = load_benchmark_report(baseline_path)
    issues = compare_with_baseline(
        current=report,
        baseline=baseline_report,
        max_slowdown_pct=float(args.max_slowdown_pct),
        max_memory_regression_pct=float(args.max_memory_regression_pct),
    )
    if issues:
        logger.info("Benchmark regression check FAILED:")
        for issue in issues:
            logger.info(f"- {issue}")
        return 2

    logger.info("Benchmark regression check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

