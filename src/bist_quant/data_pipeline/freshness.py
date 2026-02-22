from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bist_quant.data_pipeline.errors import FreshnessGateError
from bist_quant.data_pipeline.logging_utils import append_jsonl
from bist_quant.data_pipeline.schemas import validate_staleness_report
from bist_quant.data_pipeline.types import FreshnessThresholds


def _period_col_to_timestamp(col: str) -> pd.Timestamp | None:
    try:
        year_str, month_str = str(col).split("/")
        year = int(year_str)
        month = int(month_str)
        return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    except Exception:
        return None


def compute_staleness_report(
    consolidated: pd.DataFrame,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Compute per-ticker latest available quarter and staleness days."""
    if reference_date is None:
        reference_date = pd.Timestamp.now().normalize()

    if consolidated.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "latest_period",
                "period_end",
                "staleness_days",
                "has_q4_2025",
                "reference_date",
            ]
        )

    period_cols = [c for c in consolidated.columns if "/" in str(c)]
    period_ts_map = {col: _period_col_to_timestamp(col) for col in period_cols}
    period_ts_map = {k: v for k, v in period_ts_map.items() if v is not None}
    if not period_ts_map:
        return pd.DataFrame(
            columns=[
                "ticker",
                "latest_period",
                "period_end",
                "staleness_days",
                "has_q4_2025",
                "reference_date",
            ]
        )

    ordered_cols = sorted(period_ts_map.keys(), key=lambda c: period_ts_map[c])
    ticker_level = consolidated.index.get_level_values("ticker")
    any_non_na = consolidated[ordered_cols].notna().groupby(ticker_level).any()

    values = any_non_na.to_numpy(dtype=bool, copy=False)
    reversed_values = values[:, ::-1]
    has_any = reversed_values.any(axis=1)

    latest_positions = np.full(values.shape[0], -1, dtype=int)
    if values.shape[1] > 0:
        rev_argmax = reversed_values.argmax(axis=1)
        latest_positions = np.where(has_any, values.shape[1] - 1 - rev_argmax, -1)

    ordered_cols_arr = np.array(ordered_cols, dtype=object)
    ordered_ts_arr = np.array([period_ts_map[c] for c in ordered_cols], dtype="datetime64[ns]")

    latest_periods = np.where(has_any, ordered_cols_arr[latest_positions], None)
    latest_ts = np.where(has_any, ordered_ts_arr[latest_positions], np.datetime64("NaT"))

    staleness_days = np.where(
        has_any,
        (reference_date.to_datetime64() - latest_ts).astype("timedelta64[D]").astype(float),
        np.nan,
    )

    has_q4_2025 = any_non_na["2025/12"].to_numpy(dtype=bool, copy=False) if "2025/12" in any_non_na.columns else np.zeros(len(any_non_na), dtype=bool)

    report = pd.DataFrame(
        {
            "ticker": any_non_na.index.astype(str),
            "latest_period": latest_periods,
            "period_end": pd.to_datetime(latest_ts),
            "staleness_days": staleness_days,
            "has_q4_2025": has_q4_2025,
            "reference_date": reference_date,
        }
    )

    report = report.sort_values("ticker").reset_index(drop=True)
    return validate_staleness_report(report)


def summarize_quality_metrics(staleness_report: pd.DataFrame) -> dict[str, Any]:
    """Create aggregate data-quality metrics for monitoring and alerts."""
    if staleness_report.empty:
        return {
            "ticker_count": 0,
            "with_data_count": 0,
            "q4_2025_count": 0,
            "q4_2025_coverage_pct": 0.0,
            "median_staleness_days": float("nan"),
            "mean_staleness_days": float("nan"),
            "pct_gt_120_days": float("nan"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    valid_days = staleness_report["staleness_days"].dropna()
    ticker_count = int(len(staleness_report))
    with_data_count = int(valid_days.shape[0])
    q4_count = int(staleness_report["has_q4_2025"].sum())

    return {
        "ticker_count": ticker_count,
        "with_data_count": with_data_count,
        "q4_2025_count": q4_count,
        "q4_2025_coverage_pct": float(q4_count / ticker_count) if ticker_count else 0.0,
        "median_staleness_days": float(valid_days.median()) if with_data_count else float("nan"),
        "mean_staleness_days": float(valid_days.mean()) if with_data_count else float("nan"),
        "max_staleness_days": float(valid_days.max()) if with_data_count else float("nan"),
        "pct_gt_120_days": float((valid_days > 120).mean()) if with_data_count else float("nan"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def evaluate_freshness(
    quality_metrics: dict[str, Any],
    thresholds: FreshnessThresholds,
) -> list[str]:
    """Return freshness-gate violations for a metrics snapshot."""
    violations: list[str] = []

    median_days = quality_metrics.get("median_staleness_days")
    if pd.notna(median_days) and median_days > (thresholds.max_median_staleness_days + thresholds.grace_days):
        violations.append(
            "median_staleness_days "
            f"{median_days:.1f} > {thresholds.max_median_staleness_days + thresholds.grace_days}"
        )

    pct_gt_120 = quality_metrics.get("pct_gt_120_days")
    if pd.notna(pct_gt_120) and pct_gt_120 > thresholds.max_pct_over_120_days:
        violations.append(
            f"pct_gt_120_days {pct_gt_120:.2%} > {thresholds.max_pct_over_120_days:.2%}"
        )

    q4_cov = quality_metrics.get("q4_2025_coverage_pct")
    if pd.notna(q4_cov) and q4_cov < thresholds.min_q4_coverage_pct:
        violations.append(
            f"q4_2025_coverage_pct {q4_cov:.2%} < {thresholds.min_q4_coverage_pct:.2%}"
        )

    max_days = quality_metrics.get("max_staleness_days")
    if pd.notna(max_days) and max_days > (thresholds.max_max_staleness_days + thresholds.grace_days):
        violations.append(
            f"max_staleness_days {max_days:.1f} > {thresholds.max_max_staleness_days + thresholds.grace_days}"
        )

    return violations


def enforce_freshness_gate(
    *,
    quality_metrics: dict[str, Any],
    thresholds: FreshnessThresholds,
    allow_override: bool,
    alerts_log_path: Path,
) -> None:
    """Raise when freshness criteria are violated unless override is enabled."""
    violations = evaluate_freshness(quality_metrics, thresholds)
    if not violations:
        return

    append_jsonl(
        alerts_log_path,
        {
            "event": "freshness_gate_breach",
            "violations": violations,
            "quality_metrics": quality_metrics,
            "thresholds": asdict(thresholds),
            "allow_override": allow_override,
        },
    )

    if allow_override:
        return

    raise FreshnessGateError(
        "Freshness gate blocked fundamentals pipeline: " + "; ".join(violations)
    )
