#!/usr/bin/env python3
"""
Analyze XU100 seasonality by Month-of-Year and Day-of-Week.

Outputs:
- Console summary (Top 3 months, Top 2 weekdays)
- Excel workbook: Models/results/seasonality/seasonality_trends.xlsx
"""

from __future__ import annotations

import argparse
import calendar
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "Models"
SEASONALITY_RESULTS_DIR = MODELS_DIR / "results" / "seasonality"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from common.data_loader import DataLoader


WEEKDAY_NAMES = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}


def _resolve_regime_dir(project_root: Path) -> Path:
    candidates = [
        project_root / "Regime Filter",
        project_root / "Simple Regime Filter",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _build_monthly_stats(returns: pd.Series) -> pd.DataFrame:
    grouped = returns.groupby(returns.index.month)
    stats = grouped.agg(
        mean_return="mean",
        win_rate=lambda x: (x > 0).mean(),
        n_obs="count",
    )
    stats["win_rate_pct"] = stats["win_rate"] * 100.0
    stats["month_name"] = pd.Index(stats.index).map(lambda m: calendar.month_name[int(m)])
    stats.index.name = "month"

    return (
        stats.reset_index()[["month", "month_name", "n_obs", "mean_return", "win_rate_pct"]]
        .sort_values("month")
        .reset_index(drop=True)
    )


def _build_weekday_stats(returns: pd.Series) -> pd.DataFrame:
    grouped = returns.groupby(returns.index.dayofweek)
    stats = grouped.agg(
        mean_return="mean",
        win_rate=lambda x: (x > 0).mean(),
        n_obs="count",
    )
    stats["win_rate_pct"] = stats["win_rate"] * 100.0
    stats["day_name"] = pd.Index(stats.index).map(WEEKDAY_NAMES.get)
    stats.index.name = "day_of_week"

    return (
        stats.reset_index()[["day_of_week", "day_name", "n_obs", "mean_return", "win_rate_pct"]]
        .sort_values("day_of_week")
        .reset_index(drop=True)
    )


def _build_monthly_yearly_pivot(returns: pd.Series, start_year: int, end_year: int) -> pd.DataFrame:
    df = returns.to_frame("return")
    df["year"] = df.index.year
    df["month"] = df.index.month
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]

    pivot = df.pivot_table(index="year", columns="month", values="return", aggfunc="mean")
    pivot = pivot.reindex(index=range(start_year, end_year + 1), columns=range(1, 13))
    pivot.columns = [calendar.month_abbr[int(m)] for m in pivot.columns]
    pivot.index.name = "year"
    return pivot


def _build_weekday_yearly_pivot(returns: pd.Series, start_year: int, end_year: int) -> pd.DataFrame:
    df = returns.to_frame("return")
    df["year"] = df.index.year
    df["day_of_week"] = df.index.dayofweek
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]

    pivot = df.pivot_table(index="year", columns="day_of_week", values="return", aggfunc="mean")
    pivot = pivot.reindex(index=range(start_year, end_year + 1), columns=[0, 1, 2, 3, 4])
    pivot.columns = [WEEKDAY_NAMES[d] for d in pivot.columns]
    pivot.index.name = "year"
    return pivot


def _print_summary(monthly_stats: pd.DataFrame, weekday_stats: pd.DataFrame) -> None:
    fmt = lambda x: f"{x:.4f}"

    top_months = monthly_stats.sort_values("mean_return", ascending=False).head(3)
    top_weekdays = weekday_stats.sort_values("mean_return", ascending=False).head(2)

    print("\nTop 3 Months (by mean return):")
    print(top_months[["month", "month_name", "n_obs", "mean_return", "win_rate_pct"]].to_string(index=False, float_format=fmt))

    print("\nTop 2 Weekdays (by mean return):")
    print(top_weekdays[["day_of_week", "day_name", "n_obs", "mean_return", "win_rate_pct"]].to_string(index=False, float_format=fmt))

    print(f"\nDefault monthly signal months: {top_months['month'].tolist()}")
    print(f"Default weekly signal days: {top_weekdays['day_of_week'].tolist()}")


def _save_to_excel(
    output_file: Path,
    monthly_stats: pd.DataFrame,
    weekday_stats: pd.DataFrame,
    monthly_pivot: pd.DataFrame,
    weekday_pivot: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        monthly_stats.to_excel(writer, sheet_name="monthly_stats", index=False)
        weekday_stats.to_excel(writer, sheet_name="weekday_stats", index=False)
        monthly_pivot.to_excel(writer, sheet_name="monthly_yearly_trend")
        weekday_pivot.to_excel(writer, sheet_name="weekday_yearly_trend")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze XU100 month/weekday seasonality trends.")
    parser.add_argument(
        "--xu100-file",
        type=str,
        default=str(PROJECT_ROOT / "data" / "xu100_prices.csv"),
        help="Path to XU100 CSV.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=str(SEASONALITY_RESULTS_DIR / "seasonality_trends.xlsx"),
        help="Output Excel path.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2013,
        help="Start year for analysis window.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="End year for analysis window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    xu100_file = Path(args.xu100_file)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(
        data_dir=PROJECT_ROOT / "data",
        regime_model_dir=_resolve_regime_dir(PROJECT_ROOT),
    )

    xu100_prices = loader.load_xu100_prices(xu100_file).dropna().sort_index()
    returns = xu100_prices.pct_change(fill_method=None).dropna()
    returns = returns[(returns.index.year >= args.start_year) & (returns.index.year <= args.end_year)]

    if returns.empty:
        raise ValueError(
            f"No returns found in selected window {args.start_year}-{args.end_year}. "
            f"Available data spans {xu100_prices.index.min().date()} to {xu100_prices.index.max().date()}."
        )

    monthly_stats = _build_monthly_stats(returns)
    weekday_stats = _build_weekday_stats(returns)
    monthly_pivot = _build_monthly_yearly_pivot(returns, args.start_year, args.end_year)
    weekday_pivot = _build_weekday_yearly_pivot(returns, args.start_year, args.end_year)

    print(
        f"\nAnalysis window (returns): {returns.index.min().date()} to {returns.index.max().date()} "
        f"({len(returns)} observations)"
    )
    _print_summary(monthly_stats, weekday_stats)

    _save_to_excel(output_file, monthly_stats, weekday_stats, monthly_pivot, weekday_pivot)
    print(f"\nSaved seasonality workbook to: {output_file}")


if __name__ == "__main__":
    main()
