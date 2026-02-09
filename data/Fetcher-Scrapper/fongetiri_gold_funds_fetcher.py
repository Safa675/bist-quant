#!/usr/bin/env python3
"""
Fetch daily price history for Turkish gold funds from fongetiri.

The script:
1) Loads the full active fund universe from:
   https://www.fongetiri.com/fon/populer-fonlar
2) Filters funds by name:
   - strict mode: funds with standalone word "ALTIN"
   - expanded mode: strict + "KIYMETLI MADEN" funds
3) Downloads each fund's daily price series from:
   /api/v1/fund/{CODE}/chart?range=5Y
4) Saves outputs to CSV/Parquet (long + wide).

Usage:
    python data/Fetcher-Scrapper/fongetiri_gold_funds_fetcher.py
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from io import StringIO
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


POPULAR_FUNDS_URL = "https://www.fongetiri.com/fon/populer-fonlar"
CHART_API_TEMPLATE = "https://www.fongetiri.com/api/v1/fund/{code}/chart"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch daily prices for Turkish gold funds (GGK-like funds).",
    )
    parser.add_argument(
        "--range",
        default="5Y",
        help="Chart range parameter for API (default: 5Y).",
    )
    parser.add_argument(
        "--mode",
        choices=("strict", "expanded"),
        default="strict",
        help=(
            "Filter mode: strict=ALTIN only, "
            "expanded=ALTIN + KIYMETLI MADEN (default: strict)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="data",
        help="Output directory (default: data).",
    )
    parser.add_argument(
        "--sleep-ms",
        type=int,
        default=40,
        help="Pause between API calls in milliseconds (default: 40).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP timeout in seconds (default: 30).",
    )
    return parser.parse_args()


def _normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return value.upper().strip()


def _is_gold_fund(name: str, mode: str) -> bool:
    normalized = _normalize_text(name)
    has_altin = bool(re.search(r"\bALTIN\b", normalized))
    if mode == "strict":
        return has_altin

    has_precious = bool(re.search(r"\bKIYMETLI\s+MADEN", normalized))
    return has_altin or has_precious


def _find_column_name(columns: Iterable[str], needle: str) -> str:
    for col in columns:
        if needle in str(col):
            return str(col)
    raise ValueError(f"Column containing '{needle}' not found in table.")


def fetch_fund_universe(session: requests.Session, timeout: int) -> pd.DataFrame:
    resp = session.get(POPULAR_FUNDS_URL, timeout=timeout)
    resp.raise_for_status()

    tables = pd.read_html(StringIO(resp.text))
    if not tables:
        raise RuntimeError("No table found on popular funds page.")

    table = tables[0]
    fund_col = _find_column_name(table.columns, "Fonlar")
    price_col = _find_column_name(table.columns, "Anlık Fiyat")
    one_month_col = _find_column_name(table.columns, "1 Ay")

    rows: list[dict[str, str]] = []
    for raw, price_txt, one_month_txt in zip(
        table[fund_col].astype(str),
        table[price_col].astype(str),
        table[one_month_col].astype(str),
    ):
        match = re.match(r"^\s*([A-Z0-9]{3,6})\s+(.+?)\s*$", raw)
        if not match:
            continue
        code = match.group(1).upper()
        name = match.group(2).strip()
        rows.append(
            {
                "code": code,
                "fund_name": name,
                "price_text": price_txt.strip(),
                "one_month_text": one_month_txt.strip(),
            },
        )

    universe = pd.DataFrame(rows).drop_duplicates(subset=["code"]).sort_values("code")
    if universe.empty:
        raise RuntimeError("Parsed fund universe is empty.")
    return universe.reset_index(drop=True)


def fetch_fund_chart(
    session: requests.Session,
    code: str,
    chart_range: str,
    timeout: int,
) -> pd.DataFrame:
    url = CHART_API_TEMPLATE.format(code=code)
    resp = session.get(url, params={"range": chart_range}, timeout=timeout)
    resp.raise_for_status()

    payload = resp.json()
    if not payload.get("status", False):
        raise RuntimeError(f"API status=false for {code}: {payload.get('error')}")

    data = payload.get("data") or {}
    labels = data.get("labels") or []
    prices = data.get("prices") or []
    if not labels or len(labels) != len(prices):
        raise RuntimeError(
            f"Bad chart payload for {code}: labels={len(labels)} prices={len(prices)}",
        )

    series = pd.DataFrame(
        {
            "date": pd.to_datetime(labels, errors="coerce"),
            "price": pd.to_numeric(prices, errors="coerce"),
        },
    ).dropna(subset=["date", "price"])
    if series.empty:
        raise RuntimeError(f"No valid price rows for {code}.")

    return series.sort_values("date").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)

    print("Loading full fund universe...")
    universe = fetch_fund_universe(session=session, timeout=args.timeout)
    print(f"  Universe size: {len(universe)} funds")

    selected = universe[universe["fund_name"].map(lambda n: _is_gold_fund(n, args.mode))]
    selected = selected.sort_values("code").reset_index(drop=True)
    print(f"  Gold filter ({args.mode}): {len(selected)} funds")
    if selected.empty:
        raise RuntimeError("No gold funds matched filter.")

    long_frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []

    for idx, row in selected.iterrows():
        code = str(row["code"])
        name = str(row["fund_name"])
        try:
            chart = fetch_fund_chart(
                session=session,
                code=code,
                chart_range=args.range,
                timeout=args.timeout,
            )
            chart["code"] = code
            chart["fund_name"] = name
            long_frames.append(chart)
            print(f"[{idx + 1:>3}/{len(selected)}] OK   {code} rows={len(chart)}")
        except Exception as exc:  # noqa: BLE001
            failures.append({"code": code, "fund_name": name, "error": str(exc)})
            print(f"[{idx + 1:>3}/{len(selected)}] FAIL {code}: {exc}")
        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    if not long_frames:
        raise RuntimeError("No fund series could be downloaded.")

    long_df = (
        pd.concat(long_frames, ignore_index=True)
        .sort_values(["date", "code"])
        .reset_index(drop=True)
    )

    wide_df = (
        long_df.pivot_table(index="date", columns="code", values="price", aggfunc="last")
        .sort_index()
        .sort_index(axis=1)
    )

    selected_out = out_dir / "gold_funds_universe.csv"
    long_csv_out = out_dir / "gold_funds_daily_prices.csv"
    long_parquet_out = out_dir / "gold_funds_daily_prices.parquet"
    wide_csv_out = out_dir / "gold_funds_daily_prices_wide.csv"
    report_out = out_dir / "gold_funds_fetch_report.json"

    selected.to_csv(selected_out, index=False)
    long_df.to_csv(long_csv_out, index=False)
    long_df.to_parquet(long_parquet_out, index=False)
    wide_df.to_csv(wide_csv_out)

    report = {
        "range": args.range,
        "mode": args.mode,
        "universe_count": int(len(universe)),
        "selected_count": int(len(selected)),
        "downloaded_count": int(long_df["code"].nunique()),
        "start_date": str(long_df["date"].min().date()),
        "end_date": str(long_df["date"].max().date()),
        "row_count_long": int(len(long_df)),
        "row_count_wide": int(len(wide_df)),
        "failed_count": int(len(failures)),
        "failures": failures,
    }
    report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nSaved outputs:")
    print(f"  {selected_out}")
    print(f"  {long_csv_out}")
    print(f"  {long_parquet_out}")
    print(f"  {wide_csv_out}")
    print(f"  {report_out}")

    if failures:
        print(f"\nWarning: {len(failures)} funds failed. See {report_out}.")
    else:
        print("\nAll selected gold funds downloaded successfully.")


if __name__ == "__main__":
    main()
