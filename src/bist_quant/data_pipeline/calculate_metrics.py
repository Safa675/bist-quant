"""Derive fundamental metrics (ratios) from consolidated statements.

Reads the ``fundamental_data_consolidated.parquet`` produced by
``FundamentalsPipeline`` (or the borsapy_cache consolidated panel), computes
the derived ratios that signal modules expect in
``fundamental_metrics.parquet``, and writes the result to ``data_dir``.

Output schema: a ``DataFrame`` with a ``(ticker, date)`` MultiIndex and one
column per metric — the shape that ``metrics_df.loc[ticker, "debt_to_equity"]``
and ``_build_metric_panel(metrics_df, metric_name, ...)`` expect.

Usage::

    python -m bist_quant.data_pipeline.calculate_metrics
    python -m bist_quant.data_pipeline.calculate_metrics --data-dir /custom/path
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bist_quant.common.data_paths import get_data_paths
from bist_quant.common.fundamental_utils import (
    apply_lag,
    coerce_quarter_cols,
    get_consolidated_sheet,
    pick_row_from_sheet,
    sum_ttm,
)
from bist_quant.signals.fundamental_keys import (
    CAPEX_KEYS,
    CASH_KEYS,
    DIVIDENDS_PAID_KEYS,
    EBITDA_KEYS,
    GROSS_PROFIT_KEYS,
    LONG_TERM_DEBT_KEYS,
    NET_INCOME_KEYS,
    OPERATING_CF_KEYS,
    OPERATING_INCOME_KEYS,
    REVENUE_KEYS,
    SHARES_OUTSTANDING_KEYS,
    TOTAL_ASSETS_KEYS,
    TOTAL_DEBT_KEYS,
    TOTAL_EQUITY_KEYS,
    TOTAL_LIABILITIES_KEYS,
)

logger = logging.getLogger(__name__)

# Sheet names — MUST match the consolidated parquet sheet_name level.
INCOME_SHEET = "Gelir Tablosu (Çeyreklik)"
BALANCE_SHEET = "Bilanço"
CASH_FLOW_SHEET = "Nakit Akış (Çeyreklik)"

# Reporting lag (days) applied before making data point-in-time.
# Q4 (December) has 75-day audit lag; other quarters 45 days.
Q4_LAG_DAYS = 75
OTHER_LAG_DAYS = 45

# Column order in the output — sorted by consumer usage for readability.
METRIC_COLUMNS = [
    # Leverage / liquidity
    "debt_to_equity",
    "cash_ratio",
    "current_ratio",
    "operating_cash_flow",
    # Profitability
    "operating_margin",
    "gross_margin",
    "net_margin",
    "return_on_equity",
    "return_on_assets",
    # Value (market-cap dependent; may be absent)
    "earnings_yield",
    "fcf_yield",
    "ebitda_ev",
    "sales_price",
    "ocf_ev",
    # Dividends
    "dividend_yield",
    "dividend_payout_ratio",
    # Growth
    "earnings_growth_yoy",
    "revenue_growth_yoy",
]


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Compute a ratio, returning None on zero/None denominator."""
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return None
    ratio = numerator / denominator
    if not np.isfinite(ratio):
        return None
    return float(ratio)


def _safe_diff_yoy(series: pd.Series | None) -> float | None:
    """Year-over-year change (diff of 4 quarters), or None if insufficient data."""
    if series is None or len(series) < 5:
        return None
    return float(series.iloc[-1] - series.iloc[-5])


def _load_consolidated(data_dir: Path) -> pd.DataFrame | None:
    """Load the consolidated fundamentals parquet from data_dir or borsapy_cache."""
    # 1. Pipeline output (preferred).
    pipeline_path = data_dir / "fundamental_data_consolidated.parquet"
    if pipeline_path.exists():
        try:
            frame = pd.read_parquet(pipeline_path)
            if _is_valid_panel(frame):
                logger.info(f"  Loaded consolidated from pipeline: {pipeline_path}")
                return frame
        except Exception as e:
            logger.warning(f"  Failed to read {pipeline_path}: {e}")

    # 2. borsapy_cache consolidated.
    borsapy_path = data_dir / "borsapy_cache" / "financials_consolidated.parquet"
    if borsapy_path.exists():
        try:
            frame = pd.read_parquet(borsapy_path)
            if _is_valid_panel(frame):
                logger.info(f"  Loaded consolidated from borsapy_cache: {borsapy_path}")
                return frame
        except Exception as e:
            logger.warning(f"  Failed to read {borsapy_path}: {e}")

    logger.error(
        f"  No consolidated fundamentals found. Run "
        f"'bist-quant fundamentals fetch' or 'python -m bist_quant.cli.cache_cli warm' first."
    )
    return None


def _is_valid_panel(frame: pd.DataFrame) -> bool:
    """Check the frame has the expected (ticker, sheet_name, row_name) MultiIndex."""
    if frame is None or frame.empty:
        return False
    if not isinstance(frame.index, pd.MultiIndex) or frame.index.nlevels < 3:
        return False
    names = {str(n) for n in frame.index.names if n is not None}
    return {"ticker", "sheet_name", "row_name"}.issubset(names)


def _compute_ticker_metrics(
    consolidated: pd.DataFrame,
    ticker: str,
    daily_dates: pd.DatetimeIndex | None,
) -> list[dict]:
    """Compute all metrics for one ticker, returning one row per quarter date.

    If ``daily_dates`` is provided, metrics are aligned to daily dates via
    ``apply_lag`` (point-in-time with reporting delay). Otherwise they stay
    at quarter-end granularity.
    """
    inc = get_consolidated_sheet(consolidated, ticker, INCOME_SHEET)
    bs = get_consolidated_sheet(consolidated, ticker, BALANCE_SHEET)
    cf = get_consolidated_sheet(consolidated, ticker, CASH_FLOW_SHEET)

    if inc.empty and bs.empty and cf.empty:
        return []

    # Extract raw quarter series.
    def _row(sheet: pd.DataFrame, keys) -> pd.Series:
        r = pick_row_from_sheet(sheet, keys)
        return coerce_quarter_cols(r) if r is not None else pd.Series(dtype=float)

    revenue = _row(inc, REVENUE_KEYS)
    net_income = _row(inc, NET_INCOME_KEYS)
    op_income = _row(inc, OPERATING_INCOME_KEYS)
    gross_profit = _row(inc, GROSS_PROFIT_KEYS)
    ebitda = _row(inc, EBITDA_KEYS)

    total_assets = _row(bs, TOTAL_ASSETS_KEYS)
    total_equity = _row(bs, TOTAL_EQUITY_KEYS)
    total_liab = _row(bs, TOTAL_LIABILITIES_KEYS)
    cash = _row(bs, CASH_KEYS)
    total_debt = _row(bs, TOTAL_DEBT_KEYS)
    lt_debt = _row(bs, LONG_TERM_DEBT_KEYS)

    ocf = _row(cf, OPERATING_CF_KEYS)
    div_paid = _row(cf, DIVIDENDS_PAID_KEYS)

    # TTM sums for flow items (income/cash-flow), point-in-time for stock items.
    revenue_ttm = sum_ttm(revenue) if not revenue.empty else pd.Series(dtype=float)
    ni_ttm = sum_ttm(net_income) if not net_income.empty else pd.Series(dtype=float)
    op_inc_ttm = sum_ttm(op_income) if not op_income.empty else pd.Series(dtype=float)
    gp_ttm = sum_ttm(gross_profit) if not gross_profit.empty else pd.Series(dtype=float)
    ebitda_ttm = sum_ttm(ebitda) if not ebitda.empty else pd.Series(dtype=float)
    ocf_ttm = sum_ttm(ocf) if not ocf.empty else pd.Series(dtype=float)
    div_paid_ttm = sum_ttm(div_paid) if not div_paid.empty else pd.Series(dtype=float)

    # FCF = OCF - |CapEx|. CapEx row may be absent.
    capex = _row(cf, CAPEX_KEYS)
    if not ocf_ttm.empty and not capex.empty:
        capex_ttm = sum_ttm(capex.abs())
        fcf_ttm = ocf_ttm - capex_ttm
    elif not ocf_ttm.empty:
        fcf_ttm = ocf_ttm.copy()  # approximate: OCF as FCF proxy
    else:
        fcf_ttm = pd.Series(dtype=float)

    # Stock items are point-in-time (no TTM). Align them to the TTM index.
    all_dates = revenue_ttm.index.union(ni_ttm.index).union(total_assets.index).sort_values().unique()

    rows: list[dict] = []
    for dt in all_dates:
        rev_v = revenue_ttm.get(dt)
        ni_v = ni_ttm.get(dt)
        op_v = op_inc_ttm.get(dt)
        gp_v = gp_ttm.get(dt)
        ebitda_v = ebitda_ttm.get(dt)
        ocf_v = ocf_ttm.get(dt)
        div_v = div_paid_ttm.get(dt)
        fcf_v = fcf_ttm.get(dt)

        ta_v = total_assets.sort_index().asof(dt) if not total_assets.empty else None
        te_v = total_equity.sort_index().asof(dt) if not total_equity.empty else None
        tl_v = total_liab.sort_index().asof(dt) if not total_liab.empty else None
        cash_v = cash.sort_index().asof(dt) if not cash.empty else None
        debt_v = total_debt.sort_index().asof(dt) if not total_debt.empty else None

        row: dict = {"ticker": ticker, "date": dt}

        # Leverage / liquidity
        row["debt_to_equity"] = _safe_ratio(tl_v if tl_v is not None else debt_v, te_v)
        row["cash_ratio"] = _safe_ratio(cash_v, debt_v if debt_v is not None else tl_v)
        row["current_ratio"] = _safe_ratio(cash_v, tl_v)  # approx: cash / total liab
        row["operating_cash_flow"] = float(ocf_v) if ocf_v is not None and np.isfinite(ocf_v) else None

        # Profitability
        row["operating_margin"] = _safe_ratio(op_v, rev_v)
        row["gross_margin"] = _safe_ratio(gp_v, rev_v)
        row["net_margin"] = _safe_ratio(ni_v, rev_v)
        row["return_on_equity"] = _safe_ratio(ni_v, te_v)
        row["return_on_assets"] = _safe_ratio(ni_v, ta_v)

        # Value (market-cap dependent — left as None here; filled in post-hoc if prices available)
        row["earnings_yield"] = None
        row["fcf_yield"] = None
        row["ebitda_ev"] = None
        row["sales_price"] = None
        row["ocf_ev"] = None

        # Dividends
        row["dividend_yield"] = None  # market-cap dependent
        row["dividend_payout_ratio"] = _safe_ratio(div_v, ni_v)

        # Growth (YoY diff of TTM)
        rev_series = revenue_ttm.reindex(revenue_ttm.index.union([dt])).sort_index()
        ni_series = ni_ttm.reindex(ni_ttm.index.union([dt])).sort_index()
        row["earnings_growth_yoy"] = _safe_diff_yoy(ni_series.loc[:dt])
        row["revenue_growth_yoy"] = _safe_diff_yoy(rev_series.loc[:dt])

        # Store raw TTM values for post-hoc market-cap ratio computation.
        row["_raw_ni_ttm"] = float(ni_v) if ni_v is not None and np.isfinite(ni_v) else None
        row["_raw_fcf_ttm"] = float(fcf_v) if fcf_v is not None and np.isfinite(fcf_v) else None
        row["_raw_ebitda_ttm"] = float(ebitda_v) if ebitda_v is not None and np.isfinite(ebitda_v) else None
        row["_raw_revenue_ttm"] = float(rev_v) if rev_v is not None and np.isfinite(rev_v) else None
        row["_raw_ocf_ttm"] = float(ocf_v) if ocf_v is not None and np.isfinite(ocf_v) else None
        row["_raw_div_paid_ttm"] = float(div_v) if div_v is not None and np.isfinite(div_v) else None
        row["_raw_debt"] = float(debt_v) if debt_v is not None and np.isfinite(debt_v) else None
        row["_raw_cash"] = float(cash_v) if cash_v is not None and np.isfinite(cash_v) else None

        rows.append(row)

    return rows


def compute_fundamental_metrics(
    data_dir: Path | None = None,
    *,
    prices_panel: pd.DataFrame | None = None,
    shares_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute derived fundamental metrics from consolidated statements.

    Args:
        data_dir: Directory containing ``fundamental_data_consolidated.parquet``.
            Defaults to ``DataPaths.data_dir``.
        prices_panel: Optional close-price panel (Date x Ticker) for market-cap
            dependent metrics. If None, those metrics are left as NaN.
        shares_panel: Optional shares-outstanding panel (Date x Ticker).

    Returns:
        DataFrame with (ticker, date) MultiIndex and metric columns.
    """
    paths = get_data_paths() if data_dir is None else None
    resolved_dir = paths.data_dir if paths is not None else data_dir

    logger.info("📊 Computing fundamental metrics...")
    consolidated = _load_consolidated(resolved_dir)
    if consolidated is None:
        return pd.DataFrame()

    tickers = consolidated.index.get_level_values("ticker").unique().tolist()
    logger.info(f"  Processing {len(tickers)} tickers...")

    all_rows: list[dict] = []
    for i, ticker in enumerate(tickers):
        rows = _compute_ticker_metrics(consolidated, ticker, daily_dates=None)
        all_rows.extend(rows)
        if (i + 1) % 100 == 0:
            logger.info(f"    Progress: {i + 1}/{len(tickers)}")

    if not all_rows:
        logger.warning("  ⚠️  No metrics computed (no fundamental data found)")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Fill market-cap-dependent metrics if prices + shares are available.
    if prices_panel is not None and shares_panel is not None:
        df = _fill_market_cap_metrics(df, prices_panel, shares_panel)

    # Ensure all expected columns exist.
    for col in METRIC_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # Set (ticker, date) MultiIndex and sort.
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["ticker", "date"], keep="last")
    df = df.set_index(["ticker", "date"]).sort_index()

    # Drop the helper _raw_* columns used for market-cap ratio computation.
    raw_cols = [c for c in df.columns if c.startswith("_raw_")]
    df = df.drop(columns=raw_cols, errors="ignore")

    logger.info(
        f"  ✅ Computed {len(df)} metric observations for "
        f"{df.index.get_level_values('ticker').nunique()} tickers"
    )
    logger.info(f"  Metrics: {[c for c in METRIC_COLUMNS if c in df.columns]}")

    return df[METRIC_COLUMNS]


def _fill_market_cap_metrics(
    df: pd.DataFrame,
    prices: pd.DataFrame,
    shares: pd.DataFrame,
) -> pd.DataFrame:
    """Fill market-cap-dependent metrics from raw TTM values stored in ``_raw_*`` columns.

    The metrics DataFrame carries raw TTM values (``_raw_ni_ttm``,
    ``_raw_fcf_ttm``, etc.) alongside the ratios so this function can compute
    the market-cap ratios without re-reading the consolidated parquet.
    """
    market_cap = (prices * shares).dropna(how="all")
    if market_cap.empty:
        return df

    raw_cols = [
        "_raw_ni_ttm", "_raw_fcf_ttm", "_raw_ebitda_ttm",
        "_raw_revenue_ttm", "_raw_ocf_ttm", "_raw_div_paid_ttm",
    ]
    available_raws = [c for c in raw_cols if c in df.columns]
    if not available_raws:
        return df

    for idx in df.index:
        ticker, date = idx
        if ticker not in market_cap.columns:
            continue
        mc_series = market_cap[ticker].dropna()
        if mc_series.empty:
            continue
        try:
            mc = float(mc_series.asof(date))
        except (KeyError, IndexError, ValueError):
            continue
        if mc <= 0 or not np.isfinite(mc):
            continue

        # Enterprise value ≈ market cap + debt - cash.
        debt = df.at[idx, "_raw_debt"] if "_raw_debt" in df.columns else 0
        cash_val = df.at[idx, "_raw_cash"] if "_raw_cash" in df.columns else 0
        ev = mc + (debt or 0) - (cash_val or 0)

        ni = df.at[idx, "_raw_ni_ttm"] if "_raw_ni_ttm" in df.columns else None
        fcf = df.at[idx, "_raw_fcf_ttm"] if "_raw_fcf_ttm" in df.columns else None
        ebitda_v = df.at[idx, "_raw_ebitda_ttm"] if "_raw_ebitda_ttm" in df.columns else None
        rev = df.at[idx, "_raw_revenue_ttm"] if "_raw_revenue_ttm" in df.columns else None
        ocf_v = df.at[idx, "_raw_ocf_ttm"] if "_raw_ocf_ttm" in df.columns else None
        div_v = df.at[idx, "_raw_div_paid_ttm"] if "_raw_div_paid_ttm" in df.columns else None

        df.at[idx, "earnings_yield"] = _safe_ratio(ni, mc)
        df.at[idx, "fcf_yield"] = _safe_ratio(fcf, mc)
        df.at[idx, "ebitda_ev"] = _safe_ratio(ebitda_v, ev)
        df.at[idx, "sales_price"] = _safe_ratio(rev, mc)
        df.at[idx, "ocf_ev"] = _safe_ratio(ocf_v, ev)
        df.at[idx, "dividend_yield"] = _safe_ratio(div_v, mc)

    return df


def write_fundamental_metrics(
    df: pd.DataFrame,
    data_dir: Path | None = None,
) -> Path:
    """Write the metrics DataFrame to ``data_dir/fundamental_metrics.parquet``."""
    paths = get_data_paths() if data_dir is None else None
    resolved_dir = paths.data_dir if paths is not None else Path(data_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)

    output_path = resolved_dir / "fundamental_metrics.parquet"
    df.to_parquet(output_path)
    logger.info(f"💾 Written fundamental metrics: {output_path}")
    return output_path


def main() -> None:
    """CLI entry point for ``python -m bist_quant.data_pipeline.calculate_metrics``."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        prog="calculate_metrics",
        description="Derive fundamental metrics from consolidated statements.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Data directory (default: $BIST_DATA_DIR or ~/.local/share/bist-quant/data)",
    )
    args = parser.parse_args()

    df = compute_fundamental_metrics(data_dir=args.data_dir)
    if df.empty:
        print("\n❌ No metrics computed. Ensure consolidated fundamentals exist:")
        print("   bist-quant fundamentals fetch")
        print("   # or")
        print("   python -m bist_quant.cli.cache_cli warm --index XU100 --period 5y")
        sys.exit(1)

    output_path = write_fundamental_metrics(df, data_dir=args.data_dir)
    print(f"\n✅ Fundamental metrics written to: {output_path}")
    print(f"   Tickers: {df.index.get_level_values('ticker').nunique()}")
    print(f"   Observations: {len(df)}")
    print(f"   Metrics: {df.columns.tolist()}")


if __name__ == "__main__":
    main()
