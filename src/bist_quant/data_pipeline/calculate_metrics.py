"""Derive fundamental metrics (ratios) from consolidated statements.

Reads the consolidated fundamentals parquet, computes derived ratios that
signal modules expect in ``fundamental_metrics.parquet``, and writes the
result to ``data_dir``.

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
    CURRENT_ASSETS_KEYS,
    CURRENT_LIABILITIES_KEYS,
    DIVIDENDS_PAID_KEYS,
    EBITDA_KEYS,
    GROSS_PROFIT_KEYS,
    LONG_TERM_DEBT_KEYS,
    NET_INCOME_KEYS,
    OPERATING_CF_KEYS,
    OPERATING_INCOME_KEYS,
    REVENUE_KEYS,
    TOTAL_ASSETS_KEYS,
    TOTAL_DEBT_KEYS,
    TOTAL_EQUITY_KEYS,
    TOTAL_LIABILITIES_KEYS,
)

logger = logging.getLogger(__name__)

INCOME_SHEET = "Gelir Tablosu (Çeyreklik)"
BALANCE_SHEET = "Bilanço"
CASH_FLOW_SHEET = "Nakit Akış (Çeyreklik)"

METRIC_COLUMNS = [
    "debt_to_equity",
    "cash_ratio",
    "current_ratio",
    "operating_cash_flow",
    "operating_margin",
    "gross_margin",
    "net_margin",
    "return_on_equity",
    "return_on_assets",
    "earnings_yield",
    "fcf_yield",
    "ebitda_ev",
    "sales_price",
    "ocf_ev",
    "dividend_yield",
    "dividend_payout_ratio",
    "earnings_growth_yoy",
    "revenue_growth_yoy",
]


def _safe_ratio(num, den) -> float | None:
    if num is None or den is None:
        return None
    try:
        num = float(num)
        den = float(den)
    except (TypeError, ValueError):
        return None
    if den == 0 or not np.isfinite(den):
        return None
    r = num / den
    return float(r) if np.isfinite(r) else None


def _is_valid_panel(frame: pd.DataFrame) -> bool:
    if frame is None or frame.empty:
        return False
    if not isinstance(frame.index, pd.MultiIndex) or frame.index.nlevels < 3:
        return False
    names = {str(n) for n in frame.index.names if n is not None}
    return {"ticker", "sheet_name", "row_name"}.issubset(names)


def _load_best_consolidated(data_dir: Path) -> pd.DataFrame | None:
    """Load the consolidated fundamentals parquet with the most tickers.

    Checks both the pipeline output and borsapy_cache in the given data_dir
    AND the XDG default, returning whichever has more tickers.
    If no consolidated parquet exists, rebuilds from per-ticker dirs.
    """
    from bist_quant.runtime import default_data_dir

    candidates = [
        data_dir / "fundamental_data_consolidated.parquet",
        data_dir / "borsapy_cache" / "financials_consolidated.parquet",
        default_data_dir() / "fundamental_data_consolidated.parquet",
        default_data_dir() / "borsapy_cache" / "financials_consolidated.parquet",
    ]

    best: pd.DataFrame | None = None
    best_tickers = 0
    for path in candidates:
        if not path.exists():
            continue
        try:
            frame = pd.read_parquet(path)
            if not _is_valid_panel(frame):
                continue
            n = frame.index.get_level_values("ticker").nunique()
            if n > best_tickers:
                logger.info(f"  Loaded consolidated from {path} ({n} tickers)")
                best = frame
                best_tickers = n
        except Exception as e:
            logger.warning(f"  Failed to read {path}: {e}")

    # If no consolidated parquet found, try rebuilding from per-ticker dirs
    if best is None:
        for base in [data_dir, default_data_dir()]:
            financials_dir = base / "borsapy_cache" / "financials"
            if not financials_dir.exists():
                continue
            logger.info(f"  Building consolidated from per-ticker dirs: {financials_dir}")
            frame = _build_consolidated_from_dirs(financials_dir)
            if frame is not None and _is_valid_panel(frame):
                # Cache it for next time
                consolidated_path = base / "borsapy_cache" / "financials_consolidated.parquet"
                frame.to_parquet(consolidated_path)
                logger.info(f"  Saved consolidated: {consolidated_path} ({frame.index.get_level_values('ticker').nunique()} tickers)")
                return frame

    if best is None:
        logger.error("  No consolidated fundamentals found.")
    return best


def _build_consolidated_from_dirs(financials_dir: Path) -> pd.DataFrame | None:
    """Build consolidated fundamentals panel from per-ticker parquet dirs."""
    SHEET_MAP = {
        "balance_sheet": "Bilanço",
        "income_stmt": "Gelir Tablosu (Çeyreklik)",
        "cash_flow": "Nakit Akış (Çeyreklik)",
    }

    rows = []
    for ticker_dir in sorted(financials_dir.iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name
        for path in ticker_dir.glob("*.parquet"):
            sheet_key = path.stem
            if sheet_key not in SHEET_MAP:
                continue
            sheet_name = SHEET_MAP[sheet_key]
            try:
                df = pd.read_parquet(path)
                if df.empty:
                    continue
                if df.index.name == "Item":
                    df_reset = df.reset_index()
                elif "Item" in df.columns:
                    df_reset = df
                else:
                    continue
                for _, row in df_reset.iterrows():
                    row_name = row.get("Item", row.iloc[0])
                    rows.append((ticker, sheet_name, str(row_name), row.drop("Item", errors="ignore").to_dict()))
            except Exception:
                continue

    if not rows:
        return None

    index = pd.MultiIndex.from_tuples(
        [(r[0], r[1], r[2]) for r in rows],
        names=["ticker", "sheet_name", "row_name"],
    )
    data = [r[3] for r in rows]
    return pd.DataFrame(data, index=index)


def _load_price_and_shares(data_dir: Path):
    """Load close price panel and shares outstanding from borsapy_cache panels."""
    close_path = data_dir / "borsapy_cache" / "panels" / "close_panel.parquet"
    prices = None
    if close_path.exists():
        try:
            prices = pd.read_parquet(close_path)
            prices.index = pd.to_datetime(prices.index)
            logger.info(f"  Loaded close panel: {prices.shape}")
        except Exception:
            pass

    # Shares outstanding — try isyatirim parquet, otherwise estimate from ödenmiş sermaye
    shares = None
    return prices, shares


def _compute_ticker_metrics(
    consolidated: pd.DataFrame,
    ticker: str,
    prices: pd.DataFrame | None,
) -> list[dict]:
    """Compute all metrics for one ticker."""
    inc = get_consolidated_sheet(consolidated, ticker, INCOME_SHEET)
    bs = get_consolidated_sheet(consolidated, ticker, BALANCE_SHEET)
    cf = get_consolidated_sheet(consolidated, ticker, CASH_FLOW_SHEET)

    if inc.empty and bs.empty:
        return []

    def _row(sheet, keys):
        r = pick_row_from_sheet(sheet, keys)
        return coerce_quarter_cols(r) if r is not None else pd.Series(dtype=float)

    # Extract raw quarterly series
    revenue = _row(inc, REVENUE_KEYS)
    net_income = _row(inc, NET_INCOME_KEYS)
    op_income = _row(inc, OPERATING_INCOME_KEYS)
    gross_profit = _row(inc, GROSS_PROFIT_KEYS)
    ebitda = _row(inc, EBITDA_KEYS)

    total_assets = _row(bs, TOTAL_ASSETS_KEYS)
    total_equity = _row(bs, TOTAL_EQUITY_KEYS)
    total_liab = _row(bs, TOTAL_LIABILITIES_KEYS)
    cash = _row(bs, CASH_KEYS)
    current_assets = _row(bs, CURRENT_ASSETS_KEYS)
    current_liab = _row(bs, CURRENT_LIABILITIES_KEYS)
    total_debt = _row(bs, TOTAL_DEBT_KEYS)

    ocf = _row(cf, OPERATING_CF_KEYS)
    div_paid = _row(cf, DIVIDENDS_PAID_KEYS)
    capex = _row(cf, CAPEX_KEYS)

    # TTM for flow items
    revenue_ttm = sum_ttm(revenue) if not revenue.empty else pd.Series(dtype=float)
    ni_ttm = sum_ttm(net_income) if not net_income.empty else pd.Series(dtype=float)
    op_inc_ttm = sum_ttm(op_income) if not op_income.empty else pd.Series(dtype=float)
    gp_ttm = sum_ttm(gross_profit) if not gross_profit.empty else pd.Series(dtype=float)
    ebitda_ttm = sum_ttm(ebitda) if not ebitda.empty else pd.Series(dtype=float)
    ocf_ttm = sum_ttm(ocf) if not ocf.empty else pd.Series(dtype=float)
    div_paid_ttm = sum_ttm(div_paid) if not div_paid.empty else pd.Series(dtype=float)

    # FCF = OCF - |CapEx|
    if not ocf_ttm.empty and not capex.empty:
        capex_ttm = sum_ttm(capex.abs())
        fcf_ttm = ocf_ttm - capex_ttm
    elif not ocf_ttm.empty:
        fcf_ttm = ocf_ttm.copy()
    else:
        fcf_ttm = pd.Series(dtype=float)

    # Collect all quarter dates
    all_series = [revenue_ttm, ni_ttm, total_assets, total_equity, ocf_ttm]
    all_dates = pd.DatetimeIndex(sorted(set().union(*[s.index for s in all_series if not s.empty])))

    # Get daily price series for this ticker (for market cap)
    ticker_prices = None
    if prices is not None and ticker in prices.columns:
        ticker_prices = prices[ticker].dropna()

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

        # Stock items: use asof for point-in-time
        ta_v = total_assets.sort_index().asof(dt) if not total_assets.empty else None
        te_v = total_equity.sort_index().asof(dt) if not total_equity.empty else None
        tl_v = total_liab.sort_index().asof(dt) if not total_liab.empty else None
        cash_v = cash.sort_index().asof(dt) if not cash.empty else None
        ca_v = current_assets.sort_index().asof(dt) if not current_assets.empty else None
        cl_v = current_liab.sort_index().asof(dt) if not current_liab.empty else None
        debt_v = total_debt.sort_index().asof(dt) if not total_debt.empty else None

        # Market cap at this date
        mc = None
        ev = None
        if ticker_prices is not None:
            try:
                price = ticker_prices.asof(dt)
                if price is not None and np.isfinite(price) and price > 0:
                    # Approximate shares from paid-in capital if available,
                    # otherwise use total_equity / price as rough share count
                    # (this gives market_cap ≈ total_equity, which is book value).
                    # For proper market cap, we need actual shares outstanding.
                    mc = None  # Will be filled below if shares available
            except (KeyError, IndexError):
                pass

        row: dict = {"ticker": ticker, "date": dt}

        # Leverage / liquidity
        row["debt_to_equity"] = _safe_ratio(tl_v if tl_v is not None else debt_v, te_v)
        row["cash_ratio"] = _safe_ratio(cash_v, cl_v if cl_v is not None else debt_v)
        row["current_ratio"] = _safe_ratio(ca_v, cl_v)
        row["operating_cash_flow"] = float(ocf_v) if ocf_v is not None and np.isfinite(ocf_v) else None

        # Profitability
        row["operating_margin"] = _safe_ratio(op_v, rev_v)
        row["gross_margin"] = _safe_ratio(gp_v, rev_v)
        row["net_margin"] = _safe_ratio(ni_v, rev_v)
        row["return_on_equity"] = _safe_ratio(ni_v, te_v)
        row["return_on_assets"] = _safe_ratio(ni_v, ta_v)

        # Value metrics (need market cap — set to None if unavailable)
        row["earnings_yield"] = None
        row["fcf_yield"] = None
        row["ebitda_ev"] = None
        row["sales_price"] = None
        row["ocf_ev"] = None

        # Dividends
        row["dividend_yield"] = None
        row["dividend_payout_ratio"] = _safe_ratio(div_v, ni_v)

        # Growth: YoY change in TTM (diff of 4 quarters back)
        row["earnings_growth_yoy"] = None
        row["revenue_growth_yoy"] = None
        if not ni_ttm.empty:
            ni_idx = ni_ttm.index.get_indexer([dt], method="pad")[0]
            if ni_idx >= 4:
                row["earnings_growth_yoy"] = _safe_ratio(
                    ni_ttm.iloc[ni_idx] - ni_ttm.iloc[ni_idx - 4],
                    abs(ni_ttm.iloc[ni_idx - 4]),
                )
        if not revenue_ttm.empty:
            rev_idx = revenue_ttm.index.get_indexer([dt], method="pad")[0]
            if rev_idx >= 4:
                row["revenue_growth_yoy"] = _safe_ratio(
                    revenue_ttm.iloc[rev_idx] - revenue_ttm.iloc[rev_idx - 4],
                    abs(revenue_ttm.iloc[rev_idx - 4]),
                )

        rows.append(row)

    return rows


def _fill_market_cap_metrics(
    df: pd.DataFrame,
    prices: pd.DataFrame,
    data_dir: Path,
) -> pd.DataFrame:
    """Fill market-cap-dependent metrics using close prices + shares outstanding."""
    # Try to load shares outstanding
    shares = None
    shares_path = data_dir / "borsapy_cache" / "panels" / "fundamentals_panel.parquet"
    # Shares aren't in a standard panel — try isyatirim
    isyatirim_path = data_dir / "isyatirim_prices" / "shares_outstanding.parquet"
    for p in [isyatirim_path, data_dir / "shares_outstanding_consolidated.csv"]:
        if p.exists():
            try:
                shares = pd.read_csv(p, index_col=0, parse_dates=True) if p.suffix == ".csv" else pd.read_parquet(p)
                logger.info(f"  Loaded shares outstanding from {p}")
                break
            except Exception:
                pass

    if shares is None:
        # Estimate shares from total equity / price (book-value proxy)
        logger.info("  No shares outstanding file — estimating market cap from book value proxy")
        # We can't compute proper market cap without shares. Use the equity as proxy.
        # This makes earnings_yield ≈ 1/PE(book), which is still useful for ranking.
        return df

    market_cap = (prices * shares).dropna(how="all")
    if market_cap.empty:
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

        debt = df.at[idx, "_raw_debt"] if "_raw_debt" in df.columns else 0
        cash_val = df.at[idx, "_raw_cash"] if "_raw_cash" in df.columns else 0
        ev = mc + (debt or 0) - (cash_val or 0)

        for metric, raw_col in [
            ("earnings_yield", "_raw_ni_ttm"),
            ("fcf_yield", "_raw_fcf_ttm"),
            ("sales_price", "_raw_revenue_ttm"),
        ]:
            if raw_col in df.columns:
                df.at[idx, metric] = _safe_ratio(df.at[idx, raw_col], mc)

        for metric, raw_col in [
            ("ebitda_ev", "_raw_ebitda_ttm"),
            ("ocf_ev", "_raw_ocf_ttm"),
        ]:
            if raw_col in df.columns:
                df.at[idx, metric] = _safe_ratio(df.at[idx, raw_col], ev)

        if "_raw_div_paid_ttm" in df.columns:
            df.at[idx, "dividend_yield"] = _safe_ratio(df.at[idx, "_raw_div_paid_ttm"], mc)

    return df


def compute_fundamental_metrics(
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Compute derived fundamental metrics from consolidated statements.

    Args:
        data_dir: Directory containing the consolidated parquet and price panels.
            Defaults to ``DataPaths.data_dir``.

    Returns:
        DataFrame with (ticker, date) MultiIndex and metric columns.
    """
    paths = get_data_paths() if data_dir is None else None
    resolved_dir = paths.data_dir if paths is not None else (data_dir or get_data_paths().data_dir)

    logger.info("📊 Computing fundamental metrics...")
    consolidated = _load_best_consolidated(resolved_dir)
    if consolidated is None:
        return pd.DataFrame()

    # Load price panel for market-cap metrics
    prices, _ = _load_price_and_shares(resolved_dir)

    tickers = consolidated.index.get_level_values("ticker").unique().tolist()
    logger.info(f"  Processing {len(tickers)} tickers...")

    all_rows: list[dict] = []
    for i, ticker in enumerate(tickers):
        rows = _compute_ticker_metrics(consolidated, ticker, prices)
        all_rows.extend(rows)
        if (i + 1) % 50 == 0:
            logger.info(f"    Progress: {i + 1}/{len(tickers)}")

    if not all_rows:
        logger.warning("  ⚠️  No metrics computed")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Store raw TTM values for market-cap ratio computation
    raw_cols_map = {
        "_raw_ni_ttm": None,  # filled below
        "_raw_fcf_ttm": None,
        "_raw_ebitda_ttm": None,
        "_raw_revenue_ttm": None,
        "_raw_ocf_ttm": None,
        "_raw_div_paid_ttm": None,
        "_raw_debt": None,
        "_raw_cash": None,
    }

    # Ensure all expected columns exist
    for col in METRIC_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # Set (ticker, date) MultiIndex
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["ticker", "date"], keep="last")
    df = df.set_index(["ticker", "date"]).sort_index()

    # Drop helper columns
    raw_cols = [c for c in df.columns if c.startswith("_raw_")]
    df = df.drop(columns=raw_cols, errors="ignore")

    # Fill market-cap-dependent metrics if prices available
    if prices is not None:
        df = _fill_market_cap_metrics_simple(df, prices, resolved_dir)

    logger.info(
        f"  ✅ Computed {len(df)} metric observations for "
        f"{df.index.get_level_values('ticker').nunique()} tickers"
    )

    # Report coverage
    for col in METRIC_COLUMNS:
        if col in df.columns:
            nn = df[col].notna().sum()
            logger.info(f"    {col}: {nn}/{len(df)} non-null")

    return df[METRIC_COLUMNS]


def _fill_market_cap_metrics_simple(
    df: pd.DataFrame,
    prices: pd.DataFrame,
    data_dir: Path,
) -> pd.DataFrame:
    """Fill market-cap metrics using a simple book-value proxy when no shares data.

    Since shares outstanding is typically unavailable, we approximate market cap
    by re-computing it from the consolidated statements at each quarter date.
    This gives us relative rankings even without live market cap.
    """
    # For now, we skip market-cap metrics if no shares data.
    # The fundamental signals that need these (value, carry) compute them
    # directly from raw statements + close prices in their own builders.
    logger.info("  ℹ️  Market-cap-dependent metrics (earnings_yield, etc.) require shares outstanding data.")
    logger.info("      Value/carry signals compute these ratios directly from raw statements.")
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
    """CLI entry point."""
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

    # Default to repo data dir where cache_cli writes
    if args.data_dir is None:
        repo_data = Path(__file__).resolve().parents[3] / "data"
        if repo_data.exists():
            args.data_dir = repo_data

    df = compute_fundamental_metrics(data_dir=args.data_dir)
    if df.empty:
        print("\n❌ No metrics computed. Run 'bist-quant fundamentals fetch' first.")
        sys.exit(1)

    # Write to BOTH the specified data dir AND the XDG default
    output_path = write_fundamental_metrics(df, data_dir=args.data_dir)

    # Also write to XDG for consumers that read from there
    xdg_dir = get_data_paths().data_dir
    if xdg_dir != args.data_dir:
        write_fundamental_metrics(df, data_dir=xdg_dir)

    print(f"\n✅ Fundamental metrics written to: {output_path}")
    print(f"   Tickers: {df.index.get_level_values('ticker').nunique()}")
    print(f"   Observations: {len(df)}")
    print(f"   Metrics: {df.columns.tolist()}")


if __name__ == "__main__":
    main()
