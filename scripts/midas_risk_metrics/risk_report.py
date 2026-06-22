"""
Midas portfolio risk report.

Parses Midas Hesap Ekstresi (PDF) statements to reconstruct a personal
portfolio's trades, cash flows, and dividends. Fetches historical prices
from borsapy for mark-to-market valuation, then computes return and risk
metrics.

Sources of truth:
  * Midas PDF (Hesap Ekstresi) - executed trades, cash flows, dividends

Returns computed:
  * Absolute return in TRY (mark-to-market today, includes dividends)
  * Money-Weighted Return (MWR / IRR) - the actual return on your invested capital
  * Per-trade statistics (hit rate, profit factor, expectancy) - measures
    decision-making skill directly

Risk metrics (Sharpe, Sortino, Max Drawdown, etc.) on the holdings-only
value series are reported as well, but the user should be aware they are
unreliable for portfolios with high turnover: the daily series is
dominated by position-change noise (selling 80% of holdings looks like
an 80% drop, but you just took money out, you didn't lose it). The
absolute return and MWR/IRR are the headline numbers.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("midas_risk")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# Make sibling modules importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from midas_pdf_parser import parse_midas_directory, flatten_to_frames, deduplicate_trades


# ---------------------------------------------------------------------------
# 1. Price fetching (borsapy with parquet cache)
# ---------------------------------------------------------------------------

def _normalize_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Strip timezone info and normalize to midnight."""
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)
    return pd.to_datetime(idx).normalize()


def _load_cached(cache_dir: Path, symbol: str) -> pd.DataFrame | None:
    if not cache_dir:
        return None
    f = cache_dir / f"{symbol}.parquet"
    if f.exists():
        try:
            df = pd.read_parquet(f)
            df.index = _normalize_index(df.index)
            return df
        except Exception:
            return None
    return None


def _save_cached(cache_dir: Path, symbol: str, df: pd.DataFrame) -> None:
    if not cache_dir:
        return
    f = cache_dir / f"{symbol}.parquet"
    try:
        df.to_parquet(f)
    except Exception as exc:
        logger.warning("  cache write failed for %s: %s", symbol, exc)


def fetch_prices(
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Fetch daily OHLCV prices for each symbol via borsapy, with parquet cache."""
    import borsapy as bp

    cache_dir.mkdir(parents=True, exist_ok=True)
    price_cache: dict[str, pd.DataFrame] = {}
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    for symbol in sorted(set(s for s in symbols if s)):
        cached = _load_cached(cache_dir, symbol)
        if cached is not None and not cached.empty:
            price_cache[symbol] = cached
            logger.info("  cache hit  %s (%d rows)", symbol, len(cached))
            continue
        try:
            t = bp.Ticker(symbol)
            hist = t.history(start=start_str, end=end_str)
        except Exception as exc:
            logger.warning("  fetch failed for %s: %s", symbol, exc)
            continue
        if hist is None or hist.empty:
            logger.warning("  no history for %s", symbol)
            continue
        hist = hist.copy()
        hist.index = _normalize_index(hist.index)
        price_cache[symbol] = hist
        _save_cached(cache_dir, symbol, hist)
        logger.info("  fetched    %s (%d rows)", symbol, len(hist))

    return price_cache


def fetch_benchmark(
    symbol: str = "XU100",
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.Series:
    """Fetch benchmark price series (default: XU100 / BIST 100)."""
    import borsapy as bp

    t = bp.Ticker(symbol)
    start_arg = (start or pd.Timestamp.today() - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    end_arg = (end or pd.Timestamp.today()).strftime("%Y-%m-%d")
    hist = t.history(start=start_arg, end=end_arg)
    if hist is None or hist.empty:
        return pd.Series(dtype=float)
    hist = hist.copy()
    hist.index = _normalize_index(hist.index)
    return hist["Close"]


# ---------------------------------------------------------------------------
# 2. Equity curve with real cash flow
# ---------------------------------------------------------------------------

def build_equity_curve_with_cash(
    trades: pd.DataFrame,
    cash_flows: pd.DataFrame,
    price_cache: dict[str, pd.DataFrame],
    initial_cash: float = 0.0,
) -> tuple[pd.Series, dict]:
    """
    Replay trades day-by-day, mark-to-market holdings, and apply cash flows.

    Cash flow model:
      * DEPOSIT        -> +amount to cash
      * WITHDRAWAL     -> -amount from cash
      * NEMA           -> +amount to cash (interest on idle TL)
      * PROMOTION      -> +amount to cash (free-trade credits)
      * DIVIDEND       -> +amount to cash (per payment date)
      * DÖVIZ_*        -> currency conversion (excluded; net zero to portfolio)
      * DIĞER_GIDER    -> tax/withholding already in DIVIDEND; ignored
      * OTHER_INCOME   -> +amount to cash
    """
    if trades.empty and cash_flows.empty:
        return pd.Series(dtype=float), {}

    all_dates: set[pd.Timestamp] = set()
    for hist in price_cache.values():
        all_dates.update(hist.index)
    if not all_dates:
        return pd.Series(dtype=float), {}

    first_event_dates = []
    if not trades.empty:
        first_event_dates.append(trades["exec_date"].min())
    if not cash_flows.empty:
        first_event_dates.append(cash_flows["flow_date"].min())
    if not first_event_dates:
        return pd.Series(dtype=float), {}
    start = min(first_event_dates)
    end = max(all_dates)
    calendar = pd.date_range(start, end, freq="B")
    calendar = calendar[calendar <= end]

    cash = float(initial_cash)
    holdings: dict[str, float] = {}
    ledger: list[tuple[pd.Timestamp, float]] = []

    by_trade_date: dict[pd.Timestamp, list[dict]] = defaultdict(list)
    for _, tr in trades.iterrows():
        by_trade_date[pd.Timestamp(tr["exec_date"]).normalize()].append(tr.to_dict())

    by_flow_date: dict[pd.Timestamp, list[dict]] = defaultdict(list)
    for _, cf in cash_flows.iterrows():
        by_flow_date[pd.Timestamp(cf["flow_date"]).normalize()].append(cf.to_dict())

    for d in calendar:
        for cf in by_flow_date.get(d, []):
            ft = cf["flow_type"]
            amt = float(cf["amount"])
            if ft == "DEPOSIT":
                cash += amt
            elif ft == "WITHDRAWAL":
                cash -= amt
            elif ft in ("NEMA", "PROMOTION", "OTHER_INCOME", "DIVIDEND"):
                cash += amt
            elif ft in ("DÖVIZ_ALIŞ", "DÖVIZ_SATIŞ"):
                pass
            elif ft == "DIĞER_GIDER":
                pass

        for tr in by_trade_date.get(d, []):
            sym = tr["symbol"]
            qty = float(tr["quantity"])
            price = float(tr["price"])
            fee = float(tr.get("fee", 0.0))
            if tr["side"] == "BUY":
                cost = qty * price + fee
                if cost > cash + 1e-6:
                    affordable = max(0.0, (cash - fee) / price) if price > 0 else 0.0
                    if affordable <= 0:
                        continue
                    qty = affordable
                    cost = qty * price + fee
                cash -= cost
                holdings[sym] = holdings.get(sym, 0.0) + qty
            else:
                held = holdings.get(sym, 0.0)
                if held <= 0:
                    continue
                sell_qty = min(qty, held)
                proceeds = sell_qty * price - fee
                cash += proceeds
                holdings[sym] = held - sell_qty
                if holdings[sym] <= 1e-9:
                    del holdings[sym]

        d_naive = pd.Timestamp(d).tz_localize(None) if getattr(d, "tz", None) is not None else d
        equity = cash
        for sym, qty in holdings.items():
            hist = price_cache.get(sym)
            if hist is None or hist.empty:
                continue
            valid = hist.index[hist.index <= d_naive]
            if len(valid) == 0:
                continue
            last_close = float(hist.loc[valid.max(), "Close"])
            equity += qty * last_close
        ledger.append((d, equity))

    equity_curve = pd.Series(
        [v for _, v in ledger],
        index=pd.DatetimeIndex([d for d, _ in ledger]),
        name="equity",
    )

    if not equity_curve.empty:
        meaningful = equity_curve[equity_curve > 100.0]
        if not meaningful.empty:
            first_meaningful = meaningful.index[0]
            equity_curve = equity_curve.loc[equity_curve.index >= first_meaningful]

    summary = {
        "final_cash": float(cash),
        "final_holdings_value": float(equity_curve.iloc[-1] - cash) if not equity_curve.empty else 0.0,
        "final_total": float(equity_curve.iloc[-1]) if not equity_curve.empty else 0.0,
        "n_trades": int(len(trades)),
        "n_cash_flows": int(len(cash_flows)),
    }
    return equity_curve, summary


# ---------------------------------------------------------------------------
# 3. Returns metrics
# ---------------------------------------------------------------------------

def compute_total_return_metrics(
    equity_curve: pd.Series,
    cash_flows: pd.DataFrame,
    dividends: pd.DataFrame,
) -> dict:
    """Compute total return and MWR (IRR)."""
    if equity_curve.empty:
        return {}

    out: dict[str, Any] = {}

    deposits = float(cash_flows.loc[cash_flows["flow_type"] == "DEPOSIT", "amount"].sum()) \
        if not cash_flows.empty else 0.0
    withdrawals = float(cash_flows.loc[cash_flows["flow_type"] == "WITHDRAWAL", "amount"].sum()) \
        if not cash_flows.empty else 0.0
    net_invested = deposits - withdrawals

    final_equity = float(equity_curve.iloc[-1])
    if net_invested > 0:
        abs_return = (final_equity - net_invested) / net_invested
    else:
        abs_return = 0.0
    out["deposits_try"] = deposits
    out["withdrawals_try"] = withdrawals
    out["net_invested_try"] = net_invested
    out["final_equity_try"] = final_equity
    out["absolute_return_pct"] = abs_return * 100

    if not dividends.empty:
        out["total_dividends_try"] = float(dividends["net"].sum())
    else:
        out["total_dividends_try"] = 0.0

    if not cash_flows.empty:
        nema = float(cash_flows.loc[cash_flows["flow_type"] == "NEMA", "amount"].sum())
        out["total_nema_try"] = nema
    else:
        out["total_nema_try"] = 0.0

    out["mwr_irr_pct"] = _compute_irr(equity_curve, cash_flows) * 100

    return out


def _compute_irr(equity_curve: pd.Series, cash_flows: pd.DataFrame) -> float:
    """Solve IRR: investor's cash flows vs final equity value.

    Convention (investor POV):
      * Deposit (cash in)   -> negative cash flow
      * Withdrawal (cash out) -> positive cash flow
      * Final equity        -> positive cash flow (today's value)
    """
    if equity_curve.empty:
        return 0.0
    cash_flow_list: list[tuple[pd.Timestamp, float]] = []
    if not cash_flows.empty:
        for _, cf in cash_flows.iterrows():
            ft = cf["flow_type"]
            amt = float(cf["amount"])
            if ft == "DEPOSIT":
                cash_flow_list.append((pd.Timestamp(cf["flow_date"]), -amt))
            elif ft == "WITHDRAWAL":
                cash_flow_list.append((pd.Timestamp(cf["flow_date"]), amt))
    final_date = equity_curve.index.max()
    final_eq = float(equity_curve.iloc[-1])
    cash_flow_list.append((final_date, final_eq))
    if not cash_flow_list:
        return 0.0
    cash_flow_list.sort(key=lambda x: x[0])
    t0 = cash_flow_list[0][0]
    days = np.array([(d - t0).days for d, _ in cash_flow_list], dtype=float)
    flows = np.array([v for _, v in cash_flow_list], dtype=float)

    try:
        rate = 0.10
        for _ in range(200):
            discount = (1 + rate) ** (days / 365.0)
            npv = (flows / discount).sum()
            d_npv = (-flows * days / 365.0 / (1 + rate) ** (days / 365.0 + 1)).sum()
            if abs(d_npv) < 1e-10:
                break
            new_rate = rate - npv / d_npv
            if abs(new_rate - rate) < 1e-8:
                rate = new_rate
                break
            rate = new_rate
        return rate
    except (ZeroDivisionError, OverflowError):
        return 0.0


# ---------------------------------------------------------------------------
# 4. Per-trade statistics (FIFO-matched)
# ---------------------------------------------------------------------------

def compute_per_trade_stats(
    trades: pd.DataFrame,
    price_cache: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """FIFO-match buys to sells per ticker, return one row per closed trade."""
    buys_by_ticker: dict[str, list[dict]] = defaultdict(list)
    trade_results: list[dict] = []

    sorted_trades = trades.sort_values("exec_date").reset_index(drop=True)
    for _, tr in sorted_trades.iterrows():
        sym = tr["symbol"]
        if sym is None or sym not in price_cache:
            continue
        qty = float(tr["quantity"])
        price = float(tr["price"])
        date = pd.Timestamp(tr["exec_date"])
        if tr["side"] == "BUY":
            buys_by_ticker[sym].append({"date": date, "qty": qty, "price": price})
        else:
            remaining = qty
            while remaining > 1e-9 and buys_by_ticker[sym]:
                buy = buys_by_ticker[sym][0]
                matched = min(remaining, buy["qty"])
                pnl = matched * (price - buy["price"])
                hold_days = (date - buy["date"]).days
                pct_return = (price / buy["price"]) - 1
                trade_results.append({
                    "ticker": sym,
                    "buy_date": buy["date"],
                    "sell_date": date,
                    "qty": matched,
                    "buy_price": buy["price"],
                    "sell_price": price,
                    "pnl_try": pnl,
                    "pct_return": pct_return,
                    "hold_days": hold_days,
                })
                buy["qty"] -= matched
                remaining -= matched
                if buy["qty"] <= 1e-9:
                    buys_by_ticker[sym].pop(0)

    return pd.DataFrame(trade_results)


def summarize_per_trade(trade_results: pd.DataFrame) -> dict:
    if trade_results.empty:
        return {}
    winners = trade_results[trade_results["pnl_try"] > 0]
    losers = trade_results[trade_results["pnl_try"] <= 0]
    gross_profit = winners["pnl_try"].sum() if not winners.empty else 0.0
    gross_loss = abs(losers["pnl_try"].sum()) if not losers.empty else 0.0
    return {
        "n_closed_trades": int(len(trade_results)),
        "win_rate_pct": float((trade_results["pnl_try"] > 0).mean() * 100),
        "avg_return_pct": float(trade_results["pct_return"].mean() * 100),
        "median_return_pct": float(trade_results["pct_return"].median() * 100),
        "std_return_pct": float(trade_results["pct_return"].std() * 100),
        "max_win_pct": float(trade_results["pct_return"].max() * 100),
        "max_loss_pct": float(trade_results["pct_return"].min() * 100),
        "avg_hold_days": float(trade_results["hold_days"].mean()),
        "total_realized_pnl_try": float(trade_results["pnl_try"].sum()),
        "n_winners": int(len(winners)),
        "n_losers": int(len(losers)),
        "avg_win_pct": float(winners["pct_return"].mean() * 100) if not winners.empty else 0.0,
        "avg_loss_pct": float(losers["pct_return"].mean() * 100) if not losers.empty else 0.0,
        "avg_winner_hold_days": float(winners["hold_days"].mean()) if not winners.empty else 0.0,
        "avg_loser_hold_days": float(losers["hold_days"].mean()) if not losers.empty else 0.0,
        "gross_profit_try": float(gross_profit),
        "gross_loss_try": float(gross_loss),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else float("inf"),
        "expectancy_per_trade_try": float(trade_results["pnl_try"].mean()),
    }


# ---------------------------------------------------------------------------
# 5. Risk metrics via bist-quant
# ---------------------------------------------------------------------------

def compute_holdings_risk_metrics(
    trades: pd.DataFrame,
    price_cache: dict[str, pd.DataFrame],
    benchmark_curve: pd.Series | None,
) -> dict:
    """Compute risk metrics based on the holdings-weighted stock return series.

    For each day, build a portfolio value = sum(qty * close) of all held positions.
    Daily returns are winsorized at +/- 50% to suppress noise from large
    position changes (which are the dominant source of return outliers in
    personal portfolios).

    Caveat: This is NOT a clean TWR — position changes (selling 80% of holdings)
    look like -80% daily returns even though no market loss occurred. The
    user should treat Sharpe/Sortino as approximations, not as the standard
    fund-style risk metrics.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from bist_quant.analytics import PortfolioAnalytics
    if trades.empty or not price_cache:
        return {}

    holdings: dict[str, float] = {}
    sorted_trades = trades.sort_values("exec_date")
    all_dates = sorted(set().union(*[set(h.index) for h in price_cache.values()]))
    first_date = sorted_trades["exec_date"].min()
    last_date = max(sorted_trades["exec_date"].max(), all_dates[-1] if all_dates else first_date)
    calendar = pd.date_range(first_date, last_date, freq="B")
    calendar = calendar[calendar <= last_date]

    trade_iter = sorted_trades.to_dict("records")
    trade_idx = 0
    holdings_value_per_day: dict[pd.Timestamp, float] = {}

    for d in calendar:
        while trade_idx < len(trade_iter) and pd.Timestamp(trade_iter[trade_idx]["exec_date"]) <= d:
            tr = trade_iter[trade_idx]
            sym = tr["symbol"]
            qty = float(tr["quantity"])
            if tr["side"] == "BUY":
                holdings[sym] = holdings.get(sym, 0.0) + qty
            else:
                held = holdings.get(sym, 0.0)
                sell_qty = min(qty, held)
                holdings[sym] = held - sell_qty
                if holdings[sym] <= 1e-9:
                    del holdings[sym]
            trade_idx += 1
        d_naive = pd.Timestamp(d).tz_localize(None) if getattr(d, "tz", None) is not None else d
        total = 0.0
        for sym, qty in holdings.items():
            hist = price_cache.get(sym)
            if hist is None or hist.empty:
                continue
            valid = hist.index[hist.index <= d_naive]
            if len(valid) == 0:
                continue
            total += qty * float(hist.loc[valid.max(), "Close"])
        holdings_value_per_day[d] = total

    series = pd.Series(holdings_value_per_day).sort_index()
    series = series[series > 0]

    if series.empty or len(series) < 5:
        return {}

    returns = series.pct_change().dropna()
    returns = returns.clip(-0.5, 0.5)
    winsorized_curve = series.iloc[0] * (1 + returns).cumprod()

    analytics = PortfolioAnalytics.from_equity_curve(
        equity_curve=winsorized_curve,
        benchmark_curve=benchmark_curve,
        name="Midas Holdings",
    )
    return analytics.get_all_metrics()


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Midas portfolio risk report from PDF statements."
    )
    ap.add_argument(
        "--midas-dir", type=Path,
        default=Path("/home/safa/Documents/Transactions/Midas"),
    )
    ap.add_argument(
        "--cache-dir", type=Path,
        default=Path("/home/safa/Documents/Transactions/.price_cache"),
    )
    ap.add_argument(
        "--output", type=Path,
        default=Path("/home/safa/Documents/Transactions/risk_metrics_midas.json"),
    )
    args = ap.parse_args()

    if not args.midas_dir.exists():
        logger.error("Midas directory not found: %s", args.midas_dir)
        return 1

    logger.info("Parsing Midas PDFs from %s ...", args.midas_dir)
    midas_stmts = parse_midas_directory(args.midas_dir)
    midas_frames = flatten_to_frames(midas_stmts)
    midas_trades_raw = midas_frames["trades"]
    midas_trades = deduplicate_trades(midas_trades_raw)
    midas_cash = midas_frames["cash_flows"]
    midas_div = midas_frames["dividends"]
    midas_snap = midas_frames["portfolio_snapshots"]
    logger.info(
        "  Midas: %d trades (raw %d, after dedup), %d cash flows, %d dividends",
        len(midas_trades), len(midas_trades_raw),
        len(midas_cash), len(midas_div),
    )

    wd = pd.to_datetime(midas_trades["exec_date"]).dt.weekday
    weekend_mask = wd >= 5
    if weekend_mask.any():
        n_weekend = int(weekend_mask.sum())
        logger.info("  Shifting %d weekend trades to next business day", n_weekend)
        midas_trades.loc[weekend_mask, "exec_date"] = (
            pd.to_datetime(midas_trades.loc[weekend_mask, "exec_date"])
            + pd.tseries.offsets.BDay(1)
        ).values

    div_cash = midas_div.copy() if not midas_div.empty else pd.DataFrame()
    if not div_cash.empty:
        div_cash = div_cash.rename(columns={"pay_date": "flow_date", "net": "amount"})
        div_cash["flow_type"] = "DIVIDEND"
        div_cash = div_cash[["flow_date", "flow_type", "amount", "source_file"]]
    cash_combined = pd.concat([midas_cash, div_cash], ignore_index=True) if not div_cash.empty else midas_cash.copy()
    if not cash_combined.empty:
        cash_combined["flow_date"] = pd.to_datetime(cash_combined["flow_date"]).dt.normalize()
        wd_cf = cash_combined["flow_date"].dt.weekday
        weekend_cf = wd_cf >= 5
        if weekend_cf.any():
            n_cf = int(weekend_cf.sum())
            logger.info("  Shifting %d weekend cash flows to next business day", n_cf)
            cash_combined.loc[weekend_cf, "flow_date"] = (
                cash_combined.loc[weekend_cf, "flow_date"] + pd.tseries.offsets.BDay(1)
            ).values

    logger.info("Fetching prices from borsapy ...")
    all_symbols = sorted(set(midas_trades["symbol"].dropna().unique()))
    fetch_start = midas_trades["exec_date"].min() - pd.Timedelta(days=10)
    fetch_end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    price_cache = fetch_prices(all_symbols, fetch_start, fetch_end, args.cache_dir)

    logger.info("Building equity curve with cash flows ...")
    equity_curve, summary = build_equity_curve_with_cash(
        trades=midas_trades,
        cash_flows=cash_combined,
        price_cache=price_cache,
        initial_cash=0.0,
    )
    logger.info("  Final equity: %.2f TRY (cash=%.2f, holdings=%.2f)",
                summary.get("final_total", 0),
                summary.get("final_cash", 0),
                summary.get("final_holdings_value", 0))

    logger.info("Computing per-trade statistics (FIFO) ...")
    trade_results = compute_per_trade_stats(midas_trades, price_cache)
    trade_stats = summarize_per_trade(trade_results)

    logger.info("Computing total return / MWR (IRR) ...")
    return_metrics = compute_total_return_metrics(
        equity_curve=equity_curve,
        cash_flows=cash_combined,
        dividends=midas_div,
    )

    logger.info("Computing risk metrics on holdings-only series ...")
    bench_close = fetch_benchmark("XU100")
    if not bench_close.empty and not equity_curve.empty:
        bench_close = bench_close.loc[bench_close.index <= equity_curve.index.max()]
    risk_metrics = compute_holdings_risk_metrics(
        trades=midas_trades,
        price_cache=price_cache,
        benchmark_curve=bench_close,
    )

    result = {
        "summary": summary,
        "returns": return_metrics,
        "trade_stats": trade_stats,
        "risk": risk_metrics,
        "data_sources": {
            "midas_pdfs": [s.source_file for s in midas_stmts],
            "midas_trades_count": int(len(midas_trades)),
            "midas_cash_flows_count": int(len(midas_cash)),
            "midas_dividends_count": int(len(midas_div)),
            "midas_snapshots_count": int(len(midas_snap)),
        },
    }

    args.output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote metrics -> %s", args.output)

    print("\n" + "=" * 64)
    print("Midas Portfolio — Risk Report")
    print("=" * 64)
    if return_metrics:
        rm = return_metrics
        print(f"Period: {equity_curve.index.min().date()} -> {equity_curve.index.max().date()}")
        print(f"Days:   {(equity_curve.index.max() - equity_curve.index.min()).days}")
        print()
        print("Capital flows:")
        print(f"  Deposits:         {rm['deposits_try']:>14,.2f} TRY")
        print(f"  Withdrawals:      {rm['withdrawals_try']:>14,.2f} TRY")
        print(f"  Net invested:     {rm['net_invested_try']:>14,.2f} TRY")
        print(f"  Dividends:        {rm['total_dividends_try']:>14,.2f} TRY")
        print(f"  Nema (idle int.): {rm['total_nema_try']:>14,.2f} TRY")
        print()
        print("Performance:")
        print(f"  Final equity:        {rm['final_equity_try']:>14,.2f} TRY")
        print(f"  Absolute return:     {rm['absolute_return_pct']:>14.2f} %")
        print(f"  MWR / IRR (annualized): {rm['mwr_irr_pct']:>10.2f} %")
        print()
    if trade_stats:
        ts = trade_stats
        print(f"Per-trade stats ({ts['n_closed_trades']} closed trades, FIFO-matched):")
        print(f"  Win rate:         {ts['win_rate_pct']:>6.1f} %  ({ts['n_winners']} winners / {ts['n_losers']} losers)")
        print(f"  Avg return:       {ts['avg_return_pct']:>+6.2f} %  (median {ts['median_return_pct']:>+6.2f} %, std {ts['std_return_pct']:.2f} %)")
        print(f"  Max win / loss:   {ts['max_win_pct']:>+6.2f} % / {ts['max_loss_pct']:>+6.2f} %")
        print(f"  Avg hold:         {ts['avg_hold_days']:>6.1f} days  (winners {ts['avg_winner_hold_days']:.1f}d, losers {ts['avg_loser_hold_days']:.1f}d)")
        print(f"  Profit factor:    {ts['profit_factor']:>6.2f}")
        print(f"  Expectancy/trade: {ts['expectancy_per_trade_try']:>+10.2f} TRY")
        print(f"  Total realized:   {ts['total_realized_pnl_try']:>+14,.2f} TRY")
        print()
    if risk_metrics:
        m = risk_metrics
        print("Risk (holdings-only, daily, winsorized at +/-50%):")
        print(f"  Volatility:     {m['volatility']*100:>10.2f} %")
        print(f"  Max drawdown:   {m['max_drawdown']*100:>10.2f} %")
        print(f"  Sharpe:         {m['sharpe_ratio']:>10.2f}")
        print(f"  Sortino:        {m['sortino_ratio']:>10.2f}")
        print(f"  Calmar:         {m['calmar_ratio']:>10.2f}")
        print(f"  VaR (95%):      {m['var_95']*100:>10.2f} %")
        print(f"  CVaR (95%):     {m['cvar_95']*100:>10.2f} %")
        if "beta" in m:
            print(f"  Beta vs XU100:  {m['beta']:>10.2f}")
        if "alpha" in m:
            print(f"  Alpha vs XU100: {m['alpha']*100:>10.2f} %")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
