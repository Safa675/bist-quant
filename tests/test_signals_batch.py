"""Batch signal test runner - tests all registered signals with proper runtime context."""
import sys, os, time, warnings, traceback, logging

os.environ["BIST_DATA_SOURCE"] = "local"
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(level=logging.WARNING)

def log(msg):
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()

t0 = time.time()
log("=== SIGNAL MODULE TEST REPORT ===")

log("1. Importing...")
from bist_quant.common.data_loader import DataLoader
import pandas as pd
import numpy as np
log(f"   Imports done in {time.time()-t0:.1f}s")

log("2. Loading data...")
t1 = time.time()
loader = DataLoader(data_source_priority="local")

# Load prices (long format)
prices = loader.load_prices()
log(f"   Prices: {prices.shape}")

# Build close_df (wide: Date x Ticker)
if "Date" in prices.columns:
    prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce")
    idx_col = "Date"
elif not isinstance(prices.index, pd.DatetimeIndex):
    prices.index = pd.to_datetime(prices.index, errors="coerce")
    idx_col = None
else:
    idx_col = None

if "Ticker" in prices.columns:
    close_df = prices.pivot_table(
        index=idx_col or prices.index,
        columns="Ticker", values="Close"
    ).sort_index()
    
    volume_df = prices.pivot_table(
        index=idx_col or prices.index,
        columns="Ticker", values="Volume"
    ).sort_index() if "Volume" in prices.columns else pd.DataFrame(index=close_df.index)
else:
    close_df = prices.copy()
    volume_df = pd.DataFrame(index=close_df.index)

close_df.index = pd.to_datetime(close_df.index, errors="coerce")
close_df.columns = [str(c).split(".")[0].upper() for c in close_df.columns]
volume_df.index = close_df.index
if not volume_df.empty:
    volume_df.columns = [str(c).split(".")[0].upper() for c in volume_df.columns]

dates = close_df.index
log(f"   Close DF: {close_df.shape}, dates: {dates.min()} to {dates.max()}")

# Load fundamentals
try:
    fundamentals = loader.load_fundamentals()
    log(f"   Fundamentals: {len(fundamentals)} tickers")
except Exception as e:
    fundamentals = {}
    log(f"   Fundamentals: FAILED ({e})")

# Load XU100 benchmark
try:
    xu100 = loader.load_xu100_prices()
    log(f"   XU100: {len(xu100)} obs")
except Exception as e:
    xu100 = pd.Series(dtype=float)
    log(f"   XU100: FAILED ({e})")

log(f"   Data loaded in {time.time()-t1:.1f}s")

# Build runtime context
runtime_context = {
    "close_df": close_df,
    "volume_df": volume_df,
    "prices": prices,  # long format for high/low panels
    "fundamentals": fundamentals,
    "fundamentals_parquet": loader.load_fundamentals_parquet() if fundamentals else None,
    "xu100_prices": xu100,
    "dates": dates,
}

config = {
    "_runtime_context": runtime_context,
}

log("\n3. Importing factory...")
t2 = time.time()
from bist_quant.signals.factory import get_available_signals, build_signal
signals = get_available_signals()
log(f"   Factory: {len(signals)} signals in {time.time()-t2:.1f}s")

log(f"\n4. Running all {len(signals)} signals...\n")
results = {}

for name in signals:
    t_start = time.time()
    try:
        result = build_signal(name=name, dates=dates, loader=loader, config=config)
        elapsed = time.time() - t_start
        n_rows, n_cols = result.shape
        nn = int(result.count().sum())
        total = n_rows * max(n_cols, 1)
        pct = (nn / total) * 100 if total > 0 else 0
        status = "OK" if pct > 1 else "EMPTY"
        icon = "PASS" if status == "OK" else "WARN"
        results[name] = {"status": status, "shape": f"{n_rows}x{n_cols}", "pct": round(pct, 1), "time": round(elapsed, 1)}
        log(f"[{icon}] {name:40s} {n_rows:>5}x{n_cols:<5} cov={pct:5.1f}%  {elapsed:5.1f}s")
    except Exception as e:
        elapsed = time.time() - t_start
        err = str(e)[:120]
        results[name] = {"status": "ERROR", "error": err, "time": round(elapsed, 1)}
        log(f"[FAIL] {name:40s} ERROR ({elapsed:.1f}s): {err}")

log("\n" + "=" * 75)
ok = sum(1 for r in results.values() if r["status"] == "OK")
empty = sum(1 for r in results.values() if r["status"] == "EMPTY")
errs = sum(1 for r in results.values() if r["status"] == "ERROR")
log(f"SUMMARY: {ok} OK  |  {empty} EMPTY  |  {errs} ERROR  |  Total: {len(results)}")
log(f"Total time: {time.time()-t0:.1f}s")
log("=" * 75)

if errs > 0:
    log("\nERROR DETAILS:")
    for name, r in sorted(results.items()):
        if r["status"] == "ERROR":
            log(f"  {name}: {r['error']}")

if empty > 0:
    log("\nEMPTY SIGNALS (0% or near-0% coverage):")
    for name, r in sorted(results.items()):
        if r["status"] == "EMPTY":
            log(f"  {name}: {r['shape']} coverage={r['pct']}%")
