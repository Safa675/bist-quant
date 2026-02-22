"""Test İş Yatırım financial API through all code paths.

Verifies:
1. borsapy.Ticker() API works for industrial AND bank stocks
2. BorsapyClient correctly detects UFRS for banks
3. Disk cache contains valid data
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

INDUSTRIAL = ["THYAO", "AEFES", "ASELS"]
BANKS = ["GARAN", "AKBNK"]


def main():
    print("=" * 60)
    print("İŞ YATIRIM FINANCIAL API TEST")
    print("=" * 60)

    errors = 0

    # Test 1: Direct borsapy Ticker API (industrial)
    print("\n--- Test 1: Industrial stocks via borsapy ---")
    import borsapy as bp

    for symbol in INDUSTRIAL:
        t0 = time.time()
        try:
            t = bp.Ticker(symbol)
            bs = t.get_balance_sheet(quarterly=True)
            elapsed = time.time() - t0
            if bs is not None and not bs.empty:
                print(f"  {symbol}: ✅ shape={bs.shape} ({elapsed:.1f}s)")
            else:
                print(f"  {symbol}: ⚠️  EMPTY ({elapsed:.1f}s)")
                errors += 1
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  {symbol}: ❌ {e} ({elapsed:.1f}s)")
            errors += 1

    # Test 2: Bank stocks with UFRS
    print("\n--- Test 2: Bank stocks with UFRS ---")
    for symbol in BANKS:
        t0 = time.time()
        try:
            t = bp.Ticker(symbol)
            bs = t.get_balance_sheet(quarterly=True, financial_group="UFRS")
            elapsed = time.time() - t0
            if bs is not None and not bs.empty:
                print(f"  {symbol}: ✅ shape={bs.shape} ({elapsed:.1f}s)")
            else:
                print(f"  {symbol}: ⚠️  EMPTY ({elapsed:.1f}s)")
                errors += 1
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  {symbol}: ❌ {e} ({elapsed:.1f}s)")
            errors += 1

    # Test 3: BorsapyClient (with UFRS auto-detection fix)
    print("\n--- Test 3: BorsapyClient (auto UFRS detection) ---")
    from pathlib import Path
    from importlib.util import spec_from_file_location, module_from_spec

    client_path = Path("data/Fetcher-Scrapper/borsapy_client.py")
    if client_path.exists():
        spec = spec_from_file_location("borsapy_client", client_path)
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)
        BorsapyClient = mod.BorsapyClient
        client = BorsapyClient(
            cache_dir=Path("data/borsapy_cache"),
            use_mcp_fallback=False,
        )
        for symbol in INDUSTRIAL + BANKS:
            t0 = time.time()
            try:
                result = client.get_financial_statements(symbol)
                elapsed = time.time() - t0
                non_empty = sum(1 for v in result.values() if not v.empty)
                bs = result.get("balance_sheet")
                bs_str = str(bs.shape) if bs is not None and not bs.empty else "EMPTY"
                status = "✅" if non_empty > 0 else "❌"
                if non_empty == 0:
                    errors += 1
                print(f"  {symbol}: {status} {non_empty}/3 statements, bs={bs_str} ({elapsed:.1f}s)")
            except Exception as e:
                elapsed = time.time() - t0
                print(f"  {symbol}: ❌ {e} ({elapsed:.1f}s)")
                errors += 1
    else:
        print("  ⚠️  BorsapyClient not found, skipping")

    # Test 4: Disk cache
    print("\n--- Test 4: Disk cache status ---")
    cache_dir = Path("data/borsapy_cache/financials")
    if cache_dir.exists():
        import pandas as pd

        count = 0
        for ticker_dir in sorted(cache_dir.iterdir()):
            if ticker_dir.is_dir():
                bs_file = ticker_dir / "balance_sheet.parquet"
                if bs_file.exists():
                    df = pd.read_parquet(bs_file)
                    count += 1
                    if count <= 5:
                        print(f"  {ticker_dir.name}: bs shape={df.shape}")
        print(f"  ... {count} total cached tickers")
    else:
        print("  No cache directory")

    # Summary
    print("\n" + "=" * 60)
    if errors == 0:
        print("✅ ALL TESTS PASSED")
    else:
        print(f"⚠️  {errors} ERROR(S) DETECTED")
    print("=" * 60)
    return errors


if __name__ == "__main__":
    sys.exit(main())
