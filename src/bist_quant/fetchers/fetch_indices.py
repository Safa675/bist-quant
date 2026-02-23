#!/usr/bin/env python3
"""
Fetch BIST Index Historical Prices (XU030, XU100, XUTUM).
Downloads max available daily history from Yahoo Finance and caches it locally as CSV and Parquet.
Designed to reliably feed dashboard and regime engines.
"""

import argparse
import logging
from pathlib import Path
import sys

import pandas as pd
import yfinance as yf
from bist_quant.clients.borsapy_adapter import BorsapyAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("bist_quant.fetchers.indices")

# Target indices mapping (Yahoo Finance tickers)
INDICES = {
    "XU030": "XU030.IS",
    "XU100": "XU100.IS",
    "XUTUM": "XUTUM.IS"
}

def fetch_index_data(ticker_name: str, yf_symbol: str, out_dir: Path) -> None:
    """Fetch history for a single index and save to cache."""
    logger.info(f"Fetching {ticker_name} ({yf_symbol}) via yfinance...")
    
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pq = out_dir / f"{ticker_name}.parquet"
    out_csv = out_dir / f"{ticker_name}.csv"
    
    try:
        df = yf.download(yf_symbol, period="10y", interval="1d", progress=False)
        
        # Fallback to Borsapy if yfinance fails or returns very little data (like XUTUM)
        if df.empty or len(df) < 100:
            logger.warning(f"yfinance returned {len(df)} records for {ticker_name}. Attempting Borsapy fallback...")
            try:
                class MockLoader:
                    def __init__(self, data_path: Path):
                        self.data_dir = data_path
                
                # We need to give BorsapyAdapter an object that looks like DataLoader
                from bist_quant.settings import PROJECT_ROOT
                mock_loader = MockLoader(PROJECT_ROOT / "data")
                adapter = BorsapyAdapter(mock_loader)
                _ = adapter.client
                borsapy_df = adapter.client.get_history(ticker_name, period="5y", interval="1d")
                if borsapy_df is not None and not borsapy_df.empty:
                    df = borsapy_df
                    logger.info(f"Successfully retrieved {len(df)} records for {ticker_name} via Borsapy.")
            except Exception as e:
                logger.error(f"Borsapy fallback failed for {ticker_name}: {e}")
                
        if df.empty:
            logger.warning(f"No data returned for {ticker_name} from any source.")
            return
            
        # Flatten MultiIndex columns if present (yfinance > 0.2.x behavior)
        if hasattr(df.columns, "droplevel") and isinstance(df.columns, pd.MultiIndex):
            try:
                df.columns = df.columns.droplevel("Ticker")
            except Exception:
                pass
                
        df = df.reset_index()
        
        # Normalize Date column name
        if "Date" not in df.columns and "index" in df.columns:
            df = df.rename(columns={"index": "Date"})
            
        # Ensure Date is datetime just in case
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
            
        df["Ticker"] = ticker_name
        
        # Sort chronologically
        df = df.sort_values(by="Date")
        
        # Export
        df.to_parquet(out_pq)
        df.to_csv(out_csv, index=False)
        
        logger.info(f"✅ Saved {len(df)} records for {ticker_name} to {out_pq}")
        
    except Exception as exc:
        logger.error(f"❌ Failed to fetch {ticker_name}: {exc}")

def main():
    parser = argparse.ArgumentParser(description="Fetch BIST indices (XU030, XU100, XUTUM)")
    parser.add_argument(
        "--data-dir", 
        type=str, 
        help="Path to repo root data directory (defaults to auto-detect)"
    )
    args = parser.parse_args()
    
    # Auto-resolve correct paths
    if args.data_dir:
        base_dir = Path(args.data_dir)
    else:
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        base_dir = repo_root / "data"
        
    cache_dir = base_dir / "borsapy_cache" / "index_components"
    
    logger.info(f"Starting index price fetcher. Output directory: {cache_dir}")
    
    for name, symbol in INDICES.items():
        fetch_index_data(name, symbol, cache_dir)
        
    logger.info("All index fetches completed.")

if __name__ == "__main__":
    main()
