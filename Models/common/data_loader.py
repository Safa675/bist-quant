"""
Common Data Loader - Centralized data loading to eliminate redundant I/O

This module loads all fundamental data, price data, and regime predictions ONCE
and caches them in memory for use by all factor models.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import sys
import warnings
warnings.filterwarnings('ignore')

# Add regime filter to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REGIME_FILTER_DIR = PROJECT_ROOT / "Regime Filter"
if str(REGIME_FILTER_DIR) not in sys.path:
    sys.path.insert(0, str(REGIME_FILTER_DIR))

from models.ensemble_regime import EnsembleRegimeModel


class DataLoader:
    """Centralized data loader with caching"""
    
    def __init__(self, data_dir: Path, regime_model_dir: Path):
        self.data_dir = Path(data_dir)
        self.regime_model_dir = Path(regime_model_dir)
        self.fundamental_dir = self.data_dir / "fundamental_data"
        self.isyatirim_dir = self.data_dir / "price" / "isyatirim_prices"
        
        # Cache
        self._fundamentals = None
        self._prices = None
        self._close_df = None
        self._open_df = None
        self._volume_df = None
        self._regime_series = None
        self._xautry_prices = None
        self._xu100_prices = None
        self._fundamentals_parquet = None
        self._isyatirim_parquet = None
        self._shares_consolidated = None
        
    def load_prices(self, prices_file: Path) -> pd.DataFrame:
        """Load stock prices"""
        if self._prices is None:
            print("\n📊 Loading price data...")
            parquet_file = prices_file.with_suffix(".parquet")
            if parquet_file.exists():
                print(f"  📦 Using Parquet: {parquet_file.name}")
                self._prices = pd.read_parquet(
                    parquet_file,
                    columns=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"],
                )
            else:
                print(f"  📄 Using CSV: {prices_file.name}")
                self._prices = pd.read_csv(
                    prices_file,
                    usecols=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"],
                )
            if "Date" in self._prices.columns:
                self._prices["Date"] = pd.to_datetime(self._prices["Date"], errors="coerce")
            print(f"  ✅ Loaded {len(self._prices)} price records")
        return self._prices
    
    def build_close_panel(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Build close price panel (Date x Ticker)"""
        if self._close_df is None:
            print("  Building close price panel...")
            close_df = prices.pivot_table(index='Date', columns='Ticker', values='Close').sort_index()
            close_df.columns = [c.split('.')[0].upper() for c in close_df.columns]
            self._close_df = close_df
            print(f"  ✅ Close panel: {close_df.shape[0]} days × {close_df.shape[1]} tickers")
        return self._close_df
    
    def build_open_panel(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Build open price panel (Date x Ticker)"""
        if self._open_df is None:
            print("  Building open price panel...")
            open_df = prices.pivot_table(index='Date', columns='Ticker', values='Open').sort_index()
            open_df.columns = [c.split('.')[0].upper() for c in open_df.columns]
            self._open_df = open_df
            print(f"  ✅ Open panel: {open_df.shape[0]} days × {open_df.shape[1]} tickers")
        return self._open_df
    
    def build_volume_panel(self, prices: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
        """Build rolling median volume panel"""
        if self._volume_df is None:
            print(f"  Building volume panel (lookback={lookback})...")
            vol_pivot = prices.pivot_table(index="Date", columns="Ticker", values="Volume").sort_index()
            vol_pivot.columns = [c.split('.')[0].upper() for c in vol_pivot.columns]
            
            # Drop holiday rows
            valid_pct = vol_pivot.notna().mean(axis=1)
            holiday_mask = valid_pct < 0.5
            if holiday_mask.any():
                vol_clean = vol_pivot.loc[~holiday_mask]
            else:
                vol_clean = vol_pivot
            
            median_adv = vol_clean.rolling(lookback, min_periods=lookback).median()
            median_adv = median_adv.reindex(vol_pivot.index).ffill()
            self._volume_df = median_adv
            print(f"  ✅ Volume panel: {median_adv.shape[0]} days × {median_adv.shape[1]} tickers")
        return self._volume_df
    
    def load_fundamentals(self) -> Dict:
        """Load all fundamental data from Excel files"""
        if self._fundamentals is None:
            print("\n📈 Loading fundamental data...")
            fundamentals = {}
            parquet_file = self.data_dir / "fundamental_data_consolidated.parquet"
            if parquet_file.exists():
                print("  📦 Loading consolidated fundamentals (Parquet)...")
                self._fundamentals_parquet = pd.read_parquet(parquet_file)
                tickers = (
                    self._fundamentals_parquet.index.get_level_values("ticker")
                    .unique()
                    .tolist()
                )
                for ticker in tickers:
                    fundamentals[ticker] = {'path': None}
                print(f"  ✅ Loaded consolidated fundamentals for {len(tickers)} tickers")
            else:
                count = 0
                for file_path in self.fundamental_dir.rglob("*.xlsx"):
                    ticker = file_path.stem.split('.')[0].upper()
                    try:
                        fundamentals[ticker] = {
                            'path': file_path,
                            'income': None,  # Lazy load
                            'balance': None,
                            'cashflow': None,
                        }
                        count += 1
                        if count % 100 == 0:
                            print(f"  Indexed {count} tickers...")
                    except Exception:
                        continue
                print(f"  ✅ Indexed {count} fundamental data files")
            
            self._fundamentals = fundamentals
        return self._fundamentals

    def load_fundamentals_parquet(self) -> pd.DataFrame | None:
        """Load consolidated fundamentals parquet if available"""
        if self._fundamentals_parquet is None:
            parquet_file = self.data_dir / "fundamental_data_consolidated.parquet"
            if parquet_file.exists():
                print("  📦 Loading consolidated fundamentals (Parquet)...")
                self._fundamentals_parquet = pd.read_parquet(parquet_file)
        return self._fundamentals_parquet

    def load_shares_outstanding_panel(self) -> pd.DataFrame:
        """Load Date x Ticker shares outstanding panel."""
        if self._shares_consolidated is None:
            shares_file = self.data_dir / "shares_outstanding_consolidated.csv"
            if shares_file.exists():
                print("  📊 Loading consolidated shares file...")
                panel = pd.read_csv(shares_file, index_col=0, parse_dates=True)
                panel.index = pd.to_datetime(panel.index, errors="coerce")
                panel = panel.sort_index()
                panel.columns = [str(c).upper() for c in panel.columns]
                self._shares_consolidated = panel
                print(f"  ✅ Loaded shares for {panel.shape[1]} tickers")
            else:
                # Fallback: build panel from consolidated isyatirim parquet
                isy = self._load_isyatirim_parquet()
                if isy is not None and not isy.empty:
                    try:
                        daily = isy[isy["sheet_type"] == "daily"]
                        required_cols = {"ticker", "HGDG_TARIH", "SERMAYE"}
                        if not daily.empty and required_cols.issubset(daily.columns):
                            panel = daily.pivot_table(
                                index="HGDG_TARIH",
                                columns="ticker",
                                values="SERMAYE",
                                aggfunc="last",
                            )
                            panel.index = pd.to_datetime(panel.index, errors="coerce")
                            panel = panel.sort_index()
                            panel.columns = [str(c).upper() for c in panel.columns]
                            self._shares_consolidated = panel
                            print(f"  ✅ Built shares panel from isyatirim parquet for {panel.shape[1]} tickers")
                        else:
                            self._shares_consolidated = None
                    except Exception as exc:
                        print(f"  ⚠️  Failed to build shares panel from isyatirim parquet: {exc}")
                        self._shares_consolidated = None
                else:
                    print("  ⚠️  Consolidated shares file not found")
                    self._shares_consolidated = None

        if self._shares_consolidated is None:
            return pd.DataFrame()
        return self._shares_consolidated
    
    def load_shares_outstanding(self, ticker: str) -> pd.Series:
        """Load shares outstanding from consolidated file (fast!)"""
        shares_panel = self.load_shares_outstanding_panel()
        if not shares_panel.empty and ticker in shares_panel.columns:
            return shares_panel[ticker].dropna()

        # Fallback: consolidated isyatirim parquet (if available)
        isy = self._load_isyatirim_parquet()
        if isy is not None:
            try:
                daily = isy[(isy["ticker"] == ticker) & (isy["sheet_type"] == "daily")]
                if not daily.empty and "HGDG_TARIH" in daily.columns and "SERMAYE" in daily.columns:
                    series = daily.set_index("HGDG_TARIH")["SERMAYE"].dropna()
                    return series
            except Exception:
                pass
        
        # Fallback to individual Excel file (slow)
        excel_path = self.isyatirim_dir / f"{ticker}_2016_2026_daily_and_quarterly.xlsx"
        
        if not excel_path.exists():
            return pd.Series(dtype=float)
        
        try:
            df = pd.read_excel(excel_path, sheet_name='daily')
            if 'HGDG_TARIH' not in df.columns or 'SERMAYE' not in df.columns:
                return pd.Series(dtype=float)
            
            df['HGDG_TARIH'] = pd.to_datetime(df['HGDG_TARIH'])
            df = df.set_index('HGDG_TARIH').sort_index()
            return df['SERMAYE'].dropna()
        except Exception:
            return pd.Series(dtype=float)

    def _load_isyatirim_parquet(self) -> pd.DataFrame | None:
        """Load consolidated isyatirim prices parquet (used for shares fallback)"""
        if self._isyatirim_parquet is None:
            parquet_file = self.data_dir / "isyatirim_prices_consolidated.parquet"
            if parquet_file.exists():
                print("  📦 Loading consolidated isyatirim prices (Parquet)...")
                self._isyatirim_parquet = pd.read_parquet(
                    parquet_file,
                    columns=["ticker", "sheet_type", "HGDG_TARIH", "SERMAYE"],
                )
        return self._isyatirim_parquet
    
    def load_regime_predictions(self, features: pd.DataFrame) -> pd.Series:
        """Load regime model and generate predictions"""
        if self._regime_series is None:
            print("\n🎯 Loading regime model...")
            ensemble = EnsembleRegimeModel.load(self.regime_model_dir)

            print("  Generating regime predictions...")
            results = ensemble.predict(features, return_details=False)
            self._regime_series = results['ensemble_prediction']

            print(f"  ✅ Generated {len(self._regime_series)} regime predictions")
            print("\n  Regime distribution:")
            for regime, count in self._regime_series.value_counts().items():
                pct = count / len(self._regime_series) * 100
                print(f"    {regime}: {count} days ({pct:.1f}%)")

        return self._regime_series
    
    def load_xautry_prices(
        self,
        csv_path: Path,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.Series:
        """Load XAU/TRY prices"""
        if self._xautry_prices is None:
            print("\n💰 Loading XAU/TRY prices...")
            df = pd.read_csv(csv_path, parse_dates=["Date"])
            if "XAU_TRY" not in df.columns:
                raise ValueError("XAU_TRY column not found in CSV.")
            df = df.sort_values("Date")
            series = df.set_index("Date")["XAU_TRY"].astype(float)
            series.name = "XAU_TRY"
            self._xautry_prices = series
            print(f"  ✅ Loaded {len(series)} XAU/TRY observations")

        series = self._xautry_prices
        if start_date is not None:
            series = series.loc[series.index >= start_date]
        if end_date is not None:
            series = series.loc[series.index <= end_date]
        return series
    
    def load_xu100_prices(self, csv_path: Path) -> pd.Series:
        """Load XU100 benchmark prices"""
        if self._xu100_prices is None:
            print("\n📊 Loading XU100 benchmark...")
            df = pd.read_csv(csv_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()
            self._xu100_prices = df['Open'] if 'Open' in df.columns else df.iloc[:, 0]
            print(f"  ✅ Loaded {len(self._xu100_prices)} XU100 observations")
        return self._xu100_prices
    
    def load_usdtry(self) -> pd.DataFrame:
        """Load USD/TRY exchange rate data"""
        print("\n💱 Loading USD/TRY data...")
        usdtry_file = self.data_dir / "usdtry_data.csv"
        
        if not usdtry_file.exists():
            print(f"  ⚠️  USD/TRY file not found: {usdtry_file}")
            return pd.DataFrame()
        
        df = pd.read_csv(usdtry_file, parse_dates=['Date'])
        df = df.set_index('Date').sort_index()
        
        # Rename column to 'Close' for consistency
        if 'USDTRY' in df.columns:
            df = df.rename(columns={'USDTRY': 'Close'})
        
        print(f"  ✅ Loaded {len(df)} USD/TRY observations")
        return df
    
    def load_fundamental_metrics(self) -> pd.DataFrame:
        """Load pre-calculated fundamental metrics"""
        print("\n📊 Loading fundamental metrics...")
        metrics_file = self.data_dir / "fundamental_metrics.parquet"
        
        if not metrics_file.exists():
            print(f"  ⚠️  Fundamental metrics file not found: {metrics_file}")
            print(f"  Run calculate_fundamental_metrics.py to generate this file")
            return pd.DataFrame()
        
        df = pd.read_parquet(metrics_file)
        print(f"  ✅ Loaded {len(df)} metric observations")
        print(f"  Metrics: {df.columns.tolist()}")
        return df
