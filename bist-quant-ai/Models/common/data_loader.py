"""
Common Data Loader - Centralized data loading to eliminate redundant I/O

This module loads all fundamental data, price data, and regime predictions ONCE
and caches them in memory for use by all factor models.

Supports multiple data sources:
- Local parquet/CSV files (primary)
- Borsapy API (alternative/supplement)
"""

from pathlib import Path
import pandas as pd
from typing import Dict, Optional
import json
import warnings
import sys

warnings.filterwarnings('ignore')

# Add Fetcher-Scrapper to path for borsapy_client import
FETCHER_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "Fetcher-Scrapper"
if str(FETCHER_DIR) not in sys.path:
    sys.path.insert(0, str(FETCHER_DIR))

# Regime filter directory candidates (support both naming schemes)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REGIME_DIR_CANDIDATES = [
    PROJECT_ROOT / "Simple Regime Filter",
    PROJECT_ROOT / "Regime Filter",
]


class DataLoader:
    """Centralized data loader with caching and multi-source support"""

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
        self._regime_allocations = None
        self._xautry_prices = None
        self._xu100_prices = None
        self._fundamentals_parquet = None
        self._isyatirim_parquet = None
        self._shares_consolidated = None

        # Borsapy client (lazy-loaded)
        self._borsapy_client = None

    # -------------------------------------------------------------------------
    # Borsapy Integration
    # -------------------------------------------------------------------------

    @property
    def borsapy(self):
        """
        Lazy-load borsapy client.

        Returns:
            BorsapyClient instance or None if not available
        """
        if self._borsapy_client is None:
            try:
                from borsapy_client import BorsapyClient
                self._borsapy_client = BorsapyClient(
                    cache_dir=self.data_dir / "borsapy_cache"
                )
                print("  ✅ Borsapy client initialized")
            except ImportError as e:
                print(f"  ⚠️  Borsapy not available: {e}")
                print("     Install with: pip install borsapy")
                return None
        return self._borsapy_client

    def load_prices_borsapy(
        self,
        symbols: list[str] = None,
        period: str = "5y",
        index: str = "XU100",
    ) -> pd.DataFrame:
        """
        Load prices via borsapy API (alternative to local files).

        Args:
            symbols: List of symbols. If None, uses index components.
            period: Data period (e.g., "1y", "5y", "max")
            index: Index for default symbols (e.g., "XU100", "XU030")

        Returns:
            DataFrame in long format (Date, Ticker, OHLCV)
        """
        if self.borsapy is None:
            print("  ⚠️  Borsapy not available, cannot load prices")
            return pd.DataFrame()

        print(f"\n📊 Loading prices via borsapy (period={period})...")

        if symbols is None:
            symbols = self.borsapy.get_index_components(index)
            print(f"  Using {len(symbols)} symbols from {index}")

        result = self.borsapy.batch_download_to_long(
            symbols=symbols,
            period=period,
            group_by="ticker",
            add_is_suffix=False,
        )

        if result.empty:
            print("  ⚠️  No data returned from borsapy")
            return pd.DataFrame()

        loaded = result["Ticker"].dropna().nunique() if "Ticker" in result.columns else 0
        print(f"  ✅ Loaded {len(result)} price records for {loaded}/{len(symbols)} tickers")
        return result

    def get_index_components_borsapy(self, index: str = "XU100") -> list[str]:
        """
        Get index components via borsapy.

        Args:
            index: Index name (e.g., "XU100", "XU030", "XBANK")

        Returns:
            List of ticker symbols
        """
        if self.borsapy is None:
            return []
        return self.borsapy.get_index_components(index)

    def get_financials_borsapy(self, symbol: str) -> dict[str, pd.DataFrame]:
        """
        Get financial statements via borsapy.

        Args:
            symbol: Stock symbol

        Returns:
            Dict with balance_sheet, income_stmt, cashflow DataFrames
        """
        if self.borsapy is None:
            return {}
        return self.borsapy.get_financials(symbol)

    def get_dividends_borsapy(self, symbol: str) -> pd.DataFrame:
        """Get dividend history via borsapy."""
        if self.borsapy is None:
            return pd.DataFrame()
        return self.borsapy.get_dividends(symbol)

    def get_fast_info_borsapy(self, symbol: str) -> dict:
        """
        Get current quote via borsapy (15-min delayed).

        Args:
            symbol: Stock symbol

        Returns:
            Dict with current price, volume, market cap, etc.
        """
        if self.borsapy is None:
            return {}
        return self.borsapy.get_fast_info(symbol)

    def screen_stocks_borsapy(self, **filters) -> pd.DataFrame:
        """
        Screen stocks using borsapy screener.

        Example:
            loader.screen_stocks_borsapy(pe_max=10, roe_min=15, index="XU100")

        Returns:
            DataFrame with matching stocks
        """
        if self.borsapy is None:
            return pd.DataFrame()
        return self.borsapy.screen_stocks(**filters)

    def get_history_with_indicators_borsapy(
        self,
        symbol: str,
        indicators: list[str] = None,
        period: str = "2y",
    ) -> pd.DataFrame:
        """
        Get price history with technical indicators via borsapy.

        Args:
            symbol: Stock symbol
            indicators: List of indicators (e.g., ["rsi", "macd", "bb"])
            period: Data period

        Returns:
            DataFrame with OHLCV + indicator columns
        """
        if self.borsapy is None:
            return pd.DataFrame()
        return self.borsapy.get_history_with_indicators(
            symbol, indicators=indicators, period=period
        )

    # -------------------------------------------------------------------------
    # Portfolio Analytics Integration
    # -------------------------------------------------------------------------

    def create_portfolio_analytics(
        self,
        holdings: dict[str, float] = None,
        weights: dict[str, float] = None,
        returns: "pd.Series" = None,
        benchmark: str = "XU100",
        name: str = "Portfolio",
    ):
        """
        Create a PortfolioAnalytics instance.

        Args:
            holdings: Dict mapping symbol -> quantity (optional)
            weights: Dict mapping symbol -> weight (optional)
            returns: Pre-computed returns series (optional)
            benchmark: Benchmark symbol (default "XU100")
            name: Portfolio name

        Returns:
            PortfolioAnalytics instance

        Example:
            analytics = loader.create_portfolio_analytics(
                holdings={"THYAO": 100, "AKBNK": 200},
                benchmark="XU100"
            )
            print(analytics.summary())
        """
        try:
            from Models.analytics import PortfolioAnalytics
        except ImportError:
            # Try relative import
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from analytics import PortfolioAnalytics

        # If returns provided directly, use them
        if returns is not None:
            benchmark_returns = None
            if benchmark and self._close_df is not None and benchmark in self._close_df.columns:
                benchmark_returns = self._close_df[benchmark].pct_change().dropna()
            return PortfolioAnalytics(
                returns=returns,
                benchmark_returns=benchmark_returns,
                name=name,
            )

        # Otherwise, create from holdings/weights and price data
        if self._close_df is None:
            raise ValueError("Price data not loaded. Call load_prices() first.")

        # Get benchmark returns
        benchmark_returns = None
        if benchmark and benchmark in self._close_df.columns:
            benchmark_returns = self._close_df[benchmark].pct_change().dropna()

        # Create from holdings
        if holdings:
            return PortfolioAnalytics.from_holdings(
                holdings=holdings,
                close_df=self._close_df,
                benchmark_col=benchmark if benchmark in self._close_df.columns else None,
                weights=weights,
                name=name,
            )

        # Create from weights only (equal quantity assumed)
        if weights:
            holdings = {s: 1.0 for s in weights.keys()}
            return PortfolioAnalytics.from_holdings(
                holdings=holdings,
                close_df=self._close_df,
                benchmark_col=benchmark if benchmark in self._close_df.columns else None,
                weights=weights,
                name=name,
            )

        raise ValueError("Either holdings, weights, or returns must be provided")

    def analyze_strategy_performance(
        self,
        equity_curve: "pd.Series",
        benchmark_curve: "pd.Series" = None,
        name: str = "Strategy",
    ):
        """
        Analyze strategy performance from equity curve.

        Args:
            equity_curve: Cumulative strategy value series
            benchmark_curve: Cumulative benchmark value series (optional)
            name: Strategy name

        Returns:
            PortfolioAnalytics instance
        """
        try:
            from Models.analytics import PortfolioAnalytics
        except ImportError:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from analytics import PortfolioAnalytics

        return PortfolioAnalytics.from_equity_curve(
            equity_curve=equity_curve,
            benchmark_curve=benchmark_curve,
            name=name,
        )

    # -------------------------------------------------------------------------
    # Macro Events Integration
    # -------------------------------------------------------------------------

    @property
    def macro(self):
        """
        Lazy-load macro events client.

        Returns:
            MacroEventsClient instance or None if not available
        """
        if not hasattr(self, "_macro_client"):
            self._macro_client = None

        if self._macro_client is None:
            try:
                from macro_events import MacroEventsClient
                self._macro_client = MacroEventsClient()
                print("  ✅ Macro events client initialized")
            except ImportError as e:
                print(f"  ⚠️  Macro events not available: {e}")
                return None
        return self._macro_client

    def get_economic_calendar(
        self,
        days_ahead: int = 7,
        countries: list[str] = None,
    ) -> "pd.DataFrame":
        """
        Get economic calendar events.

        Args:
            days_ahead: Number of days to look ahead
            countries: Country codes (e.g., ["TR", "US"])

        Returns:
            DataFrame with economic events
        """
        if self.macro is None:
            return pd.DataFrame()
        return self.macro.get_economic_calendar(days_ahead=days_ahead, countries=countries)

    def get_inflation_data(self, periods: int = 24) -> "pd.DataFrame":
        """
        Get TCMB inflation data.

        Args:
            periods: Number of monthly periods

        Returns:
            DataFrame with inflation data
        """
        if self.macro is None:
            return pd.DataFrame()
        return self.macro.get_inflation_data(periods=periods)

    def get_bond_yields(self) -> dict:
        """
        Get Turkish government bond yields.

        Returns:
            Dict with 2y, 5y, 10y yields
        """
        if self.macro is None:
            return {}
        return self.macro.get_bond_yields()

    def get_stock_news(self, symbol: str, limit: int = 10) -> list[dict]:
        """
        Get KAP announcements/news for a stock.

        Args:
            symbol: Stock symbol
            limit: Maximum number of news items

        Returns:
            List of news items
        """
        if self.macro is None:
            return []
        return self.macro.get_stock_news(symbol, limit=limit)

    def get_macro_summary(self) -> dict:
        """
        Get comprehensive macro summary.

        Returns:
            Dict with inflation, yields, sentiment, events
        """
        if self.macro is None:
            return {}
        return self.macro.get_macro_summary()

    def load_prices(self, prices_file: Path) -> pd.DataFrame:
        """Load stock prices"""
        if self._prices is None:
            print("\n📊 Loading price data...")
            parquet_file = prices_file.with_suffix(".parquet")
            gz_file = prices_file.with_suffix(".csv.gz")
            if parquet_file.exists():
                print(f"  📦 Using Parquet: {parquet_file.name}")
                self._prices = pd.read_parquet(
                    parquet_file,
                    columns=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"],
                )
            elif gz_file.exists():
                print(f"  🗜️  Using GZip CSV: {gz_file.name}")
                self._prices = pd.read_csv(
                    gz_file,
                    compression="gzip",
                    usecols=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"],
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
            csv_gz_file = self.data_dir / "fundamental_data_consolidated.csv.gz"
            if parquet_file.exists():
                print("  📦 Loading consolidated fundamentals (Parquet)...")
                self._fundamentals_parquet = pd.read_parquet(parquet_file)
            elif csv_gz_file.exists():
                print("  🗜️  Loading consolidated fundamentals (CSV.GZ)...")
                self._fundamentals_parquet = pd.read_csv(
                    csv_gz_file,
                    compression="gzip",
                    index_col=[0, 1, 2],
                )
                self._fundamentals_parquet.index.names = ["ticker", "sheet_name", "row_name"]
                for col in self._fundamentals_parquet.columns:
                    self._fundamentals_parquet[col] = pd.to_numeric(
                        self._fundamentals_parquet[col], errors="coerce"
                    )

            if self._fundamentals_parquet is not None:
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
            csv_gz_file = self.data_dir / "fundamental_data_consolidated.csv.gz"
            if parquet_file.exists():
                print("  📦 Loading consolidated fundamentals (Parquet)...")
                self._fundamentals_parquet = pd.read_parquet(parquet_file)
            elif csv_gz_file.exists():
                print("  🗜️  Loading consolidated fundamentals (CSV.GZ)...")
                self._fundamentals_parquet = pd.read_csv(
                    csv_gz_file,
                    compression="gzip",
                    index_col=[0, 1, 2],
                )
                self._fundamentals_parquet.index.names = ["ticker", "sheet_name", "row_name"]
                for col in self._fundamentals_parquet.columns:
                    self._fundamentals_parquet[col] = pd.to_numeric(
                        self._fundamentals_parquet[col], errors="coerce"
                    )
        return self._fundamentals_parquet

    def load_shares_outstanding_panel(self) -> pd.DataFrame:
        """Load Date x Ticker shares outstanding panel."""
        if self._shares_consolidated is None:
            shares_file = self.data_dir / "shares_outstanding_consolidated.csv"
            shares_file_gz = self.data_dir / "shares_outstanding_consolidated.csv.gz"
            if shares_file.exists() or shares_file_gz.exists():
                source_file = shares_file if shares_file.exists() else shares_file_gz
                print("  📊 Loading consolidated shares file...")
                panel = pd.read_csv(
                    source_file,
                    index_col=0,
                    parse_dates=True,
                    compression="gzip" if source_file.suffix == ".gz" else "infer",
                )
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
    
    def load_regime_predictions(self, features: pd.DataFrame | None = None) -> pd.Series:
        """
        Load regime labels from regime filter outputs.

        Args:
            features: Unused legacy argument kept for backward compatibility.
        """
        del features  # Backward compatibility placeholder

        if self._regime_series is None:
            print("\n🎯 Loading regime labels...")
            candidate_files = [p / "outputs" / "regime_features.csv" for p in REGIME_DIR_CANDIDATES]
            regime_file = next((f for f in candidate_files if f.exists()), candidate_files[0])

            if not regime_file.exists():
                candidate_dirs = ", ".join(str(p / "outputs") for p in REGIME_DIR_CANDIDATES)
                raise FileNotFoundError(
                    f"Regime file not found in expected locations: {candidate_dirs}\n"
                    "Run the simplified regime pipeline to generate outputs."
                )

            regime_df = pd.read_csv(regime_file)
            if regime_df.empty:
                raise ValueError(f"Regime file is empty: {regime_file}")

            date_col = next((c for c in ("Date", "date", "DATE") if c in regime_df.columns), regime_df.columns[0])
            regime_df[date_col] = pd.to_datetime(regime_df[date_col], errors="coerce")
            regime_df = regime_df.dropna(subset=[date_col]).set_index(date_col).sort_index()

            regime_col = next(
                (c for c in ("regime_label", "simplified_regime", "regime", "detailed_regime") if c in regime_df.columns),
                None,
            )
            if regime_col is None:
                raise ValueError(
                    "No regime column found in regime file. "
                    "Expected one of: regime_label, simplified_regime, regime, detailed_regime."
                )

            self._regime_series = regime_df[regime_col].dropna().astype(str)
            if self._regime_series.empty:
                raise ValueError(f"No valid regime rows found in: {regime_file}")

            # Load regime->allocation mapping from simplified regime export.
            # This keeps portfolio sizing aligned with whichever regime config was last exported.
            self._regime_allocations = {}
            labels_file = regime_file.parent / "regime_labels.json"
            if labels_file.exists():
                try:
                    labels = json.loads(labels_file.read_text(encoding="utf-8"))
                    for payload in labels.values():
                        if not isinstance(payload, dict):
                            continue
                        regime = str(payload.get("regime", "")).strip()
                        alloc = payload.get("allocation")
                        if regime and alloc is not None:
                            try:
                                self._regime_allocations[regime] = float(alloc)
                            except (TypeError, ValueError):
                                continue
                except Exception as exc:
                    print(f"  ⚠️  Could not parse regime allocations from {labels_file.name}: {exc}")

            print(f"  ✅ Loaded {len(self._regime_series)} regime labels")
            print("\n  Regime distribution:")
            for regime, count in self._regime_series.value_counts().items():
                pct = count / len(self._regime_series) * 100
                print(f"    {regime}: {count} days ({pct:.1f}%)")
            if self._regime_allocations:
                print("  Regime allocations:")
                for regime, alloc in sorted(self._regime_allocations.items()):
                    print(f"    {regime}: {alloc:.2f}")

        return self._regime_series

    def load_regime_allocations(self) -> Dict[str, float]:
        """Get regime allocation mapping loaded from regime_labels.json when available."""
        if self._regime_series is None:
            self.load_regime_predictions()
        return dict(self._regime_allocations or {})
    
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
