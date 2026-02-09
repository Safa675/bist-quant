"""
Market Data Module
Consolidates data loading, feature engineering, and leading indicators fetching.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import sys
import warnings
import config

def _fetcher_search_paths() -> list[Path]:
    """Possible locations for Fetcher-Scrapper across layouts."""
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    return [
        current_dir / "data" / "Fetcher-Scrapper",
        project_root / "data" / "Fetcher-Scrapper",
    ]


for _fetcher_dir in _fetcher_search_paths():
    if _fetcher_dir.exists() and str(_fetcher_dir) not in sys.path:
        sys.path.append(str(_fetcher_dir))

# Import TCMB data fetcher if available
try:
    from tcmb_data_fetcher import TCMBDataFetcher
    TCMB_AVAILABLE = True
except ImportError:
    TCMB_AVAILABLE = False
    # tcmb_data_fetcher is optional; pipeline will continue with proxies.


class DataLoader:
    """Load and preprocess market data"""

    @staticmethod
    def _resolve_default_data_dir() -> Path:
        """Resolve default data directory, preferring project-level shared data."""
        current_file = Path(__file__).resolve()
        regime_filter_dir = current_file.parent
        candidates = [
            regime_filter_dir.parent / "data",  # BIST/data
            regime_filter_dir / "data",         # Regime Filter/data
        ]

        for candidate in candidates:
            if (candidate / config.XU100_FILE).exists():
                return candidate

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return candidates[0]
    
    def __init__(self, data_dir=None):
        """
        Args:
            data_dir: Path to data directory (absolute or relative to this file)
        """
        if data_dir is None:
            self.data_dir = self._resolve_default_data_dir()
        else:
            self.data_dir = Path(data_dir)
            if not self.data_dir.exists():
                self.data_dir = self._resolve_default_data_dir()

        self.xu100_data = None
        self.usdtry_data = None
        self.bist_prices = None
        
    def load_xu100(self):
        """Load and clean XU100 index data"""
        print("Loading XU100 data...")
        filepath = self.data_dir / config.XU100_FILE
        
        if not filepath.exists():
             raise FileNotFoundError(f"XU100 data file not found at {filepath}")

        # Read CSV
        df = pd.read_csv(filepath)

        if 'Date' not in df.columns:
            raise ValueError(f"XU100 file is missing required 'Date' column: {filepath}")

        # Skip a duplicated header/ticker row only when first date is invalid.
        if len(df) > 0:
            first_date = pd.to_datetime(df.iloc[0]['Date'], errors='coerce')
            if pd.isna(first_date):
                df = df.iloc[1:].copy()
        
        # Convert columns to appropriate types
        df['Date'] = pd.to_datetime(df['Date'])
        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Set date as index
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        # Remove any rows with missing close prices
        df = df[df['Close'].notna()]
        
        print(f"Loaded {len(df)} days of XU100 data from {df.index[0].date()} to {df.index[-1].date()}")
        
        self.xu100_data = df
        return df
    
    def load_usdtry(self, start_date=None, end_date=None):
        """Fetch USD/TRY exchange rate data"""
        print("Fetching USD/TRY data...")
        
        # If dates not provided, use XU100 date range
        if start_date is None and self.xu100_data is not None:
            start_date = self.xu100_data.index[0]
        if end_date is None and self.xu100_data is not None:
            end_date = self.xu100_data.index[-1]
        
        # Add buffer for calculations
        if start_date:
            start_date = pd.to_datetime(start_date) - timedelta(days=365)
        
        try:
            # Fetch from Yahoo Finance
            usdtry = yf.download(config.USDTRY_TICKER, 
                                start=start_date, 
                                end=end_date,
                                progress=False)
            
            if usdtry.empty:
                print("Warning: No USD/TRY data fetched. Using synthetic data.")
                return self._create_synthetic_usdtry()
            
            # Keep only Close price
            if isinstance(usdtry.columns, pd.MultiIndex):
                 # Handle new yfinance format
                 usdtry = usdtry['Close']
            
            # Ensure it's a DataFrame with 'USDTRY' column
            if isinstance(usdtry, pd.Series):
                usdtry = usdtry.to_frame(name='USDTRY')
            else:
                usdtry = usdtry[['Close']].copy() if 'Close' in usdtry.columns else usdtry
                usdtry.columns = ['USDTRY']
            
            print(f"Loaded {len(usdtry)} days of USD/TRY data")
            
            self.usdtry_data = usdtry
            return usdtry
            
        except Exception as e:
            print(f"Error fetching USD/TRY data: {e}")
            print("Using synthetic USD/TRY data for demonstration")
            return self._create_synthetic_usdtry()
    
    def _create_synthetic_usdtry(self):
        """Create synthetic USD/TRY data for testing"""
        if self.xu100_data is None:
            raise ValueError("Load XU100 data first")
        
        # Create synthetic USD/TRY with realistic trend
        dates = self.xu100_data.index
        n = len(dates)
        
        # Start around 2.0, end around 30.0 (realistic for 2013-2024)
        trend = np.linspace(2.0, 30.0, n)
        noise = np.random.randn(n) * 0.5
        synthetic = trend + noise
        synthetic = np.maximum(synthetic, 1.0) # Ensure positive
        
        df = pd.DataFrame({'USDTRY': synthetic}, index=dates)
        self.usdtry_data = df
        return df
    
    def load_bist_prices(self, limit_stocks=None):
        """Load individual stock prices for breadth calculations"""
        print("Loading BIST stock prices...")
        filepath = self.data_dir / config.BIST_PRICES_FILE
        
        try:
            df = pd.read_csv(filepath)
            
            # Convert date and numeric columns
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove rows with missing data
            df = df.dropna(subset=['Date', 'Close'])
            
            if limit_stocks:
                # Limit to most liquid stocks
                if 'Ticker' in df.columns:
                    top_stocks = df.groupby('Ticker')['Volume'].mean().nlargest(limit_stocks).index
                    df = df[df['Ticker'].isin(top_stocks)]
            
            print(f"Loaded prices with {len(df)} total observations")
            
            self.bist_prices = df
            return df
            
        except Exception as e:
            print(f"Warning: Could not load BIST prices: {e}")
            print("Breadth indicators will not be available")
            return None
    
    def merge_data(self):
        """Merge all data sources into a single DataFrame"""
        if self.xu100_data is None:
            raise ValueError("Load XU100 data first")
        
        # Start with XU100
        merged = self.xu100_data.copy()
        merged.columns = [f'XU100_{col}' for col in merged.columns]
        
        # Add USD/TRY if available
        if self.usdtry_data is not None:
            # Ensure indexes measure same thing (dates)
            merged = merged.join(self.usdtry_data, how='left')
            # Forward fill missing values
            merged['USDTRY'] = merged['USDTRY'].ffill()
        
        print(f"Merged data: {len(merged)} rows, {len(merged.columns)} columns")
        
        return merged
    
    def load_all(self, fetch_usdtry=True, load_stocks=False, limit_stocks=50):
        """Load all data sources"""
        self.load_xu100()
        
        if fetch_usdtry:
            self.load_usdtry()
        
        if load_stocks:
            self.load_bist_prices(limit_stocks=limit_stocks)
        
        return self.merge_data()


class LeadingIndicatorsFetcher:
    """
    Fetch leading indicators for Turkish market regime prediction
    """

    def __init__(self, cache_dir=None, tcmb_api_key=None):
        """
        Args:
            cache_dir: Directory to cache downloaded data
            tcmb_api_key: TCMB EVDS API key for Turkey-specific data
        """
        if cache_dir is None:
            current_file = Path(__file__).resolve()
            regime_filter_dir = current_file.parent
            self.cache_dir = regime_filter_dir / "data"
        else:
            self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.data = {}
        self.tcmb_data = None

        # Initialize TCMB fetcher if available
        self.tcmb_fetcher = None
        if TCMB_AVAILABLE:
            self.tcmb_fetcher = TCMBDataFetcher(api_key=tcmb_api_key, cache_dir=self.cache_dir / "tcmb_cache")
    
    def fetch_all(self, start_date='2013-01-01', end_date=None, include_tcmb=True):
        """Fetch all available leading indicators"""
        print("="*70)
        print("FETCHING LEADING INDICATORS")
        print("="*70)

        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        # Fetch each category
        self.fetch_options_indicators(start_date, end_date)
        self.fetch_credit_indicators(start_date, end_date)
        self.fetch_global_macro(start_date, end_date)
        self.fetch_turkey_macro(start_date, end_date)

        # Fetch TCMB data (Turkey-specific: VIOP30 proxy, CDS, yield curve)
        if include_tcmb:
            self.fetch_tcmb_indicators(start_date, end_date)

        # Combine all indicators
        combined = self._combine_indicators()

        if len(combined) > 0:
            print(f"\n Fetched {len(combined.columns)} leading indicators")
            print(f"   Date range: {combined.index[0]} to {combined.index[-1]}")
            print(f"   Observations: {len(combined)}")
        else:
            print("\n Warning: No data fetched")

        return combined
    
    def fetch_options_indicators(self, start_date, end_date):
        """Fetch options market indicators"""
        print("\n[1/4] Fetching Options Market Indicators...")
        
        # VIOP30 - Turkish VIX (Volatility Index)
        # Note: VIOP30 data may not be available via yfinance
        try:
            print("  - Fetching VIX (global implied volatility)...")
            vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
            if not vix.empty:
                vix_data = vix['Close'] if 'Close' in vix.columns else vix
                self.data['vix'] = vix_data
                print(f"    ✅ VIX: {len(vix)} observations")
            else:
                print("    ⚠️ VIX: No data available")
        except Exception as e:
            print(f"    ❌ VIX: Failed - {e}")
    
    def fetch_credit_indicators(self, start_date, end_date):
        """Fetch credit market indicators"""
        print("\n[2/4] Fetching Credit Market Indicators...")
        
        # US-Turkey yield spread as credit proxy
        try:
            print("  - Fetching US 10Y Treasury yield...")
            us10y = yf.download('^TNX', start=start_date, end=end_date, progress=False)
            if not us10y.empty:
                us10y_data = us10y['Close'] if 'Close' in us10y.columns else us10y
                self.data['us_10y_yield'] = us10y_data
                print(f"    ✅ US 10Y: {len(us10y)} observations")
            else:
                print("    ⚠️ US 10Y: No data available")
        except Exception as e:
            print(f"    ❌ US 10Y: Failed - {e}")
    
    def fetch_global_macro(self, start_date, end_date):
        """Fetch global macro indicators"""
        print("\n[3/4] Fetching Global Macro Indicators...")
        
        # DXY - US Dollar Index
        try:
            print("  - Fetching DXY (US Dollar Index)...")
            dxy = yf.download('DX-Y.NYB', start=start_date, end=end_date, progress=False)
            if not dxy.empty:
                dxy_data = dxy['Close'] if 'Close' in dxy.columns else dxy
                self.data['dxy'] = dxy_data
                print(f"    ✅ DXY: {len(dxy)} observations")
            else:
                print("    ⚠️ DXY: No data available")
        except Exception as e:
            print(f"    ❌ DXY: Failed - {e}")
        
        # Gold
        try:
            print("  - Fetching Gold prices...")
            gold = yf.download('GC=F', start=start_date, end=end_date, progress=False)
            if not gold.empty:
                gold_data = gold['Close'] if 'Close' in gold.columns else gold
                self.data['gold'] = gold_data
                print(f"    ✅ Gold: {len(gold)} observations")
            else:
                print("    ⚠️ Gold: No data available")
        except Exception as e:
            print(f"    ❌ Gold: Failed - {e}")
        
        # S&P 500
        try:
            print("  - Fetching S&P 500...")
            spx = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
            if not spx.empty:
                spx_data = spx['Close'] if 'Close' in spx.columns else spx
                self.data['spx'] = spx_data
                print(f"    ✅ S&P 500: {len(spx)} observations")
            else:
                print("    ⚠️ S&P 500: No data available")
        except Exception as e:
            print(f"    ❌ S&P 500: Failed - {e}")
    
    def fetch_turkey_macro(self, start_date, end_date):
        """Fetch Turkey-specific macro indicators"""
        print("\n[4/5] Fetching Turkey Macro Indicators...")
        print("  - Turkey macro data will be fetched via TCMB EVDS in next step")

    def fetch_tcmb_indicators(self, start_date, end_date):
        """Fetch Turkey-specific indicators from TCMB EVDS"""
        print("\n[5/5] Fetching TCMB Turkey-Specific Indicators...")

        if self.tcmb_fetcher is None:
            print("  Warning: TCMB fetcher not available")
            return

        try:
            # Fetch all TCMB data
            self.tcmb_data = self.tcmb_fetcher.fetch_all(start_date, end_date)

            if self.tcmb_data is not None and not self.tcmb_data.empty:
                # Add key TCMB indicators to main data dict
                tcmb_cols = [
                    'viop30_proxy',      # Turkish VIX proxy
                    'cds_proxy',         # Turkey CDS proxy
                    'yield_2y',          # 2-year bond yield
                    'yield_5y',          # 5-year bond yield
                    'yield_10y',         # 10-year bond yield
                    'yield_curve_slope', # 10Y - 2Y spread
                    'policy_rate',       # TCMB 1-week repo
                    'cpi_annual',        # CPI year-over-year
                    'inflation_exp',     # 12-month inflation expectations
                    'real_rate',         # Policy rate - inflation
                    'iv_rv_spread',      # Implied - realized vol
                    'viop30_change_5d',  # VIOP30 momentum (5d)
                    'viop30_change_20d', # VIOP30 momentum (20d)
                    'cds_change_20d',    # CDS momentum
                    'cds_ma_ratio',      # CDS vs 60-day MA
                ]

                for col in tcmb_cols:
                    if col in self.tcmb_data.columns:
                        self.data[col] = self.tcmb_data[col]
                        print(f"  + Added {col}")

                print(f"\n  Integrated {len([c for c in tcmb_cols if c in self.tcmb_data.columns])} TCMB indicators")
            else:
                print("  Warning: No TCMB data returned")

        except Exception as e:
            print(f"  Error fetching TCMB data: {e}")
            print("  Continuing without TCMB indicators...")
    
    def _combine_indicators(self):
        """Combine all fetched indicators into single DataFrame"""
        if not self.data:
            return pd.DataFrame()
        
        # Combine all series using concat
        combined = pd.concat(self.data, axis=1)
        
        # Forward fill missing values (weekends, holidays)
        combined = combined.ffill()
        
        # Calculate derived indicators
        combined = self._calculate_derived_indicators(combined)
        
        return combined
    
    def _calculate_derived_indicators(self, df):
        """Calculate derived indicators from raw data"""

        # VIX momentum
        if 'vix' in df.columns:
            df['vix_change_5d'] = df['vix'].pct_change(5)
            df['vix_change_20d'] = df['vix'].pct_change(20)
            df['vix_ma_ratio'] = df['vix'] / df['vix'].rolling(60).mean()

        # DXY momentum
        if 'dxy' in df.columns:
            df['dxy_change_5d'] = df['dxy'].pct_change(5)
            df['dxy_change_20d'] = df['dxy'].pct_change(20)

        # Gold momentum
        if 'gold' in df.columns:
            df['gold_change_20d'] = df['gold'].pct_change(20)

        # S&P 500 momentum
        if 'spx' in df.columns:
            df['spx_change_5d'] = df['spx'].pct_change(5)
            df['spx_change_20d'] = df['spx'].pct_change(20)
            df['spx_volatility'] = df['spx'].pct_change().rolling(20).std() * np.sqrt(252)

        # US yield momentum
        if 'us_10y_yield' in df.columns:
            df['us_10y_change'] = df['us_10y_yield'].diff(20)

        # VIOP30 vs VIX spread
        if 'viop30_proxy' in df.columns and 'vix' in df.columns:
            df['viop30_vix_spread'] = df['viop30_proxy'] - df['vix']

        # Turkey credit spread vs US
        if 'yield_5y' in df.columns and 'us_10y_yield' in df.columns:
            # Align yields (Turkey 5Y vs US 10Y as EM spread proxy)
            us_yield = df['us_10y_yield'] / 100 if df['us_10y_yield'].mean() > 1 else df['us_10y_yield']
            df['turkey_us_spread'] = df['yield_5y'] - us_yield

        # Real rate momentum
        if 'real_rate' in df.columns:
            df['real_rate_change_20d'] = df['real_rate'].diff(20)

        # Yield curve flattening indicator
        if 'yield_curve_slope' in df.columns:
            df['yield_curve_change'] = df['yield_curve_slope'].diff(20)
            df['yield_curve_inverted'] = (df['yield_curve_slope'] < 0).astype(int)

        return df
    
    def get_data_availability(self):
        """Report which indicators are available"""
        if not self.data:
            return "No data fetched yet"
        
        report = []
        report.append("\n" + "="*70)
        report.append("DATA AVAILABILITY REPORT")
        report.append("="*70)
        
        for name, series in self.data.items():
            coverage_pct = (1 - series.isna().sum() / len(series)) * 100
            report.append(f"{name:20s}: {len(series):5d} obs, {coverage_pct:5.1f}% coverage")
        
        return "\n".join(report)


class FeatureEngine:
    """Calculate all features needed for regime classification"""
    
    def __init__(self, data):
        """
        Args:
            data: DataFrame with XU100 prices and USD/TRY
        """
        self.data = data.copy()
        self.features = pd.DataFrame(index=data.index)
        
    def calculate_all_features(self):
        """Calculate all feature groups"""
        print("Calculating features...")
        
        self.calculate_volatility_features()
        self.calculate_drawdown_features()
        self.calculate_trend_features()
        self.calculate_momentum_features()
        self.calculate_liquidity_features()
        self.calculate_risk_features()
        
        print(f"Calculated {len(self.features.columns)} features")
        return self.features
    
    def calculate_volatility_features(self):
        """Calculate realized volatility and vol-of-vol"""
        print("  - Volatility features...")
        
        returns = self.data['XU100_Close'].pct_change()
        
        # Realized volatility (annualized)
        for window in [config.VOLATILITY_WINDOWS['short'], config.VOLATILITY_WINDOWS['long']]:
            vol = returns.rolling(window).std() * np.sqrt(252)
            self.features[f'realized_vol_{window}d'] = vol
            
            # Vol-of-vol
            vol_of_vol = vol.rolling(window).std()
            self.features[f'vol_of_vol_{window}d'] = vol_of_vol
        
        # Gap risk (overnight returns proxy - using day-to-day)
        if 'XU100_Open' in self.data.columns and 'XU100_Close' in self.data.columns:
            self.features['gap_risk'] = (self.data['XU100_Open'] / self.data['XU100_Close'].shift(1) - 1).abs()
            self.features['gap_risk_20d'] = self.features['gap_risk'].rolling(20).mean()
        
        # Downside semivariance (skew proxy)
        negative_returns = returns.copy()
        negative_returns[negative_returns > 0] = 0
        self.features['downside_semivar_20d'] = negative_returns.rolling(20).std() * np.sqrt(252)
        self.features['downside_semivar_60d'] = negative_returns.rolling(60).std() * np.sqrt(252)
        
    def calculate_drawdown_features(self):
        """Calculate drawdown metrics"""
        print("  - Drawdown features...")
        
        close = self.data['XU100_Close']
        
        for window in config.DRAWDOWN_WINDOWS:
            # Rolling maximum
            rolling_max = close.rolling(window, min_periods=1).max()
            
            # Drawdown from rolling max
            drawdown = (close - rolling_max) / rolling_max
            self.features[f'max_drawdown_{window}d'] = drawdown.rolling(window).min()
            
            # Current drawdown
            self.features[f'current_drawdown_{window}d'] = drawdown
    
    def calculate_trend_features(self):
        """Calculate trend indicators on multiple horizons"""
        print("  - Trend features...")
        
        close = self.data['XU100_Close']
        
        # Moving averages
        for window in [20, 50, 100, 200]:
            ma = close.rolling(window).mean()
            self.features[f'ma_{window}'] = ma
            self.features[f'price_to_ma_{window}'] = close / ma - 1
        
        # MA slopes (rate of change)
        slope_window = config.TREND_THRESHOLDS['ma_slope_days']
        for ma_window in [20, 50, 200]:
            ma = close.rolling(ma_window).mean()
            ma_slope = ma.pct_change(slope_window)
            self.features[f'ma_{ma_window}_slope_{slope_window}d'] = ma_slope
        
        # Price momentum (returns over different horizons)
        for window in [20, 60, 120, 252]:
            self.features[f'return_{window}d'] = close.pct_change(window)
    
    def calculate_momentum_features(self):
        """Calculate momentum on multiple time horizons"""
        print("  - Momentum features...")
        
        close = self.data['XU100_Close']
        
        # Short-term momentum (1-3 months)
        for window in config.MOMENTUM_WINDOWS['short']:
            mom = close.pct_change(window)
            self.features[f'momentum_short_{window}d'] = mom
        
        # Long-term momentum (6-12 months)
        for window in config.MOMENTUM_WINDOWS['long']:
            mom = close.pct_change(window)
            self.features[f'momentum_long_{window}d'] = mom
        
        # Momentum acceleration (change in momentum)
        if 'return_20d' in self.features.columns:
            self.features['momentum_accel_20d'] = self.features['return_20d'].diff(20)
        if 'return_60d' in self.features.columns:
            self.features['momentum_accel_60d'] = self.features['return_60d'].diff(60)
    
    def calculate_liquidity_features(self):
        """Calculate liquidity and microstructure metrics"""
        print("  - Liquidity features...")
        
        if 'XU100_Volume' not in self.data.columns:
            print("    Warning: Volume data not available")
            return

        volume = self.data['XU100_Volume']
        close = self.data['XU100_Close']
        high = self.data['XU100_High'] if 'XU100_High' in self.data.columns else close
        low = self.data['XU100_Low'] if 'XU100_Low' in self.data.columns else close
        
        # Volume metrics
        self.features['volume'] = volume
        self.features['volume_ma_20d'] = volume.rolling(20).mean()
        self.features['volume_ma_60d'] = volume.rolling(60).mean()
        self.features['volume_ratio'] = volume / self.features['volume_ma_20d']
        
        # Turnover (volume * price)
        turnover = volume * close
        self.features['turnover'] = turnover
        self.features['turnover_ma_20d'] = turnover.rolling(20).mean()
        
        # High-Low range as spread proxy
        hl_range = (high - low) / close
        self.features['hl_range'] = hl_range
        self.features['hl_range_ma_20d'] = hl_range.rolling(20).mean()
        
        # Amihud illiquidity measure: |return| / volume
        returns = close.pct_change().abs()
        self.features['amihud_illiquidity'] = returns / (volume + 1e-10)
        self.features['amihud_illiquidity_ma_20d'] = self.features['amihud_illiquidity'].rolling(20).mean()
    
    def calculate_risk_features(self):
        """Calculate risk-on/off features (USD/TRY based)"""
        print("  - Risk-on/off features...")
        
        if 'USDTRY' not in self.data.columns:
            print("    Warning: USD/TRY data not available, skipping risk features")
            return
        
        usdtry = self.data['USDTRY']
        
        # USD/TRY returns
        usdtry_returns = usdtry.pct_change()
        self.features['usdtry_return_1d'] = usdtry_returns
        
        # USD/TRY momentum
        for window in [5, 20, 60]:
            self.features[f'usdtry_momentum_{window}d'] = usdtry.pct_change(window)
        
        # USD/TRY volatility
        window = config.USDTRY_WINDOWS['vol']
        usdtry_vol = usdtry_returns.rolling(window).std() * np.sqrt(252)
        self.features[f'usdtry_vol_{window}d'] = usdtry_vol
        
        # USD/TRY trend (MA slope)
        usdtry_ma = usdtry.rolling(20).mean()
        self.features['usdtry_ma_slope_20d'] = usdtry_ma.pct_change(20)
        
        # USD/TRY acceleration
        if 'usdtry_momentum_20d' in self.features.columns:
            self.features['usdtry_accel_20d'] = self.features['usdtry_momentum_20d'].diff(20)
    
    def get_features(self):
        """Return calculated features"""
        return self.features
    
    def shift_for_prediction(self, shift_days=1):
        """
        Shift features forward to eliminate look-ahead bias.
        """
        print(f"Shifting features by {shift_days} day(s) to eliminate look-ahead bias...")
        
        shifted_features = self.features.shift(shift_days)
        
        # Critical features: those used by regime classifier
        critical_features = [
            'realized_vol_20d',
            'return_20d',
            'usdtry_momentum_20d',
            'turnover_ma_20d'
        ]
        
        # Only drop if critical features are missing
        available_critical = [f for f in critical_features if f in shifted_features.columns]
        if available_critical:
            shifted_features = shifted_features.dropna(subset=available_critical)
        else:
            # Fallback: drop rows where ALL features are NaN
            shifted_features = shifted_features.dropna(how='all')
        
        rows_before = len(self.features)
        rows_after = len(shifted_features)
        print(f"  Features: {rows_before} → {rows_after} rows (lost {rows_before - rows_after} rows)")
        
        return shifted_features

    def save_features(self, filepath):
        """Save features to CSV"""

def create_custom_indicators(xu100_data, usdtry_data, leading_data):
    """
    Create custom forward-looking indicators by combining data sources

    Args:
        xu100_data: XU100 price data
        usdtry_data: USD/TRY exchange rate
        leading_data: Leading indicators from LeadingIndicatorsFetcher

    Returns:
        DataFrame with custom indicators
    """
    if xu100_data is None or leading_data is None or leading_data.empty:
        return pd.DataFrame()
        
    custom = pd.DataFrame(index=xu100_data.index)

    # Align data first
    # Ensure leading_data covers the same range
    leading_aligned = leading_data.reindex(xu100_data.index).ffill()

    # Price-Credit Divergence
    # If stocks up but credit deteriorating = warning
    if 'us_10y_yield' in leading_aligned.columns:
        xu100_returns = xu100_data.pct_change(20)
        # Ensure xu100_returns is a Series
        if isinstance(xu100_returns, pd.DataFrame):
            xu100_returns = xu100_returns.iloc[:, 0]
        # Higher yields = worse credit conditions
        credit_stress = leading_aligned['us_10y_yield'].diff(20)
        # Ensure credit_stress is a Series
        if isinstance(credit_stress, pd.DataFrame):
            credit_stress = credit_stress.iloc[:, 0]
        custom['price_credit_divergence'] = xu100_returns + credit_stress

    # Currency-Equity Correlation
    # Normally negative in Turkey (TRY weak = stocks down)
    # If correlation breaks = regime change
    if usdtry_data is not None:
        # Align usdtry
        usdtry_aligned = usdtry_data.reindex(xu100_data.index).ffill()
        if isinstance(usdtry_aligned, pd.DataFrame):
            usdtry_aligned = usdtry_aligned.iloc[:, 0]
            
        xu100_returns_5d = xu100_data.pct_change(5)
        # Ensure Series
        if isinstance(xu100_returns_5d, pd.DataFrame):
            xu100_returns_5d = xu100_returns_5d.iloc[:, 0]
        usdtry_returns_5d = usdtry_aligned.pct_change(5)
        custom['currency_equity_corr'] = xu100_returns_5d.rolling(60).corr(usdtry_returns_5d)

    # Global Risk Appetite vs Turkey
    if 'spx' in leading_aligned.columns:
        spx_returns = leading_aligned['spx'].pct_change(20)
        if isinstance(spx_returns, pd.DataFrame):
            spx_returns = spx_returns.iloc[:, 0]
        xu100_returns = xu100_data.pct_change(20)
        if isinstance(xu100_returns, pd.DataFrame):
            xu100_returns = xu100_returns.iloc[:, 0]
        custom['turkey_vs_global'] = xu100_returns - spx_returns

    # Fear Index Composite (VIX + DXY based)
    if 'vix' in leading_aligned.columns and 'dxy' in leading_aligned.columns:
        # Normalize to 0-1
        vix = leading_aligned['vix']
        dxy = leading_aligned['dxy']
        
        # Ensure Series
        if isinstance(vix, pd.DataFrame):
            vix = vix.iloc[:, 0]
        if isinstance(dxy, pd.DataFrame):
            dxy = dxy.iloc[:, 0]
        
        vix_norm = (vix - vix.rolling(252).min()) / \
                   (vix.rolling(252).max() - vix.rolling(252).min())
        dxy_norm = (dxy - dxy.rolling(252).min()) / \
                   (dxy.rolling(252).max() - dxy.rolling(252).min())

        custom['fear_composite'] = (vix_norm + dxy_norm) / 2

    # Turkey Fear Composite (including VIOP30 and CDS)
    if 'viop30_proxy' in leading_aligned.columns and 'cds_proxy' in leading_aligned.columns:
        viop = leading_aligned['viop30_proxy']
        cds = leading_aligned['cds_proxy']
        
        # Ensure Series
        if isinstance(viop, pd.DataFrame):
            viop = viop.iloc[:, 0]
        if isinstance(cds, pd.DataFrame):
            cds = cds.iloc[:, 0]
        
        viop_norm = (viop - viop.rolling(252).min()) / \
                    (viop.rolling(252).max() - viop.rolling(252).min())
        cds_norm = (cds - cds.rolling(252).min()) / \
                   (cds.rolling(252).max() - cds.rolling(252).min())

        custom['turkey_fear_composite'] = (viop_norm + cds_norm) / 2

    # IV-RV Spread based signals
    if 'iv_rv_spread' in leading_aligned.columns:
        iv_rv = leading_aligned['iv_rv_spread']
        # High IV-RV spread = market expects more volatility (fear)
        custom['iv_rv_percentile'] = iv_rv.rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
        )

    # Credit-Equity Divergence (Turkey-specific)
    if 'cds_proxy' in leading_aligned.columns:
        xu100_returns_20d = xu100_data.pct_change(20)
        cds_change = leading_aligned['cds_proxy'].diff(20)
        # If stocks up but CDS widening = warning signal
        custom['turkey_credit_equity_div'] = xu100_returns_20d - cds_change

    # Real Rate Signal
    if 'real_rate' in leading_aligned.columns:
        # Positive real rates = tighter monetary conditions
        real_rate = leading_aligned['real_rate']
        custom['real_rate_regime'] = np.where(real_rate > 0, 1, -1)
        custom['real_rate_percentile'] = real_rate.rolling(252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
        )

    # Yield Curve Regime
    if 'yield_curve_slope' in leading_aligned.columns:
        yc_slope = leading_aligned['yield_curve_slope']
        # Inverted curve = recession signal
        custom['yield_curve_regime'] = np.where(yc_slope < 0, -1, np.where(yc_slope > 0.02, 1, 0))

    return custom
