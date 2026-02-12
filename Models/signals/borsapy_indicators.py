"""
Borsapy Technical Indicators Integration

Provides technical indicators via borsapy for signal generation.
Integrates with existing signal builders by providing panel-based
calculations across multiple tickers.

Available indicators:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- ATR (Average True Range)
- Stochastic Oscillator
- ADX (Average Directional Index)
- Supertrend
- OBV (On-Balance Volume)
- VWAP (Volume Weighted Average Price)

Usage:
    from Models.signals.borsapy_indicators import BorsapyIndicators

    # Build RSI panel for all tickers
    rsi_panel = BorsapyIndicators.build_rsi_panel(close_df, period=14)

    # Build MACD panel
    macd_panel = BorsapyIndicators.build_macd_panel(close_df)

    # Get multiple indicators at once via borsapy API
    indicators = BorsapyIndicators.fetch_indicators_for_ticker(
        "THYAO", indicators=["rsi", "macd", "bb"], period="2y"
    )
"""

from pathlib import Path
from typing import Optional
import sys

import numpy as np
import pandas as pd

# Add paths for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FETCHER_DIR = PROJECT_ROOT / "data" / "Fetcher-Scrapper"
if str(FETCHER_DIR) not in sys.path:
    sys.path.insert(0, str(FETCHER_DIR))

# Check borsapy availability
try:
    import borsapy as bp
    BORSAPY_AVAILABLE = True
except ImportError:
    BORSAPY_AVAILABLE = False
    bp = None


class BorsapyIndicators:
    """
    Technical indicator calculations using borsapy.

    Provides both:
    1. Local calculations (using price panels) - faster for backtesting
    2. API-based fetching (via borsapy) - includes all 12+ indicators
    """

    # -------------------------------------------------------------------------
    # RSI (Relative Strength Index)
    # -------------------------------------------------------------------------

    @staticmethod
    def calculate_rsi(
        prices: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """
        Calculate RSI for a single price series.

        Args:
            prices: Close prices series
            period: RSI period (default 14)

        Returns:
            RSI values (0-100)
        """
        if BORSAPY_AVAILABLE:
            try:
                return bp.calculate_rsi(prices, period=period)
            except Exception:
                pass

        # Fallback: local calculation
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def build_rsi_panel(
        close_df: pd.DataFrame,
        period: int = 14,
    ) -> pd.DataFrame:
        """
        Build RSI panel for all tickers.

        Args:
            close_df: Close prices (Date x Ticker)
            period: RSI period

        Returns:
            DataFrame (Date x Ticker) with RSI values
        """
        print(f"\n📊 Building RSI({period}) panel...")

        rsi_panel = close_df.apply(
            lambda col: BorsapyIndicators.calculate_rsi(col, period=period)
        )

        valid_pct = rsi_panel.notna().mean().mean() * 100
        print(f"  ✅ RSI panel: {rsi_panel.shape[0]} days × {rsi_panel.shape[1]} tickers")
        print(f"     Coverage: {valid_pct:.1f}%")

        return rsi_panel

    # -------------------------------------------------------------------------
    # MACD (Moving Average Convergence Divergence)
    # -------------------------------------------------------------------------

    @staticmethod
    def calculate_macd(
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        """
        Calculate MACD for a single price series.

        Args:
            prices: Close prices series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period

        Returns:
            DataFrame with columns: macd, signal, histogram
        """
        if BORSAPY_AVAILABLE:
            try:
                return bp.calculate_macd(prices, fast=fast, slow=slow, signal=signal)
            except Exception:
                pass

        # Fallback: local calculation
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return pd.DataFrame({
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram,
        })

    @staticmethod
    def build_macd_panel(
        close_df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        output: str = "histogram",
    ) -> pd.DataFrame:
        """
        Build MACD panel for all tickers.

        Args:
            close_df: Close prices (Date x Ticker)
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            output: Which MACD component ("macd", "signal", "histogram")

        Returns:
            DataFrame (Date x Ticker) with MACD component
        """
        print(f"\n📊 Building MACD({fast},{slow},{signal}) panel [{output}]...")

        def get_macd_component(col):
            macd_df = BorsapyIndicators.calculate_macd(col, fast, slow, signal)
            return macd_df[output] if output in macd_df.columns else macd_df["histogram"]

        macd_panel = close_df.apply(get_macd_component)

        valid_pct = macd_panel.notna().mean().mean() * 100
        print(f"  ✅ MACD panel: {macd_panel.shape[0]} days × {macd_panel.shape[1]} tickers")
        print(f"     Coverage: {valid_pct:.1f}%")

        return macd_panel

    # -------------------------------------------------------------------------
    # Bollinger Bands
    # -------------------------------------------------------------------------

    @staticmethod
    def calculate_bollinger_bands(
        prices: pd.Series,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> pd.DataFrame:
        """
        Calculate Bollinger Bands for a single price series.

        Args:
            prices: Close prices series
            period: MA period
            std_dev: Standard deviation multiplier

        Returns:
            DataFrame with columns: upper, middle, lower, pct_b
        """
        middle = prices.rolling(period).mean()
        std = prices.rolling(period).std()

        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)

        # %B: Where price is within the bands (0 = lower, 1 = upper)
        pct_b = (prices - lower) / (upper - lower)

        return pd.DataFrame({
            "upper": upper,
            "middle": middle,
            "lower": lower,
            "pct_b": pct_b,
        })

    @staticmethod
    def build_bollinger_panel(
        close_df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0,
        output: str = "pct_b",
    ) -> pd.DataFrame:
        """
        Build Bollinger Bands panel for all tickers.

        Args:
            close_df: Close prices (Date x Ticker)
            period: MA period
            std_dev: Standard deviation multiplier
            output: Which component ("upper", "middle", "lower", "pct_b")

        Returns:
            DataFrame (Date x Ticker) with BB component
        """
        print(f"\n📊 Building Bollinger({period}, {std_dev}) panel [{output}]...")

        def get_bb_component(col):
            bb_df = BorsapyIndicators.calculate_bollinger_bands(col, period, std_dev)
            return bb_df[output] if output in bb_df.columns else bb_df["pct_b"]

        bb_panel = close_df.apply(get_bb_component)

        valid_pct = bb_panel.notna().mean().mean() * 100
        print(f"  ✅ Bollinger panel: {bb_panel.shape[0]} days × {bb_panel.shape[1]} tickers")
        print(f"     Coverage: {valid_pct:.1f}%")

        return bb_panel

    # -------------------------------------------------------------------------
    # ATR (Average True Range)
    # -------------------------------------------------------------------------

    @staticmethod
    def calculate_atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """
        Calculate ATR for a single ticker.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period

        Returns:
            ATR values
        """
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=period, adjust=False).mean()

        return atr

    @staticmethod
    def build_atr_panel(
        high_df: pd.DataFrame,
        low_df: pd.DataFrame,
        close_df: pd.DataFrame,
        period: int = 14,
    ) -> pd.DataFrame:
        """
        Build ATR panel for all tickers.

        Args:
            high_df: High prices (Date x Ticker)
            low_df: Low prices (Date x Ticker)
            close_df: Close prices (Date x Ticker)
            period: ATR period

        Returns:
            DataFrame (Date x Ticker) with ATR values
        """
        print(f"\n📊 Building ATR({period}) panel...")

        tickers = close_df.columns
        atr_data = {}

        for ticker in tickers:
            if ticker in high_df.columns and ticker in low_df.columns:
                atr_data[ticker] = BorsapyIndicators.calculate_atr(
                    high_df[ticker], low_df[ticker], close_df[ticker], period
                )

        atr_panel = pd.DataFrame(atr_data, index=close_df.index)

        valid_pct = atr_panel.notna().mean().mean() * 100
        print(f"  ✅ ATR panel: {atr_panel.shape[0]} days × {atr_panel.shape[1]} tickers")
        print(f"     Coverage: {valid_pct:.1f}%")

        return atr_panel

    # -------------------------------------------------------------------------
    # Stochastic Oscillator
    # -------------------------------------------------------------------------

    @staticmethod
    def calculate_stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3,
    ) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            k_period: %K period
            d_period: %D smoothing period

        Returns:
            DataFrame with columns: k, d
        """
        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()

        k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
        d = k.rolling(d_period).mean()

        return pd.DataFrame({"k": k, "d": d})

    @staticmethod
    def build_stochastic_panel(
        high_df: pd.DataFrame,
        low_df: pd.DataFrame,
        close_df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3,
        output: str = "k",
    ) -> pd.DataFrame:
        """
        Build Stochastic panel for all tickers.

        Args:
            high_df: High prices (Date x Ticker)
            low_df: Low prices (Date x Ticker)
            close_df: Close prices (Date x Ticker)
            k_period: %K period
            d_period: %D period
            output: Which component ("k" or "d")

        Returns:
            DataFrame (Date x Ticker) with Stochastic values
        """
        print(f"\n📊 Building Stochastic({k_period},{d_period}) panel [{output}]...")

        tickers = close_df.columns
        stoch_data = {}

        for ticker in tickers:
            if ticker in high_df.columns and ticker in low_df.columns:
                stoch = BorsapyIndicators.calculate_stochastic(
                    high_df[ticker], low_df[ticker], close_df[ticker],
                    k_period, d_period
                )
                stoch_data[ticker] = stoch[output]

        stoch_panel = pd.DataFrame(stoch_data, index=close_df.index)

        valid_pct = stoch_panel.notna().mean().mean() * 100
        print(f"  ✅ Stochastic panel: {stoch_panel.shape[0]} days × {stoch_panel.shape[1]} tickers")
        print(f"     Coverage: {valid_pct:.1f}%")

        return stoch_panel

    # -------------------------------------------------------------------------
    # ADX (Average Directional Index)
    # -------------------------------------------------------------------------

    @staticmethod
    def calculate_adx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.DataFrame:
        """
        Calculate ADX and DI+/DI-.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ADX period

        Returns:
            DataFrame with columns: adx, di_plus, di_minus
        """
        # True Range
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

        # Smoothed averages
        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr
        minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr

        # DX and ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(span=period, adjust=False).mean()

        return pd.DataFrame({
            "adx": adx,
            "di_plus": plus_di,
            "di_minus": minus_di,
        })

    @staticmethod
    def build_adx_panel(
        high_df: pd.DataFrame,
        low_df: pd.DataFrame,
        close_df: pd.DataFrame,
        period: int = 14,
        output: str = "adx",
    ) -> pd.DataFrame:
        """
        Build ADX panel for all tickers.

        Args:
            high_df: High prices (Date x Ticker)
            low_df: Low prices (Date x Ticker)
            close_df: Close prices (Date x Ticker)
            period: ADX period
            output: Which component ("adx", "di_plus", "di_minus")

        Returns:
            DataFrame (Date x Ticker) with ADX values
        """
        print(f"\n📊 Building ADX({period}) panel [{output}]...")

        tickers = close_df.columns
        adx_data = {}

        for ticker in tickers:
            if ticker in high_df.columns and ticker in low_df.columns:
                adx_df = BorsapyIndicators.calculate_adx(
                    high_df[ticker], low_df[ticker], close_df[ticker], period
                )
                adx_data[ticker] = adx_df[output]

        adx_panel = pd.DataFrame(adx_data, index=close_df.index)

        valid_pct = adx_panel.notna().mean().mean() * 100
        print(f"  ✅ ADX panel: {adx_panel.shape[0]} days × {adx_panel.shape[1]} tickers")
        print(f"     Coverage: {valid_pct:.1f}%")

        return adx_panel

    # -------------------------------------------------------------------------
    # Supertrend
    # -------------------------------------------------------------------------

    @staticmethod
    def calculate_supertrend(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 10,
        multiplier: float = 3.0,
    ) -> pd.DataFrame:
        """
        Calculate Supertrend indicator.

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period
            multiplier: ATR multiplier

        Returns:
            DataFrame with columns: supertrend, direction (1=up, -1=down)
        """
        if BORSAPY_AVAILABLE:
            try:
                return bp.calculate_supertrend(high, low, close, period, multiplier)
            except Exception:
                pass

        # Fallback: local calculation
        atr = BorsapyIndicators.calculate_atr(high, low, close, period)
        hl2 = (high + low) / 2

        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)

        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(index=close.index, dtype=float)

        supertrend.iloc[0] = upper_band.iloc[0]
        direction.iloc[0] = 1

        for i in range(1, len(close)):
            if close.iloc[i] > supertrend.iloc[i - 1]:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1

        return pd.DataFrame({
            "supertrend": supertrend,
            "direction": direction,
        })

    @staticmethod
    def build_supertrend_panel(
        high_df: pd.DataFrame,
        low_df: pd.DataFrame,
        close_df: pd.DataFrame,
        period: int = 10,
        multiplier: float = 3.0,
        output: str = "direction",
    ) -> pd.DataFrame:
        """
        Build Supertrend panel for all tickers.

        Args:
            high_df: High prices (Date x Ticker)
            low_df: Low prices (Date x Ticker)
            close_df: Close prices (Date x Ticker)
            period: ATR period
            multiplier: ATR multiplier
            output: Which component ("supertrend" or "direction")

        Returns:
            DataFrame (Date x Ticker) with Supertrend values
        """
        print(f"\n📊 Building Supertrend({period}, {multiplier}) panel [{output}]...")

        tickers = close_df.columns
        st_data = {}

        for ticker in tickers:
            if ticker in high_df.columns and ticker in low_df.columns:
                st_df = BorsapyIndicators.calculate_supertrend(
                    high_df[ticker], low_df[ticker], close_df[ticker],
                    period, multiplier
                )
                st_data[ticker] = st_df[output]

        st_panel = pd.DataFrame(st_data, index=close_df.index)

        valid_pct = st_panel.notna().mean().mean() * 100
        print(f"  ✅ Supertrend panel: {st_panel.shape[0]} days × {st_panel.shape[1]} tickers")
        print(f"     Coverage: {valid_pct:.1f}%")

        return st_panel

    # -------------------------------------------------------------------------
    # API-based Fetching (via borsapy)
    # -------------------------------------------------------------------------

    @staticmethod
    def fetch_indicators_for_ticker(
        symbol: str,
        indicators: list[str] = None,
        period: str = "2y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch price history with indicators via borsapy API.

        This uses borsapy's built-in TradingView integration for
        accurate indicator calculations.

        Args:
            symbol: Stock symbol (e.g., "THYAO")
            indicators: List of indicators (e.g., ["rsi", "macd", "bb"])
                       If None, returns all available
            period: Data period
            interval: Data interval

        Returns:
            DataFrame with OHLCV + indicator columns
        """
        if not BORSAPY_AVAILABLE:
            print("  ⚠️  Borsapy not available")
            return pd.DataFrame()

        try:
            ticker = bp.Ticker(symbol)
            if indicators:
                return ticker.history_with_indicators(
                    period=period, interval=interval, indicators=indicators
                )
            else:
                return ticker.history_with_indicators(period=period, interval=interval)
        except Exception as e:
            print(f"  ⚠️  Failed to fetch indicators for {symbol}: {e}")
            return pd.DataFrame()

    @staticmethod
    def fetch_indicators_batch(
        symbols: list[str],
        indicators: list[str] = None,
        period: str = "2y",
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch indicators for multiple tickers via borsapy API.

        Args:
            symbols: List of stock symbols
            indicators: List of indicators
            period: Data period

        Returns:
            Dict mapping symbol -> DataFrame with indicators
        """
        if not BORSAPY_AVAILABLE:
            print("  ⚠️  Borsapy not available")
            return {}

        print(f"\n📊 Fetching indicators for {len(symbols)} tickers...")

        results = {}
        for i, symbol in enumerate(symbols):
            try:
                df = BorsapyIndicators.fetch_indicators_for_ticker(
                    symbol, indicators=indicators, period=period
                )
                if not df.empty:
                    results[symbol] = df
            except Exception:
                continue

            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(symbols)} tickers...")

        print(f"  ✅ Fetched indicators for {len(results)} tickers")
        return results


# -------------------------------------------------------------------------
# Convenience Functions
# -------------------------------------------------------------------------

def build_multi_indicator_panel(
    close_df: pd.DataFrame,
    high_df: pd.DataFrame = None,
    low_df: pd.DataFrame = None,
    indicators: list[str] = None,
) -> dict[str, pd.DataFrame]:
    """
    Build multiple indicator panels at once.

    Args:
        close_df: Close prices (Date x Ticker)
        high_df: High prices (optional, for ATR/Stoch/etc.)
        low_df: Low prices (optional)
        indicators: List of indicators to build
                   Default: ["rsi", "macd", "bb"]

    Returns:
        Dict mapping indicator name -> DataFrame panel
    """
    if indicators is None:
        indicators = ["rsi", "macd", "bb"]

    bi = BorsapyIndicators
    panels = {}

    for ind in indicators:
        ind_lower = ind.lower()

        if ind_lower == "rsi":
            panels["rsi"] = bi.build_rsi_panel(close_df)

        elif ind_lower == "macd":
            panels["macd"] = bi.build_macd_panel(close_df)

        elif ind_lower in ("bb", "bollinger"):
            panels["bb"] = bi.build_bollinger_panel(close_df)

        elif ind_lower == "atr" and high_df is not None and low_df is not None:
            panels["atr"] = bi.build_atr_panel(high_df, low_df, close_df)

        elif ind_lower in ("stoch", "stochastic") and high_df is not None and low_df is not None:
            panels["stoch"] = bi.build_stochastic_panel(high_df, low_df, close_df)

        elif ind_lower == "adx" and high_df is not None and low_df is not None:
            panels["adx"] = bi.build_adx_panel(high_df, low_df, close_df)

        elif ind_lower == "supertrend" and high_df is not None and low_df is not None:
            panels["supertrend"] = bi.build_supertrend_panel(high_df, low_df, close_df)

    return panels
