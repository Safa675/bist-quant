"""
Simple Regime Filter for BIST XU100
====================================
2 dimensions. 4 regimes. That's it.

Dimension 1: TREND → Price vs 200-day MA
Dimension 2: VOLATILITY → 20-day realized vol percentile

Regime Map:
    Uptrend  + Low/Normal Vol → Bull     (full stocks)
    Uptrend  + High Vol       → Recovery (partial stocks)
    Downtrend + Low/Normal Vol → Bear     (no stocks)
    Downtrend + High Vol       → Stress   (no stocks)

Output: One regime label per trading day.
"""

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION — Everything in one place, no external config file needed
# ============================================================================

CONFIG = {
    # Trend detection
    # MA=50 works best for BIST's volatile, mean-reverting market.
    # Faster trend detection catches reversals earlier.
    # Tuning results: MA50 (3.88 Sharpe) >> MA100 (2.51) >> MA150 (2.26) >> MA200 (2.03)
    'ma_window': 50,                 # 50-day moving average for trend

    # Volatility classification
    'vol_window': 20,                # 20-day realized volatility
    'vol_lookback': 252,             # Rolling percentile lookback (1 year)
    'vol_high_percentile': 80,       # Above 80th percentile = High Vol

    # Regime persistence (anti-whipsaw)
    # No hysteresis works best — BIST moves fast, delays cost returns.
    # Tuning: NoHyst (2.54 avg Sharpe) >> WithHyst (1.58 avg Sharpe)
    'hysteresis_days': 0,            # No delay — switch immediately
    'stress_immediate': True,        # Stress triggers immediately (no delay)

    # Regime allocations (stock %, rest goes to gold)
    # Recovery=1.0 because high-vol uptrends have the HIGHEST returns (83.9%)
    'allocations': {
        'Bull': 1.0,                 # 100% stocks
        'Recovery': 1.0,             # 100% stocks (high-vol uptrends are the best)
        'Bear': 0.0,                 # 0% stocks, 100% gold
        'Stress': 0.0,               # 0% stocks, 100% gold
    },

    # Data
    'xu100_file': 'xu100_prices.csv',
    'usdtry_ticker': 'TRY=X',
}


# ============================================================================
# DATA LOADING — Minimal, just what we need
# ============================================================================

class DataLoader:
    """Load XU100 price data. That's all we need."""

    def __init__(self, data_dir=None):
        pass

    def load_xu100(self) -> pd.DataFrame:
        """Load XU100 index data → DatetimeIndex, columns: Open/High/Low/Close/Volume"""
        from bist_quant.common.data_paths import get_data_paths
        
        filepath = get_data_paths().xu100_prices
        if not filepath.exists():
            raise FileNotFoundError(f"XU100 data not found: {filepath}")

        if filepath.suffix == '.parquet':
            df = pd.read_parquet(filepath)
        else:
            df = pd.read_csv(filepath)

        # Skip header row if present
        if len(df) > 0 and 'Date' in df.columns:
            first_date = pd.to_datetime(df.iloc[0]['Date'], errors='coerce')
            if pd.isna(first_date):
                df = df.iloc[1:].copy()

        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
        else:
            df.index = pd.to_datetime(df.index, errors='coerce')

        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.sort_index()
        
        # Check for both "Close" or "close"
        if 'Close' not in df.columns and 'close' in df.columns:
            df['Close'] = df['close']
            
        if 'Close' in df.columns:
            df = df[df['Close'].notna()]

        logger.info(f"Loaded {len(df)} days: {df.index[0].date()} → {df.index[-1].date()}")
        return df


# ============================================================================
# SIMPLE REGIME CLASSIFIER — The core logic
# ============================================================================

class SimpleRegimeClassifier:
    """
    2-dimension regime classifier.

    Dimension 1: TREND
        Price > 200 MA → Uptrend
        Price < 200 MA → Downtrend

    Dimension 2: VOLATILITY
        20d vol percentile < 80th → Normal
        20d vol percentile ≥ 80th → High

    Regimes:
        Uptrend  + Normal Vol → Bull
        Uptrend  + High Vol   → Recovery
        Downtrend + Normal Vol → Bear
        Downtrend + High Vol   → Stress
    """

    def __init__(self, config=None):
        self.config = config or CONFIG
        self.features = None
        self.raw_regimes = None
        self.regimes = None

    def classify(self, prices: pd.Series) -> pd.Series:
        """
        Classify regimes from price series.

        Args:
            prices: pd.Series of closing prices with DatetimeIndex

        Returns:
            pd.Series of regime labels ('Bull', 'Bear', 'Recovery', 'Stress')
        """
        # Step 1: Calculate features
        self.features = self._calculate_features(prices)

        # Step 2: Raw classification
        self.raw_regimes = self._classify_raw(self.features)

        # Step 3: Apply persistence filter (anti-whipsaw)
        self.regimes = self._apply_persistence(self.raw_regimes)

        return self.regimes

    def _calculate_features(self, prices: pd.Series) -> pd.DataFrame:
        """Calculate the 2 features we need: MA trend + vol percentile"""
        features = pd.DataFrame(index=prices.index)

        # 1. Trend: Price vs 200-day MA
        ma = prices.rolling(self.config['ma_window']).mean()
        features['ma_200'] = ma
        features['above_ma'] = (prices > ma).astype(int)

        # 2. Volatility: 20-day realized vol → rolling percentile
        returns = prices.pct_change()
        vol_raw = returns.rolling(self.config['vol_window']).std() * np.sqrt(252)
        features['realized_vol_20d'] = vol_raw

        # Rolling percentile of volatility
        lookback = self.config['vol_lookback']
        features['vol_percentile'] = vol_raw.rolling(
            lookback, min_periods=lookback // 2
        ).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)

        # High vol flag
        features['high_vol'] = (
            features['vol_percentile'] >= self.config['vol_high_percentile']
        ).astype(int)

        return features.dropna()

    def _classify_raw(self, features: pd.DataFrame) -> pd.Series:
        """Apply the 2x2 regime mapping"""
        regimes = pd.Series('Unknown', index=features.index, dtype=object)

        uptrend = features['above_ma'] == 1
        downtrend = features['above_ma'] == 0
        high_vol = features['high_vol'] == 1
        normal_vol = features['high_vol'] == 0

        regimes[uptrend & normal_vol] = 'Bull'
        regimes[uptrend & high_vol] = 'Recovery'
        regimes[downtrend & normal_vol] = 'Bear'
        regimes[downtrend & high_vol] = 'Stress'

        return regimes

    def _apply_persistence(self, raw: pd.Series) -> pd.Series:
        """
        Hysteresis filter: don't switch regime until new one persists for N days.
        Exception: Stress switches immediately.
        """
        hysteresis = self.config['hysteresis_days']
        immediate_stress = self.config['stress_immediate']

        smoothed = raw.copy()
        current = raw.iloc[0]
        pending_count = 0

        for i in range(1, len(raw)):
            signal = raw.iloc[i]

            if signal == current:
                pending_count = 0
            else:
                pending_count += 1

            # Stress triggers immediately
            if immediate_stress and signal == 'Stress':
                current = 'Stress'
                pending_count = 0
            elif pending_count >= hysteresis:
                current = signal
                pending_count = 0

            smoothed.iloc[i] = current

        return smoothed

    def get_current_regime(self) -> dict:
        """Get most recent regime"""
        if self.regimes is None:
            raise ValueError("Call classify() first")

        return {
            'date': self.regimes.index[-1],
            'regime': self.regimes.iloc[-1],
            'allocation': self.config['allocations'].get(self.regimes.iloc[-1], 0.5),
            'above_ma': bool(self.features['above_ma'].iloc[-1]),
            'vol_percentile': float(self.features['vol_percentile'].iloc[-1]),
            'realized_vol': float(self.features['realized_vol_20d'].iloc[-1]),
        }

    def get_distribution(self) -> pd.DataFrame:
        """Get regime distribution summary"""
        if self.regimes is None:
            raise ValueError("Call classify() first")

        counts = self.regimes.value_counts()
        pcts = self.regimes.value_counts(normalize=True) * 100

        return pd.DataFrame({
            'Count': counts,
            'Percent': pcts.round(1),
        }).sort_values('Count', ascending=False)

    def get_transitions(self) -> int:
        """Count regime transitions"""
        if self.regimes is None:
            raise ValueError("Call classify() first")
        return (self.regimes != self.regimes.shift(1)).sum()


# ============================================================================
# BACKTESTER — Simple, no-nonsense
# ============================================================================

class SimpleBacktester:
    """Backtest a regime-based strategy: stocks in Bull/Recovery, gold in Bear/Stress."""

    def __init__(self, prices: pd.Series, regimes: pd.Series, allocations: dict = None):
        self.allocations = allocations or CONFIG['allocations']

        # Align
        common = prices.index.intersection(regimes.index)
        self.prices = prices.loc[common]
        self.regimes = regimes.loc[common]
        self.returns = self.prices.pct_change()

    def run(self) -> dict:
        """Run backtest → returns dict of metrics"""
        positions = self.regimes.map(self.allocations).fillna(0.5)
        strategy_returns = self.returns * positions

        cum = (1 + strategy_returns).cumprod()
        total_return = cum.iloc[-1] - 1
        n_years = len(strategy_returns) / 252
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        annual_vol = strategy_returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        max_dd = ((cum - cum.cummax()) / cum.cummax()).min()

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'cum_returns': cum,
        }

    def run_buy_and_hold(self) -> dict:
        """Buy & hold baseline"""
        cum = (1 + self.returns).cumprod()
        total_return = cum.iloc[-1] - 1
        n_years = len(self.returns) / 252
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        annual_vol = self.returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        max_dd = ((cum - cum.cummax()) / cum.cummax()).min()

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'cum_returns': cum,
        }

    def regime_quality(self) -> dict:
        """How well do regimes separate good vs bad markets?"""
        quality = {}
        for regime in ['Bull', 'Bear', 'Recovery', 'Stress']:
            mask = self.regimes == regime
            if mask.sum() > 0:
                r = self.returns[mask]
                quality[regime] = {
                    'count': int(mask.sum()),
                    'pct': float(mask.mean() * 100),
                    'avg_annual_return': float(r.mean() * 252),
                    'volatility': float(r.std() * np.sqrt(252)),
                }
        return quality


# ============================================================================
# EXPORT — For portfolio engine compatibility
# ============================================================================

import os
from pathlib import Path

from bist_quant.settings import get_output_dir


class RegimeExporter:
    """Export regime labels for downstream consumers (portfolio engine, etc.)"""

    def __init__(self, classifier: SimpleRegimeClassifier, output_dir: str = None):
        self.classifier = classifier
        if output_dir is None:
            # Use config utility - defaults to CWD or BIST_QUANT_OUTPUT_DIR env var
            self.output_dir = get_output_dir("regime", "simple_regime")
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_csv(self, filename: str = "regime_features.csv") -> Path:
        """
        Export features + regime labels to CSV.
        Compatible with portfolio engine's data_loader.load_regime_predictions().
        """
        if self.classifier.regimes is None:
            raise ValueError("Classify regimes first")

        df = self.classifier.features.copy()
        df['simplified_regime'] = self.classifier.regimes

        # Map to 5-state format for portfolio engine compatibility
        # (Portfolio engine expects Bull/Bear/Stress/Choppy/Recovery)
        # We don't use Choppy, but map our 4 regimes to the expected labels
        df['regime_label'] = self.classifier.regimes

        filepath = self.output_dir / filename
        df.to_csv(filepath)
        logger.info(f"Exported to: {filepath}")
        return filepath

    def export_json(self, filename: str = "regime_labels.json") -> Path:
        """Export regime labels as JSON"""
        if self.classifier.regimes is None:
            raise ValueError("Classify regimes first")

        data = {}
        for date, regime in self.classifier.regimes.items():
            data[date.strftime('%Y-%m-%d')] = {
                'regime': regime,
                'allocation': CONFIG['allocations'].get(regime, 0.5),
            }

        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Exported to: {filepath}")
        return filepath


# ============================================================================
# PIPELINE — Full run in one call
# ============================================================================

def run_pipeline(data_dir=None, export=True, verbose=True):
    """
    Run the complete simple regime filter pipeline.

    Returns:
        dict with classifier, backtester, results, and current regime
    """
    if verbose:
        logger.info("=" * 70)
        logger.info("SIMPLE REGIME FILTER")
        logger.info("=" * 70)
        logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("Method:  Price vs 200 MA + Volatility Percentile")
        logger.info("Regimes: Bull | Bear | Recovery | Stress")
        logger.info("=" * 70)

    # Load data
    loader = DataLoader(data_dir)
    xu100 = loader.load_xu100()
    prices = xu100['Close']

    # Classify regimes
    classifier = SimpleRegimeClassifier()
    regimes = classifier.classify(prices)

    if verbose:
        logger.info("\nRegime Distribution:")
        dist = classifier.get_distribution()
        for regime, row in dist.iterrows():
            alloc = CONFIG['allocations'].get(regime, 0.5)
            logger.info(f"  {regime:10s}: {row['Count']:5.0f} days ({row['Percent']:5.1f}%)  → {alloc:.0%} stocks")

        logger.info(f"\nTransitions: {classifier.get_transitions()}")
        avg_duration = len(regimes) / max(classifier.get_transitions(), 1)
        logger.info(f"Avg duration: {avg_duration:.1f} days")

        current = classifier.get_current_regime()
        logger.info(f"\nCurrent Regime ({current['date'].date()}):")
        logger.info(f"  Regime:     {current['regime']}")
        logger.info(f"  Allocation: {current['allocation']:.0%} stocks")
        logger.info(f"  Above MA:   {current['above_ma']}")
        logger.info(f"  Vol %ile:   {current['vol_percentile']:.0f}th")
        logger.info(f"  Vol:        {current['realized_vol']:.1%}")

    # Backtest
    backtester = SimpleBacktester(prices, regimes)
    results = backtester.run()
    bh_results = backtester.run_buy_and_hold()
    quality = backtester.regime_quality()

    if verbose:
        logger.info(f"\n{'PERFORMANCE':=^70}")
        logger.info(f"\n{'Strategy':<25} {'Return':>10} {'Sharpe':>8} {'MaxDD':>10} {'Vol':>8}")
        logger.info("-" * 63)
        logger.info(f"{'Simple Regime Filter':<25} {results['annual_return']:>9.1%} "
              f"{results['sharpe_ratio']:>8.2f} {results['max_drawdown']:>9.1%} "
              f"{results['annual_volatility']:>7.1%}")
        logger.info(f"{'Buy & Hold':<25} {bh_results['annual_return']:>9.1%} "
              f"{bh_results['sharpe_ratio']:>8.2f} {bh_results['max_drawdown']:>9.1%} "
              f"{bh_results['annual_volatility']:>7.1%}")

        logger.info(f"\n{'REGIME QUALITY':=^70}")
        bull_ret = quality.get('Bull', {}).get('avg_annual_return', 0)
        bear_ret = quality.get('Bear', {}).get('avg_annual_return', 0)
        separation = bull_ret - bear_ret
        logger.info(f"Bull avg return:   {bull_ret:>7.1%}")
        logger.info(f"Bear avg return:   {bear_ret:>7.1%}")
        logger.warning(f"Regime separation: {separation:>7.1%} {'✓' if separation > 0.30 else '⚠️'}")

    # Export
    if export:
        exporter = RegimeExporter(classifier)
        exporter.export_csv()
        exporter.export_json()

    if verbose:
        logger.info(f"\n{'=' * 70}")
        logger.info("DONE")
        logger.info(f"{'=' * 70}")

    return {
        'classifier': classifier,
        'backtester': backtester,
        'results': results,
        'buy_hold': bh_results,
        'quality': quality,
        'current': classifier.get_current_regime(),
    }


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_pipeline()
