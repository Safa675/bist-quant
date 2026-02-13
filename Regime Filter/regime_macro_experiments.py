"""
Macro-Augmented Regime Experiments
===================================
Wraps the existing SimpleRegimeClassifier with macro overlays.

RULE: simple_regime.py is NEVER modified.
This module IMPORTS from it and extends behavior conditionally.

Strategies:
  A) risk_index   — Composite macro risk score scales allocation
  B) usdtry_override — Currency crisis forces Stress regime
  C) cds_gate     — CDS spike forces Stress regime
  D) all          — All three combined

Usage:
    from regime_macro_experiments import MacroAugmentedRegime, MacroConfig

    regime = MacroAugmentedRegime(macro_mode='risk_index')
    results = regime.run(data_dir='../data')
    regime.print_comparison(results)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import logging
import warnings
warnings.filterwarnings('ignore')

# Import existing simple regime filter (untouched)
from simple_regime import (
    SimpleRegimeClassifier,
    SimpleBacktester,
    DataLoader,
    CONFIG as SIMPLE_CONFIG,
)

from macro_features import MacroFeatures, MacroConfig

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-5s | %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger('macro_experiment')


# ============================================================================
# MACRO-AUGMENTED REGIME CLASSIFIER
# ============================================================================

class MacroAugmentedRegime:
    """
    Wraps SimpleRegimeClassifier with optional macro overlays.

    Modes:
        'none'            — Baseline: identical to simple_regime.py
        'risk_index'      — Strategy A: composite risk score scales allocation
        'usdtry_override' — Strategy B: currency crisis → force Stress
        'cds_gate'        — Strategy C: CDS spike → force Stress
        'all'             — All three combined
    """

    VALID_MODES = ('none', 'risk_index', 'usdtry_override', 'cds_gate', 'all')

    def __init__(self, macro_mode: str = 'none', config: MacroConfig = None):
        if macro_mode not in self.VALID_MODES:
            raise ValueError(f"Invalid macro_mode '{macro_mode}'. Choose from {self.VALID_MODES}")

        self.macro_mode = macro_mode
        self.config = config or MacroConfig()
        self.base_classifier = SimpleRegimeClassifier()
        self.macro = None

        # Store intermediate results for analysis
        self.base_regimes = None
        self.augmented_regimes = None
        self.macro_signals = None
        self.allocation_series = None

        log.info(f"MacroAugmentedRegime initialized: mode={macro_mode}")

    # ----------------------------------------------------------------
    # Core: classify regimes
    # ----------------------------------------------------------------

    def classify(self, prices: pd.Series, data_dir=None) -> pd.Series:
        """
        Classify regimes with optional macro overlay.

        Args:
            prices: XU100 closing prices with DatetimeIndex
            data_dir: path to data directory (for macro loading)

        Returns:
            pd.Series of regime labels ('Bull', 'Bear', 'Recovery', 'Stress')
        """
        # Step 1: Run baseline simple regime filter
        self.base_regimes = self.base_classifier.classify(prices)
        log.info(f"Base regime classified: {len(self.base_regimes)} days")

        # Step 2: If no macro, return baseline (PARITY GUARANTEE)
        if self.macro_mode == 'none':
            self.augmented_regimes = self.base_regimes.copy()
            self.macro_signals = pd.DataFrame(index=self.base_regimes.index)
            log.info("Macro mode=none → returning baseline")
            return self.augmented_regimes

        # Step 3: Load macro features
        self.macro = MacroFeatures(data_dir=data_dir, config=self.config)
        self.macro.load()

        # Step 4: Apply macro overlays
        self.augmented_regimes = self._apply_macro(self.base_regimes)
        return self.augmented_regimes

    def _apply_macro(self, base: pd.Series) -> pd.Series:
        """Apply selected macro overlay(s) to base regimes."""
        result = base.copy()
        signals = {}

        # Strategy B: USDTRY Crisis Override
        if self.macro_mode in ('usdtry_override', 'all'):
            crisis_flag = self.macro.compute_usdtry_crisis_flag()
            crisis_flag = crisis_flag.reindex(result.index).fillna(False)
            signals['usdtry_crisis'] = crisis_flag

            n_overrides = 0
            for date in result.index:
                if crisis_flag.get(date, False):
                    if result[date] != 'Stress':
                        n_overrides += 1
                    result[date] = 'Stress'

            log.info(f"USDTRY Override: {n_overrides} days forced to Stress "
                     f"({crisis_flag.sum()} crisis days total)")

        # Strategy C: CDS Stress Gate
        if self.macro_mode in ('cds_gate', 'all'):
            cds_flag = self.macro.compute_cds_stress_flag()
            cds_flag = cds_flag.reindex(result.index).fillna(False)
            signals['cds_stress'] = cds_flag

            n_overrides = 0
            for date in result.index:
                if cds_flag.get(date, False):
                    if result[date] != 'Stress':
                        n_overrides += 1
                    result[date] = 'Stress'

            log.info(f"CDS Gate: {n_overrides} days forced to Stress "
                     f"({cds_flag.sum()} stress days total)")

        # Note: risk_index does NOT change regimes — it changes allocations.
        # It's applied in get_allocations() instead.
        if self.macro_mode in ('risk_index', 'all'):
            risk_mult = self.macro.compute_risk_multiplier()
            risk_mult = risk_mult.reindex(result.index).fillna(1.0)
            signals['risk_multiplier'] = risk_mult
            log.info(f"Risk Index: avg multiplier={risk_mult.mean():.3f}, "
                     f"min={risk_mult.min():.3f}")

        self.macro_signals = pd.DataFrame(signals, index=result.index)
        return result

    # ----------------------------------------------------------------
    # Allocation: get per-day stock allocation
    # ----------------------------------------------------------------

    def get_allocations(self) -> pd.Series:
        """
        Get per-day allocation (0.0 to 1.0) based on regime + macro.

        For 'risk_index' mode: allocation = base_allocation * risk_multiplier
        For override modes: allocation follows the (possibly overridden) regime
        """
        if self.augmented_regimes is None:
            raise ValueError("Call classify() first")

        # Map regime → base allocation
        alloc = self.augmented_regimes.map(SIMPLE_CONFIG['allocations']).fillna(0.5)

        # Strategy A: Risk Index scales the allocation
        if self.macro_mode in ('risk_index', 'all'):
            if 'risk_multiplier' in self.macro_signals.columns:
                multiplier = self.macro_signals['risk_multiplier']
                alloc = alloc * multiplier
                alloc = alloc.clip(0.0, 1.0)
                log.info(f"Risk Index applied: avg allocation {alloc.mean():.3f}")

        self.allocation_series = alloc
        return alloc

    # ----------------------------------------------------------------
    # Backtesting
    # ----------------------------------------------------------------

    def backtest(self, prices: pd.Series) -> dict:
        """Run backtest using augmented regimes/allocations."""
        if self.augmented_regimes is None:
            raise ValueError("Call classify() first")

        allocations = self.get_allocations()

        # Align
        common = prices.index.intersection(allocations.index)
        p = prices.loc[common]
        a = allocations.loc[common]

        returns = p.pct_change()
        strategy_returns = returns * a

        cum = (1 + strategy_returns).cumprod()
        total_return = cum.iloc[-1] - 1
        n_years = len(strategy_returns) / 252
        cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        annual_vol = strategy_returns.std() * np.sqrt(252)
        sharpe = cagr / annual_vol if annual_vol > 0 else 0
        max_dd = ((cum - cum.cummax()) / cum.cummax()).min()

        # Regime transitions (whipsaw metric)
        transitions = (self.augmented_regimes != self.augmented_regimes.shift(1)).sum()
        avg_duration = len(self.augmented_regimes) / max(transitions, 1)

        # Regime distribution
        regime_dist = self.augmented_regimes.value_counts(normalize=True).to_dict()

        return {
            'mode': self.macro_mode,
            'cagr': cagr,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'volatility': annual_vol,
            'total_return': total_return,
            'transitions': transitions,
            'avg_regime_duration': avg_duration,
            'regime_distribution': regime_dist,
            'cum_returns': cum,
        }

    # ----------------------------------------------------------------
    # Full pipeline
    # ----------------------------------------------------------------

    def run(self, data_dir=None) -> dict:
        """
        Run full pipeline: load data → classify → backtest.

        Args:
            data_dir: path to data/ directory

        Returns:
            dict of backtest results
        """
        if data_dir is None:
            data_dir = Path(__file__).resolve().parent.parent / "data"
        else:
            data_dir = Path(data_dir)

        log.info(f"{'='*60}")
        log.info(f"Running experiment: macro_mode={self.macro_mode}")
        log.info(f"{'='*60}")

        # Load XU100 prices
        loader = DataLoader(data_dir)
        xu100 = loader.load_xu100()
        prices = xu100['Close']

        # Classify
        self.classify(prices, data_dir=data_dir)

        # Backtest
        results = self.backtest(prices)

        log.info(f"Results: CAGR={results['cagr']*100:.1f}% "
                 f"Sharpe={results['sharpe']:.2f} "
                 f"MaxDD={results['max_dd']*100:.1f}% "
                 f"Transitions={results['transitions']}")

        return results

    # ----------------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------------

    def get_override_dates(self) -> pd.DataFrame:
        """Show dates where macro changed the regime from baseline."""
        if self.base_regimes is None or self.augmented_regimes is None:
            raise ValueError("Call classify() first")

        changed = self.base_regimes != self.augmented_regimes
        if not changed.any():
            return pd.DataFrame(columns=['date', 'base_regime', 'macro_regime'])

        dates = changed[changed].index
        return pd.DataFrame({
            'date': dates,
            'base_regime': self.base_regimes.loc[dates].values,
            'macro_regime': self.augmented_regimes.loc[dates].values,
        })


# ============================================================================
# COMPARISON UTILITIES
# ============================================================================

def compare_experiments(results_list: list) -> pd.DataFrame:
    """
    Build a comparison table from multiple experiment results.

    Args:
        results_list: list of dicts from MacroAugmentedRegime.run()

    Returns:
        pd.DataFrame with one row per experiment
    """
    rows = []
    for r in results_list:
        rows.append({
            'Mode': r['mode'],
            'CAGR': f"{r['cagr']*100:.1f}%",
            'Sharpe': f"{r['sharpe']:.2f}",
            'MaxDD': f"{r['max_dd']*100:.1f}%",
            'Vol': f"{r['volatility']*100:.1f}%",
            'Transitions': r['transitions'],
            'Avg Duration': f"{r['avg_regime_duration']:.1f}d",
            'Total Return': f"{r['total_return']*100:.0f}%",
        })
    return pd.DataFrame(rows)


def print_comparison(results_list: list):
    """Pretty-print comparison table."""
    df = compare_experiments(results_list)
    print("\n" + "=" * 90)
    print("MACRO EXPERIMENT COMPARISON")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)


# ============================================================================
# ENTRY POINT (quick test)
# ============================================================================

if __name__ == "__main__":
    import sys

    data_dir = Path(__file__).resolve().parent.parent / "data"
    mode = sys.argv[1] if len(sys.argv) > 1 else 'none'

    exp = MacroAugmentedRegime(macro_mode=mode)
    results = exp.run(data_dir=data_dir)

    print(f"\nMode: {mode}")
    print(f"CAGR:  {results['cagr']*100:.1f}%")
    print(f"Sharpe: {results['sharpe']:.2f}")
    print(f"MaxDD:  {results['max_dd']*100:.1f}%")
