"""
Tune the Simple Regime Filter — Test different configurations.
Find the best combo of MA window, vol threshold, hysteresis, and allocations.
"""

import logging
import warnings
from datetime import datetime

import numpy as np
from simple_regime import CONFIG, DataLoader, SimpleBacktester, SimpleRegimeClassifier

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


def run_variant(prices, name, ma_window=200, vol_high_pct=80, hysteresis=7,
                alloc_bull=1.0, alloc_recovery=0.7, alloc_bear=0.0, alloc_stress=0.0):
    """Run a single configuration variant and return metrics."""
    config = CONFIG.copy()
    config['ma_window'] = ma_window
    config['vol_high_percentile'] = vol_high_pct
    config['hysteresis_days'] = hysteresis
    config['allocations'] = {
        'Bull': alloc_bull,
        'Recovery': alloc_recovery,
        'Bear': alloc_bear,
        'Stress': alloc_stress,
    }

    classifier = SimpleRegimeClassifier(config)
    regimes = classifier.classify(prices)

    backtester = SimpleBacktester(prices, regimes, config['allocations'])
    results = backtester.run()
    quality = backtester.regime_quality()

    transitions = classifier.get_transitions()
    avg_dur = len(regimes) / max(transitions, 1)

    bull_ret = quality.get('Bull', {}).get('avg_annual_return', 0)
    bear_ret = quality.get('Bear', {}).get('avg_annual_return', 0)
    separation = bull_ret - bear_ret

    return {
        'name': name,
        'annual_return': results['annual_return'],
        'sharpe': results['sharpe_ratio'],
        'max_dd': results['max_drawdown'],
        'vol': results['annual_volatility'],
        'transitions': transitions,
        'avg_duration': avg_dur,
        'bull_ret': bull_ret,
        'bear_ret': bear_ret,
        'separation': separation,
        'bull_pct': quality.get('Bull', {}).get('pct', 0),
        'bear_pct': quality.get('Bear', {}).get('pct', 0),
        'recovery_pct': quality.get('Recovery', {}).get('pct', 0),
        'stress_pct': quality.get('Stress', {}).get('pct', 0),
    }


def tune():
    logger.info("=" * 80)
    logger.info("SIMPLE REGIME FILTER — VARIANT TESTING")
    logger.info("=" * 80)
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")

    # Load data
    loader = DataLoader()
    xu100 = loader.load_xu100()
    prices = xu100['Close']

    variants = []

    # ===================================================================
    # Test different configurations
    # ===================================================================
    logger.info("\nRunning variants...\n")

    # --- Baseline: current config ---
    variants.append(run_variant(prices, "Baseline (200MA, hyst=7, rec=0.7)"))

    # --- Fix 1: Recovery = 1.0 (stop punishing high-vol uptrends) ---
    variants.append(run_variant(prices, "Recovery=1.0", alloc_recovery=1.0))

    # --- Fix 2: No hysteresis ---
    variants.append(run_variant(prices, "No hysteresis", hysteresis=0))

    # --- Fix 3: Both fixes ---
    variants.append(run_variant(prices, "Rec=1.0 + No hyst", alloc_recovery=1.0, hysteresis=0))

    # --- Fix 4: Lower hysteresis ---
    variants.append(run_variant(prices, "Hysteresis=3", hysteresis=3))
    variants.append(run_variant(prices, "Rec=1.0 + Hyst=3", alloc_recovery=1.0, hysteresis=3))

    # --- Different MA windows ---
    variants.append(run_variant(prices, "MA=100, Rec=1.0, NoH", ma_window=100,
                                alloc_recovery=1.0, hysteresis=0))
    variants.append(run_variant(prices, "MA=150, Rec=1.0, NoH", ma_window=150,
                                alloc_recovery=1.0, hysteresis=0))
    variants.append(run_variant(prices, "MA=50, Rec=1.0, NoH", ma_window=50,
                                alloc_recovery=1.0, hysteresis=0))

    # --- Different vol thresholds ---
    variants.append(run_variant(prices, "VolHigh=90, Rec=1.0, NoH", vol_high_pct=90,
                                alloc_recovery=1.0, hysteresis=0))
    variants.append(run_variant(prices, "VolHigh=70, Rec=1.0, NoH", vol_high_pct=70,
                                alloc_recovery=1.0, hysteresis=0))

    # --- Vol-based partial allocation during bear ---
    variants.append(run_variant(prices, "Bear=0.3 (partial avoid)", alloc_bear=0.3,
                                alloc_recovery=1.0, hysteresis=0))

    # --- Conservative: stay invested more ---
    variants.append(run_variant(prices, "Bear=0.5, Stress=0.2", alloc_bear=0.5,
                                alloc_stress=0.2, alloc_recovery=1.0, hysteresis=0))

    # --- Aggressive: all-in/all-out ---
    variants.append(run_variant(prices, "All-in/All-out (no vol)", alloc_recovery=1.0,
                                alloc_bear=0.0, alloc_stress=0.0, hysteresis=0,
                                vol_high_pct=100))  # Set vol threshold to 100 → never "high"

    # --- Small hysteresis variations with adjusted alloc ---
    variants.append(run_variant(prices, "MA=100, Hyst=3, Rec=1.0", ma_window=100,
                                alloc_recovery=1.0, hysteresis=3))
    variants.append(run_variant(prices, "MA=50, Hyst=3, Rec=1.0", ma_window=50,
                                alloc_recovery=1.0, hysteresis=3))

    # ===================================================================
    # RESULTS TABLE
    # ===================================================================
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS (sorted by Sharpe)")
    logger.info("=" * 80)

    # Sort by Sharpe
    variants.sort(key=lambda x: x['sharpe'], reverse=True)

    logger.info(f"\n{'#':<3} {'Variant':<36} {'Annual':>8} {'Sharpe':>7} {'MaxDD':>8} "
          f"{'Vol':>7} {'Trans':>6} {'Dur':>5} {'Sep':>7}")
    logger.info("-" * 92)

    for i, v in enumerate(variants, 1):
        marker = " ★" if i <= 3 else ""
        logger.info(f"{i:<3} {v['name']:<36} {v['annual_return']:>7.1%} {v['sharpe']:>7.2f} "
              f"{v['max_dd']:>7.1%} {v['vol']:>6.1%} {v['transitions']:>6} "
              f"{v['avg_duration']:>5.1f} {v['separation']:>6.1%}{marker}")

    # ===================================================================
    # DETAILED TOP 3
    # ===================================================================
    logger.info("\n\n" + "=" * 80)
    logger.info("TOP 3 VARIANTS — DETAILS")
    logger.info("=" * 80)

    for i, v in enumerate(variants[:3], 1):
        logger.info(f"\n#{i} {v['name']}")
        logger.info(f"  Annual Return:     {v['annual_return']:>7.1%}")
        logger.info(f"  Sharpe Ratio:      {v['sharpe']:>7.2f}")
        logger.info(f"  Max Drawdown:      {v['max_dd']:>7.1%}")
        logger.info(f"  Volatility:        {v['vol']:>7.1%}")
        logger.info(f"  Transitions:       {v['transitions']}")
        logger.info(f"  Avg Duration:      {v['avg_duration']:.1f} days")
        logger.info(f"  Regime Separation: {v['separation']:>7.1%}")
        if v['separation'] > 0.30:
            logger.info("  ✅ GOOD")
        elif v['separation'] > 0:
            logger.warning("  ⚠️  WEAK")
        else:
            logger.error("  ❌ INVERTED")
        logger.info(f"  Bull:     {v['bull_ret']:>6.1%} return ({v['bull_pct']:>5.1f}% of time)")
        logger.info(f"  Bear:     {v['bear_ret']:>6.1%} return ({v['bear_pct']:>5.1f}% of time)")

    # ===================================================================
    # INSIGHTS
    # ===================================================================
    logger.info("\n\n" + "=" * 80)
    logger.info("KEY INSIGHTS")
    logger.info("=" * 80)

    # Find best Sharpe
    best = variants[0]
    baseline = next(v for v in variants if 'Baseline' in v['name'])

    logger.info(f"\n  Best variant: {best['name']}")
    logger.info(f"  vs Baseline:  Sharpe {baseline['sharpe']:.2f} → {best['sharpe']:.2f} "
          f"({best['sharpe'] - baseline['sharpe']:+.2f})")
    logger.info(f"  vs Baseline:  Return {baseline['annual_return']:.1%} → {best['annual_return']:.1%} "
          f"({best['annual_return'] - baseline['annual_return']:+.1%})")

    # Check if MA=50 variants are consistently better
    ma50_variants = [v for v in variants if 'MA=50' in v['name']]
    ma200_variants = [v for v in variants if 'MA=50' not in v['name'] and 'MA=100' not in v['name']
                      and 'MA=150' not in v['name']]
    if ma50_variants and ma200_variants:
        avg_50 = np.mean([v['sharpe'] for v in ma50_variants])
        avg_200 = np.mean([v['sharpe'] for v in ma200_variants])
        logger.info(f"\n  MA=50 avg Sharpe:  {avg_50:.2f}")
        logger.info(f"  MA=200 avg Sharpe: {avg_200:.2f}")
        if avg_50 > avg_200:
            logger.info(f"  → MA=50 consistently better (+{avg_50 - avg_200:.2f})")
        else:
            logger.info("  → MA=200 holds its own")

    # Check hysteresis impact
    no_hyst = [v for v in variants if 'No h' in v['name'].lower() or 'noh' in v['name'].lower()
               or v['name'].endswith(', NoH')]
    with_hyst = [v for v in variants if v not in no_hyst]
    if no_hyst and with_hyst:
        avg_no = np.mean([v['sharpe'] for v in no_hyst])
        avg_with = np.mean([v['sharpe'] for v in with_hyst])
        logger.info(f"\n  No hysteresis avg Sharpe:   {avg_no:.2f}")
        logger.info(f"  With hysteresis avg Sharpe: {avg_with:.2f}")

    logger.info("\n" + "=" * 80)
    logger.info("DONE")
    logger.info("=" * 80)


if __name__ == "__main__":
    tune()
