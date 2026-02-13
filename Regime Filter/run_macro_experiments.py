"""
Macro Experiment Runner
========================
Runs all macro-augmented regime variants against baseline.
Includes ablation tests, walk-forward split, and sensitivity analysis.

Usage:
    python run_macro_experiments.py              # Full comparison
    python run_macro_experiments.py --sensitivity # Include sensitivity analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import warnings
warnings.filterwarnings('ignore')

from simple_regime import DataLoader, SimpleRegimeClassifier, SimpleBacktester, CONFIG
from macro_features import MacroFeatures, MacroConfig
from regime_macro_experiments import MacroAugmentedRegime, compare_experiments, print_comparison


# ============================================================================
# EXPERIMENT SUITE
# ============================================================================

def run_ablation(data_dir: Path, period_name: str = "Full",
                 start_date: str = None, end_date: str = None) -> list:
    """
    Run baseline + all macro variants on a given time period.

    Args:
        data_dir: path to data/ directory
        period_name: label for this period
        start_date: optional start date filter
        end_date: optional end date filter

    Returns:
        list of result dicts
    """
    print(f"\n{'#'*70}")
    print(f"  ABLATION: {period_name}")
    if start_date or end_date:
        print(f"  Period: {start_date or 'start'} → {end_date or 'end'}")
    print(f"{'#'*70}")

    # Load XU100 once
    loader = DataLoader(data_dir)
    xu100 = loader.load_xu100()
    prices = xu100['Close']

    # Filter by date if requested
    if start_date:
        prices = prices[prices.index >= start_date]
    if end_date:
        prices = prices[prices.index <= end_date]

    print(f"Price data: {prices.index[0].date()} → {prices.index[-1].date()} ({len(prices)} days)")

    modes = ['none', 'risk_index', 'usdtry_override', 'cds_gate', 'all']
    results = []

    for mode in modes:
        print(f"\n--- Running: {mode} ---")
        exp = MacroAugmentedRegime(macro_mode=mode)
        exp.classify(prices, data_dir=data_dir)
        r = exp.backtest(prices)
        r['period'] = period_name

        # Additional analysis: bad-market and good-market performance
        if mode != 'none' and results:
            baseline = results[0]  # 'none' is always first
            r['cagr_delta'] = r['cagr'] - baseline['cagr']
            r['sharpe_delta'] = r['sharpe'] - baseline['sharpe']
            r['maxdd_delta'] = r['max_dd'] - baseline['max_dd']

            # Count override days
            overrides = exp.get_override_dates()
            r['override_days'] = len(overrides)
        else:
            r['cagr_delta'] = 0
            r['sharpe_delta'] = 0
            r['maxdd_delta'] = 0
            r['override_days'] = 0

        results.append(r)

    return results


def run_walk_forward(data_dir: Path, split_date: str = '2020-01-01') -> dict:
    """
    Walk-forward test: train period vs test period.
    Shows if macro augmentation generalizes out-of-sample.
    """
    print(f"\n{'='*70}")
    print(f"WALK-FORWARD VALIDATION (split at {split_date})")
    print(f"{'='*70}")

    in_sample = run_ablation(data_dir, "In-Sample (2013–2019)",
                             end_date=split_date)
    out_sample = run_ablation(data_dir, "Out-of-Sample (2020–2026)",
                              start_date=split_date)

    return {'in_sample': in_sample, 'out_sample': out_sample}


def run_sensitivity(data_dir: Path) -> pd.DataFrame:
    """
    Sensitivity analysis: perturb each threshold by ±10%, ±20%.
    Check if Sharpe degrades >15% → fragile signal flag.
    """
    print(f"\n{'='*70}")
    print(f"SENSITIVITY ANALYSIS")
    print(f"{'='*70}")

    # Base config values
    base_configs = {
        'usdtry_crisis_threshold': 0.08,
        'cds_stress_percentile': 0.90,
        'risk_index_scale_factor': 0.50,
    }

    perturbations = [0.80, 0.90, 1.0, 1.10, 1.20]  # ±20%
    mode_map = {
        'usdtry_crisis_threshold': 'usdtry_override',
        'cds_stress_percentile': 'cds_gate',
        'risk_index_scale_factor': 'risk_index',
    }

    loader = DataLoader(data_dir)
    xu100 = loader.load_xu100()
    prices = xu100['Close']

    rows = []
    for param_name, base_val in base_configs.items():
        mode = mode_map[param_name]
        print(f"\n--- Perturbing {param_name} (base={base_val}) ---")

        for mult in perturbations:
            perturbed_val = base_val * mult
            config = MacroConfig(**{param_name: perturbed_val})

            exp = MacroAugmentedRegime(macro_mode=mode, config=config)
            exp.classify(prices, data_dir=data_dir)
            r = exp.backtest(prices)

            rows.append({
                'Parameter': param_name,
                'Multiplier': f"{mult:.0%}",
                'Value': f"{perturbed_val:.3f}",
                'Mode': mode,
                'CAGR': f"{r['cagr']*100:.1f}%",
                'Sharpe': r['sharpe'],
                'MaxDD': f"{r['max_dd']*100:.1f}%",
            })

    df = pd.DataFrame(rows)

    # Flag fragile: if ±20% perturbation causes >15% Sharpe degradation
    print("\n--- Fragility Check ---")
    for param in base_configs:
        param_rows = df[df['Parameter'] == param]
        base_sharpe = param_rows[param_rows['Multiplier'] == '100%']['Sharpe'].values[0]

        for _, row in param_rows.iterrows():
            if row['Multiplier'] in ('80%', '120%'):
                pct_change = abs(row['Sharpe'] - base_sharpe) / base_sharpe * 100
                status = "⚠️ FRAGILE" if pct_change > 15 else "✅ Robust"
                print(f"  {param}: {row['Multiplier']} → Sharpe {row['Sharpe']:.2f} "
                      f"({pct_change:.1f}% change) {status}")

    return df


def baseline_parity_test(data_dir: Path) -> bool:
    """
    CRITICAL TEST: Verify that macro_mode='none' produces identical
    results to running SimpleRegimeClassifier directly.
    """
    print("\n" + "=" * 70)
    print("BASELINE PARITY TEST")
    print("=" * 70)

    # 1. Run simple regime directly
    loader = DataLoader(data_dir)
    xu100 = loader.load_xu100()
    prices = xu100['Close']

    simple = SimpleRegimeClassifier()
    simple_regimes = simple.classify(prices)

    bt = SimpleBacktester(prices, simple_regimes)
    simple_results = bt.run()

    # 2. Run MacroAugmentedRegime with mode='none'
    aug = MacroAugmentedRegime(macro_mode='none')
    aug.classify(prices, data_dir=data_dir)
    aug_results = aug.backtest(prices)

    # 3. Compare
    regimes_match = (simple_regimes == aug.augmented_regimes).all()
    sharpe_match = abs(simple_results['sharpe_ratio'] - aug_results['sharpe']) < 1e-6
    cagr_match = abs(simple_results['annual_return'] - aug_results['cagr']) < 1e-6

    print(f"  Regimes identical:     {'✅ PASS' if regimes_match else '❌ FAIL'}")
    print(f"  Sharpe identical:      {'✅ PASS' if sharpe_match else '❌ FAIL'} "
          f"({simple_results['sharpe_ratio']:.6f} vs {aug_results['sharpe']:.6f})")
    print(f"  CAGR identical:        {'✅ PASS' if cagr_match else '❌ FAIL'} "
          f"({simple_results['annual_return']:.6f} vs {aug_results['cagr']:.6f})")

    passed = regimes_match and sharpe_match and cagr_match
    print(f"\n  Overall: {'✅ PARITY CONFIRMED' if passed else '❌ PARITY BROKEN!'}")
    return passed


# ============================================================================
# PRETTY PRINTING
# ============================================================================

def print_full_results(results: list):
    """Print detailed comparison table."""
    print_comparison(results)

    # Delta analysis
    if len(results) > 1:
        baseline = results[0]
        print("\nDelta vs Baseline:")
        print(f"{'Mode':<20} {'CAGR Δ':>10} {'Sharpe Δ':>10} {'MaxDD Δ':>10} {'Overrides':>10}")
        print("-" * 65)
        for r in results[1:]:
            print(f"{r['mode']:<20} "
                  f"{r['cagr_delta']*100:>+9.1f}% "
                  f"{r['sharpe_delta']:>+10.2f} "
                  f"{r['maxdd_delta']*100:>+9.1f}% "
                  f"{r['override_days']:>10}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Macro Regime Experiment Runner')
    parser.add_argument('--sensitivity', action='store_true',
                        help='Include sensitivity analysis')
    parser.add_argument('--walkforward', action='store_true',
                        help='Include walk-forward validation')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to data directory')
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else Path(__file__).resolve().parent.parent / "data"

    print("=" * 70)
    print("MACRO-AUGMENTED REGIME FILTER EXPERIMENTS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 1. Parity test (always runs first)
    parity_ok = baseline_parity_test(data_dir)
    if not parity_ok:
        print("\n⛔ ABORTING: Baseline parity broken. Fix before running experiments!")
        exit(1)

    # 2. Full ablation
    full_results = run_ablation(data_dir, "Full Period (2013–2026)")
    print_full_results(full_results)

    # 3. Walk-forward (optional)
    if args.walkforward:
        wf = run_walk_forward(data_dir)
        print("\n\nIN-SAMPLE RESULTS:")
        print_full_results(wf['in_sample'])
        print("\n\nOUT-OF-SAMPLE RESULTS:")
        print_full_results(wf['out_sample'])

    # 4. Sensitivity (optional)
    if args.sensitivity:
        sens = run_sensitivity(data_dir)
        print("\nSENSITIVITY TABLE:")
        print(sens.to_string(index=False))

    print(f"\n{'='*70}")
    print("EXPERIMENTS COMPLETE")
    print(f"{'='*70}")
