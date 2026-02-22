"""
Evaluate the Simple Regime Filter against baselines.
Runs automatically — no plots, just results.
"""

import logging
import warnings
from datetime import datetime

import pandas as pd
from simple_regime import DataLoader, SimpleBacktester, SimpleRegimeClassifier

logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


def evaluate():
    logger.info("=" * 70)
    logger.info("SIMPLE REGIME FILTER — FULL EVALUATION")
    logger.info("=" * 70)
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")

    # Load data
    loader = DataLoader()
    xu100 = loader.load_xu100()
    prices = xu100['Close']

    # ===================================================================
    # 1. Simple Regime Filter (our new model)
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("1. SIMPLE REGIME FILTER (200 MA + Vol)")
    logger.info("=" * 70)

    classifier = SimpleRegimeClassifier()
    regimes_simple = classifier.classify(prices)

    dist = classifier.get_distribution()
    for regime, row in dist.iterrows():
        logger.info(f"  {regime:10s}: {row['Count']:5.0f} ({row['Percent']:5.1f}%)")

    logger.info(f"  Transitions: {classifier.get_transitions()}")
    avg_dur = len(regimes_simple) / max(classifier.get_transitions(), 1)
    logger.info(f"  Avg duration: {avg_dur:.1f} days")

    backtester_simple = SimpleBacktester(prices, regimes_simple)
    results_simple = backtester_simple.run()
    quality_simple = backtester_simple.regime_quality()

    # ===================================================================
    # 2. Pure 200 MA (binary: above=100%, below=0%)
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("2. SIMPLE 200 MA (Pure Trend)")
    logger.info("=" * 70)

    ma_200 = prices.rolling(200).mean()
    regime_200ma = pd.Series('Bull', index=prices.index)
    regime_200ma[prices < ma_200] = 'Bear'
    regime_200ma = regime_200ma.loc[ma_200.dropna().index]

    transitions_200 = (regime_200ma != regime_200ma.shift(1)).sum()
    logger.info(f"  Transitions: {transitions_200}")
    logger.info(f"  Avg duration: {len(regime_200ma) / max(transitions_200, 1):.1f} days")

    alloc_200 = {'Bull': 1.0, 'Bear': 0.0}
    backtester_200 = SimpleBacktester(prices, regime_200ma, alloc_200)
    results_200 = backtester_200.run()

    # ===================================================================
    # 3. Simple 50 MA
    # ===================================================================
    logger.info("\n" + "=" * 70)
    logger.info("3. SIMPLE 50 MA")
    logger.info("=" * 70)

    ma_50 = prices.rolling(50).mean()
    regime_50ma = pd.Series('Bull', index=prices.index)
    regime_50ma[prices < ma_50] = 'Bear'
    regime_50ma = regime_50ma.loc[ma_50.dropna().index]

    transitions_50 = (regime_50ma != regime_50ma.shift(1)).sum()
    logger.info(f"  Transitions: {transitions_50}")
    logger.info(f"  Avg duration: {len(regime_50ma) / max(transitions_50, 1):.1f} days")

    alloc_50 = {'Bull': 1.0, 'Bear': 0.0}
    backtester_50 = SimpleBacktester(prices, regime_50ma, alloc_50)
    results_50 = backtester_50.run()

    # ===================================================================
    # 4. Buy & Hold
    # ===================================================================
    results_bh = backtester_simple.run_buy_and_hold()

    # ===================================================================
    # 5. Load old sophisticated filter for comparison (if available)
    # ===================================================================
    results_old = None
    try:
        import sys
        sys.path.insert(0, str(loader.data_dir.parent / "regime_filter"))
        from market_data import FeatureEngine
        from regime_models import RegimeClassifier
        from regime_models import SimplifiedRegimeClassifier as OldSimplified

        data_merged = pd.DataFrame({
            'XU100_Close': xu100['Close'],
            'XU100_Open': xu100.get('Open', xu100['Close']),
            'XU100_High': xu100.get('High', xu100['Close']),
            'XU100_Low': xu100.get('Low', xu100['Close']),
            'XU100_Volume': xu100.get('Volume', pd.Series(0, index=xu100.index)),
        })

        # Try to add USDTRY
        try:
            import yfinance as yf
            usdtry = yf.download('TRY=X', start=xu100.index[0] - pd.Timedelta(days=365),
                                 end=xu100.index[-1], progress=False)
            if not usdtry.empty:
                usdtry_close = usdtry['Close'] if 'Close' in usdtry.columns else usdtry
                if isinstance(usdtry_close, pd.DataFrame):
                    usdtry_close = usdtry_close.iloc[:, 0]
                data_merged['USDTRY'] = usdtry_close.reindex(data_merged.index).ffill()
        except Exception:
            pass

        engine = FeatureEngine(data_merged)
        engine.calculate_all_features()
        features = engine.shift_for_prediction(shift_days=1)

        old_classifier = RegimeClassifier(features)
        detailed = old_classifier.classify_all()
        old_simple = OldSimplified()
        old_regimes = old_simple.classify(detailed)['simplified_regime']

        alloc_old = {'Bull': 1.0, 'Bear': 0.2, 'Stress': 0.0, 'Choppy': 0.5, 'Recovery': 0.8}
        common = prices.index.intersection(old_regimes.index)
        backtester_old = SimpleBacktester(prices.loc[common], old_regimes.loc[common], alloc_old)
        results_old = backtester_old.run()
        quality_old = backtester_old.regime_quality()

        logger.info("\n" + "=" * 70)
        logger.info("5. OLD SOPHISTICATED FILTER (for comparison)")
        logger.info("=" * 70)
        old_transitions = (old_regimes != old_regimes.shift(1)).sum()
        logger.info(f"  Transitions: {old_transitions}")
        logger.info(f"  Avg duration: {len(old_regimes) / max(old_transitions, 1):.1f} days")
    except Exception as e:
        logger.info(f"\n(Could not load old filter for comparison: {e})")

    # ===================================================================
    # RESULTS COMPARISON
    # ===================================================================
    logger.info("\n\n" + "=" * 70)
    logger.info("RESULTS COMPARISON")
    logger.info("=" * 70)

    strategies = [
        ('Simple Regime (NEW)', results_simple),
        ('Pure 200 MA', results_200),
        ('Pure 50 MA', results_50),
        ('Buy & Hold', results_bh),
    ]
    if results_old is not None:
        strategies.append(('Old Sophisticated', results_old))

    # Sort by Sharpe
    strategies.sort(key=lambda x: x[1]['sharpe_ratio'], reverse=True)

    logger.info(f"\n{'Strategy':<25} {'Annual':>10} {'Sharpe':>8} {'MaxDD':>10} {'Vol':>8}")
    logger.info("-" * 63)
    for name, r in strategies:
        marker = " ←" if name == 'Simple Regime (NEW)' else ""
        logger.info(f"{name:<25} {r['annual_return']:>9.1%} "
              f"{r['sharpe_ratio']:>8.2f} {r['max_drawdown']:>9.1%} "
              f"{r['annual_volatility']:>7.1%}{marker}")

    # ===================================================================
    # REGIME QUALITY ANALYSIS
    # ===================================================================
    logger.info("\n\n" + "=" * 70)
    logger.info("REGIME QUALITY — Does it separate bull from bear?")
    logger.info("=" * 70)

    logger.info("\nSimple Regime Filter (NEW):")
    bull_ret = quality_simple.get('Bull', {}).get('avg_annual_return', 0)
    bear_ret = quality_simple.get('Bear', {}).get('avg_annual_return', 0)
    recovery_ret = quality_simple.get('Recovery', {}).get('avg_annual_return', 0)
    stress_ret = quality_simple.get('Stress', {}).get('avg_annual_return', 0)
    separation = bull_ret - bear_ret

    logger.info(f"  Bull     avg return: {bull_ret:>7.1%}  ({quality_simple.get('Bull', {}).get('pct', 0):>5.1f}% of time)")
    logger.info(f"  Recovery avg return: {recovery_ret:>7.1%}  ({quality_simple.get('Recovery', {}).get('pct', 0):>5.1f}% of time)")
    logger.info(f"  Bear     avg return: {bear_ret:>7.1%}  ({quality_simple.get('Bear', {}).get('pct', 0):>5.1f}% of time)")
    logger.info(f"  Stress   avg return: {stress_ret:>7.1%}  ({quality_simple.get('Stress', {}).get('pct', 0):>5.1f}% of time)")
    logger.info(f"\n  Bull-Bear separation: {separation:>7.1%}")
    if separation > 0.50:
        logger.info("  ✅ EXCELLENT")
    elif separation > 0.30:
        logger.info("  ✅ GOOD")
    elif separation > 0.10:
        logger.warning("  ⚠️  WEAK")
    else:
        logger.error("  ❌ BROKEN")

    if results_old is not None and quality_old:
        logger.info("\nOld Sophisticated Filter (for reference):")
        old_bull = quality_old.get('Bull', {}).get('avg_annual_return', 0)
        old_bear = quality_old.get('Bear', {}).get('avg_annual_return', 0)
        old_sep = old_bull - old_bear
        logger.info(f"  Bull avg return: {old_bull:>7.1%}")
        logger.info(f"  Bear avg return: {old_bear:>7.1%}")
        logger.info(f"  Separation:      {old_sep:>7.1%}")
        if old_sep > 0.30:
            logger.info("  ✅")
        elif old_sep > 0.10:
            logger.warning("  ⚠️")
        else:
            logger.error("  ❌")

    # ===================================================================
    # HEAD-TO-HEAD: New vs Old
    # ===================================================================
    if results_old is not None:
        logger.info("\n\n" + "=" * 70)
        logger.info("HEAD-TO-HEAD: New Simple vs Old Sophisticated")
        logger.info("=" * 70)

        metrics = [
            ('Annual Return', 'annual_return', True),
            ('Sharpe Ratio', 'sharpe_ratio', True),
            ('Max Drawdown', 'max_drawdown', False),
            ('Volatility', 'annual_volatility', False),
        ]

        for label, key, higher_better in metrics:
            new_val = results_simple[key]
            old_val = results_old[key]

            if key in ('annual_return', 'max_drawdown', 'annual_volatility'):
                new_str = f"{new_val:>8.1%}"
                old_str = f"{old_val:>8.1%}"
            else:
                new_str = f"{new_val:>8.2f}"
                old_str = f"{old_val:>8.2f}"

            if higher_better:
                winner = "NEW ✅" if new_val > old_val else "OLD"
            else:
                winner = "NEW ✅" if new_val > old_val else "OLD"
                # For drawdown, less negative = better
                if key == 'max_drawdown':
                    winner = "NEW ✅" if new_val > old_val else "OLD"
                # For vol, lower = better
                if key == 'annual_volatility':
                    winner = "NEW ✅" if new_val < old_val else "OLD"

            logger.info(f"  {label:<18} New: {new_str}  Old: {old_str}  → {winner}")

        # Improvement summary
        ret_improve = results_simple['annual_return'] - results_old['annual_return']
        sharpe_improve = results_simple['sharpe_ratio'] - results_old['sharpe_ratio']
        logger.info(f"\n  Return improvement:  {ret_improve:+.1%}")
        logger.info(f"  Sharpe improvement:  {sharpe_improve:+.2f}")

    # ===================================================================
    # CURRENT REGIME
    # ===================================================================
    current = classifier.get_current_regime()
    logger.info("\n\n" + "=" * 70)
    logger.info(f"CURRENT REGIME ({current['date'].date()})")
    logger.info("=" * 70)
    logger.info(f"  Regime:      {current['regime']}")
    logger.info(f"  Allocation:  {current['allocation']:.0%} stocks, {1-current['allocation']:.0%} gold")
    logger.info(f"  Above 200MA: {'Yes' if current['above_ma'] else 'No'}")
    logger.info(f"  Vol %ile:    {current['vol_percentile']:.0f}th percentile")
    logger.info(f"  Vol (20d):   {current['realized_vol']:.1%} annualized")

    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 70)

    return {
        'simple': results_simple,
        'ma_200': results_200,
        'ma_50': results_50,
        'buy_hold': results_bh,
        'old': results_old,
        'quality': quality_simple,
    }


if __name__ == "__main__":
    evaluate()
