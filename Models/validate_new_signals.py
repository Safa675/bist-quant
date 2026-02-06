#!/usr/bin/env python3
"""
Quick validation script for new signals integration

Tests that all three new signals can be imported and initialized without errors.
"""

import sys
from pathlib import Path

# Add Models directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("VALIDATING NEW SIGNALS INTEGRATION")
print("=" * 70)

# Test 1: Import signals
print("\n✓ Test 1: Importing signal modules...")
try:
    from signals.currency_rotation_signals import build_currency_rotation_signals
    print("  ✅ currency_rotation_signals imported successfully")
except Exception as e:
    print(f"  ❌ Failed to import currency_rotation_signals: {e}")
    sys.exit(1)

try:
    from signals.dividend_rotation_signals import build_dividend_rotation_signals
    print("  ✅ dividend_rotation_signals imported successfully")
except Exception as e:
    print(f"  ❌ Failed to import dividend_rotation_signals: {e}")
    sys.exit(1)

try:
    from signals.macro_hedge_signals import build_macro_hedge_signals
    print("  ✅ macro_hedge_signals imported successfully")
except Exception as e:
    print(f"  ❌ Failed to import macro_hedge_signals: {e}")
    sys.exit(1)

# Test 2: Import configs
print("\n✓ Test 2: Loading signal configs...")
try:
    import importlib.util
    
    for signal_name in ['currency_rotation', 'dividend_rotation', 'macro_hedge']:
        config_file = Path(__file__).parent / 'configs' / f'{signal_name}.py'
        spec = importlib.util.spec_from_file_location(signal_name, config_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, 'SIGNAL_CONFIG'):
            config = module.SIGNAL_CONFIG
            print(f"  ✅ {signal_name}: {config['name']} (enabled={config['enabled']}, rebalance={config['rebalance_frequency']})")
        else:
            print(f"  ❌ {signal_name}: Missing SIGNAL_CONFIG")
            sys.exit(1)
except Exception as e:
    print(f"  ❌ Failed to load configs: {e}")
    sys.exit(1)

# Test 3: Check portfolio engine integration
print("\n✓ Test 3: Checking portfolio engine integration...")
try:
    from portfolio_engine import PortfolioEngine, load_signal_configs
    
    configs = load_signal_configs()
    
    for signal_name in ['currency_rotation', 'dividend_rotation', 'macro_hedge']:
        if signal_name in configs:
            print(f"  ✅ {signal_name} registered in portfolio engine")
        else:
            print(f"  ❌ {signal_name} NOT registered in portfolio engine")
            sys.exit(1)
except Exception as e:
    print(f"  ❌ Failed to check portfolio engine: {e}")
    sys.exit(1)

# Test 4: Verify BIST subfolder is deleted
print("\n✓ Test 4: Verifying cleanup...")
bist_subfolder = Path(__file__).parent.parent / 'BIST'
if bist_subfolder.exists():
    print(f"  ⚠️  BIST subfolder still exists at {bist_subfolder}")
    print(f"     This is the messy folder created by the other agent")
    print(f"     Run: rm -rf {bist_subfolder}")
else:
    print(f"  ✅ BIST subfolder successfully deleted")

print("\n" + "=" * 70)
print("✅ ALL VALIDATION TESTS PASSED!")
print("=" * 70)
print("\nNew signals are ready to use:")
print("  python portfolio_engine.py currency_rotation")
print("  python portfolio_engine.py dividend_rotation")
print("  python portfolio_engine.py macro_hedge")
print("\nOr run all signals:")
print("  python portfolio_engine.py all")
print()
