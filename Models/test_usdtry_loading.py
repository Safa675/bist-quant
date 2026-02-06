#!/usr/bin/env python3
"""
Test USD/TRY data loading
"""

import sys
from pathlib import Path

# Add Models directory to path
sys.path.insert(0, str(Path(__file__).parent))

from common.data_loader import DataLoader

print("=" * 70)
print("TESTING USD/TRY DATA LOADING")
print("=" * 70)

# Initialize DataLoader
data_dir = Path("/home/safa/Documents/Markets/BIST/data")
regime_model_dir = Path("/home/safa/Documents/Markets/BIST/Regime Filter/outputs/ensemble_model")

loader = DataLoader(data_dir, regime_model_dir)

# Test USD/TRY loading
print("\n✓ Testing USD/TRY data loading...")
try:
    usdtry_df = loader.load_usdtry()
    
    if usdtry_df.empty:
        print("  ❌ USD/TRY data is empty")
    else:
        print(f"  ✅ Loaded {len(usdtry_df)} USD/TRY observations")
        print(f"  Columns: {usdtry_df.columns.tolist()}")
        print(f"  Date range: {usdtry_df.index.min()} to {usdtry_df.index.max()}")
        print(f"\n  Sample data:")
        print(usdtry_df.head())
        
except Exception as e:
    print(f"  ❌ Error loading USD/TRY: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
