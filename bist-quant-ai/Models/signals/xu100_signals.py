"""
XU100 Index Signal Construction

This is a benchmark signal that simply invests in XU100 index.
The portfolio engine will apply:
- Regime awareness (reduce/exit in Bear/Stress)
- Volatility targeting
- XAU/TRY fallback in bad regimes

This lets us compare "enhanced XU100" vs "buy and hold XU100".
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


def build_xu100_signals(
    close_df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    data_loader=None,
) -> pd.DataFrame:
    """
    Build XU100 signal - always returns XU100 as the only "stock" to hold.
    
    The portfolio engine uses this signal to decide what to hold.
    Since we only have XU100, the regime filter and risk management
    will be the only active components.
    
    Args:
        close_df: DataFrame of close prices (Date x Ticker)
        dates: DatetimeIndex to align signals to
        data_loader: Optional DataLoader instance
    
    Returns:
        DataFrame (dates x tickers) with constant score for XU100
    """
    print("\n🔧 Building XU100 benchmark signal...")
    
    # Create a signal panel where only XU100 has a score
    # All other tickers will be NaN (excluded)
    result = pd.DataFrame(index=dates, columns=['XU100'])
    
    # Give XU100 a constant score of 1.0 on all dates
    # (The actual value doesn't matter since it's the only stock)
    result['XU100'] = 1.0
    
    print(f"  ✅ XU100 signal: {result.shape[0]} days × 1 ticker (XU100 only)")
    print(f"  📊 This signal invests 100% in XU100 when regime allows")
    
    return result
