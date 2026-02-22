"""
Size Factor Signal

Economic Intuition:
------------------
The size premium (SMB - Small Minus Big) is one of the original Fama-French factors.
Smaller companies tend to outperform larger companies over the long term due to:
1. Higher risk premium - small caps are less diversified, more volatile
2. Less analyst coverage - information asymmetry creates mispricing opportunities
3. Limited institutional ownership - less efficient pricing
4. Higher growth potential - easier to double revenue from a small base

Mathematical Construction:
-------------------------
Size Score = -log(Market Cap)

Negative because smaller = better (higher expected return).
Log transformation reduces skewness and makes cross-sectional comparison meaningful.

Input Data Requirements:
-----------------------
- Daily close prices
- Shares outstanding (for market cap calculation)

Normalization:
-------------
Cross-sectional z-score by date. The inverted log market cap is standardized
so that small caps have positive z-scores, large caps have negative.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .base import (
    FactorSignal,
    FactorData,
    FactorParams,
    cross_sectional_zscore,
)


@dataclass
class SizeParams:
    """Size-specific parameters."""
    use_enterprise_value: bool = False  # Use EV instead of market cap
    log_transform: bool = True          # Apply log transformation
    include_revenue_size: bool = False  # Include revenue-based size


class SizeSignal(FactorSignal):
    """
    Size factor: favors smaller companies (small-cap premium).

    Higher scores indicate smaller companies.
    """

    @property
    def name(self) -> str:
        return "size"

    @property
    def description(self) -> str:
        return (
            "Size premium factor. Smaller companies (by market cap) receive higher "
            "scores based on the empirical finding that small caps tend to outperform "
            "large caps over the long term (Fama-French SMB factor)."
        )

    @property
    def category(self) -> str:
        return "size"

    @property
    def higher_is_better(self) -> bool:
        # Higher score = smaller company = expected outperformance
        return True

    def compute_raw_signal(
        self,
        data: FactorData,
        params: FactorParams,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Compute raw size signal (inverted log market cap).

        Returns:
            Tuple of (raw_scores, metadata)
        """
        dates = data.dates
        tickers = data.tickers
        close = data.close.reindex(index=dates, columns=tickers)

        # Get size-specific params
        size_params = SizeParams(**params.custom) if params.custom else SizeParams()

        # Calculate market cap
        if data.market_cap is not None:
            market_cap = data.market_cap.reindex(index=dates, columns=tickers)
        elif data.shares_outstanding is not None:
            shares = data.shares_outstanding.reindex(index=dates, columns=tickers)
            market_cap = close * shares
        elif data.data_loader is not None:
            # Load shares outstanding
            shares_panel = self._load_shares_panel(data.data_loader, dates, tickers)
            market_cap = close * shares_panel
        else:
            raise ValueError("Size signal requires market_cap or shares_outstanding")

        # Apply log transformation
        if size_params.log_transform:
            size_metric = np.log(market_cap.replace(0, np.nan))
        else:
            size_metric = market_cap

        # INVERT: smaller = higher score
        raw_scores = -size_metric

        metadata = {
            "components": ["market_cap"],
            "log_transformed": size_params.log_transform,
            "inverted": True,
            "coverage_pct": float(raw_scores.notna().sum().sum()) / max(raw_scores.size, 1) * 100,
        }

        return raw_scores, metadata

    def _load_shares_panel(
        self,
        data_loader,
        dates: pd.DatetimeIndex,
        tickers: pd.Index,
    ) -> pd.DataFrame:
        """Load shares outstanding and align to dates/tickers."""
        try:
            shares = data_loader.load_shares_outstanding_panel()
            if shares is not None and not shares.empty:
                shares.index = pd.to_datetime(shares.index)
                shares = shares.sort_index()
                shares.columns = pd.Index([str(c).upper() for c in shares.columns])
                return shares.reindex(index=dates, columns=tickers).ffill()
        except Exception:
            pass
        return pd.DataFrame(np.nan, index=dates, columns=tickers)

    def get_default_params(self) -> FactorParams:
        """Return size-specific default parameters."""
        return FactorParams(
            lookback_days=1,  # Size is point-in-time
            lag_days=0,       # Market cap is real-time
            custom={"log_transform": True},
        )


# =============================================================================
# POTENTIAL IMPROVEMENTS
# =============================================================================
"""
1. Size-Momentum Interaction:
   - Small caps with positive momentum may have stronger returns
   - Could add conditional signal based on recent performance

2. Liquidity-Adjusted Size:
   - Adjust size for trading liquidity (illiquid small caps are riskier)
   - Size / Volume_Percentile provides liquidity-adjusted measure

3. Residual Size:
   - Orthogonalize size against other factors (value, momentum)
   - Captures "pure" size effect

4. Regime Conditioning:
   - Size premium is pro-cyclical (works in risk-on environments)
   - Could reduce weight in high-volatility regimes

5. Sector-Neutral Size:
   - Z-score within sectors before cross-sectional ranking
   - Avoids sector concentration in small-cap portfolios
"""
