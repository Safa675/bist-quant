"""
Base Classes for Standalone Factor Signals

Provides a consistent interface for all factor implementations.
Each factor inherits from FactorSignal and implements compute_raw_signal().
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


class NormalizationMethod(Enum):
    """How to normalize raw signal values cross-sectionally."""
    ZSCORE = "zscore"              # (x - mean) / std
    RANK = "rank"                  # Percentile rank [0, 100]
    MINMAX = "minmax"              # (x - min) / (max - min) -> [0, 1]
    ROBUST_ZSCORE = "robust_zscore"  # (x - median) / MAD
    RAW = "raw"                    # No normalization


class SelectionMethod(Enum):
    """How to generate actionable selections from scores."""
    TOP_N = "top_n"                # Select top N stocks
    TOP_PCT = "top_pct"            # Select top X%
    THRESHOLD = "threshold"        # Score above threshold
    DECILE_LONG_SHORT = "decile"   # Long top decile, short bottom decile
    QUINTILE = "quintile"          # Quintile-based buckets


@dataclass
class FactorParams:
    """
    Parameters for factor signal computation.

    Attributes:
        normalization: How to normalize the raw signal
        winsorize_pct: Winsorize extreme values at this percentile (0-50)
        min_observations: Minimum observations required for valid signal
        lookback_days: Primary lookback window (factor-specific meaning)
        decay_halflife: Half-life for exponential weighting (days)
        lag_days: Days to lag signal to prevent lookahead
        custom: Factor-specific parameters
    """
    normalization: NormalizationMethod = NormalizationMethod.ZSCORE
    winsorize_pct: float = 1.0  # Winsorize at 1st/99th percentile
    min_observations: int = 20
    lookback_days: int = 252
    decay_halflife: Optional[int] = None
    lag_days: int = 0
    custom: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FactorData:
    """
    Input data container for factor computation.

    Attributes:
        close: Daily close prices (dates x tickers)
        volume: Daily volume (dates x tickers)
        fundamentals: Dict of fundamental data per ticker
        fundamentals_parquet: Consolidated fundamental data
        shares_outstanding: Shares outstanding panel (dates x tickers)
        market_cap: Market cap panel (dates x tickers)
        dates: DatetimeIndex of trading days
        tickers: Index of ticker symbols
        data_loader: Optional data loader for additional data
    """
    close: pd.DataFrame
    volume: Optional[pd.DataFrame] = None
    fundamentals: Optional[Dict] = None
    fundamentals_parquet: Optional[pd.DataFrame] = None
    shares_outstanding: Optional[pd.DataFrame] = None
    market_cap: Optional[pd.DataFrame] = None
    dates: Optional[pd.DatetimeIndex] = None
    tickers: Optional[pd.Index] = None
    data_loader: Optional[Any] = None

    def __post_init__(self):
        """Derive dates and tickers from close if not provided."""
        if self.dates is None:
            self.dates = self.close.index
        if self.tickers is None:
            self.tickers = self.close.columns


@dataclass
class SignalOutput:
    """
    Output container for factor signal computation.

    Attributes:
        scores: Normalized cross-sectional scores (dates x tickers)
        raw_scores: Raw unnormalized scores before transformation
        metadata: Additional diagnostic information
        component_scores: Individual component scores if factor is composite
    """
    scores: pd.DataFrame
    raw_scores: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    component_scores: Optional[Dict[str, pd.DataFrame]] = None


class FactorSignal(ABC):
    """
    Abstract base class for all factor signals.

    Each factor must implement:
    - name: Factor identifier
    - description: Economic intuition
    - compute_raw_signal: Core signal computation logic

    The base class provides:
    - Normalization (z-score, rank, etc.)
    - Winsorization
    - Selection methods (top N, threshold, etc.)
    - Standardized output format
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique factor identifier."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Economic intuition and rationale."""
        pass

    @property
    def category(self) -> str:
        """Factor category (value, momentum, quality, etc.)."""
        return "general"

    @property
    def higher_is_better(self) -> bool:
        """Whether higher raw values indicate stronger signal."""
        return True

    @abstractmethod
    def compute_raw_signal(
        self,
        data: FactorData,
        params: FactorParams,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Compute raw factor signal before normalization.

        Args:
            data: Input data container
            params: Factor parameters

        Returns:
            Tuple of (raw_scores DataFrame, metadata dict)
        """
        pass

    def compute_signal(
        self,
        data: FactorData,
        params: Optional[FactorParams] = None,
    ) -> SignalOutput:
        """
        Compute normalized factor signal with full output.

        This is the main entry point for factor computation.

        Args:
            data: Input data container
            params: Factor parameters (uses defaults if None)

        Returns:
            SignalOutput with normalized scores and metadata
        """
        if params is None:
            params = FactorParams()

        # Compute raw signal
        raw_scores, metadata = self.compute_raw_signal(data, params)

        # Apply lag if specified
        if params.lag_days > 0:
            raw_scores = raw_scores.shift(params.lag_days)

        # Winsorize extreme values
        if params.winsorize_pct > 0:
            raw_scores = self._winsorize(raw_scores, params.winsorize_pct)

        # Normalize
        normalized = self._normalize(raw_scores, params.normalization)

        # Add factor metadata
        metadata.update({
            "factor_name": self.name,
            "normalization": params.normalization.value,
            "higher_is_better": self.higher_is_better,
            "n_dates": len(normalized.index),
            "n_tickers": len(normalized.columns),
            "coverage": float(normalized.notna().sum().sum()) / max(normalized.size, 1),
        })

        return SignalOutput(
            scores=normalized,
            raw_scores=raw_scores,
            metadata=metadata,
        )

    def _winsorize(
        self,
        panel: pd.DataFrame,
        pct: float,
    ) -> pd.DataFrame:
        """Winsorize panel at specified percentile (cross-sectionally per date)."""
        def winsorize_row(row):
            valid = row.dropna()
            if len(valid) < 3:
                return row
            lower = np.nanpercentile(valid, pct)
            upper = np.nanpercentile(valid, 100 - pct)
            return row.clip(lower=lower, upper=upper)

        return panel.apply(winsorize_row, axis=1)

    def _normalize(
        self,
        panel: pd.DataFrame,
        method: NormalizationMethod,
    ) -> pd.DataFrame:
        """Apply cross-sectional normalization per date."""
        if method == NormalizationMethod.RAW:
            return panel

        if method == NormalizationMethod.ZSCORE:
            row_mean = panel.mean(axis=1)
            row_std = panel.std(axis=1).replace(0.0, np.nan)
            return panel.sub(row_mean, axis=0).div(row_std, axis=0)

        if method == NormalizationMethod.RANK:
            # Percentile rank [0, 100]
            ranks = panel.rank(axis=1, pct=True, method="average")
            if not self.higher_is_better:
                ranks = 1.0 - ranks
            return ranks * 100.0

        if method == NormalizationMethod.MINMAX:
            row_min = panel.min(axis=1)
            row_max = panel.max(axis=1)
            row_range = (row_max - row_min).replace(0.0, np.nan)
            return panel.sub(row_min, axis=0).div(row_range, axis=0)

        if method == NormalizationMethod.ROBUST_ZSCORE:
            row_median = panel.median(axis=1)
            row_mad = panel.sub(row_median, axis=0).abs().median(axis=1)
            row_mad = row_mad.replace(0.0, np.nan) * 1.4826  # Scale to std equiv
            return panel.sub(row_median, axis=0).div(row_mad, axis=0)

        return panel

    def select_stocks(
        self,
        scores: pd.DataFrame,
        method: SelectionMethod = SelectionMethod.TOP_N,
        n: int = 10,
        pct: float = 10.0,
        threshold: float = 0.0,
    ) -> pd.DataFrame:
        """
        Generate stock selection mask from scores.

        Args:
            scores: Normalized scores panel
            method: Selection method
            n: Number of stocks (for TOP_N)
            pct: Percentage (for TOP_PCT)
            threshold: Score threshold (for THRESHOLD)

        Returns:
            Boolean DataFrame (True = selected)
        """
        if method == SelectionMethod.TOP_N:
            def top_n_row(row):
                valid = row.dropna().nlargest(n).index
                return row.index.isin(valid)
            return scores.apply(top_n_row, axis=1)

        if method == SelectionMethod.TOP_PCT:
            cutoff = scores.quantile(1.0 - pct / 100.0, axis=1)
            return scores.ge(cutoff, axis=0) & scores.notna()

        if method == SelectionMethod.THRESHOLD:
            return (scores >= threshold) & scores.notna()

        if method == SelectionMethod.DECILE_LONG_SHORT:
            top_cutoff = scores.quantile(0.9, axis=1)
            bottom_cutoff = scores.quantile(0.1, axis=1)
            long_mask = scores.ge(top_cutoff, axis=0)
            short_mask = scores.le(bottom_cutoff, axis=0)
            # Return +1 for long, -1 for short, 0 for neutral
            return long_mask.astype(int) - short_mask.astype(int)

        if method == SelectionMethod.QUINTILE:
            # Return quintile assignment (1-5)
            return (scores.rank(axis=1, pct=True) * 5).clip(1, 5).round()

        raise ValueError(f"Unknown selection method: {method}")

    def get_default_params(self) -> FactorParams:
        """Return factor-specific default parameters."""
        return FactorParams()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def cross_sectional_zscore(panel: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score by date (convenience function)."""
    row_mean = panel.mean(axis=1)
    row_std = panel.std(axis=1).replace(0.0, np.nan)
    return panel.sub(row_mean, axis=0).div(row_std, axis=0)


def cross_sectional_rank(
    panel: pd.DataFrame,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """Cross-sectional percentile rank by date, scaled to 0-100."""
    ranks = panel.rank(axis=1, pct=True, method="average")
    if not higher_is_better:
        ranks = 1.0 - ranks
    return (ranks * 100.0).where(panel.notna())


def rolling_cumulative_return(
    daily_returns: pd.Series,
    lookback: int,
    min_obs: Optional[int] = None,
) -> pd.Series:
    """Compute rolling compounded returns using log-sum for stability."""
    if min_obs is None:
        min_obs = max(int(lookback * 0.5), 10)
    clipped = daily_returns.clip(lower=-0.99)
    log_growth = np.log1p(clipped)
    roll_log_sum = log_growth.rolling(lookback, min_periods=min_obs).sum()
    return np.expm1(roll_log_sum)


def combine_components_zscore(
    components: list,
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
) -> pd.DataFrame:
    """
    Combine multiple z-scored components using proper NaN handling.

    Computes mean only over available components for each cell.

    Args:
        components: List of (name, DataFrame) tuples
        dates: Target date index
        tickers: Target ticker columns

    Returns:
        Combined z-score panel
    """
    if not components:
        return pd.DataFrame(np.nan, index=dates, columns=tickers)

    aligned = [comp.reindex(index=dates, columns=tickers) for _, comp in components]
    stacked = np.stack([df.values for df in aligned], axis=0)

    with np.errstate(all='ignore'):
        result_values = np.nanmean(stacked, axis=0)

    all_nan_mask = np.all(np.isnan(stacked), axis=0)
    result_values[all_nan_mask] = np.nan

    return pd.DataFrame(result_values, index=dates, columns=tickers)
