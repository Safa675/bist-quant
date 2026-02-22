"""
Carry Factor Signal

Economic Intuition:
------------------
Carry measures the expected return from holding an asset assuming no
price change. In equities, carry primarily comes from:
1. Dividend Yield - cash returned to shareholders via dividends
2. Buyback Yield - cash returned via share repurchases

High-yield stocks tend to outperform because:
1. Dividends provide downside cushion
2. High yield often indicates undervaluation
3. Dividend payers are typically more mature, stable businesses
4. Buybacks signal management confidence and reduce share count

Mathematical Construction:
-------------------------
Carry Score = Dividend Yield (or Shareholder Yield if buyback data available)

Dividend Yield = Dividends TTM / Market Cap
Shareholder Yield = (Dividends + Buybacks) / Market Cap

Input Data Requirements:
-----------------------
- Dividends paid TTM
- Market cap (price × shares)
- Share buybacks (if available)

Normalization:
-------------
Cross-sectional z-score of yield.
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
class CarryParams:
    """Carry-specific parameters."""
    clip_yield_upper: float = 0.50   # Clip yield > 50% (suspicious)
    clip_yield_lower: float = 0.0    # No negative yields
    reporting_lag_days: int = 45


class CarrySignal(FactorSignal):
    """
    Carry (Dividend Yield) factor: favors high-yield stocks.

    Higher scores indicate higher dividend yield.
    """

    @property
    def name(self) -> str:
        return "carry"

    @property
    def description(self) -> str:
        return (
            "Carry/Dividend yield factor. Stocks with higher dividend yields "
            "receive higher scores. Based on the empirical finding that high-yield "
            "stocks tend to outperform, providing return even without price appreciation."
        )

    @property
    def category(self) -> str:
        return "carry"

    @property
    def higher_is_better(self) -> bool:
        return True  # Higher yield = higher expected carry return

    def compute_raw_signal(
        self,
        data: FactorData,
        params: FactorParams,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Compute dividend yield signal.

        Returns:
            Tuple of (raw_scores, metadata)
        """
        dates = data.dates
        tickers = data.tickers

        carry_params = CarryParams(**params.custom) if params.custom else CarryParams()

        # Build dividend yield panel
        div_yield_panel = self._build_dividend_yield_panel(data, carry_params)

        # Clip extreme values
        raw_scores = div_yield_panel.clip(
            lower=carry_params.clip_yield_lower,
            upper=carry_params.clip_yield_upper
        )

        metadata = {
            "components": ["dividend_yield"],
            "yield_clip": (carry_params.clip_yield_lower, carry_params.clip_yield_upper),
            "coverage_pct": float(raw_scores.notna().sum().sum()) / max(raw_scores.size, 1) * 100,
        }

        return raw_scores, metadata

    def _build_dividend_yield_panel(
        self,
        data: FactorData,
        carry_params: CarryParams,
    ) -> pd.DataFrame:
        """Build dividend yield panel from fundamental data."""
        dates = data.dates
        tickers = data.tickers
        close = data.close.reindex(index=dates, columns=tickers)

        div_yield_panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)

        # Try to load from fundamental metrics first
        if data.data_loader is not None:
            try:
                metrics_df = data.data_loader.load_fundamental_metrics()
                if not metrics_df.empty and "dividend_yield" in metrics_df.columns:
                    available_tickers = set(metrics_df.index.get_level_values(0))
                    for ticker in tickers:
                        if ticker not in available_tickers:
                            continue
                        try:
                            series = metrics_df.loc[ticker, "dividend_yield"]
                            if isinstance(series, pd.DataFrame):
                                series = series.iloc[:, 0]
                            series = series.sort_index()
                            series = series[~series.index.duplicated(keep="last")]
                            series.index = pd.to_datetime(series.index)
                            div_yield_panel[ticker] = series.reindex(dates).ffill()
                        except Exception:
                            continue

                    # If we got decent coverage, use it
                    if div_yield_panel.notna().sum().sum() > 1000:
                        return div_yield_panel
            except Exception:
                pass

        # Fallback: compute from cash flow statement
        if data.data_loader is not None:
            try:
                fundamentals_parquet = data.data_loader.load_fundamentals_parquet()
                if fundamentals_parquet is not None:
                    div_yield_panel = self._compute_from_cash_flow(
                        data, fundamentals_parquet, carry_params
                    )
            except Exception:
                pass

        return div_yield_panel

    def _compute_from_cash_flow(
        self,
        data: FactorData,
        fundamentals_parquet: pd.DataFrame,
        carry_params: CarryParams,
    ) -> pd.DataFrame:
        """Compute dividend yield from cash flow statements."""
        dates = data.dates
        tickers = data.tickers
        close = data.close.reindex(index=dates, columns=tickers)

        div_yield_panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype=float)

        # Get market cap
        if data.market_cap is not None:
            market_cap = data.market_cap.reindex(index=dates, columns=tickers)
        elif data.shares_outstanding is not None:
            shares = data.shares_outstanding.reindex(index=dates, columns=tickers)
            market_cap = close * shares
        else:
            return div_yield_panel

        try:
            from ..factor_builders import (
                CASH_FLOW_SHEET,
                DIVIDENDS_PAID_KEYS,
            )
            from ...common.utils import (
                get_consolidated_sheet,
                pick_row_from_sheet,
                coerce_quarter_cols,
                sum_ttm,
                apply_lag,
            )
        except ImportError:
            return div_yield_panel

        for ticker in tickers:
            try:
                cf = get_consolidated_sheet(fundamentals_parquet, ticker, CASH_FLOW_SHEET)
                if cf.empty:
                    continue

                div_row = pick_row_from_sheet(cf, DIVIDENDS_PAID_KEYS)
                if div_row is None:
                    continue

                divs = coerce_quarter_cols(div_row).abs()
                if divs.empty:
                    continue

                div_ttm = sum_ttm(divs)
                if div_ttm.empty:
                    continue

                # Apply lag
                div_ttm_lagged = apply_lag(div_ttm, dates)
                if div_ttm_lagged.empty or div_ttm_lagged.isna().all():
                    continue

                # Market cap for this ticker
                mcap = market_cap[ticker]

                # Dividend yield
                div_yield = div_ttm_lagged / mcap.replace(0, np.nan)
                div_yield = div_yield.replace([np.inf, -np.inf], np.nan)

                div_yield_panel[ticker] = div_yield

            except Exception:
                continue

        return div_yield_panel

    def get_default_params(self) -> FactorParams:
        """Return carry-specific default parameters."""
        return FactorParams(
            lookback_days=1,
            lag_days=0,
            winsorize_pct=2.0,
            custom={
                "clip_yield_upper": 0.50,
                "clip_yield_lower": 0.0,
            },
        )


class ShareholderYieldSignal(FactorSignal):
    """
    Shareholder Yield: combines dividends and buybacks.

    More comprehensive measure of cash returned to shareholders.
    Note: Buyback data may not be available for all markets.
    """

    @property
    def name(self) -> str:
        return "shareholder_yield"

    @property
    def description(self) -> str:
        return (
            "Shareholder yield factor. Combines dividend yield and buyback yield. "
            "More comprehensive measure of total cash returned to shareholders."
        )

    @property
    def category(self) -> str:
        return "carry"

    @property
    def higher_is_better(self) -> bool:
        return True

    def compute_raw_signal(
        self,
        data: FactorData,
        params: FactorParams,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Compute shareholder yield (dividends + buybacks)."""
        # For most markets, this falls back to dividend yield
        # as buyback data is often not available
        carry_signal = CarrySignal()
        return carry_signal.compute_raw_signal(data, params)


# =============================================================================
# POTENTIAL IMPROVEMENTS
# =============================================================================
"""
1. Net Payout Yield:
   - (Dividends + Buybacks - Issuance) / Market Cap
   - Accounts for dilution from new shares

2. Dividend Growth:
   - YoY change in dividend per share
   - Growing dividends indicate confidence

3. Payout Ratio:
   - Dividends / Earnings
   - Sustainable payout < 60-70%

4. Dividend Consistency:
   - Number of consecutive years of dividend payment
   - Stability indicator

5. Free Cash Flow Yield:
   - FCF / Market Cap
   - Capacity to pay dividends

6. Expected Dividend Growth:
   - Based on retention ratio and ROE
   - Forward-looking yield estimate
"""
