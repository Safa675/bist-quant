from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from bist_quant.common.config_manager import (
    DEFAULT_PORTFOLIO_OPTIONS,
    INVERSE_VOL_LOOKBACK,
    LIQUIDITY_QUANTILE,
    MAX_POSITION_WEIGHT,
    TARGET_DOWNSIDE_VOL,
    VOL_CAP,
    VOL_FLOOR,
    VOL_LOOKBACK,
)
from bist_quant.common.market_cap_utils import SIZE_LIQUIDITY_QUANTILE, get_size_buckets_for_date

logger = logging.getLogger(__name__)


class RiskManager:
    """Apply portfolio-level risk controls for sizing, drawdown, and execution.

    Args:
        close_df: Optional close-price panel (Date x Ticker).
        volume_df: Optional volume panel (Date x Ticker).

    Attributes:
        close_df: Price panel used by volatility-based weighting methods.
        volume_df: Volume panel used for liquidity-aware filtering.
    """

    def __init__(
        self,
        close_df: pd.DataFrame | None = None,
        volume_df: pd.DataFrame | None = None,
    ) -> None:
        self.close_df = close_df
        self.volume_df = volume_df

    def set_data(
        self,
        close_df: pd.DataFrame,
        volume_df: pd.DataFrame,
    ) -> None:
        self.close_df = close_df
        self.volume_df = volume_df

    @staticmethod
    def resolve_options(portfolio_options: dict | None) -> dict:
        opts = DEFAULT_PORTFOLIO_OPTIONS.copy()
        if portfolio_options is None:
            return opts
        if not isinstance(portfolio_options, dict):
            raise TypeError(
                f"portfolio_options must be dict or None, got {type(portfolio_options).__name__}"
            )
        opts.update(portfolio_options)
        return opts

    @staticmethod
    def print_settings(opts: dict) -> None:
        logger.info("\n🔧 Portfolio Engineering Settings:")
        logger.info(f"   Regime Filter: {'ON' if opts['use_regime_filter'] else 'OFF'}")
        logger.info(
            f"   Vol Targeting: {'ON (' + str(int(opts['target_downside_vol']*100)) + '%)' if opts['use_vol_targeting'] else 'OFF'}"
        )
        logger.info(f"   Inverse Vol Sizing: {'ON' if opts['use_inverse_vol_sizing'] else 'OFF'}")
        logger.info(
            f"   Stop Loss: {'ON (' + str(int(opts['stop_loss_threshold']*100)) + '%)' if opts['use_stop_loss'] else 'OFF'}"
        )
        logger.info(f"   Liquidity Filter: {'ON' if opts['use_liquidity_filter'] else 'OFF'}")
        if opts["use_slippage"]:
            if opts.get("use_mcap_slippage", True):
                logger.info(
                    f"   Slippage: ON (Large: {opts['slippage_bps']} bps, Mid: {opts.get('mid_cap_slippage_bps', 10.0)} bps, Small: {opts.get('small_cap_slippage_bps', 20.0)} bps)"
                )
            else:
                logger.info(f"   Slippage: ON ({opts['slippage_bps']} bps flat)")
        else:
            logger.info("   Slippage: OFF")
        logger.info(f"   Top N Stocks: {opts['top_n']}")
        logger.info(f"   Signal Lag Days: {int(opts.get('signal_lag_days', 1))}")

    def filter_by_liquidity(
        self,
        tickers: list[str],
        date: pd.Timestamp,
        liquidity_quantile: float = LIQUIDITY_QUANTILE,
    ) -> list[str]:
        if self.volume_df is None:
            return tickers

        if date not in self.volume_df.index:
            candidates = self.volume_df.index[self.volume_df.index <= date]
            if candidates.empty:
                return tickers
            date = candidates.max()

        adv = self.volume_df.loc[date, [t for t in tickers if t in self.volume_df.columns]].dropna()
        if adv.empty:
            return tickers

        threshold = adv.quantile(liquidity_quantile)
        liquid = set(adv[adv >= threshold].index)
        return [ticker for ticker in tickers if ticker in liquid]

    def inverse_downside_vol_weights(
        self,
        selected: list[str],
        date: pd.Timestamp,
        lookback: int = INVERSE_VOL_LOOKBACK,
        max_weight: float = MAX_POSITION_WEIGHT,
    ) -> pd.Series:
        if self.close_df is None:
            return pd.Series(1.0 / len(selected), index=selected)

        if date not in self.close_df.index:
            return pd.Series(1.0 / len(selected), index=selected)

        idx = self.close_df.index.get_loc(date)
        if idx < lookback:
            return pd.Series(1.0 / len(selected), index=selected)

        window_data = self.close_df.iloc[idx - lookback : idx][selected]
        returns = window_data.pct_change().dropna()

        downside_vols = []
        for ticker in selected:
            if ticker in returns.columns:
                ticker_rets = returns[ticker].dropna()
                downside_rets = ticker_rets[ticker_rets < 0]
                downside_vol = downside_rets.std() if len(downside_rets) > 2 else np.nan
            else:
                downside_vol = np.nan
            downside_vols.append(downside_vol)

        downside_vol_series = pd.Series(downside_vols, index=selected)
        inv = 1.0 / downside_vol_series.replace(0, np.nan)
        median_inv = inv.median()
        if pd.isna(median_inv) or median_inv == 0:
            return pd.Series(1.0 / len(selected), index=selected)

        inv = inv.fillna(median_inv)
        weights = inv / inv.sum()
        weights = weights.clip(upper=max_weight)
        weights = weights / weights.sum()
        return weights

    @staticmethod
    def apply_downside_vol_targeting(
        returns: pd.Series,
        target_vol: float = TARGET_DOWNSIDE_VOL,
        lookback: int = VOL_LOOKBACK,
        vol_floor: float = VOL_FLOOR,
        vol_cap: float = VOL_CAP,
    ) -> pd.Series:
        if len(returns) < lookback:
            return returns

        min_periods = lookback // 2
        negative_only = returns.where(returns < 0.0)
        total_counts = returns.rolling(lookback, min_periods=min_periods).count()
        negative_counts = negative_only.rolling(lookback, min_periods=1).count()
        rolling_downside_vol = negative_only.rolling(lookback, min_periods=1).std() * np.sqrt(252)
        rolling_downside_vol = rolling_downside_vol.where(
            (total_counts >= min_periods) & (negative_counts > 2),
        )
        leverage = target_vol / rolling_downside_vol.shift(1)
        leverage = leverage.clip(lower=vol_floor, upper=vol_cap)
        leverage = leverage.fillna(1.0)
        return returns * leverage

    @staticmethod
    def apply_stop_loss(
        current_holdings: list[str],
        stopped_out: set[str],
        entry_prices: dict[str, float],
        open_df: pd.DataFrame,
        date: pd.Timestamp,
        stop_loss_threshold: float,
    ) -> list[str]:
        holdings_to_keep = []
        for ticker in current_holdings:
            if ticker in stopped_out:
                continue
            if ticker not in entry_prices:
                holdings_to_keep.append(ticker)
                continue

            entry = entry_prices[ticker]
            current_price = (
                open_df.loc[date, ticker]
                if date in open_df.index and ticker in open_df.columns
                else np.nan
            )

            if pd.notna(current_price) and pd.notna(entry) and entry > 0:
                drawdown = (current_price / entry) - 1.0
                if drawdown < -stop_loss_threshold:
                    stopped_out.add(ticker)
                    continue

            holdings_to_keep.append(ticker)
        return holdings_to_keep

    @staticmethod
    def slippage_cost_bps(
        date: pd.Timestamp,
        new_positions: list[str],
        opts: dict,
        use_mcap_slippage: bool,
        mcap_slippage_panel: pd.DataFrame | None,
        mcap_slippage_liquidity: pd.DataFrame | None,
        small_cap_slippage_bps: float,
        mid_cap_slippage_bps: float,
    ) -> float:
        if not new_positions:
            return float(opts["slippage_bps"])

        if (
            use_mcap_slippage
            and mcap_slippage_panel is not None
            and date in mcap_slippage_panel.index
        ):
            mcaps = mcap_slippage_panel.loc[date]
            liq = (
                mcap_slippage_liquidity.loc[date]
                if mcap_slippage_liquidity is not None and date in mcap_slippage_liquidity.index
                else pd.Series(dtype=float)
            )
            _, small_caps, _ = get_size_buckets_for_date(
                mcaps,
                liq,
                liquidity_quantile=SIZE_LIQUIDITY_QUANTILE,
            )

            def get_stock_slippage(ticker: str) -> float:
                if ticker in small_caps:
                    return small_cap_slippage_bps
                if ticker in mcaps.index and not mcaps[ticker] != mcaps[ticker]:
                    mcap_pct = mcaps.rank(pct=True).get(ticker, 0.5)
                    if mcap_pct < 0.7:
                        return mid_cap_slippage_bps
                return float(opts["slippage_bps"])

            return float(np.mean([get_stock_slippage(ticker) for ticker in new_positions]))

        return float(opts["slippage_bps"])
