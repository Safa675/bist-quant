from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bist_quant.common.enums import RegimeLabel
from bist_quant.common.utils import validate_signal_panel_schema

logger = logging.getLogger(__name__)
@dataclass
class BacktestPreparationResult:
    open_df: pd.DataFrame
    signals_exec: pd.DataFrame
    regime_series_lagged: pd.Series
    xautry_fwd_ret: pd.Series
    open_fwd_ret: pd.DataFrame
    trading_days: pd.DatetimeIndex
    rebalance_days: set[pd.Timestamp]
    signal_lag_days: int


@dataclass
class RebalanceDecision:
    old_selected: set[str]
    rebalance_turnover: float
    day_signal_count: int


@dataclass
class BacktestMetrics:
    returns: pd.Series
    equity: pd.Series
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    win_rate: float
    regime_performance: dict[RegimeLabel, dict[str, float]]


class DataPreparationService:
    """Prepare aligned market/signal panels for a single backtest run."""

    def __init__(
        self,
        loader,
        data_dir: Path,
        prices: pd.DataFrame,
        regime_series: pd.Series,
        xu100_prices: pd.Series | None,
        xautry_prices: pd.Series | None,
    ) -> None:
        self.loader = loader
        self.data_dir = Path(data_dir)
        self.prices = prices
        self.regime_series = regime_series
        self.xu100_prices = xu100_prices
        self.xautry_prices = xautry_prices

    def prepare(
        self,
        signals: pd.DataFrame,
        factor_name: str,
        rebalance_freq: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        signal_lag_days: int,
    ) -> BacktestPreparationResult:
        prices_filtered = self.prices[
            (self.prices["Date"] >= start_date) & (self.prices["Date"] <= end_date)
        ].copy()

        open_df = prices_filtered.pivot_table(index="Date", columns="Ticker", values="Open").sort_index()
        open_df.columns = [column.split(".")[0].upper() for column in open_df.columns]

        if self.xu100_prices is not None and "XU100" not in open_df.columns:
            open_df["XU100"] = self.xu100_prices.reindex(open_df.index)

        if factor_name == "xu100":
            valid_xu100_mask = open_df["XU100"].notna()
            n_filtered = (~valid_xu100_mask).sum()
            if n_filtered > 0:
                logger.info(f"   Filtered {n_filtered} dates with missing XU100 data")
                open_df = open_df[valid_xu100_mask]

        if open_df.empty:
            raise ValueError("Backtest window has no valid open prices after filtering")

        signal_columns = pd.Index(signals.columns) if isinstance(signals, pd.DataFrame) else pd.Index([])
        signals = validate_signal_panel_schema(
            panel=signals,
            dates=open_df.index,
            tickers=signal_columns,
            signal_name=factor_name,
            context="backtest input signal panel",
            dtype=np.float32,
        )
        signals_exec = signals.shift(signal_lag_days)

        regime_series = self.regime_series.reindex(open_df.index).ffill()
        regime_series_lagged = regime_series.shift(1).ffill()

        xautry_series = self._resolve_xautry_series(start_date=start_date, end_date=end_date)
        xautry_prices = xautry_series.reindex(open_df.index).ffill()

        open_fwd_ret = open_df.shift(-1) / open_df - 1.0
        xautry_fwd_ret = xautry_prices.shift(-1) / xautry_prices - 1.0
        xautry_fwd_ret = xautry_fwd_ret.fillna(0.0)

        split_mask = (open_fwd_ret < -0.50) | (open_fwd_ret > 1.00)
        n_neutralised = split_mask.sum().sum()
        if n_neutralised > 0:
            open_fwd_ret = open_fwd_ret.where(~split_mask, 0.0)
            logger.info(f"   Neutralised {n_neutralised} split/corporate-action returns")

        trading_days = open_df.index
        rebalance_days = self._identify_rebalance_days(trading_days=trading_days, rebalance_freq=rebalance_freq)

        return BacktestPreparationResult(
            open_df=open_df,
            signals_exec=signals_exec,
            regime_series_lagged=regime_series_lagged,
            xautry_fwd_ret=xautry_fwd_ret,
            open_fwd_ret=open_fwd_ret,
            trading_days=trading_days,
            rebalance_days=rebalance_days,
            signal_lag_days=signal_lag_days,
        )

    def _resolve_xautry_series(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series:
        if self.xautry_prices is not None and not self.xautry_prices.empty:
            series = self.xautry_prices.copy()
            series = series.loc[(series.index >= start_date) & (series.index <= end_date)]
            return series

        return self.loader.load_xautry_prices(
            self.data_dir / "xau_try_2013_2026.csv",
            start_date=start_date,
            end_date=end_date,
        )

    @staticmethod
    def _identify_rebalance_days(
        trading_days: pd.DatetimeIndex,
        rebalance_freq: str,
    ) -> set[pd.Timestamp]:
        if rebalance_freq == "monthly":
            df = pd.DataFrame({"date": trading_days})
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month
            first_of_month = df.groupby(["year", "month"])["date"].first()
            return set(first_of_month.values)

        rebalance_days: set[pd.Timestamp] = set()
        for year in range(trading_days.min().year, trading_days.max().year + 1):
            for month, day in ((3, 15), (5, 15), (8, 15), (11, 15)):
                target = pd.Timestamp(year=year, month=month, day=day)
                valid = trading_days[trading_days >= target]
                if len(valid) > 0:
                    rebalance_days.add(valid[0])
        return rebalance_days


class RebalancingSelectionService:
    """Select holdings on rebalance dates from lagged signal panels."""

    def __init__(self, risk_manager) -> None:
        self.risk_manager = risk_manager

    def maybe_rebalance(
        self,
        date: pd.Timestamp,
        is_rebalance_day: bool,
        allocation: float,
        factor_name: str,
        signals_exec: pd.DataFrame,
        open_df: pd.DataFrame,
        position_manager,
        top_n: int,
        use_liquidity_filter: bool,
        liquidity_quantile: float,
    ) -> RebalanceDecision:
        if not is_rebalance_day:
            return RebalanceDecision(old_selected=set(), rebalance_turnover=0.0, day_signal_count=0)

        old_selected = position_manager.on_rebalance_start()

        if allocation <= 0 or date not in signals_exec.index:
            return RebalanceDecision(old_selected=old_selected, rebalance_turnover=0.0, day_signal_count=0)

        day_signals = signals_exec.loc[date].dropna()
        day_signal_count = int(day_signals.shape[0])

        available = [
            ticker
            for ticker in day_signals.index
            if ticker in open_df.columns and pd.notna(open_df.loc[date, ticker])
        ]
        if not available:
            return RebalanceDecision(
                old_selected=old_selected,
                rebalance_turnover=0.0,
                day_signal_count=day_signal_count,
            )

        if factor_name == "xu100":
            entry_prices = {ticker: float(open_df.loc[date, ticker]) for ticker in available}
            rebalance_turnover = position_manager.update_selection(
                date=date,
                new_holdings=available,
                entry_prices=entry_prices,
                old_selected=old_selected,
            )
            return RebalanceDecision(
                old_selected=old_selected,
                rebalance_turnover=rebalance_turnover,
                day_signal_count=day_signal_count,
            )

        if use_liquidity_filter:
            available = self.risk_manager.filter_by_liquidity(
                available,
                date,
                liquidity_quantile,
            )
        day_signals = day_signals[available]

        if len(day_signals) < top_n:
            return RebalanceDecision(
                old_selected=old_selected,
                rebalance_turnover=0.0,
                day_signal_count=day_signal_count,
            )

        top_stocks = day_signals.nlargest(top_n).index.tolist()
        entry_prices = {
            ticker: float(open_df.loc[date, ticker])
            for ticker in top_stocks
            if ticker in open_df.columns and pd.notna(open_df.loc[date, ticker])
        }
        rebalance_turnover = position_manager.update_selection(
            date=date,
            new_holdings=top_stocks,
            entry_prices=entry_prices,
            old_selected=old_selected,
        )
        return RebalanceDecision(
            old_selected=old_selected,
            rebalance_turnover=rebalance_turnover,
            day_signal_count=day_signal_count,
        )


class DailyReturnService:
    """Vectorized daily return engine backed by pre-aligned NumPy arrays."""

    def __init__(self, open_fwd_ret: pd.DataFrame) -> None:
        self._dates = open_fwd_ret.index
        self._tickers = open_fwd_ret.columns
        self._return_values = open_fwd_ret.to_numpy(dtype=np.float64, copy=False)

    def compute_weighted_return(
        self,
        date: pd.Timestamp,
        active_holdings: list[str],
        weights: pd.Series,
    ) -> float:
        if not active_holdings:
            return 0.0

        date_pos = self._dates.get_indexer([date])[0]
        if date_pos < 0:
            return 0.0

        holding_arr = np.asarray(active_holdings, dtype=object)
        ticker_pos = self._tickers.get_indexer(holding_arr)
        valid_mask = ticker_pos >= 0
        if not np.any(valid_mask):
            return 0.0

        valid_tickers = holding_arr[valid_mask]
        valid_pos = ticker_pos[valid_mask]
        day_returns = self._return_values[date_pos, valid_pos]
        aligned_weights = weights.reindex(valid_tickers).to_numpy(dtype=np.float64, copy=False)

        return float(
            np.dot(
                np.nan_to_num(day_returns, nan=0.0, posinf=0.0, neginf=0.0),
                np.nan_to_num(aligned_weights, nan=0.0, posinf=0.0, neginf=0.0),
            )
        )


class HoldingsHistoryAggregator:
    """Buffers holdings rows and materializes payload once per run."""

    def __init__(self) -> None:
        self._dates: list[pd.Timestamp] = []
        self._tickers: list[str] = []
        self._weights: list[float] = []
        self._regimes: list[RegimeLabel] = []
        self._allocations: list[float] = []

    def add(
        self,
        *,
        date: pd.Timestamp,
        regime: RegimeLabel,
        allocation: float,
        active_holdings: list[str],
        weights: pd.Series | None,
    ) -> None:
        if active_holdings and allocation > 0.0 and weights is not None:
            holding_arr = np.asarray(active_holdings, dtype=object)
            if holding_arr.size == 0:
                return

            raw_weights = weights.reindex(holding_arr).to_numpy(dtype=np.float64, copy=False)
            effective_weights = np.nan_to_num(raw_weights, nan=0.0) * float(allocation)
            n_names = int(holding_arr.size)

            self._dates.extend([date] * n_names)
            self._tickers.extend(holding_arr.tolist())
            self._weights.extend(effective_weights.tolist())
            self._regimes.extend([regime] * n_names)
            self._allocations.extend([float(allocation)] * n_names)
            return

        self._dates.append(date)
        self._tickers.append("XAU/TRY")
        self._weights.append(1.0)
        self._regimes.append(regime)
        self._allocations.append(0.0)

    def to_records(self) -> list[dict]:
        if not self._dates:
            return []

        frame = pd.DataFrame(
            {
                "date": self._dates,
                "ticker": self._tickers,
                "weight": self._weights,
                "regime": [str(regime) for regime in self._regimes],
                "allocation": self._allocations,
            }
        )
        # Defensive aggregation in case of duplicate date+ticker rows from upstream logic.
        grouped = frame.groupby(
            ["date", "ticker", "regime", "allocation"],
            sort=False,
            as_index=False,
        )["weight"].sum()
        return grouped.to_dict("records")


class TransactionCostModel:
    """Apply slippage and transaction cost logic."""

    def __init__(self, risk_manager) -> None:
        self.risk_manager = risk_manager

    def apply_rebalance_slippage(
        self,
        stock_return: float,
        *,
        date: pd.Timestamp,
        is_rebalance_day: bool,
        old_selected: set[str],
        active_holdings: list[str],
        rebalance_turnover: float,
        opts: dict,
        slippage_factor: float,
        use_mcap_slippage: bool,
        mcap_slippage_panel: pd.DataFrame | None,
        mcap_slippage_liquidity: pd.DataFrame | None,
        small_cap_slippage_bps: float,
        mid_cap_slippage_bps: float,
    ) -> float:
        if not (opts["use_slippage"] and is_rebalance_day and old_selected):
            return stock_return

        new_positions = list(set(active_holdings) - old_selected)
        exited_positions = list(old_selected - set(active_holdings))
        if not (new_positions or exited_positions):
            return stock_return

        avg_bps = self.risk_manager.slippage_cost_bps(
            date=date,
            new_positions=new_positions,
            opts=opts,
            use_mcap_slippage=use_mcap_slippage,
            mcap_slippage_panel=mcap_slippage_panel,
            mcap_slippage_liquidity=mcap_slippage_liquidity,
            small_cap_slippage_bps=small_cap_slippage_bps,
            mid_cap_slippage_bps=mid_cap_slippage_bps,
        )

        if use_mcap_slippage and mcap_slippage_panel is not None and date in mcap_slippage_panel.index:
            return stock_return - rebalance_turnover * (avg_bps / 10000.0) * 2
        return stock_return - rebalance_turnover * slippage_factor * 2


class BacktestMetricsService:
    """Compute portfolio-level metrics and downside-vol targeting outputs."""

    def __init__(self, risk_manager) -> None:
        self.risk_manager = risk_manager

    def compute(
        self,
        returns_df: pd.DataFrame,
        opts: dict,
        regime_allocations: dict[RegimeLabel, float],
    ) -> BacktestMetrics:
        raw_returns = returns_df["return"]
        if opts["use_vol_targeting"]:
            logger.info(f"\n📈 Applying {opts['target_downside_vol']*100:.0f}% downside volatility targeting...")
            returns = self.risk_manager.apply_downside_vol_targeting(
                raw_returns,
                target_vol=opts["target_downside_vol"],
                lookback=opts["vol_lookback"],
                vol_floor=opts["vol_floor"],
                vol_cap=opts["vol_cap"],
            )
            neg_rets = returns[returns < 0]
            realized_downside_vol = neg_rets.std() * np.sqrt(252) if len(neg_rets) > 2 else 0
            logger.info(f"   Realized downside volatility: {realized_downside_vol*100:.1f}%")
        else:
            logger.info("\n📈 Volatility targeting: OFF (using raw returns)")
            returns = raw_returns

        equity = (1 + returns).cumprod()
        total_return = float(equity.iloc[-1] - 1)
        n_years = len(returns) / 252
        cagr = float((1 + total_return) ** (1 / n_years) - 1) if n_years > 0 else 0.0

        returns_std = returns.std()
        sharpe = float(returns.mean() / returns_std * np.sqrt(252)) if returns_std > 0 else 0.0

        downside = returns[returns < 0]
        downside_std = downside.std()
        sortino = (
            float(returns.mean() / downside_std * np.sqrt(252))
            if len(downside) > 0 and downside_std > 0
            else 0.0
        )

        cummax = equity.cummax()
        drawdown = equity / cummax - 1
        max_dd = float(drawdown.min())
        win_rate = float((returns > 0).sum() / len(returns)) if len(returns) > 0 else 0.0

        regime_perf: dict[RegimeLabel, dict[str, float]] = {}
        for regime in regime_allocations:
            mask = returns_df["regime"] == regime
            if mask.sum() == 0:
                continue
            regime_returns = returns[mask]
            regime_perf[regime] = {
                "count": float(mask.sum()),
                "mean_return": float(regime_returns.mean() * 252),
                "total_return": float((1 + regime_returns).prod() - 1),
                "win_rate": float((regime_returns > 0).sum() / len(regime_returns))
                if len(regime_returns) > 0
                else 0.0,
            }

        logger.info("\n📊 Results:")
        logger.info(f"   Total Return: {total_return*100:.1f}%")
        logger.info(f"   CAGR: {cagr*100:.2f}%")
        logger.info(f"   Sharpe: {sharpe:.2f}")
        logger.info(f"   Sortino: {sortino:.2f}")
        logger.info(f"   Max Drawdown: {max_dd*100:.2f}%")
        logger.info(f"   Win Rate: {win_rate*100:.1f}%")

        return BacktestMetrics(
            returns=returns,
            equity=equity,
            total_return=total_return,
            cagr=cagr,
            sharpe=sharpe,
            sortino=sortino,
            max_drawdown=max_dd,
            win_rate=win_rate,
            regime_performance=regime_perf,
        )


class BacktestPayloadAssembler:
    """Build final backward-compatible output payload."""

    @staticmethod
    def assemble(
        metrics: BacktestMetrics,
        returns_df: pd.DataFrame,
        holdings_history: list[dict],
        sanity_df: pd.DataFrame,
        position_manager,
        signal_lag_days: int,
    ) -> dict:
        return {
            "returns": metrics.returns,
            "equity": metrics.equity,
            "total_return": metrics.total_return,
            "cagr": metrics.cagr,
            "sharpe": metrics.sharpe,
            "sortino": metrics.sortino,
            "max_drawdown": metrics.max_drawdown,
            "win_rate": metrics.win_rate,
            "xautry_returns": returns_df["xautry_return"],
            "regime_performance": metrics.regime_performance,
            "rebalance_count": position_manager.state.rebalance_count,
            "trade_count": position_manager.state.trade_count,
            "returns_df": returns_df,
            "holdings_history": holdings_history,
            "sanity_checks": sanity_df,
            "signal_lag_days": signal_lag_days,
            "trade_events": [event.__dict__ for event in position_manager.trade_events],
        }
