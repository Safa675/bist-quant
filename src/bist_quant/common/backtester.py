from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from bist_quant.common.backtest_services import (
    BacktestMetricsService,
    BacktestPayloadAssembler,
    DailyReturnService,
    DataPreparationService,
    HoldingsHistoryAggregator,
    RebalancingSelectionService,
    TransactionCostModel,
)
from bist_quant.common.config_manager import REGIME_ALLOCATIONS
from bist_quant.common.enums import RegimeLabel
from bist_quant.common.risk_manager import RiskManager

logger = logging.getLogger(__name__)


@dataclass
class TradeSummary:
    date: pd.Timestamp
    entered: int
    exited: int
    turnover: float


@dataclass
class PositionState:
    current_holdings: list[str] = field(default_factory=list)
    entry_prices: dict[str, float] = field(default_factory=dict)
    stopped_out: set[str] = field(default_factory=set)
    prev_selected: set[str] = field(default_factory=set)
    trade_count: int = 0
    rebalance_count: int = 0


class PositionManager:
    """Tracks holdings/trade state and emits deterministic turnover metrics."""

    def __init__(self) -> None:
        self.state = PositionState()
        self.trade_events: list[TradeSummary] = []

    def on_rebalance_start(self) -> set[str]:
        self.state.stopped_out.clear()
        self.state.rebalance_count += 1
        return self.state.prev_selected.copy()

    def update_selection(
        self,
        date: pd.Timestamp,
        new_holdings: list[str],
        entry_prices: dict[str, float],
        old_selected: set[str],
    ) -> float:
        self.state.current_holdings = new_holdings
        self.state.entry_prices = entry_prices
        new_positions = set(new_holdings) - old_selected
        self.state.trade_count += len(new_positions)
        self.state.prev_selected = set(new_holdings)

        union_names = old_selected.union(set(new_holdings))
        if not union_names:
            turnover = 0.0
            entered = 0
            exited = 0
        else:
            entered = len(set(new_holdings) - old_selected)
            exited = len(old_selected - set(new_holdings))
            turnover = (entered + exited) / len(union_names)
        self.trade_events.append(
            TradeSummary(date=date, entered=entered, exited=exited, turnover=turnover)
        )
        return turnover


def identify_monthly_rebalance_days(trading_days: pd.DatetimeIndex) -> set:
    df = pd.DataFrame({"date": trading_days})
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    first_of_month = df.groupby(["year", "month"])["date"].first()
    return set(first_of_month.values)


def identify_quarterly_rebalance_days(trading_days: pd.DatetimeIndex) -> set:
    rebalance_days = set()
    for year in range(trading_days.min().year, trading_days.max().year + 1):
        for month, day in ((3, 15), (5, 15), (8, 15), (11, 15)):
            target = pd.Timestamp(year=year, month=month, day=day)
            valid = trading_days[trading_days >= target]
            if len(valid) > 0:
                rebalance_days.add(valid[0])
    return rebalance_days


class Backtester:
    """Run rebalanced factor backtests using shared risk and cost services.

    Args:
        loader: Data loader used by preparation services.
        data_dir: Root data directory used for report/cache lookups.
        risk_manager: Risk management component for sizing and controls.
        build_size_market_cap_panel: Callable that builds a market-cap panel for
            size-aware slippage modeling.

    Attributes:
        close_df: Wide close-price panel used for return and risk calculations.
        volume_df: Wide volume panel used for liquidity filtering.
        regime_series: Time series of detected market regimes.
    """

    def __init__(
        self,
        loader,
        data_dir: Path,
        risk_manager: RiskManager,
        build_size_market_cap_panel: Callable,
    ) -> None:
        self.loader = loader
        self.data_dir = Path(data_dir)
        self.risk_manager = risk_manager
        self.build_size_market_cap_panel = build_size_market_cap_panel
        self.prices: pd.DataFrame | None = None
        self.close_df: pd.DataFrame | None = None
        self.volume_df: pd.DataFrame | None = None
        self.regime_series: pd.Series | None = None
        self.regime_allocations: dict[RegimeLabel, float] = REGIME_ALLOCATIONS.copy()
        self.xu100_prices: pd.Series | None = None
        self.xautry_prices: pd.Series | None = None
        self._mcap_slippage_panel_cache: pd.DataFrame | None = None

    def update_data(
        self,
        prices: pd.DataFrame,
        close_df: pd.DataFrame,
        volume_df: pd.DataFrame,
        regime_series: pd.Series,
        regime_allocations: dict[RegimeLabel, float],
        xu100_prices: pd.Series | None,
        xautry_prices: pd.Series | None = None,
    ) -> None:
        self.prices = prices
        self.close_df = close_df
        self.volume_df = volume_df
        self.regime_series = regime_series
        self.regime_allocations = regime_allocations or REGIME_ALLOCATIONS.copy()
        self.xu100_prices = xu100_prices
        self.xautry_prices = xautry_prices
        self.risk_manager.set_data(close_df=close_df, volume_df=volume_df)
        self._mcap_slippage_panel_cache = None

    def _get_cached_mcap_slippage_panel(self, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
        if self.close_df is None:
            return pd.DataFrame()
        if self._mcap_slippage_panel_cache is None:
            logger.info("   Preparing market-cap panel for size-based slippage...")
            self._mcap_slippage_panel_cache = self.build_size_market_cap_panel(
                self.close_df,
                self.close_df.index,
                self.loader,
            )
        else:
            logger.info("   Reusing cached market-cap panel for size-based slippage...")
        return self._mcap_slippage_panel_cache.reindex(
            index=trading_days,
            columns=self.close_df.columns,
        )

    def run(
        self,
        signals: pd.DataFrame,
        factor_name: str,
        rebalance_freq: str = "quarterly",
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
        portfolio_options: dict | None = None,
    ) -> dict:
        if self.prices is None or self.close_df is None or self.regime_series is None:
            raise ValueError("Backtester data not initialized. Call update_data() first.")

        opts = self.risk_manager.resolve_options(portfolio_options)
        debug_enabled = str(os.getenv("DEBUG", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        } or bool(opts.get("debug", False))

        def dbg(msg: str) -> None:
            if debug_enabled:
                logger.info(f"   [DEBUG] {msg}")

        self.risk_manager.print_settings(opts)

        signal_lag_days = int(opts.get("signal_lag_days", 1))
        if signal_lag_days < 0:
            raise ValueError(f"signal_lag_days must be >= 0, got {signal_lag_days}")
        if signal_lag_days == 0:
            logger.warning(
                "   ⚠️  signal_lag_days=0 (lookahead risk if signals use same-day close)"
            )

        backtest_start = start_date if start_date is not None else signals.index.min()
        backtest_end = end_date if end_date is not None else signals.index.max()

        prep = DataPreparationService(
            loader=self.loader,
            data_dir=self.data_dir,
            prices=self.prices,
            regime_series=self.regime_series,
            xu100_prices=self.xu100_prices,
            xautry_prices=self.xautry_prices,
        ).prepare(
            signals=signals,
            factor_name=factor_name,
            rebalance_freq=rebalance_freq,
            start_date=backtest_start,
            end_date=backtest_end,
            signal_lag_days=signal_lag_days,
        )

        dbg(
            "open_df shape="
            f"{prep.open_df.shape}, dates={prep.open_df.index.min().date()}..{prep.open_df.index.max().date()}"
        )
        dbg(f"signals shape={signals.shape}, signal_lag_days={signal_lag_days}")

        logger.info(f"Period: {prep.trading_days[0].date()} to {prep.trading_days[-1].date()}")
        logger.info(f"Trading days: {len(prep.trading_days)}")
        logger.info(f"Rebalance days: {len(prep.rebalance_days)}")

        regime_allocations = self.regime_allocations or REGIME_ALLOCATIONS
        fallback_regime = (
            RegimeLabel.BEAR
            if RegimeLabel.BEAR in regime_allocations
            else next(iter(regime_allocations), RegimeLabel.BEAR)
        )

        slippage_factor = opts["slippage_bps"] / 10000.0 if opts["use_slippage"] else 0.0
        top_n = int(opts["top_n"])
        stop_loss_threshold = float(opts["stop_loss_threshold"])
        small_cap_slippage_bps = max(
            float(opts.get("small_cap_slippage_bps", opts["slippage_bps"])),
            float(opts["slippage_bps"]),
        )
        mid_cap_slippage_bps = float(opts.get("mid_cap_slippage_bps", 10.0))
        use_mcap_slippage = bool(opts.get("use_mcap_slippage", True) and opts["use_slippage"])

        mcap_slippage_panel = None
        mcap_slippage_liquidity = None
        if use_mcap_slippage:
            mcap_slippage_panel = self._get_cached_mcap_slippage_panel(prep.trading_days)
            mcap_slippage_liquidity = self.volume_df.reindex(
                index=prep.trading_days,
                columns=self.close_df.columns,
            )

        position_manager = PositionManager()
        selection_service = RebalancingSelectionService(self.risk_manager)
        return_service = DailyReturnService(prep.open_fwd_ret)
        holdings_aggregator = HoldingsHistoryAggregator()
        cost_model = TransactionCostModel(self.risk_manager)

        portfolio_returns: list[dict] = []
        sanity_rows: list[dict] = []
        loop_days = prep.trading_days[:-1]
        lagged_regime_values = prep.regime_series_lagged.reindex(loop_days)
        xautry_fwd_values = (
            prep.xautry_fwd_ret.reindex(loop_days)
            .fillna(0.0)
            .to_numpy(dtype=np.float64, copy=False)
        )

        for day_pos, date in enumerate(loop_days):
            regime = RegimeLabel.coerce(lagged_regime_values.iat[day_pos])
            if regime is None or regime not in regime_allocations:
                regime = fallback_regime

            allocation = regime_allocations.get(regime, 0.0) if opts["use_regime_filter"] else 1.0
            is_rebalance_day = date in prep.rebalance_days

            decision = selection_service.maybe_rebalance(
                date=date,
                is_rebalance_day=is_rebalance_day,
                allocation=allocation,
                factor_name=factor_name,
                signals_exec=prep.signals_exec,
                open_df=prep.open_df,
                position_manager=position_manager,
                top_n=top_n,
                use_liquidity_filter=bool(opts["use_liquidity_filter"]),
                liquidity_quantile=float(opts["liquidity_quantile"]),
            )
            dbg(
                f"rebalance {date.date()} regime={regime} alloc={allocation:.2f} "
                f"signals={decision.day_signal_count} holdings={len(position_manager.state.current_holdings)} "
                f"turnover={decision.rebalance_turnover:.3f}"
            )

            if opts["use_stop_loss"]:
                active_holdings = self.risk_manager.apply_stop_loss(
                    current_holdings=position_manager.state.current_holdings,
                    stopped_out=position_manager.state.stopped_out,
                    entry_prices=position_manager.state.entry_prices,
                    open_df=prep.open_df,
                    date=date,
                    stop_loss_threshold=stop_loss_threshold,
                )
            else:
                active_holdings = position_manager.state.current_holdings

            weights: pd.Series | None = None
            if active_holdings and allocation > 0:
                if opts["use_inverse_vol_sizing"]:
                    weights = self.risk_manager.inverse_downside_vol_weights(
                        active_holdings,
                        date,
                        lookback=opts["inverse_vol_lookback"],
                        max_weight=opts["max_position_weight"],
                    )
                else:
                    weights = pd.Series(1.0 / len(active_holdings), index=active_holdings)

                weight_sum_raw = float(weights.sum())
                if not np.isfinite(weight_sum_raw):
                    raise ValueError(f"Invalid weight sum on {date.date()}: {weight_sum_raw}")
                if not np.isclose(weight_sum_raw, 1.0, atol=1e-6):
                    raise ValueError(
                        f"Weights do not sum to 1 on {date.date()}: {weight_sum_raw:.8f}"
                    )

                stock_return = return_service.compute_weighted_return(
                    date=date,
                    active_holdings=active_holdings,
                    weights=weights,
                )
                stock_return = cost_model.apply_rebalance_slippage(
                    stock_return,
                    date=date,
                    is_rebalance_day=is_rebalance_day,
                    old_selected=decision.old_selected,
                    active_holdings=active_holdings,
                    rebalance_turnover=decision.rebalance_turnover,
                    opts=opts,
                    slippage_factor=slippage_factor,
                    use_mcap_slippage=use_mcap_slippage,
                    mcap_slippage_panel=mcap_slippage_panel,
                    mcap_slippage_liquidity=mcap_slippage_liquidity,
                    small_cap_slippage_bps=small_cap_slippage_bps,
                    mid_cap_slippage_bps=mid_cap_slippage_bps,
                )

                xautry_ret = float(xautry_fwd_values[day_pos])
                port_ret = (
                    allocation * stock_return + (1 - allocation) * xautry_ret
                    if opts["use_regime_filter"]
                    else stock_return
                )
                effective_weight_sum = weight_sum_raw * allocation
            else:
                xautry_ret = float(xautry_fwd_values[day_pos])
                port_ret = xautry_ret if opts["use_regime_filter"] else 0.0
                weight_sum_raw = 0.0
                effective_weight_sum = 0.0

            if not np.isfinite(port_ret):
                raise ValueError(f"Non-finite portfolio return on {date.date()}: {port_ret}")

            portfolio_returns.append(
                {
                    "date": date,
                    "return": float(port_ret),
                    "xautry_return": xautry_ret,
                    "regime": regime,
                    "n_stocks": len(active_holdings),
                    "allocation": float(allocation),
                }
            )
            sanity_rows.append(
                {
                    "date": date,
                    "regime": regime,
                    "allocation": float(allocation),
                    "is_rebalance_day": bool(is_rebalance_day),
                    "signal_count": int(decision.day_signal_count),
                    "n_active_holdings": int(len(active_holdings)),
                    "weight_sum_raw": float(weight_sum_raw),
                    "effective_weight_sum": float(effective_weight_sum),
                    "rebalance_turnover": float(decision.rebalance_turnover),
                    "portfolio_return": float(port_ret),
                }
            )
            holdings_aggregator.add(
                date=date,
                regime=regime,
                allocation=float(allocation),
                active_holdings=active_holdings,
                weights=weights,
            )

        holdings_history = holdings_aggregator.to_records()
        returns_df = pd.DataFrame(portfolio_returns).set_index("date")
        if returns_df["return"].isna().any():
            bad_count = int(returns_df["return"].isna().sum())
            raise ValueError(f"Backtest produced {bad_count} NaN returns")

        sanity_df = pd.DataFrame(sanity_rows).set_index("date") if sanity_rows else pd.DataFrame()
        if not sanity_df.empty:
            invested_mask = (sanity_df["allocation"] > 0.0) & (sanity_df["n_active_holdings"] > 0)
            invested_weights = sanity_df.loc[invested_mask, "weight_sum_raw"]
            if not invested_weights.empty and (invested_weights - 1.0).abs().max() > 1e-6:
                raise ValueError(
                    "Weight-sum sanity check failed: some invested days are not fully weighted"
                )

        metrics = BacktestMetricsService(self.risk_manager).compute(
            returns_df=returns_df,
            opts=opts,
            regime_allocations=regime_allocations,
        )
        logger.info(f"   Rebalances: {position_manager.state.rebalance_count}")
        logger.info(f"   Total Trades: {position_manager.state.trade_count}")

        return BacktestPayloadAssembler.assemble(
            metrics=metrics,
            returns_df=returns_df,
            holdings_history=holdings_history,
            sanity_df=sanity_df,
            position_manager=position_manager,
            signal_lag_days=prep.signal_lag_days,
        )
