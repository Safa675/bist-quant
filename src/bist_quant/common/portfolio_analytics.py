"""Multi-asset portfolio analytics module."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Union, cast

import numpy as np
import pandas as pd

from bist_quant.clients.crypto_client import CryptoClient
from bist_quant.clients.fund_analyzer import TEFASAnalyzer
from bist_quant.clients.fx_commodities_client import FXCommoditiesClient
from bist_quant.clients.us_stock_client import USStockClient

if TYPE_CHECKING:
    from bist_quant.common.data_loader import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    sharpe_ratio: float
    sortino_ratio: float
    alpha: float
    beta: float
    volatility: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    treynor_ratio: float
    information_ratio: float


class PortfolioAnalytics:
    """Enhanced portfolio analytics with support for multiple asset classes."""

    def __init__(self):
        self.fund_analyzer = TEFASAnalyzer()
        self.crypto_client = CryptoClient()
        self.us_stock_client = USStockClient()
        self.fx_client = FXCommoditiesClient()

    async def calculate_multi_asset_portfolio_metrics(
        self,
        holdings: Dict[str, Dict[str, Union[float, int, str]]],
    ) -> Dict[str, RiskMetrics]:
        """Calculate risk metrics for a multi-asset portfolio."""
        all_returns: Dict[str, pd.Series] = {}

        for symbol, info in holdings.items():
            quantity = float(info.get("quantity", 0) or 0)
            if quantity <= 0:
                continue

            asset_type = str(info.get("asset_type", "stock"))
            try:
                series = await self._get_asset_returns(symbol, asset_type)
            except Exception as exc:
                logger.warning("Could not get returns for %s (%s): %s", symbol, asset_type, exc)
                continue

            if isinstance(series, pd.Series) and not series.empty:
                all_returns[symbol] = series

        if not all_returns:
            raise ValueError("No valid return series found for any assets")

        portfolio_returns = self._calculate_portfolio_returns(all_returns, holdings)
        metrics = self.calculate_risk_metrics(portfolio_returns)
        return {"portfolio": metrics}

    async def _get_asset_returns(self, symbol: str, asset_type: str) -> pd.Series:
        """Get daily returns for one asset."""
        normalized_type = asset_type.lower().strip()

        if normalized_type == "stock":
            try:
                import borsapy as bp
            except Exception:
                logger.warning("borsapy is not available for stock asset %s", symbol)
                return pd.Series(dtype=float)

            ticker = bp.Ticker(symbol)
            hist = ticker.history(period="1y")
            close_col = self._pick_close_col(hist)
            if close_col:
                return pd.to_numeric(hist[close_col], errors="coerce").pct_change().dropna()

        elif normalized_type == "fund":
            hist = await self.fund_analyzer.get_fund_history(symbol, period="1y")
            if not hist.empty and "nav" in hist.columns:
                return pd.to_numeric(hist["nav"], errors="coerce").pct_change().dropna()

        elif normalized_type == "crypto":
            hist = await self.crypto_client.get_crypto_history(symbol, exchange="btcturk", period="1y", interval="1d")
            if not hist.empty and "close" in hist.columns:
                return pd.to_numeric(hist["close"], errors="coerce").pct_change().dropna()

        elif normalized_type == "fx":
            hist = await self.fx_client.get_fx_history(symbol, period="1y", interval="1d")
            if not hist.empty and "close" in hist.columns:
                return pd.to_numeric(hist["close"], errors="coerce").pct_change().dropna()

        elif normalized_type == "us_stock":
            hist = await self.us_stock_client.get_us_stock_history(symbol, period="1y", interval="1d")
            if not hist.empty and "close" in hist.columns:
                return pd.to_numeric(hist["close"], errors="coerce").pct_change().dropna()

        return pd.Series(dtype=float)

    def _calculate_portfolio_returns(
        self,
        all_returns: Dict[str, pd.Series],
        holdings: Dict[str, Dict[str, Union[float, int, str]]],
    ) -> pd.Series:
        """Calculate weighted portfolio returns from asset return series."""
        frame = pd.DataFrame(all_returns)
        if frame.empty:
            return pd.Series(dtype=float)

        # Use outer alignment and neutralize missing series with 0 returns.
        frame = frame.sort_index().fillna(0.0)

        values: Dict[str, float] = {}
        total_value = 0.0
        for symbol in frame.columns:
            info = holdings.get(symbol, {})
            quantity = float(info.get("quantity", 0) or 0)
            cost_basis = float(info.get("cost_basis", 0) or 0)
            current_value = quantity * cost_basis
            if current_value > 0:
                values[symbol] = current_value
                total_value += current_value

        if total_value <= 0:
            return pd.Series(dtype=float)

        weights = pd.Series({symbol: value / total_value for symbol, value in values.items()})
        frame = frame[[col for col in frame.columns if col in weights.index]]
        if frame.empty:
            return pd.Series(dtype=float)

        weighted = frame.mul(weights, axis=1)
        portfolio = weighted.sum(axis=1)
        portfolio.name = "portfolio_return"
        return portfolio.dropna()

    def calculate_risk_metrics(self, returns: pd.Series) -> RiskMetrics:
        """Calculate risk metrics for a return series."""
        if returns.empty or len(returns) < 2:
            return RiskMetrics(
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                alpha=0.0,
                beta=0.0,
                volatility=0.0,
                max_drawdown=0.0,
                var_95=0.0,
                cvar_95=0.0,
                treynor_ratio=0.0,
                information_ratio=0.0,
            )

        returns = pd.to_numeric(returns, errors="coerce").dropna()
        if returns.empty or len(returns) < 2:
            return RiskMetrics(
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                alpha=0.0,
                beta=0.0,
                volatility=0.0,
                max_drawdown=0.0,
                var_95=0.0,
                cvar_95=0.0,
                treynor_ratio=0.0,
                information_ratio=0.0,
            )

        mean_daily = float(returns.mean())
        std_daily = float(returns.std())
        volatility = float(std_daily * np.sqrt(252))

        risk_free_daily = 0.05 / 252
        excess_daily = mean_daily - risk_free_daily

        sharpe_ratio = float((excess_daily / std_daily) * np.sqrt(252)) if std_daily != 0 else 0.0

        downside_deviation = self._calculate_downside_deviation(returns)
        sortino_ratio = float((excess_daily / downside_deviation) * np.sqrt(252)) if downside_deviation != 0 else 0.0

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = float(drawdown.min())

        var_95 = float(returns.quantile(0.05))
        tail = returns[returns <= var_95]
        cvar_95 = float(tail.mean()) if not tail.empty else 0.0

        beta = 1.0
        expected_market_daily = 0.08 / 252
        alpha = float(excess_daily - beta * (expected_market_daily - risk_free_daily))

        treynor_ratio = float(excess_daily / beta) if beta != 0 else 0.0
        information_ratio = 0.0

        return RiskMetrics(
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            alpha=alpha,
            beta=beta,
            volatility=volatility,
            max_drawdown=max_drawdown,
            var_95=var_95,
            cvar_95=cvar_95,
            treynor_ratio=treynor_ratio,
            information_ratio=information_ratio,
        )

    def _calculate_downside_deviation(self, returns: pd.Series) -> float:
        """Calculate downside deviation from negative returns."""
        negative_returns = returns[returns < 0]
        if negative_returns.empty:
            return 0.0
        return float(negative_returns.std())

    async def get_diversification_metrics(
        self,
        holdings: Dict[str, Dict[str, Union[float, int, str]]],
    ) -> Dict[str, float]:
        """Calculate simple diversification metrics."""
        values: Dict[str, float] = {}
        for symbol, info in holdings.items():
            qty = float(info.get("quantity", 0) or 0)
            cost = float(info.get("cost_basis", 0) or 0)
            value = qty * cost
            if value > 0:
                values[symbol] = value

        if not values:
            return {
                "concentration_ratio": 0.0,
                "effective_positions": 0.0,
                "avg_correlation": 0.0,
            }

        total = sum(values.values())
        weights = np.array([v / total for v in values.values()], dtype=float)

        concentration_ratio = float(weights.max())
        effective_positions = float(1.0 / np.sum(np.square(weights))) if weights.size else 0.0

        returns_map: Dict[str, pd.Series] = {}
        for symbol, info in holdings.items():
            try:
                series = await self._get_asset_returns(symbol, str(info.get("asset_type", "stock")))
            except Exception:
                continue
            if not series.empty:
                returns_map[symbol] = series.reset_index(drop=True)

        if len(returns_map) >= 2:
            corr = pd.DataFrame(returns_map).corr()
            if corr.empty:
                avg_correlation = 0.0
            else:
                mask = ~np.eye(len(corr), dtype=bool)
                vals = corr.where(mask).stack()
                avg_correlation = float(vals.mean()) if not vals.empty else 0.0
        else:
            avg_correlation = 0.0

        return {
            "concentration_ratio": concentration_ratio,
            "effective_positions": effective_positions,
            "avg_correlation": avg_correlation,
        }

    async def close_clients(self) -> None:
        """Close all async clients."""
        await self.fund_analyzer.close()
        await self.crypto_client.close()
        await self.us_stock_client.close()
        await self.fx_client.close()

    @staticmethod
    def _pick_close_col(history: pd.DataFrame) -> str | None:
        if history is None or history.empty:
            return None
        for candidate in ("Close", "close", "Adj Close", "adj_close"):
            if candidate in history.columns:
                return candidate
        return None


class PortfolioAnalyticsAdapter:
    """Adapter for PortfolioAnalytics helper methods used by DataLoader."""

    def __init__(self, loader: "DataLoader") -> None:
        self._loader = loader

    @staticmethod
    def _portfolio_analytics_cls() -> type[Any]:
        try:
            from bist_quant.analytics import PortfolioAnalytics as PortfolioAnalyticsCls
        except ImportError:
            from analytics import PortfolioAnalytics as PortfolioAnalyticsCls
        return cast(type[Any], PortfolioAnalyticsCls)

    def create_portfolio_analytics(
        self,
        holdings: dict[str, float] | None = None,
        weights: dict[str, float] | None = None,
        returns: pd.Series | None = None,
        benchmark: str = "XU100",
        name: str = "Portfolio",
    ) -> Any:
        portfolio_analytics_cls = self._portfolio_analytics_cls()
        close_df = getattr(self._loader, "_close_df", None)

        if returns is not None:
            benchmark_returns = None
            if benchmark and isinstance(close_df, pd.DataFrame) and benchmark in close_df.columns:
                benchmark_returns = close_df[benchmark].pct_change().dropna()
            return portfolio_analytics_cls(
                returns=returns,
                benchmark_returns=benchmark_returns,
                name=name,
            )

        if not isinstance(close_df, pd.DataFrame):
            raise ValueError("Price data not loaded. Call load_prices() first.")

        if holdings:
            return portfolio_analytics_cls.from_holdings(
                holdings=holdings,
                close_df=close_df,
                benchmark_col=benchmark if benchmark in close_df.columns else None,
                weights=weights,
                name=name,
            )

        if weights:
            synthetic_holdings = {symbol: 1.0 for symbol in weights.keys()}
            return portfolio_analytics_cls.from_holdings(
                holdings=synthetic_holdings,
                close_df=close_df,
                benchmark_col=benchmark if benchmark in close_df.columns else None,
                weights=weights,
                name=name,
            )

        raise ValueError("Either holdings, weights, or returns must be provided")

    def analyze_strategy_performance(
        self,
        equity_curve: pd.Series,
        benchmark_curve: pd.Series | None = None,
        name: str = "Strategy",
    ) -> Any:
        portfolio_analytics_cls = self._portfolio_analytics_cls()
        return portfolio_analytics_cls.from_equity_curve(
            equity_curve=equity_curve,
            benchmark_curve=benchmark_curve,
            name=name,
        )
