"""Common Data Loader - Centralized data loading to eliminate redundant I/O.

This module provides a :class:`DataLoader` facade that delegates to
specialised sub-loaders in the :mod:`bist_quant.common.loaders` package.
Expensive tables are cached after first use to reduce repeated I/O
during backtests.

Supports multiple data sources:
- Local parquet/CSV files (primary)
- Borsapy API (alternative/supplement)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Final

import pandas as pd

from bist_quant.clients.borsapy_adapter import BorsapyAdapter, StockData
from bist_quant.clients.fixed_income_provider import (
    DEFAULT_FALLBACK_RISK_FREE_RATE,
    FixedIncomeProvider,
)
from bist_quant.clients.derivatives_provider import DerivativesProvider
from bist_quant.clients.fx_enhanced_provider import FXEnhancedProvider
from bist_quant.clients.macro_adapter import MacroAdapter
from bist_quant.common.enums import RegimeLabel
from bist_quant.common.loaders import (
    FundamentalsLoader,
    PriceLoader,
    RegimeLoader,
    SharesLoader,
)
from bist_quant.common.panel_cache import PanelCache
from bist_quant.settings import PROJECT_ROOT
from .data_paths import DataPaths, get_data_paths

if TYPE_CHECKING:
    from bist_quant.common.portfolio_analytics import PortfolioAnalyticsAdapter
else:
    PortfolioAnalyticsAdapter = Any

try:
    from bist_quant.common.portfolio_analytics import (
        PortfolioAnalyticsAdapter as _PortfolioAnalyticsAdapter,
    )
except Exception:
    _PortfolioAnalyticsAdapter = None

logger = logging.getLogger(__name__)
FETCHER_DIR: Final[Path] = PROJECT_ROOT / "src" / "bist_quant" / "fetcher"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


class DataLoader:
    """Load market, fundamental, and regime datasets with reusable in-memory caches.

    The loader provides a unified facade that delegates to specialised
    sub-loaders for prices, fundamentals, shares, and regime data.

    Args:
        data_dir: Optional root directory for market data files.
        regime_model_dir: Optional regime output directory override.
        macro_events_path: Optional path to a custom macro events module.
        data_paths: Pre-resolved data paths object.

    Attributes:
        data_dir: Resolved root data directory.
        regime_model_dir: Resolved directory for regime artifacts.
        panel_cache: Shared cache for precomputed panel DataFrames.
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        regime_model_dir: Path | None = None,
        macro_events_path: Path | None = None,
        data_paths: DataPaths | None = None,
        data_source_priority: str | None = None,
    ):
        if data_paths is not None:
            self.paths = data_paths
        elif data_dir is None and regime_model_dir is None:
            self.paths = get_data_paths()
        else:
            # Preserve explicit constructor overrides for backward compatibility.
            self.paths = DataPaths()
            if data_dir is not None:
                self.paths.data_dir = Path(data_dir).expanduser().resolve()
            if regime_model_dir is not None:
                self.paths.regime_dir = Path(regime_model_dir).expanduser().resolve()

        self.data_dir = Path(self.paths.data_dir)
        self.regime_model_dir = Path(self.paths.regime_dir)
        self.fundamental_dir = self.paths.fundamentals_dir
        self.isyatirim_dir = self.data_dir / "price" / "isyatirim_prices"

        resolved_macro_events = (
            Path(macro_events_path)
            if macro_events_path is not None
            else (
                PROJECT_ROOT
                / "src"
                / "bist_quant"
                / "data_pipeline"
                / "fetcher_scripts"
                / "macro_events.py"
            )
        )
        if not resolved_macro_events.exists():
            resolved_macro_events = FETCHER_DIR / "tcmb_data_fetcher.py"

        # Data source priority: "auto" | "borsapy" | "local"
        self._data_source_priority = (
            (data_source_priority or os.getenv("BIST_DATA_SOURCE", "borsapy")).strip().lower()
        )
        if self._data_source_priority == "auto":
            self._data_source_priority = "borsapy"

        # Shared cache
        self.panel_cache = PanelCache(
            max_entries=_env_int("BIST_PANEL_CACHE_MAX_ENTRIES", 32),
        )

        # ── Adapters (retain for backward-compat façade properties) ──
        self._borsapy_adapter = BorsapyAdapter(self)
        self._macro_adapter = MacroAdapter(self, macro_events_path=resolved_macro_events)
        self._portfolio_analytics_adapter = (
            _PortfolioAnalyticsAdapter(self) if _PortfolioAnalyticsAdapter is not None else None
        )

        # Lazy provider caches
        self._fixed_income_provider: FixedIncomeProvider | None = None
        self._derivatives_provider: DerivativesProvider | None = None
        self._fx_enhanced_provider: FXEnhancedProvider | None = None
        self._risk_free_rate: float | None = None

        # ── Sub-loaders ──────────────────────────────────────────────────
        self.price_loader = PriceLoader(
            paths=self.paths,
            data_dir=self.data_dir,
            data_source_priority=self._data_source_priority,
            borsapy_adapter=self._borsapy_adapter,
            panel_cache=self.panel_cache,
        )
        self.fundamentals_loader = FundamentalsLoader(
            paths=self.paths,
            fundamental_dir=self.fundamental_dir,
            data_source_priority=self._data_source_priority,
        )
        self.shares_loader = SharesLoader(
            paths=self.paths,
            data_dir=self.data_dir,
            isyatirim_dir=self.isyatirim_dir,
            panel_cache=self.panel_cache,
            price_loader=self.price_loader,
            fundamentals_loader=self.fundamentals_loader,
        )
        self.regime_loader = RegimeLoader(
            paths=self.paths,
            regime_model_dir=self.regime_model_dir,
        )

    # -------------------------------------------------------------------------
    # Adapter Facades
    # -------------------------------------------------------------------------

    @property
    def borsapy_adapter(self) -> BorsapyAdapter:
        return self._borsapy_adapter

    @property
    def macro_adapter(self) -> MacroAdapter:
        return self._macro_adapter

    @property
    def portfolio_analytics(self) -> PortfolioAnalyticsAdapter:
        if self._portfolio_analytics_adapter is None:
            raise ImportError(
                "Portfolio analytics requires optional client dependencies. "
                "Install the relevant extras before using DataLoader.portfolio_analytics."
            )
        return self._portfolio_analytics_adapter

    @property
    def borsapy(self):
        return self.borsapy_adapter.client

    @property
    def macro(self):
        return self.macro_adapter.client

    @property
    def economic_calendar(self):
        return self.macro_adapter.economic_calendar

    @property
    def fixed_income(self) -> FixedIncomeProvider:
        """Lazy fixed income provider facade."""
        if self._fixed_income_provider is None:
            self._fixed_income_provider = FixedIncomeProvider()
        return self._fixed_income_provider

    @property
    def derivatives(self) -> DerivativesProvider:
        """Lazy derivatives provider facade."""
        if self._derivatives_provider is None:
            self._derivatives_provider = DerivativesProvider()
        return self._derivatives_provider

    @property
    def fx_enhanced(self) -> FXEnhancedProvider:
        """Lazy enhanced FX provider facade."""
        if self._fx_enhanced_provider is None:
            self._fx_enhanced_provider = FXEnhancedProvider()
        return self._fx_enhanced_provider

    # -------------------------------------------------------------------------
    # Risk-free rate helpers (kept here — cross-cutting concern)
    # -------------------------------------------------------------------------

    @staticmethod
    def _coerce_rate_to_decimal(value: Any) -> float | None:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not pd.notna(parsed):
            return None
        # Percent form (e.g. 38.0) -> decimal annualized (0.38)
        if abs(parsed) > 1.5:
            return parsed / 100.0
        return parsed

    def _risk_free_rate_from_csv(self) -> float | None:
        # 1) Dedicated historical deposit-rate cache (if present).
        deposit_file = self.data_dir / "tcmb_deposit_rates.csv"
        if deposit_file.exists():
            try:
                frame = pd.read_csv(deposit_file)
                for col in ("deposit_rate", "risk_free_rate", "rate", "value"):
                    if col in frame.columns:
                        series = pd.to_numeric(frame[col], errors="coerce").dropna()
                        if not series.empty:
                            candidate = self._coerce_rate_to_decimal(series.iloc[-1])
                            if candidate is not None:
                                return float(candidate)
            except Exception as exc:
                logger.warning("  ⚠️  Failed to read fallback rate file %s: %s", deposit_file, exc)

        # 2) Generic macro indicators fallback.
        tcmb_file = self.paths.tcmb_indicators
        if tcmb_file.exists():
            try:
                frame = pd.read_csv(tcmb_file)
                candidates = (
                    "risk_free_rate",
                    "policy_rate",
                    "deposit_rate",
                    "tr_10y_yield",
                    "turkey_10y_yield",
                    "bond_10y",
                    "ten_year_yield",
                    "10y",
                )
                for col in candidates:
                    if col not in frame.columns:
                        continue
                    series = pd.to_numeric(frame[col], errors="coerce").dropna()
                    if series.empty:
                        continue
                    candidate = self._coerce_rate_to_decimal(series.iloc[-1])
                    if candidate is not None:
                        return float(candidate)
            except Exception as exc:
                logger.warning(
                    "  ⚠️  Failed to read fallback indicators file %s: %s", tcmb_file, exc
                )

        return None

    @property
    def risk_free_rate(self) -> float:
        """Annual risk-free rate as decimal (live borsapy -> CSV -> fallback)."""
        if self._risk_free_rate is None:
            resolved: float | None = None

            try:
                resolved = self._coerce_rate_to_decimal(self.fixed_income.get_risk_free_rate())
            except Exception as exc:
                logger.warning("  ⚠️  Live risk-free rate fetch failed: %s", exc)

            if resolved is None:
                resolved = self._risk_free_rate_from_csv()

            if resolved is None:
                resolved = float(DEFAULT_FALLBACK_RISK_FREE_RATE)

            self._risk_free_rate = float(resolved)
            logger.info("  ✅ Risk-free rate resolved: %.4f", self._risk_free_rate)

        return self._risk_free_rate

    # -------------------------------------------------------------------------
    # Price delegations → PriceLoader
    # -------------------------------------------------------------------------

    def load_prices(
        self,
        prices_file: Path | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        symbols: list[str] | None = None,
    ) -> pd.DataFrame:
        return self.price_loader.load_prices(
            prices_file=prices_file,
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
        )

    def build_close_panel(self, prices: pd.DataFrame) -> pd.DataFrame:
        return self.price_loader.build_close_panel(prices)

    def build_open_panel(self, prices: pd.DataFrame) -> pd.DataFrame:
        return self.price_loader.build_open_panel(prices)

    def build_volume_panel(self, prices: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
        return self.price_loader.build_volume_panel(prices, lookback=lookback)

    def load_xautry_prices(
        self,
        csv_path: Path | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.Series:
        return self.price_loader.load_xautry_prices(
            csv_path=csv_path,
            start_date=start_date,
            end_date=end_date,
        )

    def load_xu100_prices(self, csv_path: Path | None = None) -> pd.Series:
        return self.price_loader.load_xu100_prices(csv_path=csv_path)

    def load_xu100(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.Series:
        return self.price_loader.load_xu100(start_date=start_date, end_date=end_date)

    def load_usdtry(self) -> pd.DataFrame:
        return self.price_loader.load_usdtry()

    def load_oil_prices(self) -> pd.DataFrame | None:
        return self.price_loader.load_oil_prices()

    def load_prices_borsapy(
        self,
        symbols: list[str] | None = None,
        period: str = "5y",
        index: str = "XU100",
    ) -> pd.DataFrame:
        return self.price_loader.load_prices_borsapy(
            symbols=symbols,
            period=period,
            index=index,
        )

    # -------------------------------------------------------------------------
    # Fundamentals delegations → FundamentalsLoader
    # -------------------------------------------------------------------------

    def load_fundamentals(self) -> Dict:
        return self.fundamentals_loader.load_fundamentals(
            borsapy_adapter=self._borsapy_adapter,
        )

    def load_fundamentals_parquet(self) -> pd.DataFrame | None:
        return self.fundamentals_loader.load_fundamentals_parquet()

    def load_fundamental_metrics(self) -> pd.DataFrame:
        return self.fundamentals_loader.load_fundamental_metrics(data_dir=self.data_dir)

    # -------------------------------------------------------------------------
    # Shares / market-cap delegations → SharesLoader
    # -------------------------------------------------------------------------

    def load_shares_outstanding_panel(self) -> pd.DataFrame:
        return self.shares_loader.load_shares_outstanding_panel()

    def load_shares_outstanding(self, ticker: str) -> pd.Series:
        return self.shares_loader.load_shares_outstanding(ticker=ticker)

    def load_market_caps(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        symbols: list[str] | None = None,
    ) -> pd.DataFrame:
        return self.shares_loader.load_market_caps(
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
        )

    # -------------------------------------------------------------------------
    # Regime delegations → RegimeLoader
    # -------------------------------------------------------------------------

    def load_regime_predictions(self, features: pd.DataFrame | None = None) -> pd.Series:
        return self.regime_loader.load_regime_predictions(features=features)

    def load_regime_allocations(self) -> Dict[RegimeLabel, float]:
        return self.regime_loader.load_regime_allocations()

    def load_regime_labels(self) -> dict:
        return self.regime_loader.load_regime_labels()

    # -------------------------------------------------------------------------
    # Simple DataPaths-based methods (no delegation needed)
    # -------------------------------------------------------------------------

    def load_sector_mapping(self) -> pd.DataFrame:
        """Load BIST sector classification mapping."""
        sector_file = self.paths.sector_classification
        if not sector_file.exists():
            return pd.DataFrame()
        if sector_file.suffix == ".parquet":
            return pd.read_parquet(sector_file)
        return pd.read_csv(sector_file)

    def validate_data(self) -> dict:
        """Validate required data files using DataPaths."""
        return self.paths.validate()

    # -------------------------------------------------------------------------
    # Adapter pass-throughs (retained for backward compatibility)
    # -------------------------------------------------------------------------

    def create_portfolio_analytics(
        self,
        holdings: dict[str, float] | None = None,
        weights: dict[str, float] | None = None,
        returns: pd.Series | None = None,
        benchmark: str = "XU100",
        name: str = "Portfolio",
    ):
        return self.portfolio_analytics.create_portfolio_analytics(
            holdings=holdings,
            weights=weights,
            returns=returns,
            benchmark=benchmark,
            name=name,
        )

    def analyze_strategy_performance(
        self,
        equity_curve: pd.Series,
        benchmark_curve: pd.Series | None = None,
        name: str = "Strategy",
    ):
        return self.portfolio_analytics.analyze_strategy_performance(
            equity_curve=equity_curve,
            benchmark_curve=benchmark_curve,
            name=name,
        )

    def get_economic_calendar(
        self,
        days_ahead: int = 7,
        countries: list[str] | None = None,
    ) -> pd.DataFrame:
        return self.macro_adapter.get_economic_calendar(
            days_ahead=days_ahead,
            countries=countries,
        )

    def get_inflation_data(self, periods: int = 24) -> pd.DataFrame:
        return self.macro_adapter.get_inflation_data(periods=periods)

    def get_bond_yields(self) -> dict:
        return self.macro_adapter.get_bond_yields()

    def get_stock_news(self, symbol: str, limit: int = 10) -> list[dict]:
        return self.macro_adapter.get_stock_news(symbol=symbol, limit=limit)

    def get_macro_summary(self) -> dict:
        return self.macro_adapter.get_macro_summary()

    def technical_scan(
        self,
        condition: str | None = None,
        universe: str | list[str] = "XU100",
        interval: str = "1d",
        conditions: list[str] | None = None,
    ) -> pd.DataFrame:
        """Run technical scans via ``TechnicalScannerEngine``."""
        try:
            from server.engines.technical_scanner import TechnicalScannerEngine
        except ImportError as exc:
            raise ImportError(
                "technical_scan() requires the server package. "
                'Install with `pip install -e ".[server]"` or call '
                "server.engines.technical_scanner.TechnicalScannerEngine directly."
            ) from exc

        scanner = TechnicalScannerEngine()
        if conditions:
            return scanner.scan_multi(
                universe=universe,
                conditions=conditions,
                interval=interval,
            )
        if condition is None:
            raise ValueError("condition is required when conditions is not provided.")
        return scanner.scan(
            universe=universe,
            condition=condition,
            interval=interval,
        )

    # -------------------------------------------------------------------------
    # Deprecated borsapy passthroughs (keep for backward-compat with warnings)
    # -------------------------------------------------------------------------

    def get_index_components_borsapy(self, index: str = "XU100") -> list[str]:
        """.. deprecated:: Use ``borsapy_adapter.get_index_components()``."""
        import warnings

        warnings.warn(
            "get_index_components_borsapy() is deprecated, "
            "use borsapy_adapter.get_index_components() directly",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.borsapy_adapter.get_index_components(index=index)

    def get_financials_borsapy(self, symbol: str) -> dict[str, pd.DataFrame]:
        """.. deprecated:: Use ``borsapy_adapter.get_financials()``."""
        import warnings

        warnings.warn(
            "get_financials_borsapy() is deprecated, use borsapy_adapter.get_financials() directly",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.borsapy_adapter.get_financials(symbol=symbol)

    def get_financial_ratios_borsapy(self, symbol: str) -> pd.DataFrame:
        """.. deprecated:: Use ``borsapy_adapter.get_financial_ratios()``."""
        import warnings

        warnings.warn(
            "get_financial_ratios_borsapy() is deprecated, "
            "use borsapy_adapter.get_financial_ratios() directly",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.borsapy_adapter.get_financial_ratios(symbol=symbol)

    def get_dividends_borsapy(self, symbol: str) -> pd.DataFrame:
        """.. deprecated:: Use ``borsapy_adapter.get_dividends()``."""
        import warnings

        warnings.warn(
            "get_dividends_borsapy() is deprecated, use borsapy_adapter.get_dividends() directly",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.borsapy_adapter.get_dividends(symbol=symbol)

    def get_fast_info_borsapy(self, symbol: str) -> dict:
        """.. deprecated:: Use ``borsapy_adapter.get_fast_info()``."""
        import warnings

        warnings.warn(
            "get_fast_info_borsapy() is deprecated, use borsapy_adapter.get_fast_info() directly",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.borsapy_adapter.get_fast_info(symbol=symbol)

    def screen_stocks_borsapy(
        self,
        template: str | None = None,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """.. deprecated:: Use ``borsapy_adapter.screen_stocks()``."""
        import warnings

        warnings.warn(
            "screen_stocks_borsapy() is deprecated, use borsapy_adapter.screen_stocks() directly",
            DeprecationWarning,
            stacklevel=2,
        )
        merged_filters = dict(filters or {})
        merged_filters.update(kwargs)
        return self.borsapy_adapter.screen_stocks(template=template, filters=merged_filters)

    def get_stock_data_borsapy(
        self,
        symbol: str,
        period: str = "1mo",
        indicators: list[str] | None = None,
    ) -> StockData | None:
        """.. deprecated:: Use ``borsapy_adapter.get_stock_data()``."""
        import warnings

        warnings.warn(
            "get_stock_data_borsapy() is deprecated, use borsapy_adapter.get_stock_data() directly",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.borsapy_adapter.get_stock_data(
            symbol=symbol,
            period=period,
            indicators=indicators,
        )

    def get_history_with_indicators_borsapy(
        self,
        symbol: str,
        indicators: list[str] | None = None,
        period: str = "2y",
    ) -> pd.DataFrame:
        """.. deprecated:: Use ``borsapy_adapter.get_history_with_indicators()``."""
        import warnings

        warnings.warn(
            "get_history_with_indicators_borsapy() is deprecated, "
            "use borsapy_adapter.get_history_with_indicators() directly",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.borsapy_adapter.get_history_with_indicators(
            symbol=symbol,
            indicators=indicators,
            period=period,
        )
