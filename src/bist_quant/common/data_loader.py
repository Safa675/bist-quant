"""
Common Data Loader - Centralized data loading to eliminate redundant I/O

This module loads all fundamental data, price data, and regime predictions ONCE
and caches them in memory for use by all factor models.

Supports multiple data sources:
- Local parquet/CSV files (primary)
- Borsapy API (alternative/supplement)
"""

import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Final

import pandas as pd

from bist_quant.clients.borsapy_adapter import BorsapyAdapter, StockData
from bist_quant.common.enums import RegimeLabel
from bist_quant.clients.fixed_income_provider import (
    DEFAULT_FALLBACK_RISK_FREE_RATE,
    FixedIncomeProvider,
)
from bist_quant.clients.derivatives_provider import DerivativesProvider
from bist_quant.clients.fx_enhanced_provider import FXEnhancedProvider
from bist_quant.clients.macro_adapter import MacroAdapter
from bist_quant.common.panel_cache import PanelCache
from bist_quant.common.portfolio_analytics import PortfolioAnalyticsAdapter
from bist_quant.settings import PROJECT_ROOT
from .data_paths import DataPaths, get_data_paths

logger = logging.getLogger(__name__)
FETCHER_DIR: Final[Path] = PROJECT_ROOT / "src" / "bist_quant" / "fetcher"
REGIME_DIR_CANDIDATES = [
    PROJECT_ROOT / "outputs" / "regime" / "simple_regime",
    PROJECT_ROOT / "outputs" / "regime",
    PROJECT_ROOT / "regime_filter",
    PROJECT_ROOT / "Simple Regime Filter",
    PROJECT_ROOT / "Regime Filter",
]


def _normalize_dt_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Strip timezone and floor to midnight for reliable date alignment."""
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.floor("D")


def _normalize_dt_series(s: pd.Series) -> pd.Series:
    """Strip timezone and floor to midnight for a datetime Series."""
    if hasattr(s.dt, "tz") and s.dt.tz is not None:
        s = s.dt.tz_localize(None)
    return s.dt.floor("D")


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

    The loader provides a unified interface for local parquet/CSV inputs plus
    optional borsapy/macro adapters. Expensive tables are cached after first use
    to reduce repeated I/O during backtests.

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
        self.fetcher_dir = PROJECT_ROOT / "src" / "bist_quant" / "data_pipeline" / "fetcher_scripts"

        resolved_macro_events = (
            Path(macro_events_path)
            if macro_events_path is not None
            else self.fetcher_dir / "macro_events.py"
        )
        if not resolved_macro_events.exists():
            resolved_macro_events = FETCHER_DIR / "tcmb_data_fetcher.py"

        # Data source priority: "auto" | "borsapy" | "local"
        # Since we migrated to strictly borsapy, auto now defaults entirely to borsapy
        self._data_source_priority = (
            data_source_priority
            or os.getenv("BIST_DATA_SOURCE", "borsapy")
        ).strip().lower()
        if self._data_source_priority == "auto":
            self._data_source_priority = "borsapy"

        # Cache
        self._fundamentals = None
        self._prices = None
        self._close_df = None
        self._open_df = None
        self._volume_df = None
        self._volume_lookback = None
        self._regime_series = None
        self._regime_allocations = None
        self._xautry_prices = None
        self._xu100_prices = None
        self._fundamentals_parquet = None
        self._isyatirim_parquet = None
        self._shares_consolidated = None
        self._fixed_income_provider: FixedIncomeProvider | None = None
        self._derivatives_provider: DerivativesProvider | None = None
        self._fx_enhanced_provider: FXEnhancedProvider | None = None
        self._risk_free_rate: float | None = None
        self.panel_cache = PanelCache(
            max_entries=_env_int("BIST_PANEL_CACHE_MAX_ENTRIES", 32),
        )

        self._borsapy_adapter = BorsapyAdapter(self)
        self._macro_adapter = MacroAdapter(self, macro_events_path=resolved_macro_events)
        self._portfolio_analytics_adapter = PortfolioAnalyticsAdapter(self)
        self._native_borsapy_module = None

        # Freshness gate controls (defaults are strict for production safety).
        self._fundamentals_freshness_gate_enabled = os.getenv(
            "BIST_ENFORCE_FUNDAMENTAL_FRESHNESS",
            "1",
        ).strip().lower() not in {"0", "false", "no", "off"}
        self._allow_stale_fundamentals = os.getenv(
            "BIST_ALLOW_STALE_FUNDAMENTALS",
            "0",
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._freshness_threshold_overrides = {
            "max_median_staleness_days": _env_int("BIST_MAX_MEDIAN_STALENESS_DAYS", 120),
            "max_pct_over_120_days": _env_float("BIST_MAX_PCT_OVER_120_DAYS", 0.90),
            "min_q4_coverage_pct": _env_float("BIST_MIN_Q4_2025_COVERAGE_PCT", 0.10),
            "max_max_staleness_days": _env_int("BIST_MAX_MAX_STALENESS_DAYS", 500),
            "grace_days": _env_int("BIST_STALENESS_GRACE_DAYS", 0),
        }

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
                logger.warning("  ⚠️  Failed to read fallback indicators file %s: %s", tcmb_file, exc)

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

    def _resolve_index_components_native(self, index: str = "XU100") -> list[str]:
        """Best-effort index component resolution via native borsapy module."""
        if self._native_borsapy_module is None:
            try:
                import borsapy as bp  # type: ignore[import-not-found]

                self._native_borsapy_module = bp
            except Exception:
                return []

        bp = self._native_borsapy_module
        if bp is None or not hasattr(bp, "index"):
            return []

        try:
            idx = bp.index(index)
        except Exception as exc:
            logger.warning(f"  ⚠️  Native borsapy index lookup failed for {index}: {exc}")
            return []

        symbols = getattr(idx, "component_symbols", None)
        if isinstance(symbols, list) and symbols:
            return [str(item).upper().split(".")[0] for item in symbols if item]

        components = getattr(idx, "components", None)
        if isinstance(components, list):
            out: list[str] = []
            seen: set[str] = set()
            for item in components:
                if not isinstance(item, dict):
                    continue
                symbol = str(item.get("symbol", "")).upper().split(".")[0]
                if not symbol or symbol in seen:
                    continue
                seen.add(symbol)
                out.append(symbol)
            return out

        return []

    def _borsapy_download_to_long(
        self,
        symbols: list[str],
        period: str = "5y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Download prices via native borsapy and normalize to long format."""
        if not symbols:
            return pd.DataFrame()

        if self._native_borsapy_module is None:
            try:
                import borsapy as bp  # type: ignore[import-not-found]

                self._native_borsapy_module = bp
                logger.info("  ✅ Native borsapy module initialized")
            except Exception as exc:
                logger.warning(f"  ⚠️  Native borsapy unavailable for fallback download: {exc}")
                return pd.DataFrame()

        bp = self._native_borsapy_module
        if bp is None:
            return pd.DataFrame()

        try:
            raw = bp.download(
                symbols,
                period=period,
                interval=interval,
                group_by="ticker",
                progress=False,
            )
        except Exception as exc:
            logger.warning(f"  ⚠️  Native borsapy fallback download failed: {exc}")
            return pd.DataFrame()

        if raw is None or raw.empty:
            return pd.DataFrame()

        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        frames: list[pd.DataFrame] = []

        if isinstance(raw.columns, pd.MultiIndex):
            lvl0 = raw.columns.get_level_values(0)
            if {"Open", "High", "Low", "Close"}.issubset(set(lvl0)):
                raw = raw.swaplevel(0, 1, axis=1).sort_index(axis=1)

            for ticker in dict.fromkeys(raw.columns.get_level_values(0)):
                sub = raw[ticker]
                if not isinstance(sub, pd.DataFrame) or sub.empty:
                    continue
                sub = sub.rename_axis("Date").reset_index()
                sub["Ticker"] = str(ticker).upper().split(".")[0]
                for col in required_cols:
                    if col not in sub.columns:
                        sub[col] = pd.NA
                frames.append(sub[["Date", "Ticker", *required_cols]])
        else:
            sub = raw.rename_axis("Date").reset_index()
            ticker = symbols[0]
            sub["Ticker"] = str(ticker).upper().split(".")[0]
            for col in required_cols:
                if col not in sub.columns:
                    sub[col] = pd.NA
            frames.append(sub[["Date", "Ticker", *required_cols]])

        if not frames:
            return pd.DataFrame()

        out = pd.concat(frames, ignore_index=True)
        if "Date" in out.columns:
            out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        return out

    def load_prices_borsapy(
        self,
        symbols: list[str] | None = None,
        period: str = "5y",
        index: str = "XU100",
    ) -> pd.DataFrame:
        resolved_symbols = [str(item).upper().split(".")[0] for item in (symbols or []) if item]
        if not resolved_symbols and symbols is None:
            resolved_symbols = self.borsapy_adapter.get_index_components(index=index)
        if not resolved_symbols and symbols is None:
            resolved_symbols = self._resolve_index_components_native(index=index)

        adapter_result = self.borsapy_adapter.load_prices(
            symbols=resolved_symbols or symbols,
            period=period,
            index=index,
        )
        if not adapter_result.empty:
            return adapter_result

        if not resolved_symbols:
            logger.warning("  ⚠️  No symbols resolved for borsapy load; returning empty frame")
            return pd.DataFrame()

        logger.warning("  ⚠️  Adapter returned no data, trying native borsapy fallback...")
        fallback = self._borsapy_download_to_long(
            symbols=resolved_symbols,
            period=period,
            interval="1d",
        )
        if fallback.empty:
            logger.warning("  ⚠️  Native borsapy fallback also returned no data")
        else:
            loaded = fallback["Ticker"].dropna().nunique() if "Ticker" in fallback.columns else 0
            logger.info(
                f"  ✅ Native fallback loaded {len(fallback)} price records for {loaded}/{len(resolved_symbols)} tickers"
            )
        return fallback

    def get_index_components_borsapy(self, index: str = "XU100") -> list[str]:
        """Get index components via borsapy. Prefer ``get_index_components_borsapy``."""
        warnings.warn(
            "get_index_components_borsapy() is deprecated, "
            "use borsapy_adapter.get_index_components() directly",
            DeprecationWarning, stacklevel=2,
        )
        return self.borsapy_adapter.get_index_components(index=index)

    def get_financials_borsapy(self, symbol: str) -> dict[str, pd.DataFrame]:
        """.. deprecated:: Use ``borsapy_adapter.get_financials()``."""
        warnings.warn(
            "get_financials_borsapy() is deprecated, "
            "use borsapy_adapter.get_financials() directly",
            DeprecationWarning, stacklevel=2,
        )
        return self.borsapy_adapter.get_financials(symbol=symbol)

    def get_financial_ratios_borsapy(self, symbol: str) -> pd.DataFrame:
        """.. deprecated:: Use ``borsapy_adapter.get_financial_ratios()``."""
        warnings.warn(
            "get_financial_ratios_borsapy() is deprecated, "
            "use borsapy_adapter.get_financial_ratios() directly",
            DeprecationWarning, stacklevel=2,
        )
        return self.borsapy_adapter.get_financial_ratios(symbol=symbol)

    def get_dividends_borsapy(self, symbol: str) -> pd.DataFrame:
        """.. deprecated:: Use ``borsapy_adapter.get_dividends()``."""
        warnings.warn(
            "get_dividends_borsapy() is deprecated, "
            "use borsapy_adapter.get_dividends() directly",
            DeprecationWarning, stacklevel=2,
        )
        return self.borsapy_adapter.get_dividends(symbol=symbol)

    def get_fast_info_borsapy(self, symbol: str) -> dict:
        """.. deprecated:: Use ``borsapy_adapter.get_fast_info()``."""
        warnings.warn(
            "get_fast_info_borsapy() is deprecated, "
            "use borsapy_adapter.get_fast_info() directly",
            DeprecationWarning, stacklevel=2,
        )
        return self.borsapy_adapter.get_fast_info(symbol=symbol)

    def screen_stocks_borsapy(
        self,
        template: str | None = None,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """.. deprecated:: Use ``borsapy_adapter.screen_stocks()``."""
        warnings.warn(
            "screen_stocks_borsapy() is deprecated, "
            "use borsapy_adapter.screen_stocks() directly",
            DeprecationWarning, stacklevel=2,
        )
        merged_filters = dict(filters or {})
        merged_filters.update(kwargs)
        return self.borsapy_adapter.screen_stocks(template=template, filters=merged_filters)

    def technical_scan(
        self,
        condition: str | None = None,
        universe: str | list[str] = "XU100",
        interval: str = "1d",
        conditions: list[str] | None = None,
    ) -> pd.DataFrame:
        """Run technical scans via ``TechnicalScannerEngine``."""
        from bist_quant.engines.technical_scanner import TechnicalScannerEngine

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

    def get_stock_data_borsapy(
        self,
        symbol: str,
        period: str = "1mo",
        indicators: list[str] | None = None,
    ) -> StockData | None:
        """.. deprecated:: Use ``borsapy_adapter.get_stock_data()``."""
        warnings.warn(
            "get_stock_data_borsapy() is deprecated, "
            "use borsapy_adapter.get_stock_data() directly",
            DeprecationWarning, stacklevel=2,
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
        warnings.warn(
            "get_history_with_indicators_borsapy() is deprecated, "
            "use borsapy_adapter.get_history_with_indicators() directly",
            DeprecationWarning, stacklevel=2,
        )
        return self.borsapy_adapter.get_history_with_indicators(
            symbol=symbol,
            indicators=indicators,
            period=period,
        )

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

    def get_economic_calendar(
        self,
        days_ahead: int = 7,
        countries: list[str] | None = None,
    ) -> pd.DataFrame:
        return self.macro_adapter.get_economic_calendar(
            days_ahead=days_ahead,
            countries=countries,
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

    def get_inflation_data(self, periods: int = 24) -> pd.DataFrame:
        return self.macro_adapter.get_inflation_data(periods=periods)

    def get_bond_yields(self) -> dict:
        return self.macro_adapter.get_bond_yields()

    def get_stock_news(self, symbol: str, limit: int = 10) -> list[dict]:
        return self.macro_adapter.get_stock_news(symbol=symbol, limit=limit)

    def get_macro_summary(self) -> dict:
        return self.macro_adapter.get_macro_summary()

    @staticmethod
    def _canonical_csv_path(path: Path) -> Path:
        text = str(path)
        if text.endswith(".csv.gz"):
            return path.with_suffix("")
        if path.suffix == ".parquet":
            return path.with_suffix(".csv")
        if path.suffix == ".csv":
            return path
        return path.with_suffix(".csv")

    @staticmethod
    def _price_source_candidates(path: Path) -> list[Path]:
        csv_path = DataLoader._canonical_csv_path(path)
        return [
            csv_path.with_suffix(".parquet"),
            Path(f"{csv_path}.gz"),
            csv_path,
        ]

    @staticmethod
    def _is_fundamentals_panel(frame: pd.DataFrame) -> bool:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return False
        if not isinstance(frame.index, pd.MultiIndex) or frame.index.nlevels < 3:
            return False
        names = {str(name) for name in frame.index.names if name is not None}
        required = {"ticker", "sheet_name", "row_name"}
        return required.issubset(names)

    def _load_consolidated_fundamentals(self) -> pd.DataFrame | None:
        """Load consolidated fundamentals directly from borsapy_cache, building on the fly if needed."""
        consolidated_path = self.paths.borsapy_cache_dir / "financials_consolidated.parquet"
        
        # 1. Try to load pre-built consolidated form
        if consolidated_path.exists():
            try:
                frame = pd.read_parquet(consolidated_path)
                if self._is_fundamentals_panel(frame):
                    logger.info("  📦 Loaded consolidated fundamentals from borsapy_cache")
                    return frame
            except Exception as e:
                logger.warning(f"  ⚠️  Failed to read {consolidated_path}: {e}")
        
        # 2. Build on the fly from borsapy_cache/financials
        financials_dir = self.paths.borsapy_cache_dir / "financials"
        if not financials_dir.exists():
            return None
            
        logger.info("  🔄 Building consolidated fundamentals from borsapy_cache...")
        
        SHEET_MAP = {
            "balance_sheet": "Bilanço",
            "income_stmt": "Gelir Tablosu (Çeyreklik)",
            "cash_flow": "Nakit Akış (Çeyreklik)"
        }
        
        rows = []
        count = 0
        for ticker_dir in financials_dir.iterdir():
            if not ticker_dir.is_dir():
                continue
            ticker = ticker_dir.name
            
            for path in ticker_dir.glob("*.parquet"):
                sheet_key = path.stem
                if sheet_key not in SHEET_MAP:
                    continue
                sheet_name = SHEET_MAP[sheet_key]
                
                try:
                    df = pd.read_parquet(path)
                    if df.empty or df.index.name != "Item":
                        continue
                        
                    # Reset index so 'Item' becomes a column
                    df_reset = df.reset_index()
                    for _, row in df_reset.iterrows():
                        row_name = row["Item"]
                        values = row.drop("Item").to_dict()
                        if not values:
                            continue
                        s = pd.Series(values, name=(ticker, sheet_name, row_name))
                        rows.append(s)
                except Exception:
                    continue
            count += 1
            if count % 100 == 0:
                logger.info(f"    Indexed {count} tickers...")
                
        if not rows:
            return None
            
        panel = pd.DataFrame(rows)
        panel.index = pd.MultiIndex.from_tuples(
            panel.index.tolist(),
            names=["ticker", "sheet_name", "row_name"],
        )
        # Drop fully NaN columns
        panel = panel.dropna(axis=1, how="all")
        
        try:
            panel.to_parquet(consolidated_path)
            logger.info("  💾 Saved new built consolidated fundamentals to borsapy_cache")
        except Exception as e:
            logger.warning(f"  ⚠️  Failed to write {consolidated_path}: {e}")
            
        return panel

    def _should_use_borsapy_for(self, category: str = "prices") -> bool:
        """Determine whether borsapy should be used for *category*.

        In ``auto`` and ``borsapy`` modes *all* categories go through
        borsapy (disk-cache → API fetch).  Only ``local`` mode uses the
        old manually-downloaded files in ``data/``.
        """
        prio = self._data_source_priority
        if prio == "local":
            return False
        # "auto" and "borsapy" both use borsapy for everything
        return True

    def _load_prices_via_borsapy(
        self,
        symbols: list[str] | None = None,
        period: str = "5y",
        index: str = "XUTUM",
    ) -> pd.DataFrame:
        """Attempt to load prices through the borsapy adapter + disk cache."""
        try:
            result = self.load_prices_borsapy(
                symbols=symbols, period=period, index=index,
            )
            if not result.empty:
                return result
        except Exception as exc:
            logger.warning("  ⚠️  Borsapy price fetch failed: %s", exc)
        return pd.DataFrame()

    def load_prices(
        self,
        prices_file: Path | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        symbols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Load stock prices.

        In ``auto`` / ``borsapy`` mode the **only** local data source is
        ``borsapy_cache/`` — old manually-downloaded files in ``data/``
        are **not** consulted.  If the cache is empty the data is fetched
        live from borsapy and persisted.

        ``local`` mode preserves the legacy behaviour (read from
        ``data/bist_prices_full.*``).
        """
        if self._prices is None:
            logger.info("\n📊 Loading price data...")

            # --- Borsapy / cache path (auto or borsapy mode) ---------------
            if prices_file is None and self._should_use_borsapy_for("prices"):
                borsapy_prices = self._load_prices_via_borsapy(
                    symbols=symbols, index="XUTUM",
                )
                if not borsapy_prices.empty:
                    self._prices = borsapy_prices
                    logger.info(
                        "  ✅ Loaded %d price records via borsapy (cache)",
                        len(self._prices),
                    )
                else:
                    raise RuntimeError(
                        "Borsapy price fetch returned no data. "
                        "Run 'python -m bist_quant.cli.cache_cli warm' to "
                        "populate the cache, or set BIST_DATA_SOURCE=local "
                        "to use old local files."
                    )

            # --- Legacy local file path (only when mode=local) -------------
            if self._prices is None:
                requested_path = (
                    Path(prices_file) if prices_file is not None else self.paths.prices_file
                )
                source = next(
                    (c for c in self._price_source_candidates(requested_path) if c.exists()), None
                )
                if source is None:
                    candidates = ", ".join(
                        str(c) for c in self._price_source_candidates(requested_path)
                    )
                    raise FileNotFoundError(f"Price file not found. Tried: {candidates}")

                if source.suffix == ".parquet":
                    logger.info(f"  📦 Using legacy Parquet: {source.name}")
                    self._prices = pd.read_parquet(
                        source,
                        columns=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"],
                    )
                elif source.suffix == ".gz":
                    logger.info(f"  🗜️  Using legacy CSV.GZ: {source.name}")
                    self._prices = pd.read_csv(
                        source,
                        compression="gzip",
                        usecols=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"],
                    )
                else:
                    logger.info(f"  📄 Using legacy CSV: {source.name}")
                    self._prices = pd.read_csv(
                        source,
                        usecols=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"],
                    )
                if "Date" in self._prices.columns:
                    self._prices["Date"] = pd.to_datetime(self._prices["Date"], errors="coerce")
                logger.info(f"  ✅ Loaded {len(self._prices)} price records (legacy)")

        prices = self._prices.copy()
        if start_date is not None:
            if "Date" in prices.columns:
                prices = prices[pd.to_datetime(prices["Date"], errors="coerce") >= start_date]
            elif isinstance(prices.index, pd.DatetimeIndex):
                prices = prices[prices.index >= start_date]
        if end_date is not None:
            if "Date" in prices.columns:
                prices = prices[pd.to_datetime(prices["Date"], errors="coerce") <= end_date]
            elif isinstance(prices.index, pd.DatetimeIndex):
                prices = prices[prices.index <= end_date]

        if symbols:
            normalized = {str(symbol).upper().split(".")[0] for symbol in symbols}
            if "Ticker" in prices.columns:
                tickers = prices["Ticker"].astype(str).str.upper().str.split(".").str[0]
                prices = prices[tickers.isin(normalized)]
            elif not prices.empty:
                keep = [
                    col for col in prices.columns if str(col).upper().split(".")[0] in normalized
                ]
                prices = prices[keep]

        return prices

    def build_close_panel(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Build close price panel (Date x Ticker)"""
        if self._close_df is None:
            logger.info("  Building close price panel...")
            prices = prices.copy()
            prices["Ticker"] = prices["Ticker"].apply(lambda x: str(x).split(".")[0].upper())
            close_df = prices.pivot_table(
                index="Date",
                columns="Ticker",
                values="Close",
                aggfunc="last",
            ).sort_index()
            self._close_df = close_df
            logger.info(f"  ✅ Close panel: {close_df.shape[0]} days × {close_df.shape[1]} tickers")
        return self._close_df

    def build_open_panel(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Build open price panel (Date x Ticker)"""
        if self._open_df is None:
            logger.info("  Building open price panel...")
            prices = prices.copy()
            prices["Ticker"] = prices["Ticker"].apply(lambda x: str(x).split(".")[0].upper())
            open_df = prices.pivot_table(
                index="Date",
                columns="Ticker",
                values="Open",
                aggfunc="last",
            ).sort_index()
            self._open_df = open_df
            logger.info(f"  ✅ Open panel: {open_df.shape[0]} days × {open_df.shape[1]} tickers")
        return self._open_df

    def build_volume_panel(self, prices: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
        """Build rolling median volume panel"""
        panel_cache = getattr(self, "panel_cache", None)
        cache_key = None
        if panel_cache is not None:
            cache_key = panel_cache.make_key(
                "volume",
                lookback=int(lookback),
                rows=int(len(prices)),
                date_start=(
                    str(pd.to_datetime(prices["Date"], errors="coerce").min())
                    if "Date" in prices.columns
                    else None
                ),
                date_end=(
                    str(pd.to_datetime(prices["Date"], errors="coerce").max())
                    if "Date" in prices.columns
                    else None
                ),
                ticker_count=(
                    int(prices["Ticker"].nunique()) if "Ticker" in prices.columns else None
                ),
            )
            cached = panel_cache.get(cache_key)
            if isinstance(cached, pd.DataFrame):
                self._volume_df = cached
                self._volume_lookback = int(lookback)
                return cached

        if self._volume_df is None or self._volume_lookback != int(lookback):
            logger.info(f"  Building volume panel (lookback={lookback})...")
            prices = prices.copy()
            prices["Ticker"] = prices["Ticker"].apply(lambda x: str(x).split(".")[0].upper())
            vol_pivot = prices.pivot_table(
                index="Date",
                columns="Ticker",
                values="Volume",
                aggfunc="last",
            ).sort_index()

            # Drop holiday rows
            valid_pct = vol_pivot.notna().mean(axis=1)
            holiday_mask = valid_pct < 0.5
            if holiday_mask.any():
                vol_clean = vol_pivot.loc[~holiday_mask]
            else:
                vol_clean = vol_pivot

            median_adv = vol_clean.rolling(lookback, min_periods=lookback).median()
            median_adv = median_adv.reindex(vol_pivot.index).ffill()
            self._volume_df = median_adv
            self._volume_lookback = int(lookback)
            if panel_cache is not None and cache_key is not None:
                panel_cache.set(cache_key, median_adv)
            logger.info(
                f"  ✅ Volume panel: {median_adv.shape[0]} days × {median_adv.shape[1]} tickers"
            )
        return self._volume_df

    def load_fundamentals(self) -> Dict:
        """Load all fundamental data.

        Priority:
        1. Consolidated parquet (existing behavior)
        2. Per-company xlsx files (existing fallback)
        3. Borsapy financial statements (supplementary fill when
           ``data_source_priority`` is ``"borsapy"`` and local data
           is missing)
        """
        if self._fundamentals is None:
            logger.info("\n📈 Loading fundamental data...")
            fundamentals = {}
            
            # 1. Try to load the consolidated panel from the new cache logic
            self._fundamentals_parquet = self._load_consolidated_fundamentals()
            if self._fundamentals_parquet is not None:
                self._enforce_fundamentals_freshness_gate(self._fundamentals_parquet)
                tickers = (
                    self._fundamentals_parquet.index.get_level_values("ticker")
                    .unique()
                    .tolist()
                )
                for ticker in tickers:
                    fundamentals[ticker] = {"path": None, "borsapy": True}
                logger.info(f"  ✅ Loaded consolidated fundamentals for {len(tickers)} tickers")
            else:
                # 2. Legacy fallback to individual xlsx files
                count = 0
                for file_path in self.fundamental_dir.rglob("*.xlsx"):
                    ticker = file_path.stem.split(".")[0].upper()
                    try:
                        fundamentals[ticker] = {
                            "path": file_path,
                            "income": None,  # Lazy load
                            "balance": None,
                            "cashflow": None,
                        }
                        count += 1
                    except Exception:
                        continue
                if count > 0:
                    logger.info(f"  ✅ Indexed {count} fundamental data files")

                # 3. Borsapy supplementary fill
                if count == 0 and self._should_use_borsapy_for("fundamentals"):
                    self._borsapy_fundamentals_fill(fundamentals)

            self._fundamentals = fundamentals
        return self._fundamentals

    def _borsapy_fundamentals_fill(
        self, fundamentals: Dict,
    ) -> None:
        """Try to populate fundamental dict from borsapy financial statements."""
        try:
            logger.info("  🔄 Attempting borsapy fundamentals fill...")
            # Get universe from borsapy index components
            symbols = self.borsapy_adapter.get_index_components(index="XU100")
            if not symbols:
                logger.warning("  ⚠️  Could not resolve symbols for borsapy fundamentals fill")
                return

            filled = 0
            for sym in symbols:
                if sym in fundamentals:
                    continue
                try:
                    stmts = self.borsapy_adapter.get_financials(symbol=sym)
                    if stmts and any(
                        isinstance(v, pd.DataFrame) and not v.empty
                        for v in stmts.values()
                    ):
                        fundamentals[sym] = {"path": None, "borsapy": True}
                        filled += 1
                except Exception:
                    continue

            if filled:
                logger.info(f"  ✅ Borsapy fill added {filled} tickers")
            else:
                logger.warning("  ⚠️  Borsapy fundamentals fill returned no data")
        except Exception as exc:
            logger.warning(f"  ⚠️  Borsapy fundamentals fill failed: {exc}")

    def load_fundamentals_parquet(self) -> pd.DataFrame | None:
        """Load consolidated fundamentals parquet if available"""
        if self._fundamentals_parquet is None:
            self._fundamentals_parquet = self._load_consolidated_fundamentals()
            if self._fundamentals_parquet is not None:
                self._enforce_fundamentals_freshness_gate(self._fundamentals_parquet)
        return self._fundamentals_parquet

    @staticmethod
    def _turkish_expected_publication_date(reference_date: pd.Timestamp | None = None) -> tuple[str, pd.Timestamp]:
        """Return the latest quarter whose deadline has passed.

        Turkish reporting deadlines:
        - Q1/Q2/Q3: 45 calendar days after quarter end
        - Q4: 75 calendar days after year end

        Returns (quarter_label, deadline).
        """
        if reference_date is None:
            reference_date = pd.Timestamp.now().normalize()

        year = reference_date.year
        # (quarter_end_month, quarter_label, deadline_days)
        deadlines = [
            (3, f"{year}/3",   45),   # Q1
            (6, f"{year}/6",   45),   # Q2
            (9, f"{year}/9",   45),   # Q3
            (12, f"{year}/12", 75),   # Q4
        ]
        # Also check prior year Q4 — it may be the latest publishable one
        deadlines.insert(0, (12, f"{year - 1}/12", 75))

        latest_q = None
        latest_deadline = None
        for end_month, label, days in deadlines:
            q_year = int(label.split("/")[0])
            quarter_end = pd.Timestamp(year=q_year, month=end_month, day=1) + pd.offsets.MonthEnd(0)
            deadline = quarter_end + pd.Timedelta(days=days)
            if reference_date >= deadline:
                if latest_deadline is None or deadline > latest_deadline:
                    latest_q = label
                    latest_deadline = deadline

        if latest_q is None:
            # Fallback: previous year Q3
            latest_q = f"{year - 1}/9"
            latest_deadline = pd.Timestamp(year=year - 1, month=9, day=30) + pd.Timedelta(days=45)
        return latest_q, latest_deadline

    def _enforce_fundamentals_freshness_gate(self, panel: pd.DataFrame) -> None:
        """Calendar-aware freshness gate for Turkish financial reporting."""
        if panel is None or panel.empty:
            return
        if not self._fundamentals_freshness_gate_enabled:
            return
        if self._allow_stale_fundamentals:
            return

        expected_q, deadline = self._turkish_expected_publication_date()
        logger.info(f"  📅 Turkish calendar: expecting at least {expected_q} (deadline was {deadline:%Y-%m-%d})")

        # Check if the expected quarter column exists in the panel
        if isinstance(panel.columns, pd.Index):
            available_periods = [c for c in panel.columns if "/" in str(c)]
            if expected_q in available_periods:
                # Count how many tickers have data for this quarter
                if isinstance(panel.index, pd.MultiIndex) and "ticker" in panel.index.names:
                    ticker_level = panel.index.get_level_values("ticker")
                    coverage = panel[expected_q].groupby(ticker_level).apply(lambda s: s.notna().any())
                    pct = coverage.mean()
                else:
                    pct = panel[expected_q].notna().mean()
                logger.info(f"  📊 Coverage for {expected_q}: {pct:.1%}")
            else:
                logger.debug(f"  ℹ️  Expected quarter {expected_q} not found in data columns")
        # Gate is advisory-only in new architecture — never block

    def load_shares_outstanding_panel(self) -> pd.DataFrame:
        """Load Date x Ticker shares outstanding panel."""
        panel_cache = getattr(self, "panel_cache", None)
        cache_key = None
        if panel_cache is not None:
            cache_key = panel_cache.make_key(
                "shares_outstanding",
                data_dir=str(self.data_dir),
            )
            cached = panel_cache.get(cache_key)
            if isinstance(cached, pd.DataFrame):
                self._shares_consolidated = cached

        if self._shares_consolidated is None:
            shares_csv = self.paths.shares_outstanding
            shares_parquet = shares_csv.with_suffix(".parquet")
            shares_csv_gz = shares_csv.with_suffix(f"{shares_csv.suffix}.gz")
            panel: pd.DataFrame | None = None

            if shares_parquet.exists():
                logger.info("  📦 Loading consolidated shares file (Parquet)...")
                panel = pd.read_parquet(shares_parquet)
            elif shares_csv_gz.exists() or shares_csv.exists():
                source = shares_csv_gz if shares_csv_gz.exists() else shares_csv
                logger.info(f"  📊 Loading consolidated shares file ({source.name})...")
                panel = pd.read_csv(
                    source,
                    index_col=0,
                    parse_dates=True,
                    compression="gzip" if source.suffix == ".gz" else "infer",
                )

            if panel is not None:
                if "Date" in panel.columns and not isinstance(panel.index, pd.DatetimeIndex):
                    panel = panel.set_index("Date")
                panel.index = pd.to_datetime(panel.index, errors="coerce")
                panel = panel.sort_index()
                panel.columns = [str(c).upper() for c in panel.columns]
                self._shares_consolidated = panel
                logger.info(f"  ✅ Loaded shares for {panel.shape[1]} tickers")
            else:
                # Fallback: build panel from consolidated isyatirim parquet
                isy = self._load_isyatirim_parquet()
                if isy is not None and not isy.empty:
                    try:
                        daily = isy[isy["sheet_type"] == "daily"]
                        required_cols = {"ticker", "HGDG_TARIH", "SERMAYE"}
                        if not daily.empty and required_cols.issubset(daily.columns):
                            panel = daily.pivot_table(
                                index="HGDG_TARIH",
                                columns="ticker",
                                values="SERMAYE",
                                aggfunc="last",
                            )
                            panel.index = pd.to_datetime(panel.index, errors="coerce")
                            panel = panel.sort_index()
                            panel.columns = [str(c).upper() for c in panel.columns]
                            self._shares_consolidated = panel
                            logger.info(
                                f"  ✅ Built shares panel from isyatirim parquet for {panel.shape[1]} tickers"
                            )
                        else:
                            self._shares_consolidated = None
                    except Exception as exc:
                        logger.warning(
                            f"  ⚠️  Failed to build shares panel from isyatirim parquet: {exc}"
                        )
                        self._shares_consolidated = None
                else:
                    logger.warning("  ⚠️  Consolidated shares file not found")
                    self._shares_consolidated = None

        if self._shares_consolidated is None:
            return pd.DataFrame()
        if panel_cache is not None and cache_key is not None:
            panel_cache.set(cache_key, self._shares_consolidated)
        return self._shares_consolidated

    def load_shares_outstanding(self, ticker: str) -> pd.Series:
        """Load shares outstanding from consolidated file or fundamentals (fast!)"""
        shares_panel = self.load_shares_outstanding_panel()
        if not shares_panel.empty and ticker in shares_panel.columns:
            return shares_panel[ticker].dropna()

        # Extract from fundamentals parquet (Ödenmiş Sermaye)
        try:
            from bist_quant.common.utils import get_consolidated_sheet, pick_row_from_sheet, coerce_quarter_cols
            fund_parquet = self.load_fundamentals_parquet()
            if fund_parquet is not None:
                bs = get_consolidated_sheet(fund_parquet, ticker, "Bilanço")
                if not bs.empty:
                    sermaye_row = pick_row_from_sheet(bs, ("Ödenmiş Sermaye", "ÖDENMİŞ SERMAYE"))
                    if sermaye_row is not None and not sermaye_row.empty:
                        sermaye_parsed = coerce_quarter_cols(sermaye_row)
                        if not sermaye_parsed.empty:
                            return sermaye_parsed.dropna()
        except Exception as e:
            logger.warning(f"Error extracting shares from fundamentals for {ticker}: {e}")

        # Fallback: consolidated isyatirim parquet (if available)
        isy = self._load_isyatirim_parquet()
        if isy is not None:
            try:
                daily = isy[(isy["ticker"] == ticker) & (isy["sheet_type"] == "daily")]
                if not daily.empty and "HGDG_TARIH" in daily.columns and "SERMAYE" in daily.columns:
                    series = daily.set_index("HGDG_TARIH")["SERMAYE"].dropna()
                    return series
            except Exception:
                pass

        # Fallback to individual Excel file (slow)
        excel_path = self.isyatirim_dir / f"{ticker}_2016_2026_daily_and_quarterly.xlsx"

        if not excel_path.exists():
            return pd.Series(dtype=float)

        try:
            df = pd.read_excel(excel_path, sheet_name="daily")
            if "HGDG_TARIH" not in df.columns or "SERMAYE" not in df.columns:
                return pd.Series(dtype=float)

            df["HGDG_TARIH"] = pd.to_datetime(df["HGDG_TARIH"])
            df = df.set_index("HGDG_TARIH").sort_index()
            return df["SERMAYE"].dropna()
        except Exception:
            return pd.Series(dtype=float)

    def _load_isyatirim_parquet(self) -> pd.DataFrame | None:
        """Load consolidated isyatirim prices parquet (used for shares fallback)"""
        if self._isyatirim_parquet is None:
            legacy_file = self.data_dir / "isyatirim_prices_consolidated.parquet"
            parquet_file = (
                self.paths.isyatirim_prices if self.paths.isyatirim_prices.exists() else legacy_file
            )
            if parquet_file.exists():
                logger.info("  📦 Loading consolidated isyatirim prices (Parquet)...")
                try:
                    self._isyatirim_parquet = pd.read_parquet(
                        parquet_file,
                        columns=["ticker", "sheet_type", "HGDG_TARIH", "SERMAYE"],
                    )
                except Exception:
                    frame = pd.read_parquet(
                        parquet_file,
                        columns=["HGDG_HS_KODU", "HGDG_TARIH", "SERMAYE"],
                    )
                    frame = frame.rename(columns={"HGDG_HS_KODU": "ticker"})
                    frame["sheet_type"] = "daily"
                    self._isyatirim_parquet = frame
        return self._isyatirim_parquet

    def load_regime_predictions(self, features: pd.DataFrame | None = None) -> pd.Series:
        """
        Load regime labels from regime filter outputs.

        Args:
            features: Unused legacy argument kept for backward compatibility.
        """
        del features  # Backward compatibility placeholder

        if self._regime_series is None:
            logger.info("\n🎯 Loading regime labels...")
            candidate_files: list[Path] = []
            direct_regime_file = self.regime_model_dir / "regime_features.csv"
            candidate_files.append(direct_regime_file)
            if self.regime_model_dir.name.lower() != "outputs":
                candidate_files.append(self.regime_model_dir / "outputs" / "regime_features.csv")
            if self.regime_model_dir.parent != self.regime_model_dir:
                candidate_files.append(
                    self.regime_model_dir.parent / "outputs" / "regime_features.csv"
                )
            candidate_files.extend(
                [p / "regime_features.csv" for p in REGIME_DIR_CANDIDATES]
            )
            candidate_files.extend(
                [p / "outputs" / "regime_features.csv" for p in REGIME_DIR_CANDIDATES]
            )
            regime_file = next((f for f in candidate_files if f.exists()), candidate_files[0])

            if not regime_file.exists():
                candidate_dirs = ", ".join(
                    str(path.parent if path.name == "regime_features.csv" else path)
                    for path in candidate_files
                )
                raise FileNotFoundError(
                    f"Regime file not found in expected locations: {candidate_dirs}\n"
                    "Run the simplified regime pipeline to generate outputs."
                )

            regime_df = pd.read_csv(regime_file)
            if regime_df.empty:
                raise ValueError(f"Regime file is empty: {regime_file}")

            date_col = next(
                (c for c in ("Date", "date", "DATE") if c in regime_df.columns),
                regime_df.columns[0],
            )
            regime_df[date_col] = pd.to_datetime(regime_df[date_col], errors="coerce")
            regime_df = regime_df.dropna(subset=[date_col]).set_index(date_col).sort_index()

            regime_col = next(
                (
                    c
                    for c in ("regime_label", "simplified_regime", "regime", "detailed_regime")
                    if c in regime_df.columns
                ),
                None,
            )
            if regime_col is None:
                raise ValueError(
                    "No regime column found in regime file. "
                    "Expected one of: regime_label, simplified_regime, regime, detailed_regime."
                )

            raw_regimes = regime_df[regime_col].dropna()
            coerced = raw_regimes.map(RegimeLabel.coerce)
            coerced = coerced[coerced.notna()]
            self._regime_series = coerced.astype(object)
            if self._regime_series.empty:
                raise ValueError(f"No valid regime rows found in: {regime_file}")

            # Load regime->allocation mapping from simplified regime export.
            # This keeps portfolio sizing aligned with whichever regime config was last exported.
            self._regime_allocations = {}
            labels_file = regime_file.parent / "regime_labels.json"
            if labels_file.exists():
                try:
                    labels = json.loads(labels_file.read_text(encoding="utf-8"))
                    for payload in labels.values():
                        if not isinstance(payload, dict):
                            continue
                        regime = RegimeLabel.coerce(payload.get("regime"))
                        alloc = payload.get("allocation")
                        if regime is not None and alloc is not None:
                            try:
                                self._regime_allocations[regime] = float(alloc)
                            except (TypeError, ValueError):
                                continue
                except Exception as exc:
                    logger.warning(
                        f"  ⚠️  Could not parse regime allocations from {labels_file.name}: {exc}"
                    )

            logger.info(f"  ✅ Loaded {len(self._regime_series)} regime labels")
            logger.info("\n  Regime distribution:")
            for regime, count in self._regime_series.astype(str).value_counts().items():
                pct = count / len(self._regime_series) * 100
                logger.info(f"    {regime}: {count} days ({pct:.1f}%)")
            if self._regime_allocations:
                logger.info("  Regime allocations:")
                for regime, alloc in sorted(
                    self._regime_allocations.items(),
                    key=lambda item: item[0].value if hasattr(item[0], "value") else str(item[0]),
                ):
                    logger.info(f"    {regime}: {alloc:.2f}")

        return self._regime_series

    def load_regime_allocations(self) -> Dict[RegimeLabel, float]:
        """Get regime allocation mapping loaded from regime_labels.json when available."""
        if self._regime_series is None:
            self.load_regime_predictions()
        return dict(self._regime_allocations or {})

    def load_xautry_prices(
        self,
        csv_path: Path | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.Series:
        """Load XAU/TRY prices"""
        if self._xautry_prices is None:
            logger.info("\n💰 Loading XAU/TRY prices...")
            target_path = Path(csv_path) if csv_path is not None else self.paths.gold_try_file
            if target_path.suffix == ".parquet":
                df = pd.read_parquet(target_path)
                if "Date" not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                    df = df.reset_index()
            else:
                df = pd.read_csv(target_path, parse_dates=["Date"])
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            if "XAU_TRY" not in df.columns:
                # Try common column name variants
                for col in ("xau_try", "Close", "close", "price"):
                    if col in df.columns:
                        df = df.rename(columns={col: "XAU_TRY"})
                        break
            if "XAU_TRY" not in df.columns:
                raise ValueError(f"XAU_TRY column not found in {target_path.name}. Columns: {list(df.columns)}")
            # Normalize dates to naive dates (midnight) to prevent overlap issues
            if "Date" in df.columns:
                df["Date"] = _normalize_dt_series(pd.to_datetime(df["Date"], errors="coerce"))
                series = df.set_index("Date")["XAU_TRY"].astype(float)
            else:
                series = df["XAU_TRY"].astype(float)
                if isinstance(series.index, pd.DatetimeIndex):
                    series.index = _normalize_dt_index(series.index)

            series.name = "XAU_TRY"
            self._xautry_prices = series
            logger.info(f"  ✅ Loaded {len(series)} XAU/TRY observations")

        series = self._xautry_prices
        if start_date is not None:
            series = series.loc[series.index >= pd.Timestamp(start_date).floor("D")]
        if end_date is not None:
            series = series.loc[series.index <= pd.Timestamp(end_date).floor("D")]
        return series

    def load_xu100_prices(self, csv_path: Path | None = None) -> pd.Series:
        """Load XU100 benchmark prices (borsapy-first with local fallback)."""
        if self._xu100_prices is None:
            logger.info("\n📊 Loading XU100 benchmark...")

            # Try borsapy first
            if csv_path is None and self._should_use_borsapy_for("xu100"):
                try:
                    hist = self.borsapy_adapter.client
                    if hist is not None:
                        xu100_df = hist.get_history("XU100", period="5y", interval="1d")
                        if xu100_df is not None and not xu100_df.empty:
                            if "Close" in xu100_df.columns:
                                xu100_df.index = _normalize_dt_index(pd.to_datetime(xu100_df.index, errors="coerce"))
                                self._xu100_prices = xu100_df["Close"].sort_index()
                                logger.info(
                                    "  ✅ Loaded %d XU100 observations via borsapy",
                                    len(self._xu100_prices),
                                )
                                return self._xu100_prices
                except Exception as exc:
                    logger.warning("  ⚠️  Borsapy XU100 fetch failed: %s", exc)

            # Local file fallback
            target_path = Path(csv_path) if csv_path is not None else self.paths.xu100_prices
            if target_path.suffix == ".parquet":
                df = pd.read_parquet(target_path)
            else:
                df = pd.read_csv(target_path)
            if "Date" in df.columns:
                df["Date"] = _normalize_dt_series(pd.to_datetime(df["Date"], errors="coerce"))
                df = df.set_index("Date")
            elif isinstance(df.index, pd.DatetimeIndex):
                df.index = _normalize_dt_index(df.index)
            df = df.sort_index()
            # Prefer close for return calculations and benchmark alignment.
            if "Close" in df.columns:
                self._xu100_prices = df["Close"]
            elif "close" in df.columns:
                self._xu100_prices = df["close"]
            else:
                self._xu100_prices = df.iloc[:, 0]
            logger.info(f"  ✅ Loaded {len(self._xu100_prices)} XU100 observations")
        return self._xu100_prices

    def load_usdtry(self) -> pd.DataFrame:
        """Load USD/TRY exchange rate data"""
        logger.info("\n💱 Loading USD/TRY data...")
        usdtry_file = self.paths.usdtry_file

        if not usdtry_file.exists():
            logger.warning(f"  ⚠️  USD/TRY file not found: {usdtry_file}")
            return pd.DataFrame()

        if usdtry_file.suffix == ".parquet":
            df = pd.read_parquet(usdtry_file)
            if "Date" not in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
            if "Date" in df.columns:
                df["Date"] = _normalize_dt_series(pd.to_datetime(df["Date"], errors="coerce"))
                df = df.set_index("Date")
            df = df.sort_index()
        else:
            df = pd.read_csv(usdtry_file, parse_dates=["Date"])
            if "Date" in df.columns:
                df["Date"] = _normalize_dt_series(pd.to_datetime(df["Date"], errors="coerce"))
            df = df.set_index("Date").sort_index()

        # Rename column to 'Close' for consistency
        if "USDTRY" in df.columns:
            df = df.rename(columns={"USDTRY": "Close"})

        logger.info(f"  ✅ Loaded {len(df)} USD/TRY observations")
        return df

    def load_market_caps(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        symbols: list[str] | None = None,
    ) -> pd.DataFrame:
        """Load or compute Date x Ticker market capitalization panel."""
        prices = self.load_prices(start_date=start_date, end_date=end_date, symbols=symbols)
        if prices.empty:
            return pd.DataFrame()

        if {"Date", "Ticker", "Close"}.issubset(set(prices.columns)):
            close_panel = self.build_close_panel(prices)
        else:
            close_panel = prices.copy()

        shares = self.load_shares_outstanding_panel()
        if shares.empty:
            return close_panel

        shares = shares.reindex(close_panel.index).ffill()
        shares = shares.reindex(columns=close_panel.columns)
        return close_panel * shares

    def load_regime_labels(self) -> dict:
        """Load regime labels JSON as a dictionary."""
        regime_file = self.paths.regime_labels
        if regime_file.exists():
            with open(regime_file, encoding="utf-8") as handle:
                return json.load(handle)
        return {}

    def load_xu100(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.Series:
        """Load XU100 series using canonical DataPaths resolution."""
        series = self.load_xu100_prices()
        out = series.copy()
        if start_date is not None:
            out = out[out.index >= start_date]
        if end_date is not None:
            out = out[out.index <= end_date]
        return out

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

    def load_fundamental_metrics(self) -> pd.DataFrame:
        """Load pre-calculated fundamental metrics"""
        logger.info("\n📊 Loading fundamental metrics...")
        metrics_file = self.data_dir / "fundamental_metrics.parquet"

        if not metrics_file.exists():
            logger.warning(f"  ⚠️  Fundamental metrics file not found: {metrics_file}")
            logger.info("  Run calculate_fundamental_metrics.py to generate this file")
            return pd.DataFrame()

        df = pd.read_parquet(metrics_file)
        logger.info(f"  ✅ Loaded {len(df)} metric observations")
        logger.info(f"  Metrics: {df.columns.tolist()}")
        return df
