from __future__ import annotations

import logging
from dataclasses import dataclass
from importlib import util
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from bist_quant.common.data_loader import DataLoader

logger = logging.getLogger(__name__)




@dataclass
class StockData:
    """Container for per-symbol data pulled via borsapy integrations."""

    symbol: str
    quote: pd.Series
    history: pd.DataFrame
    technical_indicators: pd.DataFrame
    financials: dict[str, pd.DataFrame]


class BorsapyAdapter:
    """Encapsulates borsapy-related DataLoader integrations."""

    def __init__(self, loader: "DataLoader") -> None:
        self._loader = loader
        # We can dynamically look up configs using paths module if necessary, or pass via env, 
        # but for now we'll hardcode relative to root since config is structured.
        from bist_quant.settings import PROJECT_ROOT
        self._config_path = PROJECT_ROOT / "configs" / "borsapy_config.yaml"
        self._client: Any | None = None

    @property
    def client(self) -> Any | None:
        if self._client is None:
            try:
                from bist_quant.clients.borsapy_client import BorsapyClient
            except ImportError as exc:
                logger.warning(f"  ⚠️  Borsapy not available: {exc}")
                logger.info("     Ensure `bist_quant.clients.borsapy_client` is importable or install `borsapy`.")
                return None

            try:
                self._client = BorsapyClient(
                    cache_dir=self._loader.data_dir / "borsapy_cache",
                    use_mcp_fallback=True,
                    config_path=self._config_path,
                )
            except Exception as exc:
                logger.warning(f"  ⚠️  Borsapy client initialization failed: {exc}")
                return None
            logger.info("  ✅ Borsapy client initialized")
        return self._client

    def load_prices(
        self,
        symbols: list[str] | None = None,
        period: str = "5y",
        index: str = "XU100",
    ) -> pd.DataFrame:
        if self.client is None:
            logger.warning("  ⚠️  Borsapy not available, cannot load prices")
            return pd.DataFrame()

        logger.info(f"\n📊 Loading prices via borsapy (period={period})...")

        resolved_symbols = symbols
        if resolved_symbols is None:
            resolved_symbols = self.client.get_index_components(index)
            logger.info(f"  Using {len(resolved_symbols)} symbols from {index}")

        result = self.client.batch_download_to_long(
            symbols=resolved_symbols,
            period=period,
            group_by="ticker",
            add_is_suffix=False,
        )

        if result.empty:
            logger.warning("  ⚠️  No data returned from borsapy")
            return pd.DataFrame()

        loaded = result["Ticker"].dropna().nunique() if "Ticker" in result.columns else 0
        logger.info(
            f"  ✅ Loaded {len(result)} price records for {loaded}/{len(resolved_symbols)} tickers"
        )
        return result

    def get_index_components(self, index: str = "XU100") -> list[str]:
        if self.client is None:
            return []
        result = self.client.get_index_components(index)
        if not isinstance(result, list):
            return []
        return [str(symbol) for symbol in result]

    def get_financials(self, symbol: str) -> dict[str, pd.DataFrame]:
        if self.client is None:
            return {}
        try:
            if hasattr(self.client, "get_financial_statements"):
                result = self.client.get_financial_statements(symbol)
            else:
                result = self.client.get_financials(symbol)
        except Exception as exc:
            logger.warning(f"  ⚠️  Failed to load financials for {symbol}: {exc}")
            return {}

        if not isinstance(result, dict):
            return {}

        normalized = {
            "balance_sheet": result.get("balance_sheet", pd.DataFrame()),
            "income_stmt": result.get("income_stmt", pd.DataFrame()),
        }
        cash_flow = result.get("cash_flow", result.get("cashflow", pd.DataFrame()))
        normalized["cashflow"] = cash_flow
        normalized["cash_flow"] = cash_flow
        return normalized

    def get_financial_ratios(self, symbol: str) -> pd.DataFrame:
        if self.client is None or not hasattr(self.client, "get_financial_ratios"):
            return pd.DataFrame()
        try:
            result = self.client.get_financial_ratios(symbol)
        except Exception as exc:
            logger.warning(f"  ⚠️  Failed to load financial ratios for {symbol}: {exc}")
            return pd.DataFrame()
        return result if isinstance(result, pd.DataFrame) else pd.DataFrame()

    def get_dividends(self, symbol: str) -> pd.DataFrame:
        if self.client is None:
            return pd.DataFrame()
        return self.client.get_dividends(symbol)

    def get_fast_info(self, symbol: str) -> dict[str, Any]:
        if self.client is None:
            return {}
        result = self.client.get_fast_info(symbol)
        return result if isinstance(result, dict) else {}

    def screen_stocks(
        self,
        template: str | None = None,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        if self.client is None:
            return pd.DataFrame()
        merged_filters = dict(filters or {})
        merged_filters.update(kwargs)
        return self.client.screen_stocks(template=template, filters=merged_filters)

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        return str(symbol).upper().split(".")[0].strip()

    @staticmethod
    def _normalize_key(value: str) -> str:
        chars = [ch.lower() for ch in str(value) if ch.isalnum()]
        return "".join(chars)

    @classmethod
    def _extract_first_value(cls, payload: dict[str, Any], candidates: tuple[str, ...]) -> Any:
        if not payload:
            return None
        normalized_payload = {cls._normalize_key(key): value for key, value in payload.items()}
        for candidate in candidates:
            normalized_candidate = cls._normalize_key(candidate)
            if not normalized_candidate:
                continue
            if normalized_candidate in normalized_payload:
                return normalized_payload[normalized_candidate]
            if len(normalized_candidate) >= 6:
                for key, value in normalized_payload.items():
                    if normalized_candidate in key:
                        return value
        return None

    @staticmethod
    def _as_float(value: Any) -> float | None:
        if value is None or value == "":
            return None
        try:
            parsed = float(value)
        except Exception:
            return None
        if pd.isna(parsed):
            return None
        return parsed

    @classmethod
    def _as_int(cls, value: Any) -> int | None:
        parsed = cls._as_float(value)
        if parsed is None:
            return None
        return int(parsed)

    @staticmethod
    def _flatten_payload(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for key, item in value.items():
                if isinstance(item, dict):
                    for sub_key, sub_value in item.items():
                        out[f"{key}_{sub_key}"] = sub_value
                else:
                    out[str(key)] = item
            return out
        if isinstance(value, pd.Series):
            return {str(key): item for key, item in value.to_dict().items()}
        if isinstance(value, pd.DataFrame):
            if value.empty:
                return {}
            row = value.iloc[-1].to_dict()
            return {str(key): item for key, item in row.items()}
        if isinstance(value, list) and value and isinstance(value[0], dict):
            return {str(key): item for key, item in value[0].items()}
        return {}

    def _safe_get_ticker(self, symbol: str) -> Any | None:
        if self.client is None:
            return None
        try:
            if hasattr(self.client, "get_ticker"):
                return self.client.get_ticker(symbol)
        except Exception:
            return None
        return None

    @staticmethod
    def _normalize_recommendation(value: Any) -> str | None:
        text = str(value or "").strip().upper()
        if not text:
            return None
        mapping = {
            "BUY": "AL",
            "OUTPERFORM": "AL",
            "OVERWEIGHT": "AL",
            "HOLD": "TUT",
            "NEUTRAL": "TUT",
            "UNDERWEIGHT": "SAT",
            "SELL": "SAT",
            "AL": "AL",
            "TUT": "TUT",
            "SAT": "SAT",
        }
        return mapping.get(text, text)

    def screen_stocks_isyatirim(
        self,
        template: str | None = None,
        **filters: Any,
    ) -> pd.DataFrame:
        """İş Yatırım fundamental screener via bp.screen_stocks() / bp.Screener()."""
        if self.client is None:
            return pd.DataFrame()
        try:
            result = self.client.screen_stocks(template=template, filters=filters)
        except Exception as exc:
            logger.warning(f"  ⚠️  İş Yatırım screen failed: {exc}")
            return pd.DataFrame()
        return result if isinstance(result, pd.DataFrame) else pd.DataFrame()

    def get_analyst_data(self, symbol: str) -> dict[str, Any]:
        """Analyst targets + recommendations via bp.Ticker(symbol).analyst_price_targets."""
        if self.client is None:
            return {}

        normalized_symbol = self._normalize_symbol(symbol)
        price_targets_raw: Any = None
        recommendations_raw: Any = None
        base_payload: dict[str, Any] = {}

        try:
            if hasattr(self.client, "get_analyst_targets"):
                result = self.client.get_analyst_targets(normalized_symbol)
                if isinstance(result, dict):
                    base_payload = dict(result)
                    price_targets_raw = result.get("price_targets")
                    recommendations_raw = result.get("recommendations")
        except Exception as exc:
            logger.debug(f"Analyst target client method failed for {normalized_symbol}: {exc}")

        if price_targets_raw is None or recommendations_raw is None:
            ticker = self._safe_get_ticker(normalized_symbol)
            if ticker is not None:
                try:
                    if price_targets_raw is None:
                        price_targets_raw = getattr(ticker, "analyst_price_targets", None)
                except Exception:
                    pass
                try:
                    if recommendations_raw is None:
                        recommendations_raw = getattr(ticker, "recommendations", None)
                except Exception:
                    pass

        price_targets = self._flatten_payload(price_targets_raw)
        recommendations = self._flatten_payload(recommendations_raw)
        merged = dict(base_payload)
        merged.update(price_targets)
        merged.update(recommendations)

        target_price = self._as_float(
            self._extract_first_value(
                merged,
                (
                    "target_price",
                    "price_target",
                    "mean_target_price",
                    "target_mean",
                    "analyst_target",
                    "consensus_target",
                ),
            )
        )
        target_high = self._as_float(
            self._extract_first_value(
                merged,
                ("target_high", "high_target_price", "price_target_high", "high"),
            )
        )
        target_low = self._as_float(
            self._extract_first_value(
                merged,
                ("target_low", "low_target_price", "price_target_low", "low"),
            )
        )
        target_median = self._as_float(
            self._extract_first_value(
                merged,
                ("target_median", "median_target_price", "price_target_median", "median"),
            )
        )
        target_count = self._as_int(
            self._extract_first_value(
                merged,
                ("target_count", "number_of_analysts", "analyst_count", "count"),
            )
        )

        recommendation = self._normalize_recommendation(
            self._extract_first_value(
                merged,
                (
                    "recommendation",
                    "to_grade",
                    "rating",
                    "consensus_recommendation",
                    "recommendation_key",
                ),
            )
        )
        recommendation_score = self._as_float(
            self._extract_first_value(
                merged,
                ("recommendation_score", "rating_score", "score"),
            )
        )

        upside_potential = self._as_float(
            self._extract_first_value(
                merged,
                ("upside_potential", "target_upside", "upside"),
            )
        )
        last_price = self._as_float(
            self._extract_first_value(
                merged,
                ("last_price", "price", "close", "current_price"),
            )
        )
        if upside_potential is None and target_price is not None and last_price is not None and last_price > 0:
            upside_potential = ((target_price / last_price) - 1.0) * 100.0

        forward_pe = self._as_float(
            self._extract_first_value(
                merged,
                ("forward_pe", "fwd_pe", "pe_forward", "ileri_fk"),
            )
        )

        return {
            "symbol": normalized_symbol,
            "analyst_target_price": target_price,
            "analyst_target_high": target_high,
            "analyst_target_low": target_low,
            "analyst_target_median": target_median,
            "analyst_target_count": target_count,
            "recommendation": recommendation,
            "recommendation_score": recommendation_score,
            "upside_potential": upside_potential,
            "forward_pe": forward_pe,
        }

    def get_foreign_ownership(self, symbol: str) -> dict[str, Any]:
        """Foreign ownership + 1W/1M change."""
        if self.client is None:
            return {}

        normalized_symbol = self._normalize_symbol(symbol)
        merged: dict[str, Any] = {}

        try:
            if hasattr(self.client, "get_foreign_ownership"):
                result = self.client.get_foreign_ownership(normalized_symbol)
                merged.update(self._flatten_payload(result))
        except Exception as exc:
            logger.debug(f"Foreign ownership client method failed for {normalized_symbol}: {exc}")

        ticker = self._safe_get_ticker(normalized_symbol)
        if ticker is not None:
            for attr in (
                "foreign_ownership",
                "foreign_ratio",
                "foreign_shareholders",
                "foreign_holders",
                "ownership",
                "shareholders",
            ):
                try:
                    merged.update(self._flatten_payload(getattr(ticker, attr, None)))
                except Exception:
                    continue

        try:
            fast_info = self.get_fast_info(normalized_symbol)
            if isinstance(fast_info, dict):
                merged.update(fast_info)
        except Exception:
            pass

        foreign_ratio = self._as_float(
            self._extract_first_value(
                merged,
                (
                    "foreign_ratio",
                    "foreign_ownership",
                    "foreign_ownership_ratio",
                    "foreign_share",
                    "yabanci_orani",
                    "yabanci_payi",
                ),
            )
        )
        foreign_change_1w = self._as_float(
            self._extract_first_value(
                merged,
                (
                    "foreign_change_1w",
                    "foreign_ratio_change_1w",
                    "foreign_ownership_change_1w",
                    "foreign_1w_change",
                    "yabanci_degisim_1h",
                ),
            )
        )
        foreign_change_1m = self._as_float(
            self._extract_first_value(
                merged,
                (
                    "foreign_change_1m",
                    "foreign_ratio_change_1m",
                    "foreign_ownership_change_1m",
                    "foreign_1m_change",
                    "yabanci_degisim_1a",
                ),
            )
        )
        float_ratio = self._as_float(
            self._extract_first_value(
                merged,
                ("float_ratio", "free_float", "free_float_ratio", "fiili_dolasim_orani"),
            )
        )

        return {
            "symbol": normalized_symbol,
            "foreign_ratio": foreign_ratio,
            "foreign_change_1w": foreign_change_1w,
            "foreign_change_1m": foreign_change_1m,
            "float_ratio": float_ratio,
        }

    def get_screener_criteria(self) -> list[dict[str, Any]]:
        """Available filter criteria from İş Yatırım API."""
        if self.client is None:
            return []

        def _normalize(value: Any) -> list[dict[str, Any]]:
            if isinstance(value, list):
                out: list[dict[str, Any]] = []
                for row in value:
                    if isinstance(row, dict):
                        out.append(dict(row))
                    elif isinstance(row, str):
                        out.append({"key": row, "label": row})
                return out
            if isinstance(value, dict):
                out = []
                for key, row in value.items():
                    if isinstance(row, dict):
                        item = dict(row)
                        item.setdefault("key", key)
                        item.setdefault("label", str(item.get("name") or key))
                        out.append(item)
                    else:
                        out.append({"key": str(key), "label": str(row)})
                return out
            return []

        try:
            if hasattr(self.client, "get_screener_criteria"):
                rows = _normalize(self.client.get_screener_criteria())
                if rows:
                    return rows
        except Exception as exc:
            logger.debug(f"Screener criteria client method failed: {exc}")

        sources: list[Any] = [self.client]
        try:
            import borsapy as bp  # type: ignore[import-not-found]

            sources.extend([getattr(bp, "Screener", None), bp])
        except Exception:
            pass

        for source in sources:
            if source is None:
                continue
            for attr in ("SCREENER_CRITERIA", "SCREENING_CRITERIA", "CRITERIA", "FILTERS"):
                rows = _normalize(getattr(source, attr, None))
                if rows:
                    return rows

        return []

    def get_history_with_indicators(
        self,
        symbol: str,
        indicators: list[str] | None = None,
        period: str = "2y",
    ) -> pd.DataFrame:
        if self.client is None:
            return pd.DataFrame()
        return self.client.get_history_with_indicators(
            symbol,
            indicators=indicators,
            period=period,
        )

    def get_stock_data(
        self,
        symbol: str,
        period: str = "1mo",
        indicators: list[str] | None = None,
    ) -> StockData | None:
        """Get quote/history/technicals/financials in one call."""
        if self.client is None:
            return None

        history = self.client.get_history(symbol, period=period)
        technical = self._calculate_technical_indicators(
            symbol=symbol,
            history=history,
            period=period,
            indicators=indicators,
        )
        financials = self.get_financials(symbol)

        quote_payload = self.get_fast_info(symbol)
        quote_series = pd.Series(quote_payload if isinstance(quote_payload, dict) else {}, dtype="object")

        return StockData(
            symbol=symbol,
            quote=quote_series,
            history=history if isinstance(history, pd.DataFrame) else pd.DataFrame(),
            technical_indicators=technical,
            financials=financials,
        )

    def _calculate_technical_indicators(
        self,
        symbol: str,
        history: pd.DataFrame,
        period: str,
        indicators: list[str] | None = None,
    ) -> pd.DataFrame:
        if self.client is None:
            return pd.DataFrame()
        if indicators is None:
            indicators = ["rsi", "macd", "bb"]
        try:
            return self.client.get_history_with_indicators(
                symbol,
                indicators=indicators,
                period=period,
            )
        except Exception:
            return history if isinstance(history, pd.DataFrame) else pd.DataFrame()
