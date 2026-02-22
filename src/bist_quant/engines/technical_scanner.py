"""Technical scanner engine wrapping borsapy scan() and TechnicalScanner."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

LOGGER = logging.getLogger("bist_quant.engines.technical_scanner")

try:  # optional dependency
    import borsapy as bp
except Exception:  # pragma: no cover
    bp = None


def _normalize_symbol(value: Any) -> str:
    symbol = str(value or "").strip().upper()
    if not symbol:
        return ""
    return symbol.split(".")[0]


class TechnicalScannerEngine:
    """Expression-based technical scanner over index universes or symbol lists."""

    PREDEFINED: dict[str, str] = {
        "oversold": "rsi < 30",
        "overbought": "rsi > 70",
        "golden_cross": "sma_20 crosses_above sma_50",
        "death_cross": "sma_20 crosses_below sma_50",
        "bollinger_squeeze_low": "close < bb_lower",
        "bollinger_squeeze_high": "close > bb_upper",
        "macd_bullish_cross": "macd crosses_above signal",
        "supertrend_bullish": "supertrend_direction == 1",
        "high_volume_momentum": "rsi > 50 and volume > 5000000 and change_percent > 2",
        "dip_buying": "close below_pct sma_200 0.90",
    }

    def __init__(self, borsapy_module: Any | None = None) -> None:
        self._bp = bp if borsapy_module is None else borsapy_module

    def scan(
        self,
        universe: str | list[str] = "XU100",
        condition: str = "rsi < 30",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Run a technical scan and return symbols plus indicator fields."""
        resolved_condition = str(condition or "").strip()
        if not resolved_condition:
            raise ValueError("Scan condition must be a non-empty string.")
        raw = self._run_scan(universe=universe, condition=resolved_condition, interval=interval)
        return self._normalize_scan_frame(raw)

    def scan_multi(
        self,
        universe: str | list[str] = "XU100",
        conditions: list[str] | tuple[str, ...] = (),
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Run multiple conditions and return symbols satisfying all conditions."""
        normalized = [str(item).strip() for item in conditions if str(item).strip()]
        if not normalized:
            raise ValueError("At least one technical condition is required.")

        frames: list[pd.DataFrame] = []
        for idx, condition in enumerate(normalized):
            frame = self.scan(universe=universe, condition=condition, interval=interval)
            if frame.empty:
                return pd.DataFrame(columns=["symbol"])
            frame = frame.drop_duplicates(subset=["symbol"], keep="last").copy()
            frame["scan_condition"] = condition
            if idx > 0:
                overlap = {col for col in frame.columns if col != "symbol"}
                frame = frame.rename(columns={col: f"{col}__c{idx}" for col in overlap})
            frames.append(frame)

        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on="symbol", how="inner")

        merged["matched_conditions"] = len(normalized)
        merged["scan_query"] = " and ".join(normalized)
        return merged

    def predefined_scans(self) -> dict[str, str]:
        """Return named predefined scan expressions."""
        return dict(self.PREDEFINED)

    def _run_scan(self, universe: str | list[str], condition: str, interval: str) -> Any:
        if self._bp is None:
            LOGGER.info("borsapy is unavailable; returning empty technical scan result.")
            return pd.DataFrame()

        primary = self._run_scan_via_function(universe=universe, condition=condition, interval=interval)
        if primary is not None:
            return primary
        fallback = self._run_scan_via_scanner(universe=universe, condition=condition, interval=interval)
        if fallback is not None:
            return fallback
        return pd.DataFrame()

    def _run_scan_via_function(self, universe: str | list[str], condition: str, interval: str) -> Any | None:
        scan_fn = getattr(self._bp, "scan", None)
        if not callable(scan_fn):
            return None

        kwargs_candidates = [
            {"universe": universe, "condition": condition, "interval": interval},
            {"symbols": universe, "condition": condition, "interval": interval},
        ]
        for kwargs in kwargs_candidates:
            try:
                return scan_fn(**kwargs)
            except TypeError:
                continue
            except Exception as exc:
                LOGGER.debug("bp.scan failed: %s", exc)
                return None

        try:
            return scan_fn(universe, condition, interval)
        except Exception as exc:
            LOGGER.debug("bp.scan positional call failed: %s", exc)
            return None

    def _run_scan_via_scanner(self, universe: str | list[str], condition: str, interval: str) -> Any | None:
        scanner_cls = getattr(self._bp, "TechnicalScanner", None)
        if scanner_cls is None:
            return None

        scanner = None
        for kwargs in (
            {"universe": universe, "interval": interval},
            {"symbols": universe, "interval": interval},
            {},
        ):
            try:
                scanner = scanner_cls(**kwargs)
                break
            except TypeError:
                continue
            except Exception as exc:
                LOGGER.debug("TechnicalScanner init failed: %s", exc)
                return None
        if scanner is None:
            try:
                scanner = scanner_cls(universe, interval)
            except Exception:
                return None

        for method_name in ("scan", "run", "filter"):
            method = getattr(scanner, method_name, None)
            if not callable(method):
                continue
            for kwargs in (
                {"condition": condition, "universe": universe, "interval": interval},
                {"condition": condition},
                {"query": condition, "universe": universe, "interval": interval},
                {"query": condition},
            ):
                try:
                    return method(**kwargs)
                except TypeError:
                    continue
                except Exception as exc:
                    LOGGER.debug("TechnicalScanner.%s failed: %s", method_name, exc)
                    return None
            try:
                return method(condition)
            except Exception:
                continue
        return None

    @staticmethod
    def _coerce_dataframe(value: Any) -> pd.DataFrame:
        if isinstance(value, pd.DataFrame):
            return value.copy()
        if isinstance(value, pd.Series):
            return value.to_frame().T
        if isinstance(value, list):
            if not value:
                return pd.DataFrame()
            if all(isinstance(item, dict) for item in value):
                return pd.DataFrame(value)
            return pd.DataFrame({"value": value})
        if isinstance(value, dict):
            if not value:
                return pd.DataFrame()
            return pd.DataFrame([value])
        return pd.DataFrame()

    def _normalize_scan_frame(self, payload: Any) -> pd.DataFrame:
        frame = self._coerce_dataframe(payload)
        if frame.empty:
            return pd.DataFrame(columns=["symbol"])

        symbol_col = None
        for candidate in ("symbol", "Symbol", "ticker", "Ticker", "code", "Code"):
            if candidate in frame.columns:
                symbol_col = candidate
                break

        if symbol_col is None:
            return pd.DataFrame(columns=["symbol"])

        normalized = frame.copy()
        normalized["symbol"] = normalized[symbol_col].map(_normalize_symbol)
        normalized = normalized[normalized["symbol"] != ""]
        normalized = normalized.drop_duplicates(subset=["symbol"], keep="last")
        if symbol_col != "symbol":
            normalized = normalized.drop(columns=[symbol_col], errors="ignore")

        ordered_columns = ["symbol"] + [col for col in normalized.columns if col != "symbol"]
        return normalized.loc[:, ordered_columns].reset_index(drop=True)
