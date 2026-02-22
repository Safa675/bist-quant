"""TradingView TA consensus signals as a cross-sectional factor."""

from __future__ import annotations

import logging
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd

from bist_quant.signals._context import get_runtime_context, require_context

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import borsapy as bp
except Exception:  # pragma: no cover - optional dependency
    bp = None


def _as_int(value: Any, default: int, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(parsed, minimum)


def _as_float(value: Any, default: float, minimum: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = default
    if not np.isfinite(parsed):
        parsed = default
    return max(parsed, minimum)


class TAConsensusSignals:
    """Maps TradingView consensus to a [-1, +1] factor score."""

    CONSENSUS_MAP = {
        "STRONG_BUY": 1.0,
        "BUY": 0.5,
        "NEUTRAL": 0.0,
        "SELL": -0.5,
        "STRONG_SELL": -1.0,
    }
    _ALIAS_MAP = {
        "AL": "BUY",
        "SAT": "SELL",
        "TUT": "NEUTRAL",
        "STRONGBUY": "STRONG_BUY",
        "STRONGSELL": "STRONG_SELL",
        "STRONG BUY": "STRONG_BUY",
        "STRONG SELL": "STRONG_SELL",
    }

    def __init__(
        self,
        *,
        borsapy_module: Any | None = None,
        batch_size: int = 20,
        request_sleep_seconds: float = 0.0,
        batch_pause_seconds: float = 0.0,
    ) -> None:
        self._bp = borsapy_module if borsapy_module is not None else bp
        self.batch_size = _as_int(batch_size, default=20, minimum=1)
        self.request_sleep_seconds = _as_float(request_sleep_seconds, default=0.0, minimum=0.0)
        self.batch_pause_seconds = _as_float(batch_pause_seconds, default=0.0, minimum=0.0)

    @staticmethod
    def _normalize_symbol(symbol: Any) -> str:
        text = str(symbol or "").strip().upper()
        if not text:
            return ""
        return text.split(".")[0]

    @classmethod
    def _normalize_recommendation(cls, value: Any) -> str:
        text = str(value or "").strip().upper()
        if not text:
            return "NEUTRAL"
        text = text.replace("-", "_")
        text = cls._ALIAS_MAP.get(text, text)
        text = cls._ALIAS_MAP.get(text.replace("_", " "), text)
        text = cls._ALIAS_MAP.get(text.replace("_", ""), text)
        return text if text in cls.CONSENSUS_MAP else "NEUTRAL"

    @staticmethod
    def _safe_int(value: Any) -> int:
        try:
            return int(value)
        except Exception:
            return 0

    @staticmethod
    def _sleep(seconds: float) -> None:
        if seconds > 0.0:
            time.sleep(seconds)

    @classmethod
    def _normalize_symbols(cls, symbols: Iterable[Any]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for raw in symbols:
            symbol = cls._normalize_symbol(raw)
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            out.append(symbol)
        return out

    def _fetch_signals(self, symbols: list[str], interval: str) -> dict[str, dict[str, Any]]:
        if self._bp is None:
            LOGGER.debug("borsapy unavailable; returning empty TA consensus payloads")
            return {}

        payloads: dict[str, dict[str, Any]] = {}
        for idx, symbol in enumerate(symbols):
            if idx > 0:
                self._sleep(self.request_sleep_seconds)
            try:
                ticker = self._bp.Ticker(symbol)
                raw = ticker.ta_signals(interval=interval)
                if isinstance(raw, dict):
                    payloads[symbol] = raw
            except Exception as exc:
                LOGGER.debug("ta_signals failed for %s: %s", symbol, exc)

            is_batch_boundary = (idx + 1) % self.batch_size == 0
            has_more = (idx + 1) < len(symbols)
            if is_batch_boundary and has_more:
                self._sleep(self.batch_pause_seconds)

        return payloads

    def _extract_vote_counts(self, payload: dict[str, Any]) -> tuple[int, int, int]:
        summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
        if not isinstance(summary, dict):
            summary = {}

        buy = self._safe_int(summary.get("buy"))
        sell = self._safe_int(summary.get("sell"))
        neutral = self._safe_int(summary.get("neutral"))
        if buy > 0 or sell > 0 or neutral > 0:
            return buy, sell, neutral

        # Some payloads may omit summary counts; derive from indicator-level votes.
        for section_name in ("oscillators", "moving_averages"):
            section = payload.get(section_name, {})
            if not isinstance(section, dict):
                continue
            compute = section.get("compute", {})
            if not isinstance(compute, dict):
                continue
            for signal in compute.values():
                normalized = self._normalize_recommendation(signal)
                if normalized in {"BUY", "STRONG_BUY"}:
                    buy += 1
                elif normalized in {"SELL", "STRONG_SELL"}:
                    sell += 1
                else:
                    neutral += 1
        return buy, sell, neutral

    def build_consensus_panel(
        self,
        symbols: list[str],
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Cross-sectional consensus scores."""
        ordered_symbols = self._normalize_symbols(symbols)
        columns = [
            "symbol",
            "consensus",
            "consensus_score",
            "buy_count",
            "sell_count",
            "neutral_count",
            "oscillator_consensus",
            "moving_average_consensus",
        ]
        if not ordered_symbols:
            return pd.DataFrame(columns=columns)

        payloads = self._fetch_signals(ordered_symbols, interval=interval)
        rows: list[dict[str, Any]] = []

        for symbol in ordered_symbols:
            payload = payloads.get(symbol)
            if not isinstance(payload, dict):
                rows.append(
                    {
                        "symbol": symbol,
                        "consensus": None,
                        "consensus_score": np.nan,
                        "buy_count": 0,
                        "sell_count": 0,
                        "neutral_count": 0,
                        "oscillator_consensus": None,
                        "moving_average_consensus": None,
                    }
                )
                continue

            summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
            oscillators = (
                payload.get("oscillators", {}) if isinstance(payload.get("oscillators"), dict) else {}
            )
            moving_averages = (
                payload.get("moving_averages", {})
                if isinstance(payload.get("moving_averages"), dict)
                else {}
            )

            consensus = self._normalize_recommendation(summary.get("recommendation"))
            oscillator_consensus = self._normalize_recommendation(oscillators.get("recommendation"))
            moving_average_consensus = self._normalize_recommendation(
                moving_averages.get("recommendation")
            )
            buy_count, sell_count, neutral_count = self._extract_vote_counts(payload)

            rows.append(
                {
                    "symbol": symbol,
                    "consensus": consensus,
                    "consensus_score": float(self.CONSENSUS_MAP.get(consensus, 0.0)),
                    "buy_count": int(buy_count),
                    "sell_count": int(sell_count),
                    "neutral_count": int(neutral_count),
                    "oscillator_consensus": oscillator_consensus,
                    "moving_average_consensus": moving_average_consensus,
                }
            )

        return pd.DataFrame(rows, columns=columns)

    def build_oscillator_panel(
        self,
        symbols: list[str],
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Per-indicator oscillator signals (RSI=BUY, MACD=SELL, etc.) for all symbols."""
        ordered_symbols = self._normalize_symbols(symbols)
        if not ordered_symbols:
            return pd.DataFrame()

        payloads = self._fetch_signals(ordered_symbols, interval=interval)
        rows: dict[str, dict[str, str]] = {}
        for symbol in ordered_symbols:
            payload = payloads.get(symbol, {})
            oscillators = (
                payload.get("oscillators", {}) if isinstance(payload, dict) else {}
            )
            compute = oscillators.get("compute", {}) if isinstance(oscillators, dict) else {}
            if not isinstance(compute, dict):
                rows[symbol] = {}
                continue
            rows[symbol] = {
                str(indicator): self._normalize_recommendation(signal)
                for indicator, signal in compute.items()
            }

        panel = pd.DataFrame.from_dict(rows, orient="index")
        panel.index.name = "symbol"
        return panel.reindex(index=ordered_symbols)

    def build_signal_panel(
        self,
        symbols: list[str],
        dates: pd.DatetimeIndex,
        interval: str = "1d",
        fillna_value: float | None = 0.0,
    ) -> pd.DataFrame:
        """Build date-indexed panel by broadcasting cross-sectional consensus scores."""
        ordered_symbols = self._normalize_symbols(symbols)
        date_index = pd.DatetimeIndex(dates) if len(dates) else pd.DatetimeIndex([])
        if not ordered_symbols:
            return pd.DataFrame(index=date_index, columns=[], dtype="float64")
        if date_index.empty:
            return pd.DataFrame(index=date_index, columns=ordered_symbols, dtype="float64")

        consensus = self.build_consensus_panel(ordered_symbols, interval=interval)
        if consensus.empty:
            base = pd.Series(dtype="float64", index=ordered_symbols)
        else:
            base = consensus.set_index("symbol")["consensus_score"].astype("float64")
            base = base.reindex(ordered_symbols)

        values = np.tile(base.to_numpy(dtype="float64"), (len(date_index), 1))
        panel = pd.DataFrame(values, index=date_index, columns=ordered_symbols, dtype="float64")
        if fillna_value is not None:
            panel = panel.fillna(float(fillna_value))
        return panel


def build_ta_consensus_from_config(
    dates: pd.DatetimeIndex,
    loader: Any,
    config: dict[str, Any],
    signal_params: dict[str, Any],
) -> pd.DataFrame:
    """Factory-compatible builder for TA consensus signal panels."""
    del loader
    context = get_runtime_context(config)
    close_df = require_context("ta_consensus", context, "close_df")

    interval = str(signal_params.get("interval", "1d"))
    batch_size = _as_int(signal_params.get("batch_size"), default=20, minimum=1)
    request_sleep_seconds = _as_float(signal_params.get("request_sleep_seconds"), default=0.0)
    batch_pause_seconds = _as_float(signal_params.get("batch_pause_seconds"), default=0.0)

    raw_fillna = signal_params.get("fillna_value", 0.0)
    fillna_value = None if raw_fillna is None else _as_float(raw_fillna, default=0.0)

    builder = TAConsensusSignals(
        batch_size=batch_size,
        request_sleep_seconds=request_sleep_seconds,
        batch_pause_seconds=batch_pause_seconds,
    )
    return builder.build_signal_panel(
        symbols=[str(symbol) for symbol in close_df.columns],
        dates=dates,
        interval=interval,
        fillna_value=fillna_value,
    )


BUILDERS = {
    "ta_consensus": build_ta_consensus_from_config,
}

