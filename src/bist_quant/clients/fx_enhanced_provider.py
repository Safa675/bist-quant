"""
Enhanced FX provider: bank rates, institution rates, intraday FX data.

This provider supplements MCP-driven FX snapshots with borsapy-native
bank/institution tables and TradingView-backed intraday bars.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class FXEnhancedProvider:
    """Resilient accessor for enhanced FX and institution-rate datasets."""

    _BANK_COLUMNS = (
        "bank",
        "bank_name",
        "currency",
        "buy",
        "sell",
        "spread",
        "mid",
        "timestamp",
        "source",
    )
    _INSTITUTION_COLUMNS = (
        "institution",
        "institution_name",
        "asset",
        "buy",
        "sell",
        "spread",
        "mid",
        "timestamp",
        "source",
    )
    _INTRADAY_COLUMNS = (
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "asset",
        "interval",
        "period",
        "source",
    )
    _INTRADAY_SUPPORTED = {"USD", "EUR", "GBP", "XAU", "XAG", "BRENT", "WTI"}
    _INTRADAY_ALIASES = {
        "USD": "USD",
        "USDTL": "USD",
        "USDTRY": "USD",
        "USD/TRY": "USD",
        "EUR": "EUR",
        "EURTL": "EUR",
        "EURTRY": "EUR",
        "EUR/TRY": "EUR",
        "GBP": "GBP",
        "GBPTL": "GBP",
        "GBPTRY": "GBP",
        "GBP/TRY": "GBP",
        "XAU": "XAU",
        "XAUUSD": "XAU",
        "ONSALTIN": "XAU",
        "ONS-ALTIN": "XAU",
        "GOLD": "XAU",
        "XAG": "XAG",
        "XAGUSD": "XAG",
        "SILVER": "XAG",
        "BRENT": "BRENT",
        "UKOIL": "BRENT",
        "WTI": "WTI",
        "USOIL": "WTI",
    }
    _INSTITUTION_ASSET_ALIASES = {
        "gram-altin": "gram-altin",
        "gram_altin": "gram-altin",
        "gram altin": "gram-altin",
        "xau": "gram-altin",
        "gold": "gram-altin",
        "gram-gumus": "gram-gumus",
        "gram_gumus": "gram-gumus",
        "gram gumus": "gram-gumus",
        "xag": "gram-gumus",
        "silver": "gram-gumus",
        "ons-altin": "ons-altin",
        "ons_altin": "ons-altin",
        "ons altin": "ons-altin",
        "xauusd": "ons-altin",
        "gram-platin": "gram-platin",
        "gram_platin": "gram-platin",
        "gram platin": "gram-platin",
        "xpt": "gram-platin",
        "platinum": "gram-platin",
    }

    def __init__(self, borsapy_module: Any | None = None) -> None:
        self._bp = borsapy_module
        self._import_attempted = borsapy_module is not None

    def _get_bp(self) -> Any | None:
        if self._bp is not None:
            return self._bp
        if self._import_attempted:
            return None
        self._import_attempted = True
        try:
            import borsapy as bp  # type: ignore[import-not-found]

            self._bp = bp
        except Exception as exc:
            logger.info("  FXEnhancedProvider: borsapy unavailable: %s", exc)
            self._bp = None
        return self._bp

    @staticmethod
    def _call_if_callable(obj: Any, *args: Any, **kwargs: Any) -> Any:
        if not callable(obj):
            return None
        try:
            return obj(*args, **kwargs)
        except TypeError:
            try:
                return obj(*args)
            except Exception:
                return None
        except Exception:
            return None

    @staticmethod
    def _as_frame(payload: Any) -> pd.DataFrame:
        if isinstance(payload, pd.DataFrame):
            return payload.copy()
        if isinstance(payload, (list, tuple)):
            try:
                return pd.DataFrame(payload)
            except Exception:
                return pd.DataFrame()
        if isinstance(payload, dict):
            if payload and all(
                not isinstance(v, (list, tuple, dict, pd.Series, pd.DataFrame))
                for v in payload.values()
            ):
                return pd.DataFrame([payload])
            try:
                return pd.DataFrame(payload)
            except Exception:
                return pd.DataFrame([payload])
        return pd.DataFrame()

    @staticmethod
    def _pick_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
        lookup = {str(col).strip().lower(): col for col in frame.columns}
        for candidate in candidates:
            hit = lookup.get(candidate.lower())
            if hit is not None:
                return str(hit)
        return None

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            parsed = float(value)
            return parsed if math.isfinite(parsed) else None
        text = str(value).strip()
        if not text:
            return None
        text = text.replace("%", "").replace(",", ".")
        parsed_chars = "".join(ch for ch in text if ch in "0123456789+-eE.")
        if not parsed_chars:
            return None
        try:
            parsed = float(parsed_chars)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None

    @staticmethod
    def _to_timestamp_series(series: pd.Series) -> pd.Series:
        ts = pd.to_datetime(series, errors="coerce")
        try:
            if ts.dt.tz is not None:
                ts = ts.dt.tz_convert(None)
        except Exception:
            pass
        return ts

    @classmethod
    def _empty_bank_rates(cls) -> pd.DataFrame:
        return pd.DataFrame(columns=list(cls._BANK_COLUMNS))

    @classmethod
    def _empty_institution_rates(cls) -> pd.DataFrame:
        return pd.DataFrame(columns=list(cls._INSTITUTION_COLUMNS))

    @classmethod
    def _empty_intraday(cls) -> pd.DataFrame:
        return pd.DataFrame(columns=list(cls._INTRADAY_COLUMNS))

    @staticmethod
    def _normalize_bank_currency(currency: str) -> str:
        token = str(currency or "").strip().upper()
        if "/" in token:
            token = token.split("/", 1)[0]
        if len(token) > 3 and token.endswith("TRY"):
            token = token[:3]
        return token

    def _normalize_bank_frame(self, frame: pd.DataFrame, currency: str) -> pd.DataFrame:
        if frame.empty:
            return self._empty_bank_rates()

        work = frame.copy()
        bank_col = self._pick_column(work, ["bank", "institution", "code", "slug", "name"])
        bank_name_col = self._pick_column(work, ["bank_name", "institution_name", "display_name", "title", "name"])
        buy_col = self._pick_column(work, ["buy", "bid", "buying", "alis"])
        sell_col = self._pick_column(work, ["sell", "ask", "selling", "satis"])
        spread_col = self._pick_column(work, ["spread", "spread_pct", "makas"])
        currency_col = self._pick_column(work, ["currency", "asset", "symbol"])
        timestamp_col = self._pick_column(work, ["timestamp", "update_time", "datetime", "date", "time"])

        out = pd.DataFrame(index=work.index)
        out["bank"] = work[bank_col].astype("string").str.strip() if bank_col else pd.NA
        out["bank_name"] = work[bank_name_col].astype("string").str.strip() if bank_name_col else pd.NA
        out["currency"] = (
            work[currency_col].astype("string").str.strip().str.upper() if currency_col else currency.upper()
        )
        out["buy"] = work[buy_col].map(self._to_float) if buy_col else pd.NA
        out["sell"] = work[sell_col].map(self._to_float) if sell_col else pd.NA
        out["spread"] = work[spread_col].map(self._to_float) if spread_col else pd.NA

        calculated_spread = (
            (pd.to_numeric(out["sell"], errors="coerce") - pd.to_numeric(out["buy"], errors="coerce"))
            / pd.to_numeric(out["buy"], errors="coerce")
            * 100.0
        )
        out["spread"] = pd.to_numeric(out["spread"], errors="coerce").fillna(calculated_spread)
        out["mid"] = (
            pd.to_numeric(out["buy"], errors="coerce")
            + pd.to_numeric(out["sell"], errors="coerce")
        ) / 2.0

        if timestamp_col:
            ts = self._to_timestamp_series(work[timestamp_col])
        else:
            ts = pd.Series(pd.Timestamp(datetime.now()), index=out.index)
        out["timestamp"] = ts.fillna(pd.Timestamp(datetime.now()))
        out["source"] = "bank_rates"

        out["bank"] = out["bank"].fillna(out["bank_name"]).fillna("")
        out["bank_name"] = out["bank_name"].fillna(out["bank"])
        out["currency"] = out["currency"].fillna(currency.upper()).astype("string").str.upper()

        out = out[
            pd.to_numeric(out["buy"], errors="coerce").notna()
            & pd.to_numeric(out["sell"], errors="coerce").notna()
        ]
        if out.empty:
            return self._empty_bank_rates()

        out = out.sort_values("bank").reset_index(drop=True)
        return out.loc[:, list(self._BANK_COLUMNS)]

    @staticmethod
    def _normalize_institution_asset(asset: str) -> str:
        token = str(asset or "").strip().lower().replace("_", "-")
        return FXEnhancedProvider._INSTITUTION_ASSET_ALIASES.get(token, token)

    def _normalize_institution_frame(self, frame: pd.DataFrame, asset: str) -> pd.DataFrame:
        if frame.empty:
            return self._empty_institution_rates()

        work = frame.copy()
        inst_col = self._pick_column(work, ["institution", "bank", "code", "slug", "name"])
        inst_name_col = self._pick_column(work, ["institution_name", "bank_name", "display_name", "title", "name"])
        buy_col = self._pick_column(work, ["buy", "bid", "buying", "alis"])
        sell_col = self._pick_column(work, ["sell", "ask", "selling", "satis"])
        spread_col = self._pick_column(work, ["spread", "spread_pct", "makas"])
        asset_col = self._pick_column(work, ["asset", "currency", "symbol"])
        timestamp_col = self._pick_column(work, ["timestamp", "update_time", "datetime", "date", "time"])

        out = pd.DataFrame(index=work.index)
        out["institution"] = work[inst_col].astype("string").str.strip() if inst_col else pd.NA
        out["institution_name"] = (
            work[inst_name_col].astype("string").str.strip() if inst_name_col else pd.NA
        )
        out["asset"] = work[asset_col].astype("string").str.strip() if asset_col else asset
        out["buy"] = work[buy_col].map(self._to_float) if buy_col else pd.NA
        out["sell"] = work[sell_col].map(self._to_float) if sell_col else pd.NA
        out["spread"] = work[spread_col].map(self._to_float) if spread_col else pd.NA

        calculated_spread = (
            (pd.to_numeric(out["sell"], errors="coerce") - pd.to_numeric(out["buy"], errors="coerce"))
            / pd.to_numeric(out["buy"], errors="coerce")
            * 100.0
        )
        out["spread"] = pd.to_numeric(out["spread"], errors="coerce").fillna(calculated_spread)
        out["mid"] = (
            pd.to_numeric(out["buy"], errors="coerce")
            + pd.to_numeric(out["sell"], errors="coerce")
        ) / 2.0

        if timestamp_col:
            ts = self._to_timestamp_series(work[timestamp_col])
        else:
            ts = pd.Series(pd.Timestamp(datetime.now()), index=out.index)
        out["timestamp"] = ts.fillna(pd.Timestamp(datetime.now()))
        out["source"] = "institution_rates"

        out["institution"] = out["institution"].fillna(out["institution_name"]).fillna("")
        out["institution_name"] = out["institution_name"].fillna(out["institution"])
        out["asset"] = out["asset"].fillna(asset).astype("string")

        out = out[
            pd.to_numeric(out["buy"], errors="coerce").notna()
            & pd.to_numeric(out["sell"], errors="coerce").notna()
        ]
        if out.empty:
            return self._empty_institution_rates()

        out = out.sort_values("institution").reset_index(drop=True)
        return out.loc[:, list(self._INSTITUTION_COLUMNS)]

    @staticmethod
    def _normalize_intraday_asset(currency: str) -> str | None:
        token = str(currency or "").strip().upper().replace(" ", "")
        mapped = FXEnhancedProvider._INTRADAY_ALIASES.get(token, token)
        return mapped if mapped in FXEnhancedProvider._INTRADAY_SUPPORTED else None

    def _normalize_intraday_frame(
        self,
        frame: pd.DataFrame,
        asset: str,
        interval: str,
        period: str,
    ) -> pd.DataFrame:
        if frame.empty:
            return self._empty_intraday()

        work = frame.copy()
        if isinstance(work.index, pd.DatetimeIndex):
            work = work.sort_index().reset_index()
        elif not isinstance(work.index, pd.RangeIndex):
            work = work.reset_index()

        ts_col = self._pick_column(work, ["timestamp", "datetime", "date", "time", "index"])
        open_col = self._pick_column(work, ["open", "o"])
        high_col = self._pick_column(work, ["high", "h"])
        low_col = self._pick_column(work, ["low", "l"])
        close_col = self._pick_column(work, ["close", "c", "last", "price", "value"])
        vol_col = self._pick_column(work, ["volume", "vol", "v"])

        out = pd.DataFrame(index=work.index)
        if ts_col:
            out["timestamp"] = self._to_timestamp_series(work[ts_col])
        else:
            out["timestamp"] = pd.NaT
        out["open"] = work[open_col].map(self._to_float) if open_col else pd.NA
        out["high"] = work[high_col].map(self._to_float) if high_col else pd.NA
        out["low"] = work[low_col].map(self._to_float) if low_col else pd.NA
        out["close"] = work[close_col].map(self._to_float) if close_col else pd.NA
        out["volume"] = work[vol_col].map(self._to_float) if vol_col else 0.0
        out["asset"] = asset
        out["interval"] = str(interval)
        out["period"] = str(period)
        out["source"] = "tradingview"

        for col in ("open", "high", "low"):
            out[col] = pd.to_numeric(out[col], errors="coerce")
        close_numeric = pd.to_numeric(out["close"], errors="coerce")
        out["close"] = close_numeric
        out["open"] = out["open"].fillna(close_numeric)
        out["high"] = out["high"].fillna(close_numeric)
        out["low"] = out["low"].fillna(close_numeric)

        out = out[out["timestamp"].notna() & out["close"].notna()]
        if out.empty:
            return self._empty_intraday()

        out = out.sort_values("timestamp").reset_index(drop=True)
        return out.loc[:, list(self._INTRADAY_COLUMNS)]

    def get_bank_rates(self, currency: str = "USD") -> pd.DataFrame:
        """All bank buying/selling rates for a currency."""
        bp = self._get_bp()
        if bp is None:
            return self._empty_bank_rates()

        normalized_currency = self._normalize_bank_currency(currency)
        fx_cls = getattr(bp, "FX", None)
        if not callable(fx_cls):
            return self._empty_bank_rates()

        try:
            fx = fx_cls(normalized_currency)
            payload = getattr(fx, "bank_rates", None)
            if callable(payload):
                payload = self._call_if_callable(payload)
        except Exception as exc:
            logger.info(
                "  FXEnhancedProvider: bank rates unavailable for %s: %s",
                normalized_currency,
                exc,
            )
            return self._empty_bank_rates()

        return self._normalize_bank_frame(self._as_frame(payload), normalized_currency)

    def get_institution_rates(self, asset: str = "gram-altin") -> pd.DataFrame:
        """Gold/silver institution rates (kuyumcular + banks)."""
        bp = self._get_bp()
        if bp is None:
            return self._empty_institution_rates()

        normalized_asset = self._normalize_institution_asset(asset)
        fx_cls = getattr(bp, "FX", None)
        if not callable(fx_cls):
            return self._empty_institution_rates()

        try:
            fx = fx_cls(normalized_asset)
            payload = getattr(fx, "institution_rates", None)
            if callable(payload):
                payload = self._call_if_callable(payload)
        except Exception as exc:
            logger.info(
                "  FXEnhancedProvider: institution rates unavailable for %s: %s",
                normalized_asset,
                exc,
            )
            return self._empty_institution_rates()

        return self._normalize_institution_frame(self._as_frame(payload), normalized_asset)

    def get_intraday(
        self,
        currency: str = "USD",
        interval: str = "1h",
        period: str = "5d",
    ) -> pd.DataFrame:
        """Intraday FX history via TradingView-backed borsapy FX.history()."""
        bp = self._get_bp()
        if bp is None:
            return self._empty_intraday()

        normalized_asset = self._normalize_intraday_asset(currency)
        if normalized_asset is None:
            logger.info(
                "  FXEnhancedProvider: intraday not supported for '%s' (supported: %s)",
                currency,
                sorted(self._INTRADAY_SUPPORTED),
            )
            return self._empty_intraday()

        fx_cls = getattr(bp, "FX", None)
        if not callable(fx_cls):
            return self._empty_intraday()

        try:
            fx = fx_cls(normalized_asset)
            history_fn = getattr(fx, "history", None)
            payload = self._call_if_callable(history_fn, period=period, interval=interval)
            if payload is None:
                payload = self._call_if_callable(history_fn, period, interval)
        except Exception as exc:
            logger.info(
                "  FXEnhancedProvider: intraday history unavailable for %s: %s",
                normalized_asset,
                exc,
            )
            return self._empty_intraday()

        return self._normalize_intraday_frame(
            self._as_frame(payload),
            asset=normalized_asset,
            interval=interval,
            period=period,
        )

    def get_carry_spread(self) -> dict[str, Any]:
        """Bank buy/sell spread summary as a simple carry-trade proxy."""
        rates = self.get_bank_rates(currency="USD")
        if rates.empty:
            return {
                "currency": "USD",
                "bank_count": 0,
                "avg_buy": None,
                "avg_sell": None,
                "avg_spread_pct": None,
                "min_spread_pct": None,
                "max_spread_pct": None,
                "best_bank": None,
                "worst_bank": None,
                "timestamp": datetime.now().isoformat(),
            }

        buy = pd.to_numeric(rates["buy"], errors="coerce")
        sell = pd.to_numeric(rates["sell"], errors="coerce")
        spread = pd.to_numeric(rates["spread"], errors="coerce")

        valid = rates[(buy.notna()) & (sell.notna())]
        if valid.empty:
            return {
                "currency": "USD",
                "bank_count": 0,
                "avg_buy": None,
                "avg_sell": None,
                "avg_spread_pct": None,
                "min_spread_pct": None,
                "max_spread_pct": None,
                "best_bank": None,
                "worst_bank": None,
                "timestamp": datetime.now().isoformat(),
            }

        valid_spread = pd.to_numeric(valid["spread"], errors="coerce").dropna()
        best_bank = None
        worst_bank = None
        if not valid_spread.empty:
            best_row = valid.loc[valid_spread.idxmin()]
            worst_row = valid.loc[valid_spread.idxmax()]
            best_bank = str(best_row.get("bank_name") or best_row.get("bank") or "")
            worst_bank = str(worst_row.get("bank_name") or worst_row.get("bank") or "")

        return {
            "currency": "USD",
            "bank_count": int(len(valid)),
            "avg_buy": float(pd.to_numeric(valid["buy"], errors="coerce").mean()),
            "avg_sell": float(pd.to_numeric(valid["sell"], errors="coerce").mean()),
            "avg_spread_pct": float(valid_spread.mean()) if not valid_spread.empty else None,
            "min_spread_pct": float(valid_spread.min()) if not valid_spread.empty else None,
            "max_spread_pct": float(valid_spread.max()) if not valid_spread.empty else None,
            "best_bank": best_bank or None,
            "worst_bank": worst_bank or None,
            "timestamp": datetime.now().isoformat(),
        }
