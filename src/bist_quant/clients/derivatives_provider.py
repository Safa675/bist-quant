"""
VIOP derivatives data provider.

Primary source is `borsapy` (when available). Methods are defensive and
normalize multiple possible payload shapes to stable return types.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime, timezone
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class DerivativesProvider:
    """Resilient accessor for VIOP futures/options datasets."""

    def __init__(
        self,
        borsapy_module: Any | None = None,
        cache_dir: Any = None,
    ) -> None:
        self._bp = borsapy_module
        self._import_attempted = borsapy_module is not None
        self._disk_cache: Any | None = None
        if cache_dir is not None:
            try:
                from pathlib import Path as _Path
                from bist_quant.common.disk_cache import DiskCache
                self._disk_cache = DiskCache(_Path(cache_dir))
            except Exception:
                pass

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
            logger.info("  DerivativesProvider: borsapy unavailable: %s", exc)
            self._bp = None
        return self._bp

    @staticmethod
    def _as_frame(payload: Any) -> pd.DataFrame:
        if isinstance(payload, pd.DataFrame):
            return payload.copy()
        if isinstance(payload, (list, tuple)):
            if not payload:
                return pd.DataFrame()
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
        text = re.sub(r"[^0-9eE.\-+]", "", text)
        if not text:
            return None
        try:
            parsed = float(text)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None

    @staticmethod
    def _normalize_symbol(symbol: Any) -> str:
        return str(symbol or "").strip().upper().split(".")[0]

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

    def _get_viop_instance(self, bp: Any) -> Any | None:
        viop_obj = getattr(bp, "VIOP", None)
        if viop_obj is None:
            return None
        instance = self._call_if_callable(viop_obj)
        if instance is not None:
            return instance
        if not callable(viop_obj):
            return viop_obj
        return None

    def _collect_viop_frames(self) -> list[pd.DataFrame]:
        bp = self._get_bp()
        if bp is None:
            return []

        frames: list[pd.DataFrame] = []
        viop = self._get_viop_instance(bp)

        module_candidates = (
            "viop",
            "viop_contracts",
            "viop_futures",
            "viop_options",
            "futures",
            "options",
        )
        for name in module_candidates:
            fn = getattr(bp, name, None)
            payload = None
            if callable(fn):
                payload = self._call_if_callable(fn)
            if payload is None:
                continue
            frame = self._as_frame(payload)
            if not frame.empty:
                frames.append(frame)

        if viop is None:
            return frames

        method_candidates = (
            "contracts",
            "all",
            "table",
            "data",
            "latest",
            "futures",
            "get_futures",
            "futures_contracts",
            "options",
            "get_options",
            "option_chain",
        )
        for name in method_candidates:
            payload = self._call_if_callable(getattr(viop, name, None))
            if payload is None:
                continue
            frame = self._as_frame(payload)
            if not frame.empty:
                frames.append(frame)

        return frames

    def _all_contracts_frame(self) -> pd.DataFrame:
        frames = self._collect_viop_frames()
        if not frames:
            return pd.DataFrame()
        try:
            merged = pd.concat(frames, ignore_index=True, sort=False)
        except Exception:
            merged = frames[0].copy()
            for frame in frames[1:]:
                merged = merged.merge(frame, how="outer")
        return merged.drop_duplicates().reset_index(drop=True)

    @staticmethod
    def _extract_option_side(record: dict[str, Any]) -> str | None:
        for key in ("option_type", "type", "right", "side", "contract_type"):
            raw = str(record.get(key, "")).strip().lower()
            if not raw:
                continue
            if "put" in raw or "satim" in raw:
                return "put"
            if "call" in raw or "alim" in raw:
                return "call"
            if raw in {"p", "put"}:
                return "put"
            if raw in {"c", "call"}:
                return "call"

        symbol = str(record.get("symbol", "")).strip().upper()
        match = re.search(r"([CP])(?:\d+)?$", symbol)
        if match:
            return "call" if match.group(1) == "C" else "put"
        return None

    @staticmethod
    def _is_option_record(record: dict[str, Any]) -> bool:
        side = DerivativesProvider._extract_option_side(record)
        if side is not None:
            return True

        text_fields = [
            str(record.get("contract_type", "")),
            str(record.get("type", "")),
            str(record.get("instrument_type", "")),
            str(record.get("asset_type", "")),
            str(record.get("category", "")),
        ]
        text = " ".join(text_fields).lower()
        return "option" in text or "opsiyon" in text

    @staticmethod
    def _is_futures_record(record: dict[str, Any]) -> bool:
        if DerivativesProvider._is_option_record(record):
            return False
        text_fields = [
            str(record.get("contract_type", "")),
            str(record.get("type", "")),
            str(record.get("instrument_type", "")),
            str(record.get("asset_type", "")),
            str(record.get("category", "")),
        ]
        text = " ".join(text_fields).lower()
        if "future" in text or "futures" in text or "vadeli" in text:
            return True
        return True

    @staticmethod
    def _extract_expiry(record: dict[str, Any]) -> pd.Timestamp | None:
        for key in ("expiry", "expiration", "maturity", "vade", "date"):
            raw = record.get(key)
            if raw is None:
                continue
            parsed = pd.to_datetime(raw, errors="coerce")
            if pd.isna(parsed):
                continue
            return pd.Timestamp(parsed).tz_localize(None) if getattr(parsed, "tzinfo", None) else pd.Timestamp(parsed)
        return None

    @classmethod
    def _extract_last_price(cls, record: dict[str, Any]) -> float | None:
        for key in ("last_price", "last", "price", "close", "settlement", "mark", "value"):
            value = cls._to_float(record.get(key))
            if value is not None:
                return value
        return None

    def _frame_to_records(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        if frame.empty:
            return []

        symbol_col = self._pick_column(
            frame,
            ["symbol", "contract_symbol", "contract", "code", "ticker", "name"],
        )
        base_col = self._pick_column(
            frame,
            ["base_symbol", "underlying", "root", "asset", "underlier", "base"],
        )
        type_col = self._pick_column(
            frame,
            ["contract_type", "type", "instrument_type", "asset_type", "category"],
        )
        option_side_col = self._pick_column(frame, ["option_type", "right", "side"])
        expiry_col = self._pick_column(frame, ["expiry", "expiration", "maturity", "vade", "date"])
        last_col = self._pick_column(frame, ["last_price", "last", "price", "close", "settlement", "mark"])
        volume_col = self._pick_column(frame, ["volume", "vol", "trade_volume", "traded_volume"])
        oi_col = self._pick_column(frame, ["open_interest", "oi", "openinterest"])
        strike_col = self._pick_column(frame, ["strike", "strike_price", "kullanim_fiyati"])

        out: list[dict[str, Any]] = []
        for _, row in frame.iterrows():
            base_record = row.to_dict()
            record: dict[str, Any] = {}
            if symbol_col is not None:
                record["symbol"] = self._normalize_symbol(base_record.get(symbol_col))
            else:
                record["symbol"] = self._normalize_symbol(base_record.get("symbol"))
            if base_col is not None:
                record["base_symbol"] = self._normalize_symbol(base_record.get(base_col))
            if type_col is not None:
                record["contract_type"] = str(base_record.get(type_col, "")).strip().lower()
            if option_side_col is not None:
                raw_side = str(base_record.get(option_side_col, "")).strip().lower()
                if raw_side:
                    record["option_type"] = raw_side
            if expiry_col is not None:
                expiry = pd.to_datetime(base_record.get(expiry_col), errors="coerce")
                if pd.notna(expiry):
                    record["expiry"] = pd.Timestamp(expiry).date().isoformat()
            if last_col is not None:
                record["last_price"] = self._to_float(base_record.get(last_col))
            if volume_col is not None:
                record["volume"] = self._to_float(base_record.get(volume_col))
            if oi_col is not None:
                record["open_interest"] = self._to_float(base_record.get(oi_col))
            if strike_col is not None:
                record["strike"] = self._to_float(base_record.get(strike_col))

            merged = dict(base_record)
            merged.update(record)
            out.append(merged)
        return out

    @staticmethod
    def _select_records_by_base(records: list[dict[str, Any]], base_symbol: str) -> list[dict[str, Any]]:
        normalized_base = DerivativesProvider._normalize_symbol(base_symbol)
        if not normalized_base:
            return records
        selected: list[dict[str, Any]] = []
        for record in records:
            base = DerivativesProvider._normalize_symbol(record.get("base_symbol"))
            symbol = DerivativesProvider._normalize_symbol(record.get("symbol"))
            if base == normalized_base:
                selected.append(record)
                continue
            if symbol.startswith(normalized_base):
                selected.append(record)
        return selected

    @staticmethod
    def _infer_spot_symbol(base_symbol: str) -> str:
        symbol = DerivativesProvider._normalize_symbol(base_symbol)
        explicit_map = {"XU030D": "XU030", "XU100D": "XU100", "XUTUMD": "XUTUM"}
        if symbol in explicit_map:
            return explicit_map[symbol]
        if symbol.endswith("D") and symbol[:-1] in {"XU030", "XU100", "XUTUM"}:
            return symbol[:-1]
        return symbol

    def _spot_price(self, spot_symbol: str) -> float | None:
        bp = self._get_bp()
        if bp is None:
            return None

        index_fn = getattr(bp, "index", None)
        if callable(index_fn):
            idx = self._call_if_callable(index_fn, spot_symbol)
            info = getattr(idx, "info", None)
            if isinstance(info, dict):
                for key in ("last", "value", "close", "previous_close"):
                    price = self._to_float(info.get(key))
                    if price is not None:
                        return price

        ticker_cls = getattr(bp, "Ticker", None)
        if callable(ticker_cls):
            ticker = self._call_if_callable(ticker_cls, spot_symbol)
            fast_info = getattr(ticker, "fast_info", None)
            if isinstance(fast_info, dict):
                for key in ("last_price", "last", "close", "previous_close"):
                    price = self._to_float(fast_info.get(key))
                    if price is not None:
                        return price
        return None

    def get_futures(self) -> pd.DataFrame:
        """All active futures contracts."""
        if self._disk_cache is not None:
            _cached = self._disk_cache.get_dataframe("derivatives", "futures")
            if _cached is not None:
                return _cached
        frame = self._all_contracts_frame()
        if frame.empty:
            return pd.DataFrame()

        records = self._frame_to_records(frame)
        futures = [record for record in records if self._is_futures_record(record)]
        if not futures:
            return pd.DataFrame()
        _result = pd.DataFrame(futures).drop_duplicates().reset_index(drop=True)
        if self._disk_cache is not None and not _result.empty:
            self._disk_cache.set_dataframe("derivatives", "futures", _result)
        return _result

    def get_options(self) -> pd.DataFrame:
        """All active options."""
        if self._disk_cache is not None:
            _cached = self._disk_cache.get_dataframe("derivatives", "options")
            if _cached is not None:
                return _cached
        frame = self._all_contracts_frame()
        if frame.empty:
            return pd.DataFrame()

        records = self._frame_to_records(frame)
        options = [record for record in records if self._is_option_record(record)]
        if not options:
            return pd.DataFrame()
        _result = pd.DataFrame(options).drop_duplicates().reset_index(drop=True)
        if self._disk_cache is not None and not _result.empty:
            self._disk_cache.set_dataframe("derivatives", "options", _result)
        return _result

    def get_contracts(self, base_symbol: str) -> list[dict]:
        """Available contracts for a base symbol (e.g., XU030D)."""
        bp = self._get_bp()
        normalized_base = self._normalize_symbol(base_symbol)

        if bp is not None:
            viop_contracts_fn = getattr(bp, "viop_contracts", None)
            if callable(viop_contracts_fn):
                payload = None
                for kwargs in (
                    {"base_symbol": normalized_base},
                    {"symbol": normalized_base},
                    {"underlying": normalized_base},
                    {},
                ):
                    payload = self._call_if_callable(viop_contracts_fn, **kwargs)
                    if payload is not None:
                        break
                if payload is None:
                    payload = self._call_if_callable(viop_contracts_fn, normalized_base)
                frame = self._as_frame(payload)
                if not frame.empty:
                    records = self._frame_to_records(frame)
                    selected = self._select_records_by_base(records, normalized_base)
                    if selected:
                        return selected

        all_records: list[dict[str, Any]] = []
        futures = self.get_futures()
        options = self.get_options()
        if not futures.empty:
            all_records.extend(futures.to_dict("records"))
        if not options.empty:
            all_records.extend(options.to_dict("records"))
        return self._select_records_by_base(all_records, normalized_base)

    def _select_front_month(self, contracts: list[dict[str, Any]]) -> dict[str, Any] | None:
        futures = [record for record in contracts if self._is_futures_record(record)]
        if not futures:
            return None

        today = pd.Timestamp(datetime.now(timezone.utc).date())

        dated: list[tuple[pd.Timestamp, dict[str, Any]]] = []
        undated: list[dict[str, Any]] = []
        for record in futures:
            expiry = self._extract_expiry(record)
            if expiry is None:
                undated.append(record)
            else:
                dated.append((expiry.normalize(), record))

        if dated:
            future_dated = [(d, r) for d, r in dated if d >= today]
            if future_dated:
                return sorted(future_dated, key=lambda item: item[0])[0][1]
            return sorted(dated, key=lambda item: item[0])[0][1]

        return undated[0] if undated else None

    def get_futures_basis(self, base_symbol: str = "XU030D") -> dict:
        """Futures basis = futures_price - spot_price."""
        normalized_base = self._normalize_symbol(base_symbol or "XU030D")
        spot_symbol = self._infer_spot_symbol(normalized_base)
        contracts = self.get_contracts(normalized_base)
        front = self._select_front_month(contracts)

        if front is None:
            return {
                "base_symbol": normalized_base,
                "spot_symbol": spot_symbol,
                "error": "No futures contract available",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        futures_price = self._extract_last_price(front)
        spot_price = self._spot_price(spot_symbol)
        basis = None
        basis_pct = None
        regime = None
        if futures_price is not None and spot_price not in (None, 0.0):
            basis = float(futures_price - spot_price)
            basis_pct = float((basis / spot_price) * 100.0)
            if basis > 0:
                regime = "contango"
            elif basis < 0:
                regime = "backwardation"
            else:
                regime = "flat"

        expiry = self._extract_expiry(front)
        return {
            "base_symbol": normalized_base,
            "spot_symbol": spot_symbol,
            "contract": self._normalize_symbol(front.get("symbol")),
            "expiry": expiry.date().isoformat() if expiry is not None else None,
            "futures_price": futures_price,
            "spot_price": spot_price,
            "basis": basis,
            "basis_pct": basis_pct,
            "regime": regime,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_put_call_ratio(self) -> float | None:
        """Aggregate P/C ratio from VIOP options."""
        options = self.get_options()
        if options.empty:
            return None

        put_total = 0.0
        call_total = 0.0
        for record in options.to_dict("records"):
            side = self._extract_option_side(record)
            if side not in {"put", "call"}:
                continue

            weight = None
            for key in ("volume", "open_interest", "oi", "trade_volume", "traded_volume"):
                weight = self._to_float(record.get(key))
                if weight is not None:
                    break
            if weight is None:
                weight = 1.0

            if side == "put":
                put_total += float(weight)
            else:
                call_total += float(weight)

        if call_total <= 0.0:
            return None
        return float(put_total / call_total)

    def get_index_futures_premium(self) -> dict:
        """XU030 futures premium/discount vs spot."""
        basis = self.get_futures_basis(base_symbol="XU030D")
        premium_points = basis.get("basis")
        premium_pct = basis.get("basis_pct")
        return {
            "base_symbol": basis.get("base_symbol", "XU030D"),
            "spot_symbol": basis.get("spot_symbol", "XU030"),
            "contract": basis.get("contract"),
            "expiry": basis.get("expiry"),
            "premium_points": premium_points,
            "premium_pct": premium_pct,
            "regime": basis.get("regime"),
            "is_premium": bool(premium_points is not None and premium_points > 0),
            "timestamp": basis.get("timestamp", datetime.now(timezone.utc).isoformat()),
            **({"error": basis["error"]} if "error" in basis else {}),
        }

