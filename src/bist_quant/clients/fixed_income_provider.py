"""
Fixed income data provider: bond yields, TCMB rates, and Eurobonds.

Primary source is `borsapy` (when available). Methods are defensive and
normalize multiple possible payload shapes to stable return types.
"""

from __future__ import annotations

import logging
import math
import re
from datetime import datetime
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_FALLBACK_RISK_FREE_RATE = 0.28


class FixedIncomeProvider:
    """Resilient accessor for fixed income and central bank datasets."""

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
            logger.info("  FixedIncomeProvider: borsapy unavailable: %s", exc)
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
            # If all values are scalar, make single-row frame.
            if payload and all(not isinstance(v, (list, tuple, dict, pd.Series, pd.DataFrame)) for v in payload.values()):
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
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            if not math.isfinite(float(value)):
                return None
            return float(value)

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
        if not math.isfinite(parsed):
            return None
        return parsed

    @staticmethod
    def _to_percent_rate(value: Any) -> float | None:
        parsed = FixedIncomeProvider._to_float(value)
        if parsed is None:
            return None
        # If already in decimal annualized form (0.2803), convert to percent.
        if -1.5 <= parsed <= 1.5:
            return parsed * 100.0
        return parsed

    @staticmethod
    def _to_decimal_rate(value: Any) -> float | None:
        parsed = FixedIncomeProvider._to_float(value)
        if parsed is None:
            return None
        # If likely percentage (28.03), convert to decimal.
        if abs(parsed) > 1.5:
            return parsed / 100.0
        return parsed

    @staticmethod
    def _normalize_maturity(label: Any) -> str | None:
        if label is None:
            return None
        text = str(label).strip().upper()
        if not text:
            return None
        match = re.search(r"(\d+(?:\.\d+)?)\s*Y", text)
        if match:
            years = float(match.group(1))
            return f"{int(years)}Y" if years.is_integer() else f"{years:g}Y"

        # Fallback: extract first number from labels like "2 years", "10yr".
        match = re.search(r"(\d+(?:\.\d+)?)", text)
        if match:
            years = float(match.group(1))
            return f"{int(years)}Y" if years.is_integer() else f"{years:g}Y"
        return None

    @staticmethod
    def _extract_series(frame: pd.DataFrame, value_candidates: list[str]) -> pd.Series:
        if frame.empty:
            return pd.Series(dtype=float)

        work = frame.copy()
        date_col = FixedIncomeProvider._pick_column(
            work, ["date", "datetime", "timestamp", "time", "tarih"]
        )
        if date_col is not None:
            work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
            work = work.dropna(subset=[date_col]).set_index(date_col).sort_index()
        elif not isinstance(work.index, pd.DatetimeIndex):
            return pd.Series(dtype=float)
        else:
            work = work.sort_index()

        value_col = FixedIncomeProvider._pick_column(work, value_candidates)
        if value_col is None:
            numeric_cols = [
                c for c in work.columns if pd.api.types.is_numeric_dtype(work[c])
            ]
            if not numeric_cols:
                return pd.Series(dtype=float)
            value_col = str(numeric_cols[0])

        values = work[value_col].apply(FixedIncomeProvider._to_float).astype(float)
        values = values[values.notna()]
        values = values[~values.index.duplicated(keep="last")]
        return values.sort_index()

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

    def _extract_yield_points(self, frame: pd.DataFrame) -> dict[str, float]:
        if frame.empty:
            return {}

        work = frame.copy()
        maturity_col = self._pick_column(
            work, ["maturity", "tenor", "term", "vade", "label", "name", "bond", "type"]
        )
        yield_col = self._pick_column(
            work,
            ["yield", "ask_yield", "ytm", "rate", "value", "faiz", "last", "close"],
        )

        if maturity_col is None:
            if work.index.nlevels == 1 and not isinstance(work.index, pd.RangeIndex):
                work = work.reset_index().rename(columns={work.index.name or "index": "maturity"})
                maturity_col = "maturity"
            else:
                return {}

        if yield_col is None:
            numeric_cols = [
                str(col)
                for col in work.columns
                if pd.api.types.is_numeric_dtype(work[col]) and str(col) != maturity_col
            ]
            if not numeric_cols:
                return {}
            yield_col = numeric_cols[0]

        points: dict[str, float] = {}
        for _, row in work.iterrows():
            maturity = self._normalize_maturity(row.get(maturity_col))
            if maturity is None:
                continue
            value = self._to_percent_rate(row.get(yield_col))
            if value is None:
                continue
            points[maturity] = float(value)
        return points

    @staticmethod
    def _sort_maturity_keys(keys: list[str]) -> list[str]:
        def _order_key(label: str) -> float:
            match = re.search(r"(\d+(?:\.\d+)?)", str(label))
            return float(match.group(1)) if match else float("inf")

        return sorted(keys, key=_order_key)

    def get_bond_yields(self) -> dict[str, float]:
        """
        Return bond yields as percentages.

        Example:
            {"2Y": 26.42, "5Y": 27.15, "10Y": 28.03}
        """
        bp = self._get_bp()
        if bp is None:
            return {}

        points: dict[str, float] = {}

        for name in ("bonds", "bond_yields", "yield_curve"):
            payload = self._call_if_callable(getattr(bp, name, None))
            frame = self._as_frame(payload)
            parsed = self._extract_yield_points(frame)
            if parsed:
                points.update(parsed)
                break

        if not points:
            bond_obj = getattr(bp, "Bond", None)
            bond_instance = self._call_if_callable(bond_obj) if bond_obj is not None else None
            if bond_instance is not None:
                for method in ("yield_curve", "yields", "curve", "table", "data", "latest"):
                    payload = self._call_if_callable(getattr(bond_instance, method, None))
                    frame = self._as_frame(payload)
                    parsed = self._extract_yield_points(frame)
                    if parsed:
                        points.update(parsed)
                        break

        if not points:
            risk_free_raw = self._call_if_callable(getattr(bp, "risk_free_rate", None))
            risk_free = self._to_percent_rate(risk_free_raw)
            if risk_free is not None:
                points["10Y"] = float(risk_free)

        return {k: points[k] for k in self._sort_maturity_keys(list(points.keys()))}

    def get_risk_free_rate(self) -> float:
        """Return annual risk-free rate as decimal (e.g. 0.2803)."""
        bp = self._get_bp()
        if bp is not None:
            for name in ("risk_free_rate", "riskfree_rate"):
                payload = self._call_if_callable(getattr(bp, name, None))
                if payload is None:
                    continue

                if isinstance(payload, dict):
                    for key in ("rate", "value", "risk_free_rate", "10Y", "10y"):
                        candidate = self._to_decimal_rate(payload.get(key))
                        if candidate is not None:
                            return float(candidate)
                else:
                    candidate = self._to_decimal_rate(payload)
                    if candidate is not None:
                        return float(candidate)

        yields = self.get_bond_yields()
        for key in ("10Y", "5Y", "2Y"):
            if key in yields:
                candidate = self._to_decimal_rate(yields[key])
                if candidate is not None:
                    return float(candidate)

        return float(DEFAULT_FALLBACK_RISK_FREE_RATE)

    def get_yield_curve(self) -> pd.DataFrame:
        """Return DataFrame with columns: maturity, yield, change."""
        bp = self._get_bp()
        if bp is not None:
            for name in ("bonds", "bond_yields", "yield_curve"):
                payload = self._call_if_callable(getattr(bp, name, None))
                frame = self._as_frame(payload)
                if frame.empty:
                    continue

                work = frame.copy()
                maturity_col = self._pick_column(
                    work, ["maturity", "tenor", "term", "vade", "label", "name", "bond", "type"]
                )
                yield_col = self._pick_column(
                    work, ["yield", "ask_yield", "ytm", "rate", "value", "faiz", "last", "close"]
                )
                change_col = self._pick_column(work, ["change", "delta", "chg", "daily_change"])
                if maturity_col is None or yield_col is None:
                    continue

                records: list[dict[str, Any]] = []
                for _, row in work.iterrows():
                    maturity = self._normalize_maturity(row.get(maturity_col))
                    if maturity is None:
                        continue
                    yld = self._to_percent_rate(row.get(yield_col))
                    if yld is None:
                        continue
                    change = self._to_float(row.get(change_col)) if change_col is not None else None
                    records.append({"maturity": maturity, "yield": yld, "change": change})

                if records:
                    out = pd.DataFrame(records)
                    out = out.sort_values(
                        "maturity",
                        key=lambda s: s.map(
                            lambda v: float(re.search(r"(\d+(?:\.\d+)?)", str(v)).group(1))
                            if re.search(r"(\d+(?:\.\d+)?)", str(v))
                            else float("inf")
                        ),
                    )
                    return out.reset_index(drop=True)

        yields = self.get_bond_yields()
        if not yields:
            return pd.DataFrame(columns=["maturity", "yield", "change"])

        return pd.DataFrame(
            [{"maturity": key, "yield": value, "change": None} for key, value in yields.items()]
        )

    def get_tcmb_rates(self) -> dict[str, Any]:
        """
        Return current TCMB policy/corridor rates.

        Example:
            {"policy_rate": 38.0, "overnight": {...}, "late_liquidity": {...}}
        """
        bp = self._get_bp()
        if bp is None:
            return {}

        tcmb_obj = getattr(bp, "TCMB", None)
        tcmb = self._call_if_callable(tcmb_obj) if tcmb_obj is not None else None
        if tcmb is None:
            return {}

        result: dict[str, Any] = {"timestamp": datetime.now().isoformat()}

        policy_rate = self._to_float(getattr(tcmb, "policy_rate", None))
        if policy_rate is not None:
            result["policy_rate"] = policy_rate

        overnight = getattr(tcmb, "overnight", None)
        if isinstance(overnight, dict):
            result["overnight"] = overnight

        late_liquidity = getattr(tcmb, "late_liquidity", None)
        if isinstance(late_liquidity, dict):
            result["late_liquidity"] = late_liquidity

        rates_table = getattr(tcmb, "rates", None)
        frame = self._as_frame(rates_table)
        if not frame.empty:
            result["rates_detail"] = frame.to_dict("records")

        return result

    def get_tcmb_history(self, rate_type: str = "policy", period: str = "1y") -> pd.DataFrame:
        """Return normalized historical TCMB rate series with DateTime index."""
        bp = self._get_bp()
        if bp is None:
            return pd.DataFrame(columns=["rate"])

        tcmb_obj = getattr(bp, "TCMB", None)
        tcmb = self._call_if_callable(tcmb_obj) if tcmb_obj is not None else None
        if tcmb is None:
            return pd.DataFrame(columns=["rate"])

        payload = None
        method_names = (
            "history",
            "get_history",
            "rate_history",
            "get_rate_history",
            "policy_history",
            "policy_rate_history",
        )

        for method_name in method_names:
            method = getattr(tcmb, method_name, None)
            if not callable(method):
                continue

            # Try common signatures.
            for kwargs in (
                {"rate_type": rate_type, "period": period},
                {"period": period},
                {"rate_type": rate_type},
                {},
            ):
                payload = self._call_if_callable(method, **kwargs)
                if payload is not None:
                    break

            if payload is None:
                for args in ((rate_type, period), (period,), (rate_type,), ()):
                    payload = self._call_if_callable(method, *args)
                    if payload is not None:
                        break
            if payload is not None:
                break

        frame = self._as_frame(payload)
        series = self._extract_series(
            frame,
            [
                rate_type,
                f"{rate_type}_rate",
                "policy_rate",
                "rate",
                "value",
                "close",
                "faiz",
            ],
        )
        if series.empty:
            policy_rate = self._to_float(getattr(tcmb, "policy_rate", None))
            if policy_rate is None:
                return pd.DataFrame(columns=["rate"])
            return pd.DataFrame(
                {"rate": [policy_rate]},
                index=pd.DatetimeIndex([pd.Timestamp(datetime.now().date())], name="Date"),
            )

        out = series.to_frame(name="rate")
        out.index = pd.to_datetime(out.index, errors="coerce")
        out = out[out.index.notna()].sort_index()
        out = out[~out.index.duplicated(keep="last")]
        out.index.name = "Date"
        return out

    def get_eurobonds(self, currency: str | None = None) -> pd.DataFrame:
        """Return Turkish sovereign Eurobond table."""
        bp = self._get_bp()
        if bp is None:
            return pd.DataFrame()

        payload = None
        for name in ("eurobonds", "eurobond", "get_eurobonds"):
            payload = self._call_if_callable(getattr(bp, name, None))
            if payload is not None:
                break

        if payload is None:
            eurobond_obj = getattr(bp, "Eurobond", None)
            eurobond_instance = self._call_if_callable(eurobond_obj) if eurobond_obj is not None else None
            if eurobond_instance is not None:
                for method in ("all", "list", "table", "data", "latest"):
                    payload = self._call_if_callable(getattr(eurobond_instance, method, None))
                    if payload is not None:
                        break

        frame = self._as_frame(payload)
        if frame.empty:
            return frame

        if currency:
            ccy_col = self._pick_column(frame, ["currency", "ccy", "curr"])
            if ccy_col is not None:
                want = str(currency).upper()
                frame = frame[
                    frame[ccy_col].astype(str).str.upper().str.contains(want, na=False)
                ]

        return frame.reset_index(drop=True)

    def get_spread_index(self) -> float:
        """
        Return average USD Eurobond spread proxy.

        Current implementation follows the integration plan:
        average `ask_yield` across USD-denominated bonds.
        """
        eurobonds = self.get_eurobonds(currency="USD")
        if eurobonds.empty:
            return float("nan")

        spread_col = self._pick_column(
            eurobonds, ["ask_yield", "yield", "ytm", "spread", "value", "rate"]
        )
        if spread_col is None:
            return float("nan")

        values = eurobonds[spread_col].apply(self._to_float).dropna()
        if values.empty:
            return float("nan")

        return float(values.mean())

