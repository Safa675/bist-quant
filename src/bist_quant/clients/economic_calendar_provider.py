"""
Economic calendar provider using borsapy EconomicCalendar.

Coverage focus: TR, US, EU, DE, GB, JP, CN.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from typing import Any, Callable

import pandas as pd

logger = logging.getLogger(__name__)


class EconomicCalendarProvider:
    """Resilient accessor for upcoming macro-economic events."""

    SUPPORTED_COUNTRIES = ("TR", "US", "EU", "DE", "GB", "JP", "CN")
    _CANONICAL_COLUMNS = (
        "Date",
        "Time",
        "Country",
        "Importance",
        "Event",
        "Forecast",
        "Previous",
    )

    def __init__(
        self,
        borsapy_module: Any | None = None,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        self._bp = borsapy_module
        self._import_attempted = borsapy_module is not None
        self._now = now_fn or datetime.now

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
            logger.info("  EconomicCalendarProvider: borsapy unavailable: %s", exc)
            self._bp = None
        return self._bp

    @staticmethod
    def _period_from_days(days_ahead: int) -> str:
        if days_ahead <= 1:
            return "1d"
        if days_ahead <= 7:
            return "1w"
        if days_ahead <= 14:
            return "2w"
        return "1mo"

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

    @classmethod
    def _empty_events(cls) -> pd.DataFrame:
        return pd.DataFrame(columns=list(cls._CANONICAL_COLUMNS))

    @staticmethod
    def _normalize_importance(importance: str | None) -> str | None:
        if importance is None:
            return None
        value = str(importance).strip().lower()
        if not value:
            return None
        mapping = {
            "medium": "mid",
            "moderate": "mid",
            "mid": "mid",
            "high": "high",
            "low": "low",
        }
        return mapping.get(value, value)

    @staticmethod
    def _normalize_country(country: str | list[str] | None) -> str | list[str] | None:
        if country is None:
            return None

        if isinstance(country, str):
            tokens = [item.strip().upper() for item in country.split(",") if item.strip()]
        elif isinstance(country, (list, tuple, set)):
            tokens = [str(item).strip().upper() for item in country if str(item).strip()]
        else:
            return None

        deduped: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            if token in seen:
                continue
            seen.add(token)
            deduped.append(token)

        if not deduped:
            return None
        if len(deduped) == 1:
            return deduped[0]
        return deduped

    @classmethod
    def _normalize_events(cls, payload: Any) -> pd.DataFrame:
        frame = cls._as_frame(payload)
        if frame.empty:
            return cls._empty_events()

        work = frame.copy()
        aliases = {
            "Date": ["date", "datetime", "event_date", "timestamp", "tarih"],
            "Time": ["time", "saat", "event_time"],
            "Country": ["country", "country_code", "nation"],
            "Importance": ["importance", "impact", "priority", "volatility"],
            "Event": ["event", "title", "name", "description"],
            "Forecast": ["forecast", "expected", "consensus"],
            "Previous": ["previous", "prior"],
        }
        rename_map: dict[str, str] = {}
        for canonical, candidates in aliases.items():
            column = cls._pick_column(work, candidates)
            if column is not None:
                rename_map[column] = canonical
        if rename_map:
            work = work.rename(columns=rename_map)

        for column in cls._CANONICAL_COLUMNS:
            if column not in work.columns:
                work[column] = pd.NA

        parsed_date = pd.to_datetime(work["Date"], errors="coerce")
        if parsed_date.notna().any():
            derived_time = parsed_date.dt.strftime("%H:%M")
            time_text = work["Time"].astype("string").fillna("").str.strip()
            missing_time = time_text.eq("") | time_text.str.lower().isin({"nan", "none", "nat"})
            has_clock = derived_time != "00:00"
            time_text = time_text.mask(missing_time & has_clock, derived_time)
            time_text = time_text.replace({"": pd.NA})
            work["Time"] = time_text
            work["Date"] = parsed_date.dt.date

        work["Country"] = (
            work["Country"]
            .astype("string")
            .fillna("")
            .str.upper()
            .str.strip()
            .replace({"": pd.NA})
        )

        importance = work["Importance"].astype("string").fillna("").str.lower().str.strip()
        importance = importance.replace(
            {
                "medium": "mid",
                "moderate": "mid",
                "med": "mid",
            }
        )
        work["Importance"] = importance.replace({"": pd.NA})

        for column in ("Event", "Forecast", "Previous"):
            work[column] = (
                work[column]
                .astype("string")
                .fillna("")
                .str.strip()
                .replace({"": pd.NA})
            )

        normalized = work.loc[:, list(cls._CANONICAL_COLUMNS)].copy()
        normalized = normalized[~(normalized["Date"].isna() & normalized["Event"].isna())]
        return normalized.reset_index(drop=True)

    @classmethod
    def _event_datetimes(cls, events: pd.DataFrame) -> pd.Series:
        if events.empty or "Date" not in events.columns:
            return pd.Series(dtype="datetime64[ns]")

        date_text = events["Date"].astype("string").fillna("").str.strip()
        time_text = (
            events["Time"].astype("string").fillna("").str.strip()
            if "Time" in events.columns
            else pd.Series("", index=events.index, dtype="string")
        )
        missing_time = time_text.eq("")

        combined = pd.to_datetime((date_text + " " + time_text).str.strip(), errors="coerce")
        fallback = pd.to_datetime(events["Date"], errors="coerce")
        timestamps = combined.fillna(fallback)
        timestamps = timestamps.where(~missing_time, fallback + pd.Timedelta(hours=23, minutes=59))

        try:
            tz = timestamps.dt.tz
            if tz is not None:
                timestamps = timestamps.dt.tz_convert(None)
        except Exception:
            pass

        return timestamps

    def get_events(
        self,
        period: str = "1w",
        country: str | list[str] | None = None,
        importance: str | None = None,
    ) -> pd.DataFrame:
        """
        Upcoming economic events.

        Returns canonical columns:
        Date, Time, Country, Importance, Event, Forecast, Previous.
        """
        bp = self._get_bp()
        if bp is None:
            return self._empty_events()

        normalized_country = self._normalize_country(country)
        normalized_importance = self._normalize_importance(importance)

        payload: Any = None
        calendar_cls = getattr(bp, "EconomicCalendar", None)
        if calendar_cls is not None:
            try:
                calendar = calendar_cls()
                payload = calendar.events(
                    period=period,
                    country=normalized_country,
                    importance=normalized_importance,
                )
            except Exception as exc:
                logger.info("  EconomicCalendar.events failed: %s", exc)

        if payload is None:
            calendar_fn = getattr(bp, "economic_calendar", None)
            if callable(calendar_fn):
                try:
                    payload = calendar_fn(
                        period=period,
                        country=normalized_country,
                        importance=normalized_importance,
                    )
                except Exception as exc:
                    logger.info("  bp.economic_calendar failed: %s", exc)

        events = self._normalize_events(payload)
        if events.empty:
            return events

        if normalized_country is not None:
            wanted = {normalized_country} if isinstance(normalized_country, str) else set(normalized_country)
            events = events[events["Country"].astype("string").fillna("").str.upper().isin(wanted)]

        if normalized_importance is not None:
            events = events[
                events["Importance"]
                .astype("string")
                .fillna("")
                .str.lower()
                .eq(normalized_importance)
            ]

        event_dt = self._event_datetimes(events)
        if not event_dt.empty:
            events = events.assign(_event_dt=event_dt)
            events = events.sort_values("_event_dt").drop(columns="_event_dt")
        return events.reset_index(drop=True)

    def get_high_impact_events(self, days_ahead: int = 7) -> pd.DataFrame:
        """High-importance events in the next N days."""
        events = self.get_events(
            period=self._period_from_days(days_ahead),
            country=list(self.SUPPORTED_COUNTRIES),
            importance="high",
        )
        if events.empty:
            return events

        event_dt = self._event_datetimes(events)
        if event_dt.empty:
            return events.reset_index(drop=True)

        now = pd.Timestamp(self._now())
        end = now + timedelta(days=max(1, int(days_ahead)))
        filtered = events[(event_dt >= now) & (event_dt <= end)]
        return filtered.reset_index(drop=True)

    def is_event_window(self, country: str = "TR", hours_before: int = 2) -> bool:
        """True if a high-impact event is imminent."""
        horizon_hours = max(0, int(hours_before))
        days_ahead = max(1, math.ceil(horizon_hours / 24) + 1)
        events = self.get_high_impact_events(days_ahead=days_ahead)
        if events.empty:
            return False

        normalized_country = self._normalize_country(country)
        if normalized_country is not None:
            wanted = (
                {normalized_country}
                if isinstance(normalized_country, str)
                else set(normalized_country)
            )
            events = events[events["Country"].astype("string").fillna("").str.upper().isin(wanted)]
        if events.empty:
            return False

        now = pd.Timestamp(self._now())
        end = now + timedelta(hours=horizon_hours)
        event_dt = self._event_datetimes(events)
        if event_dt.empty:
            event_days = pd.to_datetime(events["Date"], errors="coerce").dt.normalize().dropna()
            return bool((event_days == now.normalize()).any())

        mask = (event_dt >= now) & (event_dt <= end)
        return bool(mask.any())

    def get_tr_events_this_week(self) -> pd.DataFrame:
        """Shortcut for Turkey high-impact events this week."""
        return self.get_events(period="1w", country="TR", importance="high")
