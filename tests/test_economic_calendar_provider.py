"""Unit tests for economic-calendar integrations."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from bist_quant.common.economic_calendar_provider import EconomicCalendarProvider
from bist_quant.common.macro_adapter import MacroAdapter
from bist_quant.regime.macro_features import MacroConfig, MacroFeatures


class _DummyEconomicCalendar:
    def events(
        self,
        period: str = "1w",
        start: datetime | str | None = None,
        end: datetime | str | None = None,
        country: str | list[str] | None = None,
        importance: str | None = None,
    ) -> pd.DataFrame:
        del period, start, end, country, importance
        return pd.DataFrame(
            [
                {
                    "date": "2026-02-21",
                    "time": "10:00",
                    "country": "TR",
                    "importance": "high",
                    "event": "CBRT Rate Decision",
                    "forecast": "45.0",
                    "previous": "42.5",
                },
                {
                    "date": "2026-02-22",
                    "time": "14:30",
                    "country": "US",
                    "importance": "medium",
                    "event": "US CPI",
                    "forecast": "3.0",
                    "previous": "2.9",
                },
                {
                    "date": "2026-02-23",
                    "time": "15:00",
                    "country": "US",
                    "importance": "high",
                    "event": "FOMC Minutes",
                    "forecast": pd.NA,
                    "previous": pd.NA,
                },
                {
                    "date": "2026-03-01",
                    "time": "09:00",
                    "country": "JP",
                    "importance": "high",
                    "event": "Japan GDP",
                    "forecast": "0.9",
                    "previous": "0.7",
                },
            ]
        )


class _DummyBorsapy:
    EconomicCalendar = _DummyEconomicCalendar


def test_provider_get_events_normalizes_columns_and_filters() -> None:
    provider = EconomicCalendarProvider(borsapy_module=_DummyBorsapy())
    events = provider.get_events(period="1w", country=["TR", "US"], importance="high")

    assert list(events.columns) == [
        "Date",
        "Time",
        "Country",
        "Importance",
        "Event",
        "Forecast",
        "Previous",
    ]
    assert len(events) == 2
    assert set(events["Country"]) == {"TR", "US"}
    assert set(events["Importance"]) == {"high"}


def test_provider_high_impact_window_filtering() -> None:
    provider = EconomicCalendarProvider(
        borsapy_module=_DummyBorsapy(),
        now_fn=lambda: datetime(2026, 2, 21, 9, 0, 0),
    )
    events = provider.get_high_impact_events(days_ahead=2)

    assert len(events) == 1
    assert set(events["Event"]) == {"CBRT Rate Decision"}


def test_provider_is_event_window_detects_imminent_event() -> None:
    class _WindowCalendar:
        def events(self, *args, **kwargs) -> pd.DataFrame:
            del args, kwargs
            return pd.DataFrame(
                [
                    {
                        "date": "2026-02-21",
                        "time": "10:30",
                        "country": "TR",
                        "importance": "high",
                        "event": "TR Event",
                    },
                    {
                        "date": "2026-02-21",
                        "time": "15:00",
                        "country": "US",
                        "importance": "high",
                        "event": "US Event",
                    },
                ]
            )

    class _WindowBorsapy:
        EconomicCalendar = _WindowCalendar

    provider = EconomicCalendarProvider(
        borsapy_module=_WindowBorsapy(),
        now_fn=lambda: datetime(2026, 2, 21, 9, 0, 0),
    )

    assert provider.is_event_window(country="TR", hours_before=2) is True
    assert provider.is_event_window(country="TR", hours_before=1) is False


def test_provider_tr_events_this_week_shortcut() -> None:
    provider = EconomicCalendarProvider(borsapy_module=_DummyBorsapy())
    events = provider.get_tr_events_this_week()

    assert not events.empty
    assert set(events["Country"]) == {"TR"}
    assert set(events["Importance"]) == {"high"}


def test_macro_adapter_uses_legacy_client_when_provider_empty() -> None:
    adapter = MacroAdapter(loader=object(), macro_events_path=Path("/tmp/missing_macro_events.py"))

    class _EmptyProvider:
        @staticmethod
        def get_events(*args, **kwargs) -> pd.DataFrame:
            del args, kwargs
            return pd.DataFrame()

    class _LegacyClient:
        @staticmethod
        def get_economic_calendar(days_ahead: int = 7, countries: list[str] | None = None) -> pd.DataFrame:
            del days_ahead, countries
            return pd.DataFrame(
                [
                    {
                        "Date": "2026-02-21",
                        "Country": "TR",
                        "Event": "Legacy Event",
                    }
                ]
            )

    adapter._economic_calendar_provider = _EmptyProvider()
    adapter._client = _LegacyClient()

    events = adapter.get_economic_calendar(days_ahead=7, countries=["TR"])

    assert len(events) == 1
    assert events.iloc[0]["Event"] == "Legacy Event"


def test_macro_features_compute_event_risk_flag_marks_pre_event_days(tmp_path) -> None:
    macro = MacroFeatures(data_dir=tmp_path)
    index = pd.to_datetime(["2026-02-20", "2026-02-21", "2026-02-22", "2026-02-23"])
    macro.tcmb = pd.DataFrame({"dummy": [1, 2, 3, 4]}, index=index)
    macro.usdtry = pd.Series([1.0, 1.1, 1.2, 1.3], index=index)
    macro._loaded = True

    class _CalendarProvider:
        @staticmethod
        def get_high_impact_events(days_ahead: int = 7) -> pd.DataFrame:
            del days_ahead
            return pd.DataFrame(
                [
                    {
                        "Date": "2026-02-22",
                        "Time": "10:00",
                        "Country": "TR",
                        "Importance": "high",
                        "Event": "TR CPI",
                    },
                    {
                        "Date": "2026-02-23",
                        "Time": "15:00",
                        "Country": "US",
                        "Importance": "high",
                        "Event": "US PMI",
                    },
                    {
                        "Date": "2026-02-23",
                        "Time": "11:00",
                        "Country": "JP",
                        "Importance": "high",
                        "Event": "JP Event",
                    },
                ]
            )

    macro._economic_calendar_provider = _CalendarProvider()
    flag = macro.compute_event_risk_flag(days_ahead=7, hours_before=24, countries=("TR", "US"))

    assert bool(flag.loc[pd.Timestamp("2026-02-20")]) is False
    assert bool(flag.loc[pd.Timestamp("2026-02-21")]) is True
    assert bool(flag.loc[pd.Timestamp("2026-02-22")]) is True
    assert bool(flag.loc[pd.Timestamp("2026-02-23")]) is True


def test_macro_features_risk_index_can_use_futures_basis_factor(tmp_path) -> None:
    config = MacroConfig(
        risk_index_lookback=4,
        risk_index_weights={"cds": 0.0, "vix": 0.0, "usdtry": 0.0},
        tcmb_policy_rate_weight=0.0,
        futures_basis_weight=1.0,
    )
    macro = MacroFeatures(data_dir=tmp_path, config=config)
    index = pd.date_range("2026-02-16", periods=7, freq="D")
    macro.tcmb = pd.DataFrame(
        {
            "futures_basis_pct": [1.1, 0.9, 0.6, 0.2, -0.1, -0.4, -0.8],
        },
        index=index,
    )
    macro.usdtry = pd.Series([35.0, 35.1, 35.2, 35.0, 35.3, 35.4, 35.6], index=index)
    macro._loaded = True

    futures_risk = macro.compute_futures_basis_risk().dropna()
    risk_index = macro.compute_risk_index().dropna()

    assert not futures_risk.empty
    assert not risk_index.empty
    assert risk_index.index.equals(futures_risk.index)
    assert risk_index.iloc[-1] == pytest.approx(futures_risk.iloc[-1])


def test_macro_features_risk_index_can_use_intraday_fx_factor(tmp_path) -> None:
    config = MacroConfig(
        risk_index_lookback=4,
        risk_index_weights={"cds": 0.0, "vix": 0.0, "usdtry": 0.0},
        tcmb_policy_rate_weight=0.0,
        futures_basis_weight=0.0,
        use_intraday_fx=True,
        intraday_fx_weight=1.0,
        intraday_fx_currency="USD",
        intraday_fx_interval="1h",
        intraday_fx_period="5d",
        intraday_fx_momentum_bars=1,
        intraday_fx_lookback=3,
    )
    macro = MacroFeatures(data_dir=tmp_path, config=config)
    index = pd.date_range("2026-02-16", periods=8, freq="D")
    macro.tcmb = pd.DataFrame({"dummy": range(8)}, index=index)
    macro.usdtry = pd.Series([35.0, 35.1, 35.2, 35.0, 35.3, 35.4, 35.5, 35.6], index=index)
    macro._loaded = True

    intraday_index = pd.date_range("2026-02-19 09:00:00", periods=72, freq="h")
    intraday_data = pd.DataFrame(
        {
            "timestamp": intraday_index,
            "close": [34.0 + i * 0.01 for i in range(len(intraday_index))],
        }
    )

    class _DummyFXEnhanced:
        @staticmethod
        def get_intraday(currency: str = "USD", interval: str = "1h", period: str = "5d") -> pd.DataFrame:
            del currency, interval, period
            return intraday_data

    macro._fx_enhanced_provider = _DummyFXEnhanced()

    intraday_mom = macro.compute_intraday_fx_momentum().dropna()
    intraday_risk = macro._rolling_percentile(intraday_mom, config.intraday_fx_lookback).dropna()
    risk_index = macro.compute_risk_index().dropna()

    assert not intraday_mom.empty
    assert not intraday_risk.empty
    assert not risk_index.empty
    assert risk_index.iloc[-1] == pytest.approx(intraday_risk.iloc[-1])


def test_macro_features_compute_eurobond_stress_refreshes_snapshot(tmp_path) -> None:
    config = MacroConfig(cds_lookback=4, cds_stress_percentile=0.9)
    macro = MacroFeatures(data_dir=tmp_path, config=config)

    today = pd.Timestamp(datetime.now().date())
    index = pd.date_range(end=today - pd.Timedelta(days=1), periods=5, freq="D")
    macro.tcmb = pd.DataFrame(
        {
            "eurobond_spread": [5.0, 5.1, 5.2, 5.3, 5.2],
        },
        index=index,
    )
    macro.usdtry = pd.Series([35.0, 35.1, 35.2, 35.3, 35.4], index=index)
    macro._loaded = True

    class _FixedIncome:
        @staticmethod
        def get_spread_index() -> float:
            return 8.0

    macro._fixed_income_provider = _FixedIncome()  # type: ignore[assignment]

    stress = macro.compute_eurobond_stress()

    assert not stress.empty
    assert stress.dtype == bool
    assert bool(stress.loc[today]) is True
    assert macro.tcmb.loc[today, "eurobond_spread"] == pytest.approx(8.0)
