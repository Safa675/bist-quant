"""Golden tests for BUILDERS value/investment with empty fundamentals."""

from __future__ import annotations

from bist_quant.signals.investment_signals import build_investment_signals
from bist_quant.signals.value_signals import build_value_signals

from tests.signals.conftest import panel_checksum


class _NullFundamentalsLoader:
    def load_fundamentals_parquet(self):
        return None


class TestFundamentalSignalsGolden:
    def test_value_empty_fundamentals_shape_and_checksum(self, signal_close_df, signal_dates) -> None:
        loader = _NullFundamentalsLoader()
        result = build_value_signals({}, signal_close_df, signal_dates, data_loader=loader)
        assert result.shape[0] == len(signal_dates)
        assert list(result.columns) == list(signal_close_df.columns)
        assert result.isna().all().all()
        assert panel_checksum(result) == "7720d1a39cf51e38182ba5389dfe9f4d"

    def test_investment_empty_fundamentals_shape_and_checksum(self, signal_close_df, signal_dates) -> None:
        loader = _NullFundamentalsLoader()
        result = build_investment_signals({}, signal_close_df, signal_dates, data_loader=loader)
        assert result.shape[0] == len(signal_dates)
        assert list(result.columns) == list(signal_close_df.columns)
        assert result.isna().all().all()
        assert panel_checksum(result) == "7720d1a39cf51e38182ba5389dfe9f4d"
