"""Unit tests for DataLoader."""

from __future__ import annotations

import pandas as pd
import pytest

from bist_quant import DataLoader


class TestDataLoader:
    """Tests for DataLoader class."""

    def test_data_loader_initialization(self) -> None:
        """Test DataLoader can be initialized."""
        loader = DataLoader()
        assert loader is not None

    def test_data_loader_with_custom_path(self, temp_data_dir) -> None:
        """Test DataLoader with custom data path."""
        loader = DataLoader(data_dir=temp_data_dir, data_source_priority="local")
        assert loader.data_dir == temp_data_dir

    def test_load_prices_from_csv(self, temp_data_dir) -> None:
        """Test load_prices reads a canonical CSV input file."""
        prices_path = temp_data_dir / "bist_prices_full.csv"
        pd.DataFrame(
            {
                "Date": ["2023-01-02", "2023-01-03"],
                "Ticker": ["THYAO", "GARAN"],
                "Open": [10.0, 20.0],
                "High": [11.0, 21.0],
                "Low": [9.5, 19.5],
                "Close": [10.8, 20.5],
                "Volume": [1_000_000, 2_000_000],
            }
        ).to_csv(prices_path, index=False)

        loader = DataLoader(data_dir=temp_data_dir, data_source_priority="local")
        result = loader.load_prices()

        assert not result.empty
        assert set(["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]).issubset(
            result.columns
        )

    def test_missing_prices_file_raises(self, temp_data_dir) -> None:
        """Test load_prices raises for missing canonical files."""
        loader = DataLoader(data_dir=temp_data_dir, data_source_priority="local")
        with pytest.raises(FileNotFoundError):
            loader.load_prices()

    def test_economic_calendar_property(self) -> None:
        """Test DataLoader exposes economic calendar provider facade."""
        loader = DataLoader()
        assert loader.economic_calendar is loader.macro_adapter.economic_calendar

    def test_derivatives_property_is_lazy_singleton(self, monkeypatch) -> None:
        """Test DataLoader exposes lazy derivatives provider facade."""
        from bist_quant.common import data_loader as data_loader_module

        class DummyDerivativesProvider:
            pass

        monkeypatch.setattr(data_loader_module, "DerivativesProvider", DummyDerivativesProvider)

        loader = DataLoader(data_source_priority="local")
        first = loader.derivatives
        second = loader.derivatives

        assert isinstance(first, DummyDerivativesProvider)
        assert first is second

    def test_fx_enhanced_property_is_lazy_singleton(self, monkeypatch) -> None:
        """Test DataLoader exposes lazy enhanced FX provider facade."""
        from bist_quant.common import data_loader as data_loader_module

        class DummyFXEnhancedProvider:
            pass

        monkeypatch.setattr(data_loader_module, "FXEnhancedProvider", DummyFXEnhancedProvider)

        loader = DataLoader(data_source_priority="local")
        first = loader.fx_enhanced
        second = loader.fx_enhanced

        assert isinstance(first, DummyFXEnhancedProvider)
        assert first is second

    def test_technical_scan_delegates_to_engine(self, monkeypatch) -> None:
        """Test DataLoader.technical_scan forwards args to TechnicalScannerEngine."""
        from bist_quant.engines import technical_scanner as technical_scanner_module

        captured: dict[str, object] = {}

        class DummyScanner:
            def scan(self, universe, condition, interval):  # noqa: ANN001
                captured["single"] = {
                    "universe": universe,
                    "condition": condition,
                    "interval": interval,
                }
                return pd.DataFrame({"symbol": ["THYAO"]})

            def scan_multi(self, universe, conditions, interval):  # noqa: ANN001
                captured["multi"] = {
                    "universe": universe,
                    "conditions": conditions,
                    "interval": interval,
                }
                return pd.DataFrame({"symbol": ["THYAO"]})

        monkeypatch.setattr(technical_scanner_module, "TechnicalScannerEngine", DummyScanner)

        loader = DataLoader(data_source_priority="local")
        single = loader.technical_scan(
            condition="rsi < 30",
            universe="XU100",
            interval="1d",
        )
        multi = loader.technical_scan(
            universe="XU100",
            conditions=["rsi < 30", "macd crosses_above signal"],
            interval="1d",
        )

        assert list(single["symbol"]) == ["THYAO"]
        assert list(multi["symbol"]) == ["THYAO"]
        assert captured["single"] == {
            "universe": "XU100",
            "condition": "rsi < 30",
            "interval": "1d",
        }
        assert captured["multi"] == {
            "universe": "XU100",
            "conditions": ["rsi < 30", "macd crosses_above signal"],
            "interval": "1d",
        }
