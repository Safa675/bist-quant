"""Unit tests for PortfolioEngine and related classes."""

from __future__ import annotations

import pandas as pd

from bist_quant import (
    PortfolioEngine,
    PortfolioResult,
    SignalResult,
    get_default_options,
    run_backtest,
)


class TestPortfolioResult:
    """Tests for PortfolioResult dataclass."""

    def test_portfolio_result_creation(self) -> None:
        """Test that PortfolioResult can be instantiated."""
        result = PortfolioResult(
            returns=pd.Series([0.01, -0.02, 0.03]),
            positions=pd.DataFrame(),
            turnover=pd.Series([0.1, 0.2, 0.3]),
            transaction_costs=pd.Series([0.001, 0.002, 0.003]),
            metrics={"sharpe": 1.5, "total_return": 0.15},
        )
        assert result.metrics["sharpe"] == 1.5

    def test_portfolio_result_empty_metrics(self) -> None:
        """Test PortfolioResult with empty metrics."""
        result = PortfolioResult(
            returns=pd.Series(dtype=float),
            positions=pd.DataFrame(),
            turnover=pd.Series(dtype=float),
            transaction_costs=pd.Series(dtype=float),
            metrics={},
        )
        assert result.metrics == {}


class TestSignalResult:
    """Tests for SignalResult dataclass."""

    def test_signal_result_creation(self) -> None:
        """Test that SignalResult can be instantiated."""
        result = SignalResult(
            signals=pd.DataFrame({"THYAO": [1, 0, -1], "GARAN": [0, 1, 0]}),
            metadata={"signal_name": "momentum"},
        )
        assert result.metadata["signal_name"] == "momentum"


class TestPortfolioEngine:
    """Tests for PortfolioEngine class."""

    def test_engine_initialization(self) -> None:
        """Test PortfolioEngine can be initialized."""
        engine = PortfolioEngine()
        assert engine is not None

    def test_engine_with_options(self, sample_config: dict[str, object]) -> None:
        """Test PortfolioEngine accepts option overrides."""
        engine = PortfolioEngine(options=sample_config)
        assert engine.options["signal"] == "momentum"

    def test_get_default_options_returns_copy(self) -> None:
        """Test default options helper returns a mutable copy."""
        first = get_default_options()
        second = get_default_options()
        first["top_n"] = -1
        assert second["top_n"] != -1

    def test_run_backtest_convenience_function(self, monkeypatch) -> None:
        """Test run_backtest convenience helper delegates to PortfolioEngine."""

        def fake_run_backtest(self, signals, start_date, end_date):
            return PortfolioResult(
                returns=pd.Series([0.01, 0.02]),
                positions=pd.DataFrame(index=pd.date_range(start_date, periods=2, freq="D")),
                turnover=pd.Series([0.1, 0.2]),
                transaction_costs=pd.Series([0.001, 0.002]),
                metrics={"total_return": 0.03},
            )

        monkeypatch.setattr(PortfolioEngine, "run_backtest", fake_run_backtest)

        signals = pd.DataFrame(
            {"THYAO": [1, 0]},
            index=pd.date_range("2023-01-01", periods=2, freq="D"),
        )
        result = run_backtest(signals=signals, start_date="2023-01-01", end_date="2023-01-02")

        assert isinstance(result, PortfolioResult)
        assert result.metrics["total_return"] == 0.03
