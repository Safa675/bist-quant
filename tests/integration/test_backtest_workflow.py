"""Integration tests for complete backtest workflow."""

from __future__ import annotations

import pandas as pd
import pytest

from bist_quant import (
    Backtester,
    ConfigManager,
    DataLoader,
    PortfolioEngine,
    PortfolioResult,
    RiskManager,
    run_backtest,
)


class TestBacktestWorkflow:
    """Integration tests for end-to-end backtest workflow."""

    @pytest.mark.integration
    def test_component_wiring(self, sample_config: dict[str, object]) -> None:
        """Test core components can be wired from defaults."""
        engine = PortfolioEngine(options=sample_config)
        manager = ConfigManager.from_default_paths()

        assert isinstance(engine.data_loader, DataLoader)
        assert isinstance(engine.backtester, Backtester)
        assert isinstance(engine.risk_manager, RiskManager)
        assert len(manager.load_signal_configs()) > 0

    @pytest.mark.integration
    def test_convenience_run_backtest(self, monkeypatch) -> None:
        """Test top-level run_backtest helper delegates to PortfolioEngine."""

        def fake_run_backtest(self, signals, start_date, end_date):
            return PortfolioResult(
                returns=pd.Series([0.01, -0.01]),
                positions=pd.DataFrame(index=pd.date_range(start_date, periods=2, freq="D")),
                turnover=pd.Series([0.1, 0.0]),
                transaction_costs=pd.Series([0.001, 0.0]),
                metrics={"sharpe": 1.2},
            )

        monkeypatch.setattr(PortfolioEngine, "run_backtest", fake_run_backtest)

        signals = pd.DataFrame(
            {"THYAO": [1.0, 0.0]},
            index=pd.date_range("2024-01-01", periods=2, freq="D"),
        )
        result = run_backtest(signals=signals, start_date="2024-01-01", end_date="2024-01-02")

        assert isinstance(result, PortfolioResult)
        assert result.metrics["sharpe"] == 1.2
