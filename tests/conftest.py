"""Shared pytest fixtures for bist_quant tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_prices_df() -> pd.DataFrame:
    """Generate sample OHLCV price data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=252, freq="B")
    tickers = ["THYAO", "GARAN", "AKBNK", "EREGL", "BIMAS"]

    data: list[dict[str, object]] = []
    for idx, ticker in enumerate(tickers):
        rng = np.random.default_rng(seed=2024 + idx)
        base_price = float(rng.uniform(10.0, 100.0))
        returns = rng.normal(0.001, 0.02, len(dates))
        prices = base_price * np.cumprod(1 + returns)

        for i, date in enumerate(dates):
            data.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": float(prices[i] * rng.uniform(0.99, 1.01)),
                    "high": float(prices[i] * rng.uniform(1.00, 1.03)),
                    "low": float(prices[i] * rng.uniform(0.97, 1.00)),
                    "close": float(prices[i]),
                    "volume": int(rng.integers(100_000, 10_000_000)),
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def sample_fundamentals_df() -> pd.DataFrame:
    """Generate sample fundamental data for testing."""
    tickers = ["THYAO", "GARAN", "AKBNK", "EREGL", "BIMAS"]

    data: list[dict[str, object]] = []
    for idx, ticker in enumerate(tickers):
        rng = np.random.default_rng(seed=3000 + idx)
        data.append(
            {
                "ticker": ticker,
                "market_cap": float(rng.uniform(1e9, 1e11)),
                "pe_ratio": float(rng.uniform(5.0, 30.0)),
                "pb_ratio": float(rng.uniform(0.5, 5.0)),
                "roe": float(rng.uniform(0.05, 0.30)),
                "debt_to_equity": float(rng.uniform(0.1, 2.0)),
                "dividend_yield": float(rng.uniform(0.0, 0.10)),
            }
        )

    return pd.DataFrame(data)


@pytest.fixture
def sample_config() -> dict[str, object]:
    """Sample strategy configuration for testing."""
    return {
        "signal": "momentum",
        "lookback_period": 21,
        "holding_period": 5,
        "top_n": 10,
        "rebalance_frequency": "weekly",
    }


@pytest.fixture
def temp_data_dir(tmp_path: pytest.TempPathFactory) -> object:
    """Create a temporary data directory with placeholder parquet files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    (data_dir / "prices.parquet").touch()
    (data_dir / "fundamentals.parquet").touch()

    return data_dir
