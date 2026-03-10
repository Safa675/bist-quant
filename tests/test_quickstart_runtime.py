from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from bist_quant.common.data_manager import DataManager
from bist_quant.common.enums import RegimeLabel


class _LoaderWithoutRegimeRequirement:
    def load_prices(self):
        return pd.DataFrame(
            {
                "Date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "symbol": ["AAA", "AAA"],
                "Open": [10.0, 10.5],
                "Close": [10.5, 11.0],
                "Volume": [1000, 1100],
            }
        )

    def build_close_panel(self, prices):
        frame = prices.pivot(index="Date", columns="symbol", values="Close")
        frame.index = pd.to_datetime(frame.index)
        return frame

    def build_open_panel(self, prices):
        frame = prices.pivot(index="Date", columns="symbol", values="Open")
        frame.index = pd.to_datetime(frame.index)
        return frame

    def build_volume_panel(self, prices):
        frame = prices.pivot(index="Date", columns="symbol", values="Volume")
        frame.index = pd.to_datetime(frame.index)
        return frame

    def load_fundamentals(self):
        return pd.DataFrame()

    def load_regime_predictions(self):
        raise AssertionError(
            "regime predictions should not be required when regime filter is disabled"
        )

    def load_regime_allocations(self):
        return {}

    def load_xautry_prices(self, path):
        return pd.Series([1.0, 1.0], index=pd.to_datetime(["2024-01-02", "2024-01-03"]))

    def load_xu100_prices(self, path):
        return pd.Series([100.0, 101.0], index=pd.to_datetime(["2024-01-02", "2024-01-03"]))


def test_data_manager_skips_regime_load_when_regime_filter_disabled(
    monkeypatch, tmp_path: Path
) -> None:
    from bist_quant.common import data_manager as data_manager_module

    usdtry_file = tmp_path / "usdtry.csv"
    xu100_file = tmp_path / "xu100.csv"
    usdtry_file.write_text("date,value\n2024-01-02,1\n", encoding="utf-8")
    xu100_file.write_text("date,value\n2024-01-02,100\n", encoding="utf-8")

    monkeypatch.setattr(
        data_manager_module,
        "get_data_paths",
        lambda: SimpleNamespace(
            borsapy_cache_dir=tmp_path / "missing_cache",
            usdtry_file=usdtry_file,
            xu100_prices=xu100_file,
        ),
        raising=False,
    )

    manager = DataManager(
        data_loader=_LoaderWithoutRegimeRequirement(),
        data_dir=tmp_path,
        base_regime_allocations={RegimeLabel.BULL: 1.0},
    )

    loaded = manager.load_all(use_cache=False, require_regime=False)

    assert loaded.close_df.shape == (2, 1)
    assert loaded.regime_series.empty
