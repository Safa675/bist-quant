from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from bist_quant.common.enums import RegimeLabel

logger = logging.getLogger(__name__)
@dataclass
class LoadedMarketData:
    prices: pd.DataFrame
    close_df: pd.DataFrame
    open_df: pd.DataFrame
    volume_df: pd.DataFrame
    fundamentals: Any
    regime_series: pd.Series
    regime_allocations: dict[RegimeLabel, float]
    xautry_prices: pd.Series
    xu100_prices: pd.Series


class DataManager:
    """Centralized market/fundamental/regime data loader with caching."""

    def __init__(
        self,
        data_loader,
        data_dir: Path,
        base_regime_allocations: dict[RegimeLabel, float],
    ) -> None:
        self.loader = data_loader
        self.data_dir = Path(data_dir)
        self.base_regime_allocations = dict(base_regime_allocations)
        self._cache: LoadedMarketData | None = None

    def clear_cache(self) -> None:
        self._cache = None

    def load_all(self, use_cache: bool = True) -> LoadedMarketData:
        if use_cache and self._cache is not None:
            return self._cache

        logger.info("\n" + "=" * 70)
        logger.info("LOADING ALL DATA")
        logger.info("=" * 70)

        start_time = time.time()

        prices_file = self.data_dir / "bist_prices_full.csv"
        
        # Auto-fetch detection logic
        if not prices_file.exists():
            logger.warning("⚠️  Price data missing. Attempting to automatically fetch via yfinance pipeline...")
            script_path = self.data_dir.parent / "scripts" / "run_fetch_pipeline.sh"
            if script_path.exists():
                try:
                    subprocess.run(["bash", str(script_path), "--source", "yfinance"], check=True)
                    logger.info("✅ Auto-fetch pipeline completed successfully.")
                except subprocess.CalledProcessError as e:
                    logger.error(f"❌ Auto-fetch pipeline failed: {e}")
                    raise ValueError(f"Price data not available and auto-fetch failed: {prices_file}")
            else:
                logger.error(f"❌ Auto-fetch script not found at target: {script_path}")
                raise ValueError(f"Price data not available from {prices_file}")
            
        prices = self.loader.load_prices(prices_file)
        if prices is None or prices.empty:
            raise ValueError(f"Price data could not be loaded from {prices_file}")

        close_df = self.loader.build_close_panel(prices)
        open_df = self.loader.build_open_panel(prices)
        volume_df = self.loader.build_volume_panel(prices)
        if close_df.empty or open_df.empty or volume_df.empty:
            raise ValueError("Failed to build required price panels (close/open/volume)")

        fundamentals = self.loader.load_fundamentals()

        regime_series = self.loader.load_regime_predictions()
        loaded_allocations = self.loader.load_regime_allocations()
        if loaded_allocations:
            regime_allocations = self.base_regime_allocations.copy()
            regime_allocations.update(loaded_allocations)
            logger.info("  ✅ Using regime allocations from regime_labels.json:")
            for regime, allocation in sorted(
                regime_allocations.items(),
                key=lambda item: str(item[0]),
            ):
                logger.info(f"    {regime}: {allocation:.2f}")
        else:
            regime_allocations = self.base_regime_allocations.copy()
            logger.info("  ℹ️  Using fallback regime allocations from portfolio_engine constants.")

        xautry_file = self.data_dir / "xau_try_2013_2026.csv"
        xautry_prices = self.loader.load_xautry_prices(xautry_file)

        xu100_file = self.data_dir / "xu100_prices.csv"
        xu100_prices = self.loader.load_xu100_prices(xu100_file)

        loaded_data = LoadedMarketData(
            prices=prices,
            close_df=close_df,
            open_df=open_df,
            volume_df=volume_df,
            fundamentals=fundamentals,
            regime_series=regime_series,
            regime_allocations=regime_allocations,
            xautry_prices=xautry_prices,
            xu100_prices=xu100_prices,
        )
        self._cache = loaded_data

        elapsed = time.time() - start_time
        logger.info(f"\n✅ Data loading completed in {elapsed:.1f} seconds")
        return loaded_data

    def build_runtime_context(self, loaded_data: LoadedMarketData | None = None) -> dict[str, Any]:
        data = loaded_data or self._cache
        if data is None:
            raise ValueError("No loaded data available. Call load_all() first.")

        return {
            "prices": data.prices,
            "close_df": data.close_df,
            "open_df": data.open_df,
            "volume_df": data.volume_df,
            "fundamentals": data.fundamentals,
            "xu100_prices": data.xu100_prices,
        }
