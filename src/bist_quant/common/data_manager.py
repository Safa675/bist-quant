from __future__ import annotations

import logging
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


def _newest_mtime(directory: Path, glob_pattern: str) -> float:
    """Return the most recent mtime among files matching *glob_pattern*."""
    newest = 0.0
    for path in directory.glob(glob_pattern):
        try:
            mt = path.stat().st_mtime
            if mt > newest:
                newest = mt
        except OSError:
            continue
    return newest


def build_consolidated_prices_panel(
    cache_dir: Path,
    output_path: Path | None = None,
    *,
    force: bool = False,
    pattern: str = "*_max_1d.parquet",
) -> Path:
    """Build a single consolidated prices parquet from per-ticker cache files.

    The panel is written to *output_path* (default: ``cache_dir/../panels/prices_panel.parquet``).
    If the panel already exists and is newer than every source file, this is a no-op
    unless *force* is True.

    Returns the path to the consolidated panel.
    """
    prices_dir = cache_dir / "prices"
    if output_path is None:
        panels_dir = cache_dir / "panels"
        panels_dir.mkdir(parents=True, exist_ok=True)
        output_path = panels_dir / "prices_panel.parquet"

    # ── Staleness check ──────────────────────────────────────────────────
    if not force and output_path.exists():
        panel_mtime = output_path.stat().st_mtime
        source_mtime = _newest_mtime(prices_dir, pattern)
        if source_mtime > 0 and panel_mtime >= source_mtime:
            panel_size = output_path.stat().st_size
            # Also check that the panel isn't suspiciously small (< 100 KB)
            if panel_size > 100_000:
                logger.info(
                    "  ⚡ Consolidated prices panel is up-to-date (%d KB), skipping rebuild",
                    panel_size // 1024,
                )
                return output_path

    # ── Rebuild ──────────────────────────────────────────────────────────
    t0 = time.time()
    source_files = sorted(prices_dir.glob(pattern))
    if not source_files:
        raise FileNotFoundError(
            f"No price files matching '{pattern}' found in {prices_dir}. "
            "Run the fetch pipeline to populate borsapy_cache."
        )

    frames: list[pd.DataFrame] = []
    skipped = 0
    for path in source_files:
        try:
            df = pd.read_parquet(path)
            if df.empty:
                skipped += 1
                continue
            # Normalize index: strip timezone, rename to Date column
            if isinstance(df.index, pd.DatetimeIndex):
                idx = df.index.tz_localize(None) if df.index.tz else df.index
                df.index = idx.floor("D")
                df = df.reset_index()
                if df.columns[0] != "Date":
                    df = df.rename(columns={df.columns[0]: "Date"})
            # Ensure Ticker column
            if "Ticker" not in df.columns:
                ticker = path.stem.split("_")[0]
                df["Ticker"] = ticker
            frames.append(df)
        except Exception as exc:
            logger.debug("  Skipping %s: %s", path.name, exc)
            skipped += 1

    if not frames:
        raise ValueError("All price files were empty or unreadable.")

    panel = pd.concat(frames, ignore_index=True)

    # Ensure standard column order
    standard_cols = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
    available = [c for c in standard_cols if c in panel.columns]
    extra = [c for c in panel.columns if c not in standard_cols]
    panel = panel[available + extra]

    # Coerce Date
    panel["Date"] = pd.to_datetime(panel["Date"], errors="coerce")
    panel = panel.dropna(subset=["Date"]).sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output_path, index=False)

    elapsed = time.time() - t0
    logger.info(
        "  ✅ Built consolidated prices panel: %d rows × %d tickers in %.1fs (%d KB)",
        len(panel),
        panel["Ticker"].nunique(),
        elapsed,
        output_path.stat().st_size // 1024,
    )
    if skipped:
        logger.info("     (%d files skipped due to errors)", skipped)

    return output_path


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

    def _ensure_consolidated_panel(self) -> Path | None:
        """Build or refresh the consolidated prices panel if per-ticker files exist."""
        from bist_quant.common.data_paths import get_data_paths
        _paths = get_data_paths()
        prices_dir = _paths.borsapy_cache_dir / "prices"
        if not prices_dir.exists() or not any(prices_dir.glob("*_max_1d.parquet")):
            return None
        try:
            return build_consolidated_prices_panel(_paths.borsapy_cache_dir)
        except Exception as exc:
            logger.warning("  ⚠️  Failed to build consolidated panel: %s", exc)
            return None

    def load_all(self, use_cache: bool = True) -> LoadedMarketData:
        if use_cache and self._cache is not None:
            return self._cache

        logger.info("\n" + "=" * 70)
        logger.info("LOADING ALL DATA")
        logger.info("=" * 70)

        start_time = time.time()

        from bist_quant.common.data_paths import get_data_paths
        _paths = get_data_paths()

        # ── Ensure consolidated panel is fresh ───────────────────────────
        panel_path = self._ensure_consolidated_panel()

        # ── Load prices ──────────────────────────────────────────────────
        # If a fresh consolidated panel exists, load it directly for speed.
        # Otherwise fall back to the DataLoader's normal path.
        if panel_path is not None and panel_path.exists() and panel_path.stat().st_size > 100_000:
            logger.info("  ⚡ Loading from consolidated prices panel...")
            prices = pd.read_parquet(panel_path)
            if "Date" in prices.columns:
                prices["Date"] = pd.to_datetime(prices["Date"], errors="coerce").dt.floor("D")
            # Cache inside DataLoader so other methods can reuse it
            if self.loader._prices is None:
                self.loader._prices = prices
            logger.info(
                "  ✅ Loaded %d price records from consolidated panel",
                len(prices),
            )
        else:
            prices = self.loader.load_prices()

        if prices is None or prices.empty:
            raise ValueError(
                "Price data could not be loaded. "
                "Run the fetch pipeline to populate borsapy_cache."
            )

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

        xautry_file = _paths.usdtry_file
        xautry_prices = self.loader.load_xautry_prices(xautry_file)

        xu100_file = _paths.xu100_prices
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
