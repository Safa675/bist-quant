"""On-disk cache for five-factor axis-construction panels.

Extracted from ``five_factor_rotation_signals.py`` so the cache load/save/
staleness logic is testable and reusable independently of the main rotation
pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

# ---------------------------------------------------------------------------
# Constants (kept here so the cache module is self-contained)
# ---------------------------------------------------------------------------

AXIS_CACHE_VERSION = 2
AXIS_CACHE_FILENAME = f"multi_factor_axis_construction_v{AXIS_CACHE_VERSION}.parquet"

AXIS_PANEL_NAMES = (
    # Original panels
    "size_small_signal",
    "value_level_signal",
    "value_growth_signal",
    "profit_margin_level",
    "profit_margin_growth",
    "investment_reinvestment_signal",
    "realized_volatility",
    "market_beta",
    # New panels
    "quality_roe",
    "quality_roa",
    "quality_accruals",
    "quality_piotroski",
    "liquidity_amihud",
    "liquidity_turnover",
    "liquidity_spread_proxy",
    # Trading intensity (separate from liquidity)
    "trading_intensity_relative_volume",
    "trading_intensity_volume_trend",
    "trading_intensity_turnover_velocity",
    "sentiment_52w_high_pct",
    "sentiment_price_acceleration",
    "sentiment_reversal",
    "fundmom_margin_change",
    "fundmom_sales_accel",
    "carry_dividend_yield",
    "carry_shareholder_yield",
    "defensive_earnings_stability",
    "defensive_beta_to_market",
)

# Optional panels that may legitimately be empty in some environments.
OPTIONAL_EMPTY_CACHE_PANELS = {
    "liquidity_spread_proxy",   # Requires intraday/spread-like inputs we may not have.
    "carry_shareholder_yield",  # Often unavailable in Turkey; carry uses dividend yield.
}


# ---------------------------------------------------------------------------
# Cache functions
# ---------------------------------------------------------------------------

def resolve_axis_cache_path(data_loader, cache_path: Path | str | None = None) -> Path:
    """Resolve on-disk path for axis construction cache parquet."""
    if cache_path is not None:
        return Path(cache_path)
    if data_loader is not None and hasattr(data_loader, "data_dir"):
        return Path(data_loader.data_dir) / AXIS_CACHE_FILENAME
    return Path(__file__).resolve().parents[3] / "data" / AXIS_CACHE_FILENAME


def load_axis_construction_cache(
    cache_path: Path,
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
) -> Dict[str, pd.DataFrame]:
    """Load cached axis construction panels from parquet."""
    if not cache_path.exists():
        return {}

    try:
        cached = pd.read_parquet(cache_path)
    except Exception as exc:
        print(f"  ⚠️  Failed to read axis cache {cache_path}: {exc}")
        return {}

    if cached.empty:
        return {}

    if not isinstance(cached.columns, pd.MultiIndex):
        try:
            cached.columns = pd.MultiIndex.from_tuples(cached.columns)
        except Exception:
            print(f"  ⚠️  Axis cache columns are not multi-indexed: {cache_path}")
            return {}

    lvl0 = cached.columns.get_level_values(0)
    loaded: Dict[str, pd.DataFrame] = {}
    for panel_name in AXIS_PANEL_NAMES:
        if panel_name not in lvl0:
            continue
        panel = cached[panel_name]
        panel.index = pd.to_datetime(panel.index)
        panel.columns = pd.Index(panel.columns).astype(str)
        panel = panel.reindex(index=dates, columns=tickers)
        loaded[panel_name] = panel

    return loaded


def save_axis_construction_cache(
    cache_path: Path,
    panels: Dict[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
) -> None:
    """Persist axis construction panels to parquet."""
    to_save: Dict[str, pd.DataFrame] = {}
    for panel_name in AXIS_PANEL_NAMES:
        panel = panels.get(panel_name)
        if panel is None or panel.empty:
            continue
        to_save[panel_name] = panel.reindex(index=dates, columns=tickers).astype(float)

    if not to_save:
        return

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    wide = pd.concat(to_save, axis=1)
    wide.to_parquet(cache_path)
    print(f"  💾 Saved axis construction cache: {cache_path}")


def cache_panel_stale_reason(panel_name: str, panel: pd.DataFrame | None) -> str | None:
    """Return reason string when cached panel looks stale/corrupt, else None."""
    if panel is None:
        return "missing"
    if panel.empty:
        return "empty_shape"

    # Some panels are optional and may be all-NaN by design.
    if panel_name in OPTIONAL_EMPTY_CACHE_PANELS:
        return None

    non_na_total = int(panel.notna().sum().sum())
    if non_na_total == 0:
        return "all_nan"

    # If historical values exist but latest row is fully empty, cache is stale
    # relative to current date/ticker universe.
    latest_non_na = int(panel.iloc[-1].notna().sum()) if len(panel.index) > 0 else 0
    if latest_non_na == 0:
        return "no_latest_coverage"

    return None


__all__ = [
    "AXIS_CACHE_VERSION",
    "AXIS_CACHE_FILENAME",
    "AXIS_PANEL_NAMES",
    "OPTIONAL_EMPTY_CACHE_PANELS",
    "resolve_axis_cache_path",
    "load_axis_construction_cache",
    "save_axis_construction_cache",
    "cache_panel_stale_reason",
]
