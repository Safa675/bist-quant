"""
Data Path Resolution - BIST Quant

Provides consistent data path resolution across all modules.
Supports environment variable configuration for flexible deployment.

Environment Variables:
    BIST_DATA_DIR: Path to canonical market data directory
    BIST_REGIME_DIR: Path to regime filter outputs (optional)
    BIST_CACHE_DIR: Path to cache directory (optional)

Usage:
    from bist_quant.common.data_paths import DataPaths

    paths = DataPaths()
    prices_path = paths.prices_file
    fundamentals_path = paths.fundamentals_file
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _find_bist_root() -> Path:
    """Find BIST repository root directory."""
    current = Path(__file__).resolve()

    for parent in [current] + list(current.parents):
        # The true project root has a data directory and optionally a pyproject.toml
        if (parent / "data").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
        if (parent / "data").is_dir() and (parent / "src" / "bist_quant").is_dir():
            return parent
            
    # Fallback to finding just the data dir
    for parent in [current] + list(current.parents):
        if (parent / "data").is_dir():
            return parent

    return Path(__file__).parent.parent.parent.parent


def _resolve_path(value: Optional[Path | str]) -> Optional[Path]:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def _get_env_path(var_name: str, default: Optional[Path] = None) -> Optional[Path]:
    """Get path from environment variable."""
    value = os.environ.get(var_name)
    if value:
        path = Path(value).expanduser().resolve()
        if path.exists():
            return path
    return default


@dataclass
class DataPaths:
    """
    Centralized data path management.

    Resolves paths based on:
    1. Environment variables (highest priority)
    2. Explicit constructor arguments
    3. Auto-detected BIST repository paths (fallback)

    Attributes:
        data_dir: Root directory for market data
        regime_dir: Directory for regime filter outputs
        cache_dir: Directory for cached computations
    """

    data_dir: Optional[Path | str] = None
    regime_dir: Optional[Path | str] = None
    cache_dir: Optional[Path | str] = None

    def __post_init__(self) -> None:
        """Initialize paths from environment or defaults."""
        bist_root = _find_bist_root()

        explicit_data = _resolve_path(self.data_dir)
        explicit_regime = _resolve_path(self.regime_dir)
        explicit_cache = _resolve_path(self.cache_dir)

        self.data_dir = (
            _get_env_path("BIST_DATA_DIR")
            or explicit_data
            or (bist_root / "data")
        )
        self.regime_dir = (
            _get_env_path("BIST_REGIME_DIR")
            or explicit_regime
            or (bist_root / "outputs" / "regime" / "simple_regime")
        )
        self.cache_dir = (
            _get_env_path("BIST_CACHE_DIR")
            or explicit_cache
            or (bist_root / ".cache")
        )

    # -------------------------------------------------------------------------
    # Price Data
    # -------------------------------------------------------------------------

    @property
    def prices_file(self) -> Path:
        """Primary price data file (Parquet preferred, CSV fallback)."""
        cache = self.borsapy_cache_dir / "panels" / "prices_panel.parquet"
        if cache.exists() and cache.stat().st_size > 100_000:
            return cache
        parquet = self.data_dir / "bist_prices_full.parquet"
        if parquet.exists():
            return parquet
        return self.data_dir / "bist_prices_full.csv"

    @property
    def prices_parquet(self) -> Path:
        """Primary price data file (Parquet)."""
        cache = self.borsapy_cache_dir / "panels" / "prices_panel.parquet"
        if cache.exists():
            return cache
        return self.data_dir / "bist_prices_full.parquet"

    @property
    def isyatirim_prices(self) -> Path:
        """Consolidated IsYatirim prices."""
        return self.data_dir / "consolidated_isyatirim_prices.parquet"

    @property
    def xu100_prices(self) -> Path:
        """XU100 index prices, preferring the borsapy_cache."""
        cache_pq = self.borsapy_cache_dir / "index_components" / "XU100.parquet"
        if cache_pq.exists():
            return cache_pq
        cache_csv = self.borsapy_cache_dir / "index_components" / "XU100.csv"
        if cache_csv.exists():
            return cache_csv
        
        parquet = self.data_dir / "xu100_prices.parquet"
        if parquet.exists():
            return parquet
        return self.data_dir / "xu100_prices.csv"

    # -------------------------------------------------------------------------
    # Fundamental Data
    # -------------------------------------------------------------------------

    @property
    def fundamentals_file(self) -> Path:
        """Consolidated fundamental data (Parquet)."""
        return self.data_dir / "fundamental_data_consolidated.parquet"

    @property
    def fundamentals_csv(self) -> Path:
        """Consolidated fundamental data (CSV)."""
        return self.data_dir / "fundamental_data_consolidated.csv"

    @property
    def fundamentals_dir(self) -> Path:
        """Per-company fundamental data directory."""
        return self.data_dir / "fundamental_data"

    @property
    def shares_outstanding(self) -> Path:
        """Shares outstanding data."""
        return self.data_dir / "shares_outstanding_consolidated.csv"

    # -------------------------------------------------------------------------
    # Factor Data
    # -------------------------------------------------------------------------

    @property
    def five_factor_axes(self) -> Path:
        """Pre-computed five-factor axis construction."""
        return self.data_dir / "five_factor_axis_construction.parquet"

    @property
    def multi_factor_axes(self) -> Path:
        """Pre-computed multi-factor axis construction."""
        return self.data_dir / "multi_factor_axis_construction.parquet"

    # -------------------------------------------------------------------------
    # Reference Data
    # -------------------------------------------------------------------------

    @property
    def sector_classification(self) -> Path:
        """BIST sector classification."""
        parquet = self.data_dir / "bist_sector_classification.parquet"
        if parquet.exists():
            return parquet
        return self.data_dir / "bist_sector_classification.csv"

    @property
    def tcmb_indicators(self) -> Path:
        """Central bank (TCMB) indicators."""
        return self.data_dir / "tcmb_indicators.csv"

    # -------------------------------------------------------------------------
    # FX and Commodities
    # -------------------------------------------------------------------------

    @property
    def usdtry_file(self) -> Path:
        """USD/TRY exchange rate data (now sourced from gold cache)."""
        cache = self.borsapy_cache_dir / "gold" / "xau_try_daily.parquet"
        if cache.exists():
            return cache
        return self.data_dir / "usdtry_data.csv"

    @property
    def gold_try_file(self) -> Path:
        """Gold prices in TRY (now sourced from gold cache)."""
        cache = self.borsapy_cache_dir / "gold" / "xau_try_daily.parquet"
        if cache.exists():
            return cache
        return self.data_dir / "xau_try_2013_2026.csv"

    @property
    def gold_funds_file(self) -> Path:
        """Gold fund daily prices."""
        csv = self.data_dir / "gold_funds_daily_prices.csv"
        if csv.exists():
            return csv
        return self.data_dir / "gold_funds_daily_prices.parquet"

    # -------------------------------------------------------------------------
    # Regime Filter
    # -------------------------------------------------------------------------

    @property
    def regime_labels(self) -> Path:
        """Regime predictions JSON."""
        return self.regime_dir / "regime_labels.json"

    @property
    def regime_features(self) -> Path:
        """Regime features CSV."""
        in_data = self.data_dir / "regime_features_full.csv"
        in_regime = self.regime_dir / "regime_features.csv"

        if in_data.exists():
            return in_data
        return in_regime

    # -------------------------------------------------------------------------
    # Cache
    # -------------------------------------------------------------------------

    @property
    def signal_cache_dir(self) -> Path:
        """Cache directory for computed signals."""
        cache = self.cache_dir / "signals"
        cache.mkdir(parents=True, exist_ok=True)
        return cache

    @property
    def backtest_cache_dir(self) -> Path:
        """Cache directory for backtest results."""
        cache = self.cache_dir / "backtests"
        cache.mkdir(parents=True, exist_ok=True)
        return cache

    @property
    def borsapy_cache_dir(self) -> Path:
        """Cache directory for borsapy API data."""
        return self.data_dir / "borsapy_cache"

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate(self, strict: bool = False) -> dict:
        """
        Validate that required data files exist.

        Args:
            strict: If True, raise error on missing files

        Returns:
            Dict with validation results
        """
        required_files = [
            ("prices", self.prices_file),
            ("fundamentals", self.fundamentals_file),
            ("shares_outstanding", self.shares_outstanding),
        ]

        optional_files = [
            ("xu100_prices", self.xu100_prices),
            ("sector_classification", self.sector_classification),
            ("five_factor_axes", self.five_factor_axes),
            ("multi_factor_axes", self.multi_factor_axes),
            ("regime_labels", self.regime_labels),
            ("usdtry", self.usdtry_file),
            ("gold_try", self.gold_try_file),
        ]

        results = {
            "valid": True,
            "data_dir": str(self.data_dir),
            "data_dir_exists": self.data_dir.exists(),
            "required": {},
            "optional": {},
            "missing_required": [],
            "missing_optional": [],
        }

        for name, path in required_files:
            exists = path.exists()
            results["required"][name] = {
                "path": str(path),
                "exists": exists,
            }
            if not exists:
                results["missing_required"].append(name)
                results["valid"] = False

        for name, path in optional_files:
            exists = path.exists()
            results["optional"][name] = {
                "path": str(path),
                "exists": exists,
            }
            if not exists:
                results["missing_optional"].append(name)

        if strict and not results["valid"]:
            missing = ", ".join(results["missing_required"])
            raise FileNotFoundError(f"Missing required data files: {missing}")

        return results

    def __str__(self) -> str:
        return f"DataPaths(data_dir={self.data_dir})"

    def __repr__(self) -> str:
        return (
            "DataPaths(\n"
            f"  data_dir={self.data_dir},\n"
            f"  regime_dir={self.regime_dir},\n"
            f"  cache_dir={self.cache_dir}\n"
            ")"
        )


_default_paths: Optional[DataPaths] = None


def get_data_paths() -> DataPaths:
    """Get default DataPaths instance (singleton)."""
    global _default_paths
    if _default_paths is None:
        _default_paths = DataPaths()
    return _default_paths


def reset_data_paths() -> None:
    """Reset singleton (useful for testing)."""
    global _default_paths
    _default_paths = None


def get_prices_path() -> Path:
    """Get path to price data file."""
    return get_data_paths().prices_file


def get_fundamentals_path() -> Path:
    """Get path to fundamentals file."""
    return get_data_paths().fundamentals_file


def get_regime_labels_path() -> Path:
    """Get path to regime labels."""
    return get_data_paths().regime_labels


def validate_data_paths(strict: bool = False) -> dict:
    """Validate data paths."""
    return get_data_paths().validate(strict=strict)


__all__ = [
    "DataPaths",
    "get_data_paths",
    "reset_data_paths",
    "get_prices_path",
    "get_fundamentals_path",
    "get_regime_labels_path",
    "validate_data_paths",
]
