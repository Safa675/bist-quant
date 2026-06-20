"""Regime prediction and label loading sub-loader.

Handles loading regime predictions from CSV files, regime allocation
mappings from JSON sidecars, and raw regime labels.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from bist_quant.common.enums import RegimeLabel
from bist_quant.settings import PROJECT_ROOT

if TYPE_CHECKING:
    from bist_quant.common.data_paths import DataPaths

logger = logging.getLogger(__name__)

REGIME_DIR_CANDIDATES: list[Path] = [
    PROJECT_ROOT / "outputs" / "regime" / "simple_regime",
    PROJECT_ROOT / "outputs" / "regime",
    PROJECT_ROOT / "regime_filter",
    PROJECT_ROOT / "Simple Regime Filter",
    PROJECT_ROOT / "Regime Filter",
]


class RegimeLoader:
    """Load regime predictions, allocations, and raw labels.

    Args:
        paths: Pre-resolved data paths object.
        regime_model_dir: Directory for regime artifacts.
    """

    def __init__(
        self,
        paths: DataPaths,
        regime_model_dir: Path,
    ) -> None:
        self.paths = paths
        self.regime_model_dir = regime_model_dir
        self._regime_series: pd.Series | None = None
        self._regime_allocations: dict[RegimeLabel, float] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_regime_predictions(self, features: pd.DataFrame | None = None) -> pd.Series:
        """Load regime labels from regime filter outputs.

        Args:
            features: Unused legacy argument kept for backward compatibility.
        """
        del features  # Backward compatibility placeholder

        if self._regime_series is None:
            logger.info("\n🎯 Loading regime labels...")
            candidate_files: list[Path] = []
            direct_regime_file = self.regime_model_dir / "regime_features.csv"
            candidate_files.append(direct_regime_file)
            if self.regime_model_dir.name.lower() != "outputs":
                candidate_files.append(
                    self.regime_model_dir / "outputs" / "regime_features.csv"
                )
            if self.regime_model_dir.parent != self.regime_model_dir:
                candidate_files.append(
                    self.regime_model_dir.parent / "outputs" / "regime_features.csv"
                )
            candidate_files.extend(
                [p / "regime_features.csv" for p in REGIME_DIR_CANDIDATES]
            )
            candidate_files.extend(
                [p / "outputs" / "regime_features.csv" for p in REGIME_DIR_CANDIDATES]
            )
            regime_file = next(
                (f for f in candidate_files if f.exists()), candidate_files[0]
            )

            if not regime_file.exists():
                candidate_dirs = ", ".join(
                    str(
                        path.parent
                        if path.name == "regime_features.csv"
                        else path
                    )
                    for path in candidate_files
                )
                raise FileNotFoundError(
                    f"Regime file not found in expected locations: {candidate_dirs}\n"
                    "Run the simplified regime pipeline to generate outputs."
                )

            regime_df = pd.read_csv(regime_file)
            if regime_df.empty:
                raise ValueError(f"Regime file is empty: {regime_file}")

            date_col = next(
                (c for c in ("Date", "date", "DATE") if c in regime_df.columns),
                regime_df.columns[0],
            )
            regime_df[date_col] = pd.to_datetime(regime_df[date_col], errors="coerce")
            regime_df = regime_df.dropna(subset=[date_col]).set_index(date_col).sort_index()

            regime_col = next(
                (
                    c
                    for c in ("regime_label", "simplified_regime", "regime", "detailed_regime")
                    if c in regime_df.columns
                ),
                None,
            )
            if regime_col is None:
                raise ValueError(
                    "No regime column found in regime file. "
                    "Expected one of: regime_label, simplified_regime, regime, detailed_regime."
                )

            raw_regimes = regime_df[regime_col].dropna()
            coerced = raw_regimes.map(RegimeLabel.coerce)
            coerced = coerced[coerced.notna()]
            self._regime_series = coerced.astype(object)
            if self._regime_series.empty:
                raise ValueError(f"No valid regime rows found in: {regime_file}")

            # Load regime->allocation mapping from simplified regime export.
            self._regime_allocations = {}
            labels_file = regime_file.parent / "regime_labels.json"
            if labels_file.exists():
                try:
                    labels = json.loads(labels_file.read_text(encoding="utf-8"))
                    for payload in labels.values():
                        if not isinstance(payload, dict):
                            continue
                        regime = RegimeLabel.coerce(payload.get("regime"))
                        alloc = payload.get("allocation")
                        if regime is not None and alloc is not None:
                            try:
                                self._regime_allocations[regime] = float(alloc)
                            except (TypeError, ValueError):
                                continue
                except Exception as exc:
                    logger.warning(
                        f"  ⚠️  Could not parse regime allocations from {labels_file.name}: {exc}"
                    )

            logger.info(f"  ✅ Loaded {len(self._regime_series)} regime labels")
            logger.info("\n  Regime distribution:")
            for regime, count in self._regime_series.astype(str).value_counts().items():
                pct = count / len(self._regime_series) * 100
                logger.info(f"    {regime}: {count} days ({pct:.1f}%)")
            if self._regime_allocations:
                logger.info("  Regime allocations:")
                for regime, alloc in sorted(
                    self._regime_allocations.items(),
                    key=lambda item: item[0].value
                    if hasattr(item[0], "value")
                    else str(item[0]),
                ):
                    logger.info(f"    {regime}: {alloc:.2f}")

        return self._regime_series

    def load_regime_allocations(self) -> dict[RegimeLabel, float]:
        """Get regime allocation mapping loaded from regime_labels.json."""
        if self._regime_series is None:
            self.load_regime_predictions()
        return dict(self._regime_allocations or {})

    def load_regime_labels(self) -> dict:
        """Load regime labels JSON as a dictionary."""
        regime_file = self.paths.regime_labels
        if regime_file.exists():
            with open(regime_file, encoding="utf-8") as handle:
                return json.load(handle)
        return {}
