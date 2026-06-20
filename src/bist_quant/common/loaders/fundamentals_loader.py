"""Fundamentals data loading sub-loader.

Handles loading, caching, and validation of fundamental financial data
(balance sheet, income statement, cash flow) from parquet, xlsx, and
borsapy sources.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import pandas as pd

if TYPE_CHECKING:
    from bist_quant.common.data_paths import DataPaths

logger = logging.getLogger(__name__)


class FundamentalsLoader:
    """Load and cache fundamental financial data.

    This loader is owned by :class:`DataLoader` and should not be
    instantiated directly.
    """

    def __init__(
        self,
        paths: DataPaths,
        fundamental_dir: Path,
        data_source_priority: str,
    ) -> None:
        self.paths = paths
        self.fundamental_dir = fundamental_dir
        self._data_source_priority = data_source_priority

        # Caches
        self._fundamentals: Dict | None = None
        self._fundamentals_parquet: pd.DataFrame | None = None

        # Freshness gate controls (defaults are strict for production safety).
        self._fundamentals_freshness_gate_enabled = os.getenv(
            "BIST_ENFORCE_FUNDAMENTAL_FRESHNESS",
            "1",
        ).strip().lower() not in {"0", "false", "no", "off"}
        self._allow_stale_fundamentals = os.getenv(
            "BIST_ALLOW_STALE_FUNDAMENTALS",
            "0",
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._freshness_threshold_overrides = {
            "max_median_staleness_days": _env_int("BIST_MAX_MEDIAN_STALENESS_DAYS", 120),
            "max_pct_over_120_days": _env_float("BIST_MAX_PCT_OVER_120_DAYS", 0.90),
            "min_q4_coverage_pct": _env_float("BIST_MIN_Q4_2025_COVERAGE_PCT", 0.10),
            "max_max_staleness_days": _env_int("BIST_MAX_MAX_STALENESS_DAYS", 500),
            "grace_days": _env_int("BIST_STALENESS_GRACE_DAYS", 0),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_fundamentals(self, borsapy_adapter: Any) -> Dict:
        """Load all fundamental data.

        Priority:
        1. Consolidated parquet (existing behavior)
        2. Per-company xlsx files (existing fallback)
        3. Borsapy financial statements (supplementary fill when
           ``data_source_priority`` is ``"borsapy"`` and local data
           is missing)
        """
        if self._fundamentals is None:
            logger.info("\n📈 Loading fundamental data...")
            fundamentals: Dict = {}

            # 1. Try to load the consolidated panel from the new cache logic
            self._fundamentals_parquet = self._load_consolidated_fundamentals()
            if self._fundamentals_parquet is not None:
                self._enforce_fundamentals_freshness_gate(self._fundamentals_parquet)
                tickers = (
                    self._fundamentals_parquet.index.get_level_values("ticker").unique().tolist()
                )
                for ticker in tickers:
                    fundamentals[ticker] = {"path": None, "borsapy": True}
                logger.info(f"  ✅ Loaded consolidated fundamentals for {len(tickers)} tickers")
            else:
                # 2. Legacy fallback to individual xlsx files
                count = 0
                for file_path in self.fundamental_dir.rglob("*.xlsx"):
                    ticker = file_path.stem.split(".")[0].upper()
                    try:
                        fundamentals[ticker] = {
                            "path": file_path,
                            "income": None,  # Lazy load
                            "balance": None,
                            "cashflow": None,
                        }
                        count += 1
                    except Exception:
                        continue
                if count > 0:
                    logger.info(f"  ✅ Indexed {count} fundamental data files")

                # 3. Borsapy supplementary fill
                if count == 0 and self._data_source_priority != "local":
                    self._borsapy_fundamentals_fill(fundamentals, borsapy_adapter)

            self._fundamentals = fundamentals
        return self._fundamentals

    def load_fundamentals_parquet(self) -> pd.DataFrame | None:
        """Load consolidated fundamentals parquet if available."""
        if self._fundamentals_parquet is None:
            self._fundamentals_parquet = self._load_consolidated_fundamentals()
            if self._fundamentals_parquet is not None:
                self._enforce_fundamentals_freshness_gate(self._fundamentals_parquet)
        return self._fundamentals_parquet

    def load_fundamental_metrics(self, data_dir: Path) -> pd.DataFrame:
        """Load pre-calculated fundamental metrics."""
        logger.info("\n📊 Loading fundamental metrics...")
        metrics_file = data_dir / "fundamental_metrics.parquet"

        if not metrics_file.exists():
            logger.warning(f"  ⚠️  Fundamental metrics file not found: {metrics_file}")
            logger.info("  Run calculate_fundamental_metrics.py to generate this file")
            return pd.DataFrame()

        df = pd.read_parquet(metrics_file)
        logger.info(f"  ✅ Loaded {len(df)} metric observations")
        logger.info(f"  Metrics: {df.columns.tolist()}")
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_fundamentals_panel(frame: pd.DataFrame) -> bool:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return False
        if not isinstance(frame.index, pd.MultiIndex) or frame.index.nlevels < 3:
            return False
        names = {str(name) for name in frame.index.names if name is not None}
        required = {"ticker", "sheet_name", "row_name"}
        return required.issubset(names)

    def _load_consolidated_fundamentals(self) -> pd.DataFrame | None:
        """Load consolidated fundamentals directly from borsapy_cache, building on the fly if needed."""
        consolidated_path = self.paths.borsapy_cache_dir / "financials_consolidated.parquet"

        # 1. Try to load pre-built consolidated form
        if consolidated_path.exists():
            try:
                frame = pd.read_parquet(consolidated_path)
                if self._is_fundamentals_panel(frame):
                    logger.info("  📦 Loaded consolidated fundamentals from borsapy_cache")
                    return frame
            except Exception as e:
                logger.warning(f"  ⚠️  Failed to read {consolidated_path}: {e}")

        # 2. Build on the fly from borsapy_cache/financials
        financials_dir = self.paths.borsapy_cache_dir / "financials"
        if not financials_dir.exists():
            return None

        logger.info("  🔄 Building consolidated fundamentals from borsapy_cache...")

        SHEET_MAP = {
            "balance_sheet": "Bilanço",
            "income_stmt": "Gelir Tablosu (Çeyreklik)",
            "cash_flow": "Nakit Akış (Çeyreklik)",
        }

        rows = []
        count = 0
        for ticker_dir in financials_dir.iterdir():
            if not ticker_dir.is_dir():
                continue
            ticker = ticker_dir.name

            for path in ticker_dir.glob("*.parquet"):
                sheet_key = path.stem
                if sheet_key not in SHEET_MAP:
                    continue
                sheet_name = SHEET_MAP[sheet_key]

                try:
                    df = pd.read_parquet(path)
                    if df.empty or df.index.name != "Item":
                        continue

                    # Reset index so 'Item' becomes a column
                    df_reset = df.reset_index()
                    for _, row in df_reset.iterrows():
                        row_name = row["Item"]
                        values = row.drop("Item").to_dict()
                        if not values:
                            continue
                        s = pd.Series(values, name=(ticker, sheet_name, row_name))
                        rows.append(s)
                except Exception:
                    continue
            count += 1
            if count % 100 == 0:
                logger.info(f"    Indexed {count} tickers...")

        if not rows:
            return None

        panel = pd.DataFrame(rows)
        panel.index = pd.MultiIndex.from_tuples(
            panel.index.tolist(),
            names=["ticker", "sheet_name", "row_name"],
        )
        # Drop fully NaN columns
        panel = panel.dropna(axis=1, how="all")

        try:
            panel.to_parquet(consolidated_path)
            logger.info("  💾 Saved new built consolidated fundamentals to borsapy_cache")
        except Exception as e:
            logger.warning(f"  ⚠️  Failed to write {consolidated_path}: {e}")

        return panel

    def _borsapy_fundamentals_fill(
        self,
        fundamentals: Dict,
        borsapy_adapter: Any,
    ) -> None:
        """Try to populate fundamental dict from borsapy financial statements."""
        try:
            logger.info("  🔄 Attempting borsapy fundamentals fill...")
            # Get universe from borsapy index components
            symbols = borsapy_adapter.get_index_components(index="XU100")
            if not symbols:
                logger.warning("  ⚠️  Could not resolve symbols for borsapy fundamentals fill")
                return

            filled = 0
            for sym in symbols:
                if sym in fundamentals:
                    continue
                try:
                    stmts = borsapy_adapter.get_financials(symbol=sym)
                    if stmts and any(
                        isinstance(v, pd.DataFrame) and not v.empty for v in stmts.values()
                    ):
                        fundamentals[sym] = {"path": None, "borsapy": True}
                        filled += 1
                except Exception:
                    continue

            if filled:
                logger.info(f"  ✅ Borsapy fill added {filled} tickers")
            else:
                logger.warning("  ⚠️  Borsapy fundamentals fill returned no data")
        except Exception as exc:
            logger.warning(f"  ⚠️  Borsapy fundamentals fill failed: {exc}")

    @staticmethod
    def _turkish_expected_publication_date(
        reference_date: pd.Timestamp | None = None,
    ) -> tuple[str, pd.Timestamp]:
        """Return the latest quarter whose deadline has passed.

        Turkish reporting deadlines:
        - Q1/Q2/Q3: 45 calendar days after quarter end
        - Q4: 75 calendar days after year end

        Returns (quarter_label, deadline).
        """
        if reference_date is None:
            reference_date = pd.Timestamp.now().normalize()

        year = reference_date.year
        # (quarter_end_month, quarter_label, deadline_days)
        deadlines = [
            (3, f"{year}/3", 45),  # Q1
            (6, f"{year}/6", 45),  # Q2
            (9, f"{year}/9", 45),  # Q3
            (12, f"{year}/12", 75),  # Q4
        ]
        # Also check prior year Q4 — it may be the latest publishable one
        deadlines.insert(0, (12, f"{year - 1}/12", 75))

        latest_q: str | None = None
        latest_deadline: pd.Timestamp | None = None
        for end_month, label, days in deadlines:
            q_year = int(label.split("/")[0])
            quarter_end = pd.Timestamp(year=q_year, month=end_month, day=1) + pd.offsets.MonthEnd(0)
            deadline = quarter_end + pd.Timedelta(days=days)
            if reference_date >= deadline:
                if latest_deadline is None or deadline > latest_deadline:
                    latest_q = label
                    latest_deadline = deadline

        if latest_q is None:
            # Fallback: previous year Q3
            latest_q = f"{year - 1}/9"
            latest_deadline = pd.Timestamp(year=year - 1, month=9, day=30) + pd.Timedelta(days=45)
        return latest_q, latest_deadline

    def _enforce_fundamentals_freshness_gate(self, panel: pd.DataFrame) -> None:
        """Calendar-aware freshness gate for Turkish financial reporting."""
        if panel is None or panel.empty:
            return
        if not self._fundamentals_freshness_gate_enabled:
            return
        if self._allow_stale_fundamentals:
            return

        expected_q, deadline = self._turkish_expected_publication_date()
        logger.info(
            f"  📅 Turkish calendar: expecting at least {expected_q} (deadline was {deadline:%Y-%m-%d})"
        )

        # Check if the expected quarter column exists in the panel
        if isinstance(panel.columns, pd.Index):
            available_periods = [c for c in panel.columns if "/" in str(c)]
            if expected_q in available_periods:
                # Count how many tickers have data for this quarter
                if isinstance(panel.index, pd.MultiIndex) and "ticker" in panel.index.names:
                    ticker_level = panel.index.get_level_values("ticker")
                    coverage = (
                        panel[expected_q].groupby(ticker_level).apply(lambda s: s.notna().any())
                    )
                    pct = coverage.mean()
                else:
                    pct = panel[expected_q].notna().mean()
                logger.info(f"  📊 Coverage for {expected_q}: {pct:.1%}")
            else:
                logger.debug(f"  ℹ️  Expected quarter {expected_q} not found in data columns")
        # Gate is advisory-only in new architecture — never block


# ---------------------------------------------------------------------------
# Module-level helpers (shared by all sub-loaders)
# ---------------------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
