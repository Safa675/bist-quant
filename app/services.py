"""
Singleton service layer for the BIST Quant Research Cockpit.

All heavy objects are instantiated once per Streamlit session via
@st.cache_resource and shared across all pages.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)

# ── repo-root-relative data directory ────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent
_DATA_DIR = _REPO_ROOT / "data"

# ── availability flags ────────────────────────────────────────────────────────
try:
    from bist_quant.services.core_service import CoreBackendService  # type: ignore

    CORE_SERVICE_AVAILABLE = True
except ImportError:
    CORE_SERVICE_AVAILABLE = False
    logger.warning("bist_quant.services.core_service is not importable.")

try:
    from bist_quant.engines.factor_lab import FactorLabEngine  # type: ignore

    FACTOR_LAB_AVAILABLE = True
except ImportError:
    FACTOR_LAB_AVAILABLE = False
    logger.warning("bist_quant.engines.factor_lab is not importable.")

try:
    from bist_quant.common.data_manager import DataManager  # type: ignore

    DATA_MANAGER_AVAILABLE = True
except ImportError:
    DATA_MANAGER_AVAILABLE = False
    logger.warning("bist_quant.common.data_manager is not importable.")

try:
    from bist_quant.services.realtime_service import RealtimeService  # type: ignore

    REALTIME_AVAILABLE = True
except ImportError:
    REALTIME_AVAILABLE = False
    logger.warning("bist_quant.services.realtime_service is not importable.")

try:
    from bist_quant.regime.simple_regime import (  # type: ignore
        SimpleRegimeClassifier,
        DataLoader as RegimeDataLoader,
    )

    REGIME_AVAILABLE = True
except ImportError:
    REGIME_AVAILABLE = False
    logger.warning("bist_quant.regime.simple_regime is not importable.")

try:
    from bist_quant.common.data_loader import load_xu100_prices  # type: ignore

    DATA_LOADER_AVAILABLE = True
except ImportError:
    DATA_LOADER_AVAILABLE = False
    logger.warning("bist_quant.common.data_loader is not importable.")


# ── service factories ─────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading core service…")
def get_core_service() -> "CoreBackendService | None":
    """Return a cached `CoreBackendService` instance, or *None* on failure."""
    if not CORE_SERVICE_AVAILABLE:
        st.warning(
            "⚠️ `bist_quant` core service is not available. "
            "Run `pip install -e .[api]` from the repo root."
        )
        return None
    try:
        svc = CoreBackendService(data_dir=str(_DATA_DIR))
        logger.info("CoreBackendService initialised.")
        return svc
    except Exception as exc:
        logger.error("Failed to initialise CoreBackendService: %s", exc)
        st.warning(f"⚠️ CoreBackendService failed to start: {exc}")
        return None


@st.cache_resource(show_spinner="Loading Factor Lab…")
def get_factor_lab() -> "FactorLabEngine | None":
    """Return a cached `FactorLabEngine` instance, or *None* on failure."""
    if not FACTOR_LAB_AVAILABLE:
        st.warning("⚠️ FactorLabEngine is not available.")
        return None
    try:
        core = get_core_service()
        if core is None:
            return None
        lab = FactorLabEngine(backend_service=core)
        logger.info("FactorLabEngine initialised.")
        return lab
    except Exception as exc:
        logger.error("Failed to initialise FactorLabEngine: %s", exc)
        st.warning(f"⚠️ FactorLabEngine failed to start: {exc}")
        return None


@st.cache_resource(show_spinner="Loading data manager…")
def get_data_manager() -> "DataManager | None":
    """Return a cached `DataManager` instance, or *None* on failure."""
    if not DATA_MANAGER_AVAILABLE:
        st.warning("⚠️ DataManager is not available.")
        return None
    try:
        dm = DataManager(data_dir=str(_DATA_DIR))
        logger.info("DataManager initialised with data_dir=%s", _DATA_DIR)
        return dm
    except Exception as exc:
        logger.error("Failed to initialise DataManager: %s", exc)
        st.warning(f"⚠️ DataManager failed to start: {exc}")
        return None


@st.cache_resource(show_spinner=False)
def get_realtime_service() -> "RealtimeService | None":
    """
    Return a cached `RealtimeService`, or *None* if borsapy credentials are
    missing or the import fails.  Degrades gracefully — no exception raised.
    """
    if not REALTIME_AVAILABLE:
        logger.info("RealtimeService not available (borsapy not installed).")
        return None
    try:
        svc = RealtimeService()
        logger.info("RealtimeService initialised.")
        return svc
    except Exception as exc:
        logger.warning("RealtimeService unavailable: %s", exc)
        return None


@st.cache_data(ttl=300, show_spinner=False)
def get_regime_data() -> dict[str, Any]:
    """
    Return a comprehensive regime dict:
        {
            "label":        str,          # e.g. "Bull"
            "current":      dict,         # full get_current_regime() output
            "series":       dict,         # {date_str: regime_str} full history
            "distribution": dict,         # {regime: {Count, Percent}}
            "available":    bool,
        }

    Cached for 5 minutes.  Never raises.
    """
    fallback: dict[str, Any] = {
        "label": "Unknown",
        "current": {},
        "series": {},
        "distribution": {},
        "available": False,
    }

    if not REGIME_AVAILABLE:
        return fallback

    try:
        loader = RegimeDataLoader(data_dir=str(_DATA_DIR))
        xu100 = loader.load_xu100()
        clf = SimpleRegimeClassifier()
        regime_series = clf.classify(xu100["Close"])
        current = clf.get_current_regime()
        distribution = clf.get_distribution()

        label = current.get("regime", "Unknown")

        # Convert series index (Timestamps) to ISO strings for JSON safety
        series_dict = {
            str(d.date()): r for d, r in regime_series.items()
        }

        dist_dict = distribution.to_dict(orient="index") if hasattr(distribution, "to_dict") else {}

        return {
            "label": label,
            "current": current,
            "series": series_dict,
            "distribution": dist_dict,
            "available": True,
        }
    except Exception as exc:
        logger.warning("SimpleRegimeClassifier error: %s", exc)
        return fallback


def get_regime_classifier() -> dict[str, Any]:
    """
    Lightweight wrapper returning just enough for the sidebar badge.
    Delegates to `get_regime_data()`.
    """
    data = get_regime_data()
    return {"label": data["label"], "regime": data["current"], "available": data["available"]}


def is_realtime_connected() -> bool:
    """Return True if the RealtimeService initialised successfully."""
    return get_realtime_service() is not None


__all__ = [
    "get_core_service",
    "get_factor_lab",
    "get_data_manager",
    "get_realtime_service",
    "get_regime_data",
    "get_regime_classifier",
    "is_realtime_connected",
    "CORE_SERVICE_AVAILABLE",
    "FACTOR_LAB_AVAILABLE",
    "DATA_MANAGER_AVAILABLE",
    "REALTIME_AVAILABLE",
    "REGIME_AVAILABLE",
    "_DATA_DIR",
    "_REPO_ROOT",
]
