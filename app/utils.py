"""
Utility helpers for the BIST Quant Research Cockpit.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parent.parent
_DATA_DIR = _REPO_ROOT / "data"

# ── regime colours ────────────────────────────────────────────────────────────
_REGIME_COLORS: dict[str, str] = {
    "bull": "#2ecc71",
    "recovery": "#f39c12",
    "bear": "#e74c3c",
    "stress": "#8e44ad",
    "unknown": "#95a5a6",
}


def regime_color(regime: str) -> str:
    """Map a regime label string to a hex colour code."""
    return _REGIME_COLORS.get(regime.lower(), _REGIME_COLORS["unknown"])


# ── number formatting ─────────────────────────────────────────────────────────


def fmt_pct(v: float | None, decimals: int = 2) -> str:
    """Format a float as a signed percentage string, e.g. ``+12.34%``."""
    if v is None:
        return "—"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.{decimals}f}%"


def fmt_num(v: float | None, decimals: int = 2) -> str:
    """Format a float with thousands separators, e.g. ``1,234.56``."""
    if v is None:
        return "—"
    return f"{v:,.{decimals}f}"


# ── data loading ──────────────────────────────────────────────────────────────


def resolve_data_path(filename: str) -> Path:
    """Resolve *filename* relative to the ``data/`` directory at repo root."""
    return _DATA_DIR / filename


@st.cache_data(ttl=3600, show_spinner=False)
def load_csv_cached(path: str) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame.  Results are cached for 1 hour.

    Args:
        path: Absolute path string to the CSV file.

    Returns:
        DataFrame, or an empty DataFrame if the file is missing.
    """
    p = Path(path)
    if not p.exists():
        logger.warning("CSV not found: %s", p)
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception as exc:
        logger.error("Failed to read CSV %s: %s", p, exc)
        return pd.DataFrame()


# ── concurrency ───────────────────────────────────────────────────────────────

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="bist_quant")


def run_in_thread(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
    """
    Submit *fn* to a shared thread-pool executor.

    Returns a ``concurrent.futures.Future`` immediately.  Callers are
    responsible for polling or awaiting the result.

    Example::

        future = run_in_thread(core_service.run_backtest, payload)
        while not future.done():
            time.sleep(0.2)
            st.rerun()
        result = future.result()
    """
    return _executor.submit(fn, *args, **kwargs)


# ── streamlit helpers ─────────────────────────────────────────────────────────


def metric_card(
    label: str,
    value: str,
    delta: str | None = None,
    color: str | None = None,
) -> None:
    """
    Render an ``st.metric`` with an optional custom background colour.

    The colour is applied via a thin HTML/CSS hack that wraps the metric in a
    coloured border-left strip — compatible with Streamlit's light and dark
    themes.
    """
    if color:
        st.markdown(
            f"""
            <div style="
                border-left: 4px solid {color};
                padding-left: 0.6rem;
                margin-bottom: 0.25rem;
            ">
            """,
            unsafe_allow_html=True,
        )

    st.metric(label=label, value=value, delta=delta)

    if color:
        st.markdown("</div>", unsafe_allow_html=True)


__all__ = [
    "fmt_pct",
    "fmt_num",
    "resolve_data_path",
    "load_csv_cached",
    "run_in_thread",
    "regime_color",
    "metric_card",
]
