"""
BIST Quant Tactical Strategies.

Overlay strategies that gate equity exposure inside the backtester
event loop — alongside the regime filter. Unlike cross-sectional
stock signals, overlays return a continuous exposure scalar in [0, 1].
"""

from __future__ import annotations

from bist_quant.strategies.base import (
    TacticalOverlay,
    build_tactical_overlay,
)

__all__ = [
    "TacticalOverlay",
    "build_tactical_overlay",
]
