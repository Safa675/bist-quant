"""
BIST Quant Regime Filter.

Market regime classification utilities.
"""

from __future__ import annotations

try:
    from .simple_regime import RegimeClassifier, RegimeType
    HAS_REGIME = True
except Exception:
    try:
        from .simple_regime import SimpleRegimeClassifier as RegimeClassifier
        RegimeType = None
        HAS_REGIME = True
    except Exception:
        RegimeClassifier = None
        RegimeType = None
        HAS_REGIME = False

__all__ = [
    "RegimeClassifier",
    "RegimeType",
    "HAS_REGIME",
]
