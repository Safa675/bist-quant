"""Compatibility shim for legacy indicator imports.

Historically, tooling imported ``borsapy_indicators`` as a top-level module.
The canonical implementation now lives in
``bist_quant.signals.borsapy_indicators``. Re-exporting here preserves
backward compatibility for scripts/tests still using the old path.
"""

from bist_quant.signals.borsapy_indicators import BORSAPY_AVAILABLE
from bist_quant.signals.borsapy_indicators import BorsapyIndicators
from bist_quant.signals.borsapy_indicators import build_multi_indicator_panel

__all__ = [
    "BORSAPY_AVAILABLE",
    "BorsapyIndicators",
    "build_multi_indicator_panel",
]
