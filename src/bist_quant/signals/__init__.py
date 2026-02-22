"""
BIST Quant Signal Builders.

74+ signal builders for quantitative analysis including momentum, value,
quality, technical indicators, factor rotation strategies, and composites.
"""

from __future__ import annotations

from pathlib import Path

_SIGNALS_DIR = Path(__file__).parent


def list_available_signals() -> list[str]:
    """List available signal module names."""
    signals: list[str] = []
    for file_path in _SIGNALS_DIR.glob("*_signals.py"):
        if not file_path.name.startswith("_") and not file_path.name.startswith("test_"):
            signals.append(file_path.stem.replace("_signals", ""))
    return sorted(signals)


def get_signal_module(signal_name: str):
    """
    Dynamically import a signal module.

    Args:
        signal_name: Name like "momentum", "value", etc.

    Returns:
        Imported module
    """
    import importlib

    module_name = f".{signal_name}_signals"
    return importlib.import_module(module_name, package="bist_quant.signals")


# Convenience imports
try:
    from .momentum_signals import *
except Exception:
    pass

try:
    from .value_signals import *
except Exception:
    pass

try:
    from .profitability_signals import *
except Exception:
    pass

try:
    from .quality_momentum_signals import *
except Exception:
    pass

try:
    from .factor_builders import *
except Exception:
    pass

try:
    from .factor_axes import *
except Exception:
    pass


__all__ = [
    "list_available_signals",
    "get_signal_module",
]
