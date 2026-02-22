"""Signal factory helpers for strategy signal builder discovery and execution."""

from typing import Any

import pandas as pd

from bist_quant.signals.composite import BUILDERS as COMPOSITE_BUILDERS
from bist_quant.signals.momentum import BUILDERS as MOMENTUM_BUILDERS
from bist_quant.signals.protocol import SignalBuilder
from bist_quant.signals.quality import BUILDERS as QUALITY_BUILDERS
from bist_quant.signals.ta_consensus_signals import BUILDERS as TA_CONSENSUS_BUILDERS
from bist_quant.signals.technical import BUILDERS as TECHNICAL_BUILDERS
from bist_quant.signals.value import BUILDERS as VALUE_BUILDERS

ConfigDict = dict[str, Any]

BUILDERS: dict[str, SignalBuilder] = {
    **MOMENTUM_BUILDERS,
    **VALUE_BUILDERS,
    **QUALITY_BUILDERS,
    **TECHNICAL_BUILDERS,
    **COMPOSITE_BUILDERS,
    **TA_CONSENSUS_BUILDERS,
}


def get_available_signals() -> list[str]:
    """Return all registered signal names in deterministic sorted order."""
    return sorted(BUILDERS.keys())


def _resolve_signal_params(name: str, config: ConfigDict) -> ConfigDict:
    signal_params = config.get("signal_params", {})
    if signal_params is None:
        signal_params = {}
    if not isinstance(signal_params, dict):
        raise TypeError(
            f"Signal '{name}' expects config['signal_params'] to be dict, got {type(signal_params).__name__}"
        )

    legacy_params = config.get("parameters", {})
    if legacy_params is None:
        legacy_params = {}
    if not isinstance(legacy_params, dict):
        raise TypeError(
            f"Signal '{name}' expects config['parameters'] to be dict, got {type(legacy_params).__name__}"
        )

    merged_params = dict(legacy_params)
    merged_params.update(signal_params)
    return merged_params


def build_signal(
    name: str,
    dates: pd.DatetimeIndex,
    loader: Any,
    config: ConfigDict,
) -> pd.DataFrame:
    """Build one signal DataFrame via registered builder implementations.

    Args:
        name: Registered signal name.
        dates: Trading dates to build the signal for.
        loader: Data loader dependency passed through to the builder.
        config: Strategy configuration including ``parameters`` or
            ``signal_params`` dictionaries.

    Returns:
        Signal panel as a ``pandas.DataFrame``.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Signal name must be a non-empty string")
    if not isinstance(dates, pd.DatetimeIndex):
        raise TypeError(f"'dates' must be pd.DatetimeIndex, got {type(dates).__name__}")
    if not isinstance(config, dict):
        raise TypeError(f"'config' must be dict, got {type(config).__name__}")

    builder = BUILDERS.get(name)
    if builder is None:
        available = ", ".join(get_available_signals())
        raise ValueError(f"Unknown signal: {name}. Available signals: {available}")

    signal_params = _resolve_signal_params(name, config)
    result = builder(dates=dates, loader=loader, config=config, signal_params=signal_params)

    if not isinstance(result, pd.DataFrame):
        raise TypeError(
            f"Signal builder '{name}' must return pd.DataFrame, got {type(result).__name__}"
        )

    return result


class SignalFactory:
    """Class-style wrapper around module-level signal factory functions.

    This preserves a discoverable object-oriented API while delegating to the
    underlying functional implementation used internally.
    """

    @staticmethod
    def get_available_signals() -> list[str]:
        """Return all registered signal names."""
        return get_available_signals()

    @staticmethod
    def build(
        name: str,
        dates: pd.DatetimeIndex,
        loader: Any,
        config: ConfigDict,
    ) -> pd.DataFrame:
        """Build a signal by name using the registered builder map."""
        return build_signal(name=name, dates=dates, loader=loader, config=config)
