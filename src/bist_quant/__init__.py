"""
BIST Quant - Quantitative Research and Backtesting Library for Borsa Istanbul.

A comprehensive Python library for quantitative finance research, backtesting,
and portfolio optimization focused on Borsa Istanbul (BIST) markets.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any
import warnings

__version__ = "0.3.2"
__author__ = "BIST Quant Team"

# =============================================================================
# CORE COMPONENTS
# =============================================================================

from .portfolio import (
    DEFAULT_PORTFOLIO_OPTIONS,
    PortfolioEngine,
    PortfolioResult,
    SignalResult,
    get_default_options,
    run_backtest,
)

from .common.data_loader import DataLoader
from .common.config_manager import ConfigManager, load_config, load_signal_configs
from .common.data_paths import (
    DataPaths,
    get_data_paths,
    get_fundamentals_path,
    get_prices_path,
    validate_data_paths,
)

# =============================================================================
# BACKTESTING & RISK
# =============================================================================

from .common.backtester import Backtester
from .common.risk_manager import RiskManager
from .common.report_generator import ReportGenerator

# =============================================================================
# MULTI-ASSET CLIENTS (OPTIONAL)
# =============================================================================

try:
    from .clients.crypto_client import CryptoClient
except Exception:
    CryptoClient = None

try:
    from .clients.us_stock_client import USStockClient
except Exception:
    USStockClient = None

try:
    from .clients.fx_commodities_client import FXCommoditiesClient
except Exception:
    FXCommoditiesClient = None

try:
    from .clients.fx_enhanced_provider import FXEnhancedProvider
except Exception:
    FXEnhancedProvider = None

try:
    from .clients.fund_analyzer import FundAnalyzer
except Exception:
    FundAnalyzer = None

try:
    from .clients.borsapy_adapter import BorsapyAdapter
except Exception:
    BorsapyAdapter = None

# =============================================================================
# REGIME FILTER
# =============================================================================

try:
    from .regime import HAS_REGIME as HAS_REGIME_FILTER
    from .regime import RegimeClassifier
except Exception:
    RegimeClassifier = None
    HAS_REGIME_FILTER = False

# =============================================================================
# LEGACY-COMPAT EXPORTS
# =============================================================================

_DEPRECATED_IMPORT_MAP = {
    "BackendPaths": "bist_quant.runtime",
    "RuntimePaths": "bist_quant.runtime",
    "RuntimePathError": "bist_quant.runtime",
    "resolve_runtime_paths": "bist_quant.runtime",
}
_DEPRECATED_NAMES = set(_DEPRECATED_IMPORT_MAP)


def __getattr__(name: str) -> Any:
    """Lazy loading with deprecation warnings for legacy compatibility exports."""
    if name in _DEPRECATED_NAMES:
        module = _DEPRECATED_IMPORT_MAP[name]
        warnings.warn(
            f"Importing {name} from bist_quant is deprecated and will be removed in v1.0. "
            f"Please use 'from {module} import {name}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(import_module(module), name)

    raise AttributeError(name)


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Version
    "__version__",
    # Core
    "PortfolioEngine",
    "PortfolioResult",
    "SignalResult",
    "DEFAULT_PORTFOLIO_OPTIONS",
    "run_backtest",
    "get_default_options",
    # Data
    "DataLoader",
    "DataPaths",
    "get_data_paths",
    "get_prices_path",
    "get_fundamentals_path",
    "validate_data_paths",
    # Configuration
    "ConfigManager",
    "load_config",
    "load_signal_configs",
    # Backtesting & Risk
    "Backtester",
    "RiskManager",
    "ReportGenerator",
    # Multi-asset
    "CryptoClient",
    "USStockClient",
    "FXCommoditiesClient",
    "FXEnhancedProvider",
    "FundAnalyzer",
    "BorsapyAdapter",
    # Regime
    "RegimeClassifier",
    "HAS_REGIME_FILTER",
    "BackendPaths",
    "RuntimePaths",
    "RuntimePathError",
    "resolve_runtime_paths",
]
