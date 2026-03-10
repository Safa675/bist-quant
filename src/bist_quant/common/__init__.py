"""
BIST Quant Common Utilities.

Core infrastructure for backtesting, data management, and analysis.
"""

from __future__ import annotations

# Data loading and paths
from .data_loader import DataLoader
from .data_paths import (
    DataPaths,
    get_data_paths,
    get_fundamentals_path,
    get_prices_path,
    get_regime_labels_path,
    reset_data_paths,
    validate_data_paths,
)

# Configuration
from .config_manager import (
    ConfigManager,
    list_available_strategies,
    load_config,
    load_signal_configs,
)

# Backtesting infrastructure
from .backtester import Backtester
from .backtest_services import (
    BacktestMetrics,
    BacktestMetricsService,
    BacktestPayloadAssembler,
    BacktestPreparationResult,
    DataPreparationService,
    DailyReturnService,
    HoldingsHistoryAggregator,
    RebalanceDecision,
    RebalancingSelectionService,
    TransactionCostModel,
)
from .risk_manager import RiskManager
from .report_generator import ReportGenerator

# Data management
from .data_manager import DataManager
from .panel_cache import PanelCache
from .staleness import staleness_summary as check_data_staleness

# Multi-asset clients
try:
    from bist_quant.clients.crypto_client import CryptoClient
except Exception:
    CryptoClient = None

try:
    from bist_quant.clients.us_stock_client import USStockClient
except Exception:
    USStockClient = None

try:
    from bist_quant.clients.fx_commodities_client import FXCommoditiesClient
except Exception:
    FXCommoditiesClient = None

try:
    from bist_quant.clients.fund_analyzer import FundAnalyzer
except Exception:
    FundAnalyzer = None

try:
    from bist_quant.clients.borsapy_adapter import BorsapyAdapter
except Exception:
    BorsapyAdapter = None

try:
    from bist_quant.clients.macro_adapter import MacroAdapter
except Exception:
    MacroAdapter = None

try:
    from bist_quant.clients.fixed_income_provider import FixedIncomeProvider
except Exception:
    FixedIncomeProvider = None

try:
    from bist_quant.clients.derivatives_provider import DerivativesProvider
except Exception:
    DerivativesProvider = None

try:
    from bist_quant.clients.economic_calendar_provider import EconomicCalendarProvider
except Exception:
    EconomicCalendarProvider = None

try:
    from bist_quant.clients.fx_enhanced_provider import FXEnhancedProvider
except Exception:
    FXEnhancedProvider = None

# Utilities
from .utils import *
from .enums import *
from .market_cap_utils import *

# Benchmarking
try:
    from .benchmarking import (
        BenchmarkConfig,
        run_backtester_benchmark,
        run_benchmark_suite,
        run_pipeline_benchmark,
    )
except Exception:
    BenchmarkConfig = None
    run_backtester_benchmark = None
    run_benchmark_suite = None
    run_pipeline_benchmark = None

# Portfolio analytics
try:
    from .portfolio_analytics import PortfolioAnalytics, PortfolioAnalyticsAdapter, RiskMetrics
except Exception:
    PortfolioAnalytics = None
    PortfolioAnalyticsAdapter = None
    RiskMetrics = None

__all__ = [
    # Data
    "DataLoader",
    "DataPaths",
    "get_data_paths",
    "reset_data_paths",
    "get_prices_path",
    "get_fundamentals_path",
    "get_regime_labels_path",
    "validate_data_paths",
    "DataManager",
    "PanelCache",
    "check_data_staleness",
    # Configuration
    "ConfigManager",
    "load_config",
    "load_signal_configs",
    "list_available_strategies",
    # Backtesting
    "Backtester",
    "BacktestPreparationResult",
    "RebalanceDecision",
    "BacktestMetrics",
    "DataPreparationService",
    "RebalancingSelectionService",
    "DailyReturnService",
    "HoldingsHistoryAggregator",
    "TransactionCostModel",
    "BacktestMetricsService",
    "BacktestPayloadAssembler",
    "RiskManager",
    "ReportGenerator",
    "BenchmarkConfig",
    "run_backtester_benchmark",
    "run_pipeline_benchmark",
    "run_benchmark_suite",
    "PortfolioAnalytics",
    "PortfolioAnalyticsAdapter",
    "RiskMetrics",
    # Multi-asset
    "CryptoClient",
    "USStockClient",
    "FXCommoditiesClient",
    "FundAnalyzer",
    "BorsapyAdapter",
    "MacroAdapter",
    "FixedIncomeProvider",
    "DerivativesProvider",
    "EconomicCalendarProvider",
    "FXEnhancedProvider",
]
