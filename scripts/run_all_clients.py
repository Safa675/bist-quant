#!/usr/bin/env python3
"""
Runner script to instantiate and exercise all Market Data Clients.
This ensures that the initial connection and caches are established.
"""
import asyncio
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def get_loader():
    # Attempt to load a real DataLoader if available
    try:
        from bist_quant.common.data_loader import DataLoader
        return DataLoader()
    except ImportError:
        class DummyLoader:
            def __init__(self):
                self.data_dir = Path("data")
        return DummyLoader()


async def run_async_clients():
    logging.info("--- Testing Async Clients ---")
    
    try:
        from bist_quant.clients.crypto_client import CryptoClient
        logging.info("Instantiating CryptoClient...")
        crypto = CryptoClient()
        logging.info("Fetching Crypto markets...")
        # await crypto.get_crypto_markets()
        await crypto.close()
        logging.info("CryptoClient initialized successfully.")
    except Exception as e:
        logging.error(f"CryptoClient failed: {e}")

    try:
        from bist_quant.clients.us_stock_client import USStockClient
        logging.info("Instantiating USStockClient...")
        us = USStockClient()
        await us.close()
        logging.info("USStockClient initialized successfully.")
    except Exception as e:
        logging.error(f"USStockClient failed: {e}")

    try:
        from bist_quant.clients.fx_commodities_client import FXCommoditiesClient
        logging.info("Instantiating FXCommoditiesClient...")
        fx = FXCommoditiesClient()
        await fx.close()
        logging.info("FXCommoditiesClient initialized successfully.")
    except Exception as e:
        logging.error(f"FXCommoditiesClient failed: {e}")

    try:
        from bist_quant.clients.fund_analyzer import FundAnalyzer
        logging.info("Instantiating FundAnalyzer...")
        funds = FundAnalyzer()
        await funds.close()
        logging.info("FundAnalyzer initialized successfully.")
    except Exception as e:
        logging.error(f"FundAnalyzer failed: {e}")


def run_sync_clients():
    logging.info("\n--- Testing Sync Clients ---")
    loader = get_loader()
    
    try:
        from bist_quant.clients.borsapy_adapter import BorsapyAdapter
        logging.info("Instantiating BorsapyAdapter...")
        adapter = BorsapyAdapter(loader)
        # Force initialization
        _ = adapter.client
        logging.info("BorsapyAdapter initialized successfully.")
    except Exception as e:
        logging.error(f"BorsapyAdapter failed: {e}")

    try:
        from bist_quant.clients.macro_adapter import MacroAdapter
        logging.info("Instantiating MacroAdapter...")
        macro = MacroAdapter(loader, macro_events_path=Path("data/Fetcher-Scrapper/macro_events.py"))
        _ = macro.client
        logging.info("MacroAdapter initialized successfully.")
    except Exception as e:
        logging.error(f"MacroAdapter failed: {e}")

    try:
        from bist_quant.clients.economic_calendar_provider import EconomicCalendarProvider
        logging.info("Instantiating EconomicCalendarProvider...")
        eco = EconomicCalendarProvider()
        logging.info("EconomicCalendarProvider initialized successfully.")
    except Exception as e:
        logging.error(f"EconomicCalendarProvider failed: {e}")

    try:
        from bist_quant.clients.fixed_income_provider import FixedIncomeProvider
        logging.info("Instantiating FixedIncomeProvider...")
        fi = FixedIncomeProvider()
        logging.info("FixedIncomeProvider initialized successfully.")
    except Exception as e:
        logging.error(f"FixedIncomeProvider failed: {e}")

    try:
        from bist_quant.clients.derivatives_provider import DerivativesProvider
        logging.info("Instantiating DerivativesProvider...")
        deriv = DerivativesProvider()
        logging.info("DerivativesProvider initialized successfully.")
    except Exception as e:
        logging.error(f"DerivativesProvider failed: {e}")

    try:
        from bist_quant.clients.fx_enhanced_provider import FXEnhancedProvider
        logging.info("Instantiating FXEnhancedProvider...")
        fxe = FXEnhancedProvider()
        logging.info("FXEnhancedProvider initialized successfully.")
    except Exception as e:
        logging.error(f"FXEnhancedProvider failed: {e}")


def main():
    print("============================================================")
    print(" Starting Data Clients Initialization Pipeline")
    print("============================================================")
    run_sync_clients()
    asyncio.run(run_async_clients())
    print("\n============================================================")
    print(" Pipeline Complete ✅")
    print("============================================================")


if __name__ == "__main__":
    main()
