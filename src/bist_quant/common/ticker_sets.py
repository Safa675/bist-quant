"""Canonical lists of bank and financial tickers.

Governs endpoint routing and configuration options where UFRS accounting
is required on İş Yatırım endpoints.
"""

# Banks
BANK_TICKERS = {
    "AKBNK",
    "ALBRK",
    "DENIZ",
    "GARAN",
    "HALKB",
    "ICBCT",
    "ISCTR",
    "KLNMA",
    "QNBFB",
    "QNBFK",
    "QNBTR",
    "SKBNK",
    "TSKB",
    "VAKBN",
    "YKBNK",
}

# Financial services / insurance
FINANCE_TICKERS = {
    "AGESA",
    "AKGRT",
    "ANHYT",
    "ANSGR",
    "AVHOL",
    "AVIVA",
    "GUSGR",
    "HDFGS",
    "ISFIN",
    "ISGSY",
    "ISYAT",
    "RAYSG",
    "SEKFK",
    "TURSG",
    "VAKFN",
    "VKFYO",
}

# Combined set for UFRS accounting endpoint routing
UFRS_TICKERS = BANK_TICKERS | FINANCE_TICKERS
