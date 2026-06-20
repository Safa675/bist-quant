"""Canonical Turkish financial-statement key aliases.

Across the signal modules the same fundamental concept (e.g. "total assets")
appears under many Turkish spellings, capitalisations and whitespace variants.
``_turkish_match`` already normalises case/whitespace before comparing, so
uppercase / lowercase / indented variants are equivalent at lookup time.
This module collects every variant that occurs in the codebase into a single
superset per concept, ordered by preference (first match wins in
``pick_row_from_sheet`` / ``pick_row``).

Importing the constants from here keeps each signal module tiny and makes it
trivial to add a new spelling variant without hunting through 14 files.
"""

# ---------------------------------------------------------------------------
# Balance sheet keys
# ---------------------------------------------------------------------------

TOTAL_ASSETS_KEYS = (
    "Toplam Varlıklar",
    "Toplam Aktifler",
    "TOPLAM VARLIKLAR",
    "Varlık Toplamı",
    "    Toplam Varlıklar",
)

TOTAL_EQUITY_KEYS = (
    "Özkaynaklar",
    "Toplam Özkaynaklar",
    "    Toplam Özkaynaklar",
    "TOPLAM ÖZKAYNAKLAR",
    "Ana Ortaklığa Ait Özkaynaklar",
)

TOTAL_LIABILITIES_KEYS = (
    "Toplam Yükümlülükler",
    "Toplam Borçlar",
)

CURRENT_ASSETS_KEYS = (
    "Dönen Varlıklar",
    "Toplam Dönen Varlıklar",
    "DÖNEN VARLIKLAR",
)

CURRENT_LIABILITIES_KEYS = (
    "Kısa Vadeli Yükümlülükler",
    "Toplam Kısa Vadeli Yükümlülükler",
    "KISA VADELİ YÜKÜMLÜLÜKLER",
)

# Short-term / current debt
CURRENT_DEBT_KEYS = (
    "Finansal Borçlar",
    "Kısa Vadeli Finansal Borçlar",
    "Kısa Vadeli Borçlanmalar",
)

# Long-term debt
LONG_TERM_DEBT_KEYS = (
    "Uzun Vadeli Borçlanmalar",
    "Uzun Vadeli Yükümlülükler",
    "Uzun Vadeli Finansal Borçlar",
    "Finansal Borçlar",
)

# All financial debt (short + long). Used where the broadest "total debt" proxy
# is needed (e.g. small-cap leverage, value EV).
TOTAL_DEBT_KEYS = (
    "Finansal Borçlar",
    "Kısa Vadeli Finansal Borçlar",
    "Uzun Vadeli Finansal Borçlar",
    "FİNANSAL BORÇLAR",
)

CASH_KEYS = (
    "Nakit ve Nakit Benzerleri",
    "Nakit ve Nakit Benzeri Varlıklar",
    "NAKİT VE NAKİT BENZERLERİ",
    "  Nakit ve Nakit Benzerleri",
)

INCOME_TAX_PAYABLE_KEYS = (
    "Dönem Karı Vergi Yükümlülüğü",
    "Cari Dönem Vergisi ile İlgili Yükümlülükler",
    "Kısa Vadeli Borç Karşılıkları",
)

SHARES_OUTSTANDING_KEYS = (
    "Ödenmiş Sermaye",
)

# ---------------------------------------------------------------------------
# Income statement keys
# ---------------------------------------------------------------------------

REVENUE_KEYS = (
    "Satış Gelirleri",
    "Toplam Hasılat",
    "Hasılat",
    "Net Satışlar",
    "SATIŞ GELİRLERİ",
)

NET_INCOME_KEYS = (
    "Dönem Karı (Zararı)",
    "Dönem Net Karı (Zararı)",
    "Dönem Net Karı veya Zararı",
    "Net Dönem Karı (Zararı)",
    "Dönem Net Kar/Zararı",
    "DÖNEM KARI (ZARARI)",
    "    Dönem Karı (Zararı)",
    "Sürdürülen Faaliyetler Dönem Karı (Zararı)",
    "SÜRDÜRÜLEN FAALİYETLER DÖNEM KARI/ZARARI",
    "Ana Ortaklık Payları",
    "ANA ORTAKLIK PAYLARI",
)

OPERATING_INCOME_KEYS = (
    "Faaliyet Karı (Zararı)",
    "Finansman Geliri (Gideri) Öncesi Faaliyet Karı (Zararı)",
    "FAALİYET KARI (ZARARI)",
)

GROSS_PROFIT_KEYS = (
    "Brüt Kar (Zarar)",
    "Ticari Faaliyetlerden Brüt Kar (Zarar)",
    "TİCARİ FAALİYETLERDEN BRÜT KAR (ZARAR)",
    "Brüt Kar",
)

DEPRECIATION_KEYS = (
    "Amortisman ve İtfa Giderleri",
    "Amortisman ve İtfa Gideri",
    "Amortisman Giderleri",
)

RD_KEYS = (
    "Araştırma ve Geliştirme Giderleri (-)",
    "Araştırma ve Geliştirme Giderleri",
)

# ---------------------------------------------------------------------------
# Cash-flow statement keys
# ---------------------------------------------------------------------------

OPERATING_CF_KEYS = (
    "İşletme Faaliyetlerinden Nakit Akışları",
    "İşletme Faaliyetlerinden Kaynaklanan Net Nakit",
    "İşletme Faaliyetlerinden Kaynaklanan Net Nakit",
    "Faaliyetlerden Elde Edilen Nakit Akışları",
    "İŞLETME FAALİYETLERİNDEN NAKİT AKIŞLARI",
    "İşletme Faaliyetlerinden Nakit Akışları",
    " İşletme Faaliyetlerinden Kaynaklanan Net Nakit",
    "Esas Faaliyetlerden Kaynaklanan Net Nakit",
)

# Alias kept for legacy import sites that spell it CFO_KEYS.
CFO_KEYS = OPERATING_CF_KEYS

CAPEX_KEYS = (
    "Maddi ve Maddi Olmayan Duran Varlıkların Alımından Kaynaklanan Nakit Çıkışları",
    "SABİT SERMAYE YATIRIMLARI",
    " Sabit Sermaye Yatırımları",
)

DIVIDENDS_PAID_KEYS = (
    "Ödenen Temettüler",
    "Ödenen Kar Payları",
    "Temettü Ödemeleri",
    "Kar Payı Ödemeleri",
)

# ---------------------------------------------------------------------------
# Derived / value-specific aliases
# ---------------------------------------------------------------------------

# Book value = total equity (book value of equity).
BOOK_VALUE_KEYS = (
    "Toplam Özkaynaklar",
    "    Toplam Özkaynaklar",
    "Ana Ortaklığa Ait Özkaynaklar",
    "Özkaynaklar",
)

EBITDA_KEYS = (
    "FAVÖK",
    "Faiz Amortisman ve Vergi Öncesi Kar",
)

__all__ = [
    # Balance sheet
    "TOTAL_ASSETS_KEYS",
    "TOTAL_EQUITY_KEYS",
    "TOTAL_LIABILITIES_KEYS",
    "CURRENT_ASSETS_KEYS",
    "CURRENT_LIABILITIES_KEYS",
    "CURRENT_DEBT_KEYS",
    "LONG_TERM_DEBT_KEYS",
    "TOTAL_DEBT_KEYS",
    "CASH_KEYS",
    "INCOME_TAX_PAYABLE_KEYS",
    "SHARES_OUTSTANDING_KEYS",
    # Income statement
    "REVENUE_KEYS",
    "NET_INCOME_KEYS",
    "OPERATING_INCOME_KEYS",
    "GROSS_PROFIT_KEYS",
    "DEPRECIATION_KEYS",
    "RD_KEYS",
    # Cash flow
    "OPERATING_CF_KEYS",
    "CFO_KEYS",
    "CAPEX_KEYS",
    "DIVIDENDS_PAID_KEYS",
    # Derived
    "BOOK_VALUE_KEYS",
    "EBITDA_KEYS",
]
