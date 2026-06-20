"""Sub-loaders extracted from :mod:`bist_quant.common.data_loader`.

Each sub-loader is a mixin-like class that carries a specific responsibility
(e.g. prices, fundamentals, shares, regime).  The top-level
:class:`~bist_quant.common.data_loader.DataLoader` facade delegates to these
internally while keeping the public API unchanged for all callers.
"""

from bist_quant.common.loaders.fundamentals_loader import FundamentalsLoader
from bist_quant.common.loaders.price_loader import PriceLoader
from bist_quant.common.loaders.regime_loader import RegimeLoader
from bist_quant.common.loaders.shares_loader import SharesLoader

__all__ = [
    "FundamentalsLoader",
    "PriceLoader",
    "RegimeLoader",
    "SharesLoader",
]
