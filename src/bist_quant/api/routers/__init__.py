"""Router sub-package — each module exposes one APIRouter."""

from bist_quant.api.routers.factors import router as factors_router
from bist_quant.api.routers.screener import router as screener_router
from bist_quant.api.routers.analytics import router as analytics_router
from bist_quant.api.routers.optimization import router as optimization_router
from bist_quant.api.routers.professional import router as professional_router
from bist_quant.api.routers.compliance import router as compliance_router
from bist_quant.api.routers.signal_construction import router as signal_construction_router

__all__ = [
    "factors_router",
    "screener_router",
    "analytics_router",
    "optimization_router",
    "professional_router",
    "compliance_router",
    "signal_construction_router",
]
