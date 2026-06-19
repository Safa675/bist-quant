"""Router sub-package — each module exposes one APIRouter."""

from server.api.routers.factors import router as factors_router
from server.api.routers.screener import router as screener_router
from server.api.routers.analytics import router as analytics_router
from server.api.routers.optimization import router as optimization_router
from server.api.routers.professional import router as professional_router
from server.api.routers.compliance import router as compliance_router
from server.api.routers.signal_construction import router as signal_construction_router

__all__ = [
    "factors_router",
    "screener_router",
    "analytics_router",
    "optimization_router",
    "professional_router",
    "compliance_router",
    "signal_construction_router",
]
