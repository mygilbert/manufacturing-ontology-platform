"""
FastAPI 라우터 모듈
"""

from .dashboard import router as dashboard_router
from .alerts import router as alerts_router
from .websocket import router as websocket_router

__all__ = ["dashboard_router", "alerts_router", "websocket_router"]
