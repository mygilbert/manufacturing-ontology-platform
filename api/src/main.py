"""
Manufacturing Ontology Platform - FastAPI 메인 서버
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from routers import ontology, analytics, realtime, agent

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    # 시작
    logger.info("Starting Manufacturing Ontology Platform API...")

    # 데이터베이스 연결 초기화
    try:
        from services.ontology_service import ontology_service
        from services.analytics_service import analytics_service

        await ontology_service.initialize()
        await analytics_service.initialize()
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")

    yield

    # 종료
    logger.info("Shutting down...")
    try:
        await ontology_service.close()
        await analytics_service.close()
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# FastAPI 앱 생성
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
Manufacturing Ontology Platform API

팔란티어 온톨로지 개념을 적용한 제조 데이터 실시간 분석 플랫폼

## 주요 기능

* **온톨로지 관리** - 설비, 공정, Lot, 웨이퍼 등 객체 및 관계 관리
* **실시간 분석** - FDC/SPC 데이터 실시간 분석 및 알림
* **이상 감지** - Isolation Forest, Autoencoder 기반 이상 탐지
* **예측 분석** - 설비 고장 예측, 품질 예측
* **GraphQL** - 유연한 온톨로지 쿼리
* **AI Agent** - Ollama LLM 기반 FDC 분석 Agent (근본원인 분석, 알람 점검)
""",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS 미들웨어
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 라우터 등록
app.include_router(ontology.router, prefix="/api/ontology", tags=["Ontology"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["Analytics"])
app.include_router(realtime.router, prefix="/api/realtime", tags=["Real-time"])
app.include_router(agent.router, prefix="/api/agent", tags=["AI Agent"])


# 헬스체크
@app.get("/health", tags=["Health"])
async def health_check():
    """서버 상태 확인"""
    return {
        "status": "healthy",
        "version": settings.app_version,
    }


@app.get("/", tags=["Root"])
async def root():
    """API 루트"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "graphql": "/graphql",
    }


# 에러 핸들러
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "message": str(exc)},
    )


# GraphQL 라우터 (Strawberry)
try:
    from graphql_app import graphql_app
    app.include_router(graphql_app, prefix="/graphql")
except ImportError:
    logger.warning("GraphQL not available")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )
