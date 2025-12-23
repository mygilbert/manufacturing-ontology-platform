"""
FastAPI 메인 애플리케이션
실시간 이상 감지 경보 시스템
"""

import os
import sys
import asyncio
import numpy as np
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# 모듈 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '..', 'src'))

from .config import AlertConfig, config
from .detector import EnsembleAnomalyDetector, create_feature_vector
from .alert_manager import AlertManager
from .data_simulator import RealTimeDataSimulator
from .websocket_manager import ws_manager
from .models import Measurement, DetectionResult, SystemStatus
from .routers import dashboard_router, alerts_router, websocket_router
from .routers.dashboard import set_app_state

# 전역 상태
app_state: Dict[str, Any] = {
    "config": config,
    "detector": None,
    "alert_manager": None,
    "simulator": None,
    "ws_manager": ws_manager,
    "is_running": False,
    "should_start": False,
    "start_time": None,
    "data_source": "",
    "monitoring_task": None
}


async def monitoring_loop():
    """실시간 모니터링 루프"""
    detector = app_state["detector"]
    alert_manager = app_state["alert_manager"]
    simulator = app_state["simulator"]

    print("\n" + "=" * 60)
    print("Starting Real-time Monitoring Loop")
    print("=" * 60)

    app_state["is_running"] = True
    app_state["start_time"] = datetime.now()

    try:
        async for window_data in simulator.stream():
            if not app_state["is_running"]:
                break

            # 특성 벡터 생성
            feature_vector = np.array(window_data['feature_vector']).reshape(1, -1)

            # 이상 감지
            detection = detector.predict(feature_vector)
            detection.window_index = window_data['window_index']
            detection.sensor_values = window_data['values']

            # 측정값 객체 생성
            measurement = Measurement(
                timestamp=window_data['timestamp'],
                window_index=window_data['window_index'],
                values=window_data['values'],
                std_values=window_data['std_values'],
                min_values=window_data['min_values'],
                max_values=window_data['max_values'],
                sample_count=window_data['sample_count']
            )

            # WebSocket으로 측정값 전송
            await ws_manager.send_measurement(measurement.model_dump())

            # WebSocket으로 감지 결과 전송
            await ws_manager.send_detection(detection.model_dump())

            # 경보 처리
            alert = alert_manager.process_detection(detection, window_data['values'])
            if alert:
                await ws_manager.send_alert(alert.model_dump())

            # 진행 상황 출력 (10개 윈도우마다)
            if window_data['window_index'] % 10 == 0:
                progress = window_data['progress']
                total_alerts = len(alert_manager.alert_history)
                print(f"[Monitor] Window {window_data['window_index']:,} | "
                      f"Progress: {progress:.1f}% | "
                      f"Score: {detection.ensemble_score:.3f} | "
                      f"Alerts: {total_alerts}")

    except Exception as e:
        print(f"[Monitor] Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        app_state["is_running"] = False
        print("\n" + "=" * 60)
        print("Monitoring Loop Ended")
        print(f"Total Windows: {detector.total_processed}")
        print(f"Total Alerts: {len(alert_manager.alert_history)}")
        print("=" * 60 + "\n")


async def initialize_system(data_path: str = None):
    """시스템 초기화"""
    if data_path is None:
        data_path = config.DATA_PATH

    print("\n" + "=" * 60)
    print("Initializing Real-time Alert System")
    print("=" * 60)

    # 시뮬레이터 초기화
    simulator = RealTimeDataSimulator(data_path, config, config.SIMULATION_SPEED)
    app_state["simulator"] = simulator
    app_state["data_source"] = data_path

    # 감지기 초기화
    detector = EnsembleAnomalyDetector(config)
    app_state["detector"] = detector

    # 경보 관리자 초기화
    alert_manager = AlertManager(config)
    app_state["alert_manager"] = alert_manager

    # 초기 학습
    print("\n[Init] Training anomaly detection models...")
    training_data = simulator.get_training_data()
    detector.fit(training_data)

    print("\n[Init] System initialized successfully!")
    print(f"  - Data source: {data_path}")
    print(f"  - Training samples: {len(training_data)}")
    print(f"  - Simulation speed: {config.SIMULATION_SPEED}x")
    print(f"  - Alert thresholds: WARNING={config.SCORE_THRESHOLD_WARNING}, "
          f"CRITICAL={config.SCORE_THRESHOLD_CRITICAL}, "
          f"EMERGENCY={config.SCORE_THRESHOLD_EMERGENCY}")
    print("=" * 60 + "\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 수명주기 관리"""
    # 시작 시
    print("\n[App] Starting FDC Real-time Alert System...")

    # 시스템 초기화
    await initialize_system()

    # 앱 상태 설정
    set_app_state(app_state)

    # 백그라운드 태스크 시작 (should_start 플래그 모니터링)
    async def check_start_flag():
        while True:
            if app_state.get("should_start") and not app_state.get("is_running"):
                app_state["should_start"] = False
                app_state["monitoring_task"] = asyncio.create_task(monitoring_loop())
            await asyncio.sleep(0.5)

    start_checker = asyncio.create_task(check_start_flag())

    yield

    # 종료 시
    print("\n[App] Shutting down...")
    app_state["is_running"] = False
    start_checker.cancel()

    if app_state.get("monitoring_task"):
        app_state["monitoring_task"].cancel()


# FastAPI 앱 생성
app = FastAPI(
    title="FDC Real-time Alert System",
    description="실시간 이상 감지 경보 시스템",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(dashboard_router)
app.include_router(alerts_router)
app.include_router(websocket_router)

# 정적 파일 서빙
static_dir = os.path.join(script_dir, "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root():
    """메인 페이지 (대시보드)"""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": "FDC Real-time Alert System",
        "version": "1.0.0",
        "endpoints": {
            "dashboard": "/",
            "api_status": "/api/status",
            "api_alerts": "/api/alerts/history",
            "websocket_dashboard": "/ws/dashboard",
            "websocket_alerts": "/ws/alerts"
        }
    }


@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=True
    )
