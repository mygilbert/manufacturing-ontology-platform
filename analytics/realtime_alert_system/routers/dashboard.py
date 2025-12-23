"""
대시보드 API 라우터
시스템 상태 조회 및 제어
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, Optional
from datetime import datetime

from ..models import SystemStatus, ControlCommand

router = APIRouter(prefix="/api", tags=["dashboard"])

# 전역 상태 (main.py에서 주입)
_app_state: Dict[str, Any] = {}


def get_app_state() -> Dict[str, Any]:
    """앱 상태 의존성"""
    return _app_state


def set_app_state(state: Dict[str, Any]):
    """앱 상태 설정"""
    global _app_state
    _app_state = state


@router.get("/status", response_model=SystemStatus)
async def get_status(state: Dict = Depends(get_app_state)) -> SystemStatus:
    """
    현재 시스템 상태 조회

    Returns:
        SystemStatus: 시스템 상태 정보
    """
    detector = state.get("detector")
    alert_manager = state.get("alert_manager")
    simulator = state.get("simulator")

    is_running = state.get("is_running", False)

    return SystemStatus(
        is_running=is_running,
        is_model_fitted=detector.is_fitted if detector else False,
        total_processed=detector.total_processed if detector else 0,
        total_alerts=len(alert_manager.alert_history) if alert_manager else 0,
        alerts_by_severity=alert_manager.alerts_by_severity if alert_manager else {},
        current_window=detector.total_processed if detector else 0,
        last_update=detector.last_update if detector else None,
        start_time=state.get("start_time"),
        data_source=state.get("data_source", "")
    )


@router.get("/statistics")
async def get_statistics(state: Dict = Depends(get_app_state)) -> Dict:
    """
    상세 통계 조회

    Returns:
        통계 정보
    """
    detector = state.get("detector")
    alert_manager = state.get("alert_manager")
    simulator = state.get("simulator")
    ws_manager = state.get("ws_manager")

    return {
        "detector": {
            "is_fitted": detector.is_fitted if detector else False,
            "total_processed": detector.total_processed if detector else 0,
            "algorithms": detector.get_algorithm_names() if detector else []
        },
        "alerts": alert_manager.get_statistics() if alert_manager else {},
        "simulator": simulator.get_status() if simulator else {},
        "websocket": ws_manager.get_statistics() if ws_manager else {},
        "timestamp": datetime.now().isoformat()
    }


@router.post("/control/start")
async def start_monitoring(state: Dict = Depends(get_app_state)) -> Dict:
    """
    모니터링 시작

    Returns:
        시작 결과
    """
    if state.get("is_running"):
        raise HTTPException(status_code=400, detail="Monitoring is already running")

    # 모니터링 시작 플래그 설정
    state["should_start"] = True

    return {
        "status": "starting",
        "message": "Monitoring start requested",
        "timestamp": datetime.now().isoformat()
    }


@router.post("/control/stop")
async def stop_monitoring(state: Dict = Depends(get_app_state)) -> Dict:
    """
    모니터링 중지

    Returns:
        중지 결과
    """
    if not state.get("is_running"):
        raise HTTPException(status_code=400, detail="Monitoring is not running")

    simulator = state.get("simulator")
    if simulator:
        simulator.stop()

    state["is_running"] = False

    return {
        "status": "stopped",
        "message": "Monitoring stopped",
        "timestamp": datetime.now().isoformat()
    }


@router.post("/control/reset")
async def reset_system(state: Dict = Depends(get_app_state)) -> Dict:
    """
    시스템 리셋

    Returns:
        리셋 결과
    """
    if state.get("is_running"):
        raise HTTPException(status_code=400, detail="Stop monitoring first")

    detector = state.get("detector")
    alert_manager = state.get("alert_manager")
    simulator = state.get("simulator")

    if detector:
        detector.reset()
    if alert_manager:
        alert_manager.reset()
    if simulator:
        simulator.reset()

    state["start_time"] = None

    return {
        "status": "reset",
        "message": "System reset completed",
        "timestamp": datetime.now().isoformat()
    }


@router.get("/config")
async def get_config(state: Dict = Depends(get_app_state)) -> Dict:
    """
    현재 설정 조회

    Returns:
        설정 정보
    """
    config = state.get("config")
    if not config:
        return {}

    return {
        "thresholds": {
            "warning": config.SCORE_THRESHOLD_WARNING,
            "critical": config.SCORE_THRESHOLD_CRITICAL,
            "emergency": config.SCORE_THRESHOLD_EMERGENCY
        },
        "algorithms": {
            "zscore_threshold": config.ZSCORE_THRESHOLD,
            "cusum_threshold": config.CUSUM_THRESHOLD,
            "if_contamination": config.IF_CONTAMINATION,
            "lof_neighbors": config.LOF_N_NEIGHBORS
        },
        "aggregation_window": config.AGGREGATION_WINDOW_SEC,
        "simulation_speed": config.SIMULATION_SPEED,
        "sensor_columns": config.SENSOR_COLUMNS
    }
