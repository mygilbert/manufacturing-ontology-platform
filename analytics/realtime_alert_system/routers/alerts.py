"""
경보 API 라우터
경보 이력 조회 및 관리
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, List, Optional
from datetime import datetime

from ..models import Alert, SeverityLevel, AlertHistoryRequest
from .dashboard import get_app_state

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.get("/history", response_model=List[Alert])
async def get_alert_history(
    limit: int = Query(default=100, ge=1, le=1000, description="조회할 경보 수"),
    severity: Optional[SeverityLevel] = Query(default=None, description="심각도 필터"),
    state: Dict = Depends(get_app_state)
) -> List[Alert]:
    """
    경보 이력 조회

    Args:
        limit: 조회할 경보 수 (기본 100, 최대 1000)
        severity: 심각도 필터 (선택)

    Returns:
        경보 목록 (최신순)
    """
    alert_manager = state.get("alert_manager")
    if not alert_manager:
        return []

    return alert_manager.get_history(limit=limit, severity=severity)


@router.get("/latest", response_model=Optional[Alert])
async def get_latest_alert(
    state: Dict = Depends(get_app_state)
) -> Optional[Alert]:
    """
    가장 최근 경보 조회

    Returns:
        최신 경보 또는 None
    """
    alert_manager = state.get("alert_manager")
    if not alert_manager or not alert_manager.alert_history:
        return None

    return alert_manager.alert_history[-1]


@router.get("/count")
async def get_alert_count(
    state: Dict = Depends(get_app_state)
) -> Dict:
    """
    경보 수 통계

    Returns:
        심각도별 경보 수
    """
    alert_manager = state.get("alert_manager")
    if not alert_manager:
        return {
            "total": 0,
            "by_severity": {"WARNING": 0, "CRITICAL": 0, "EMERGENCY": 0}
        }

    return {
        "total": len(alert_manager.alert_history),
        "by_severity": alert_manager.alerts_by_severity.copy()
    }


@router.get("/{alert_id}", response_model=Alert)
async def get_alert_by_id(
    alert_id: str,
    state: Dict = Depends(get_app_state)
) -> Alert:
    """
    특정 경보 조회

    Args:
        alert_id: 경보 ID

    Returns:
        경보 정보
    """
    alert_manager = state.get("alert_manager")
    if not alert_manager:
        raise HTTPException(status_code=404, detail="Alert not found")

    for alert in alert_manager.alert_history:
        if alert.id == alert_id:
            return alert

    raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")


@router.delete("/clear")
async def clear_alerts(
    state: Dict = Depends(get_app_state)
) -> Dict:
    """
    경보 이력 삭제

    Returns:
        삭제 결과
    """
    alert_manager = state.get("alert_manager")
    if not alert_manager:
        return {"status": "no_manager", "cleared": 0}

    count = len(alert_manager.alert_history)
    alert_manager.reset()

    return {
        "status": "cleared",
        "cleared": count,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/summary")
async def get_alert_summary(
    hours: int = Query(default=1, ge=1, le=24, description="조회할 시간 범위"),
    state: Dict = Depends(get_app_state)
) -> Dict:
    """
    경보 요약 (시간별)

    Args:
        hours: 조회할 시간 범위

    Returns:
        시간별 경보 요약
    """
    alert_manager = state.get("alert_manager")
    if not alert_manager:
        return {"summary": [], "total": 0}

    # 최근 N시간 경보 필터링
    from datetime import timedelta
    cutoff = datetime.now() - timedelta(hours=hours)

    recent_alerts = [
        a for a in alert_manager.alert_history
        if a.timestamp >= cutoff
    ]

    # 심각도별 집계
    by_severity = {"WARNING": 0, "CRITICAL": 0, "EMERGENCY": 0}
    for alert in recent_alerts:
        if alert.severity.value in by_severity:
            by_severity[alert.severity.value] += 1

    # 시간대별 집계
    hourly = {}
    for alert in recent_alerts:
        hour_key = alert.timestamp.strftime("%Y-%m-%d %H:00")
        if hour_key not in hourly:
            hourly[hour_key] = 0
        hourly[hour_key] += 1

    return {
        "time_range_hours": hours,
        "total": len(recent_alerts),
        "by_severity": by_severity,
        "hourly_distribution": hourly,
        "timestamp": datetime.now().isoformat()
    }
