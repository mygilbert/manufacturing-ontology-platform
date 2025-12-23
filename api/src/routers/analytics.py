"""
분석 API 라우터

이상 감지, SPC 분석, 예측 분석 API
"""
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Query, Path, Body
from pydantic import BaseModel, Field

router = APIRouter()


# ============================================================
# Pydantic 모델
# ============================================================

class AnomalyDetectionRequest(BaseModel):
    """이상 감지 요청"""
    equipment_id: str
    feature_names: List[str]
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class AnomalyResponse(BaseModel):
    """이상 감지 결과"""
    anomaly_id: str
    equipment_id: str
    detected_at: datetime
    severity: str
    anomaly_score: float
    features: Dict[str, float]
    message: Optional[str] = None


class SPCAnalysisRequest(BaseModel):
    """SPC 분석 요청"""
    equipment_id: str
    item_id: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    chart_type: str = "individual"


class SPCAnalysisResponse(BaseModel):
    """SPC 분석 결과"""
    equipment_id: str
    item_id: str
    chart_type: str
    ucl: float
    cl: float
    lcl: float
    cpk: Optional[float] = None
    ppk: Optional[float] = None
    ooc_count: int = 0
    violations: List[str] = []


class CapabilityRequest(BaseModel):
    """공정 능력 분석 요청"""
    equipment_id: str
    item_id: str
    usl: float
    lsl: float
    target: Optional[float] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class CapabilityResponse(BaseModel):
    """공정 능력 분석 결과"""
    cp: float
    cpk: float
    pp: Optional[float] = None
    ppk: Optional[float] = None
    ppm_total: float
    level: str
    recommendations: List[str]


class PredictionRequest(BaseModel):
    """예측 요청"""
    equipment_id: str
    prediction_type: str = "failure"  # failure, quality
    horizon_hours: int = 24


class PredictionResponse(BaseModel):
    """예측 결과"""
    equipment_id: str
    prediction_type: str
    predicted_at: datetime
    probability: float
    risk_level: str
    details: Dict[str, Any] = {}


# ============================================================
# 이상 감지 API
# ============================================================

@router.get("/anomalies")
async def list_anomalies(
    equipment_id: Optional[str] = Query(None, description="설비 ID"),
    severity: Optional[str] = Query(None, description="심각도"),
    since: Optional[datetime] = Query(None, description="시작 시간"),
    limit: int = Query(100, ge=1, le=1000),
):
    """이상 감지 결과 목록 조회"""
    from services.analytics_service import analytics_service

    anomalies = await analytics_service.list_anomalies(
        equipment_id=equipment_id,
        severity=severity,
        since=since,
        limit=limit,
    )
    return anomalies


@router.post("/anomalies/detect")
async def detect_anomalies(request: AnomalyDetectionRequest):
    """실시간 이상 감지"""
    from services.analytics_service import analytics_service

    try:
        result = await analytics_service.detect_anomalies(
            equipment_id=request.equipment_id,
            feature_names=request.feature_names,
            start_time=request.start_time,
            end_time=request.end_time,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/anomalies/train/{equipment_id}")
async def train_anomaly_model(
    equipment_id: str = Path(..., description="설비 ID"),
    model_type: str = Query("isolation_forest", regex="^(isolation_forest|autoencoder)$"),
    feature_names: List[str] = Body(...),
    lookback_days: int = Query(30, ge=1, le=365),
):
    """이상 감지 모델 학습"""
    from services.analytics_service import analytics_service

    try:
        result = await analytics_service.train_anomaly_model(
            equipment_id=equipment_id,
            model_type=model_type,
            feature_names=feature_names,
            lookback_days=lookback_days,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/anomalies/models")
async def list_anomaly_models(
    equipment_id: Optional[str] = Query(None, description="설비 ID"),
):
    """학습된 이상 감지 모델 목록"""
    from services.analytics_service import analytics_service

    models = await analytics_service.list_anomaly_models(equipment_id)
    return models


# ============================================================
# SPC 분석 API
# ============================================================

@router.get("/spc/control-chart")
async def get_control_chart(
    equipment_id: str = Query(..., description="설비 ID"),
    item_id: str = Query(..., description="측정 항목 ID"),
    chart_type: str = Query("individual", description="관리도 타입"),
    since: Optional[datetime] = Query(None, description="시작 시간"),
    until: Optional[datetime] = Query(None, description="종료 시간"),
    limit: int = Query(100, ge=1, le=1000),
):
    """관리도 데이터 조회"""
    from services.analytics_service import analytics_service

    result = await analytics_service.get_control_chart(
        equipment_id=equipment_id,
        item_id=item_id,
        chart_type=chart_type,
        since=since,
        until=until,
        limit=limit,
    )
    return result


@router.post("/spc/analyze")
async def analyze_spc(request: SPCAnalysisRequest):
    """SPC 분석 수행"""
    from services.analytics_service import analytics_service

    try:
        result = await analytics_service.analyze_spc(
            equipment_id=request.equipment_id,
            item_id=request.item_id,
            chart_type=request.chart_type,
            start_time=request.start_time,
            end_time=request.end_time,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/spc/violations")
async def get_spc_violations(
    equipment_id: Optional[str] = Query(None, description="설비 ID"),
    item_id: Optional[str] = Query(None, description="측정 항목 ID"),
    rule: Optional[str] = Query(None, description="위반 규칙"),
    since: Optional[datetime] = Query(None, description="시작 시간"),
    limit: int = Query(100, ge=1, le=1000),
):
    """SPC 규칙 위반 목록"""
    from services.analytics_service import analytics_service

    violations = await analytics_service.get_spc_violations(
        equipment_id=equipment_id,
        item_id=item_id,
        rule=rule,
        since=since,
        limit=limit,
    )
    return violations


# ============================================================
# 공정 능력 API
# ============================================================

@router.post("/capability/analyze")
async def analyze_capability(request: CapabilityRequest):
    """공정 능력 분석"""
    from services.analytics_service import analytics_service

    try:
        result = await analytics_service.analyze_capability(
            equipment_id=request.equipment_id,
            item_id=request.item_id,
            usl=request.usl,
            lsl=request.lsl,
            target=request.target,
            start_time=request.start_time,
            end_time=request.end_time,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/capability/trend")
async def get_capability_trend(
    equipment_id: str = Query(..., description="설비 ID"),
    item_id: str = Query(..., description="측정 항목 ID"),
    usl: float = Query(..., description="규격 상한"),
    lsl: float = Query(..., description="규격 하한"),
    period: str = Query("daily", regex="^(hourly|daily|weekly)$"),
    days: int = Query(30, ge=1, le=365),
):
    """공정 능력 추세 분석"""
    from services.analytics_service import analytics_service

    result = await analytics_service.get_capability_trend(
        equipment_id=equipment_id,
        item_id=item_id,
        usl=usl,
        lsl=lsl,
        period=period,
        days=days,
    )
    return result


@router.get("/capability/comparison")
async def compare_equipment_capability(
    equipment_ids: str = Query(..., description="설비 ID 목록 (쉼표 구분)"),
    item_id: str = Query(..., description="측정 항목 ID"),
    usl: float = Query(..., description="규격 상한"),
    lsl: float = Query(..., description="규격 하한"),
    days: int = Query(7, ge=1, le=90),
):
    """설비간 공정 능력 비교"""
    from services.analytics_service import analytics_service

    equipment_list = equipment_ids.split(",")

    result = await analytics_service.compare_capability(
        equipment_ids=equipment_list,
        item_id=item_id,
        usl=usl,
        lsl=lsl,
        days=days,
    )
    return result


# ============================================================
# 예측 분석 API
# ============================================================

@router.get("/prediction/failure")
async def predict_equipment_failure(
    equipment_id: str = Query(..., description="설비 ID"),
    horizon_hours: int = Query(24, ge=1, le=168),
):
    """설비 고장 예측"""
    from services.analytics_service import analytics_service

    result = await analytics_service.predict_failure(
        equipment_id=equipment_id,
        horizon_hours=horizon_hours,
    )
    return result


@router.get("/prediction/quality")
async def predict_quality(
    process_id: str = Query(..., description="공정 ID"),
    lot_id: Optional[str] = Query(None, description="Lot ID"),
):
    """품질 예측"""
    from services.analytics_service import analytics_service

    result = await analytics_service.predict_quality(
        process_id=process_id,
        lot_id=lot_id,
    )
    return result


@router.post("/prediction/train/failure/{equipment_id}")
async def train_failure_model(
    equipment_id: str = Path(..., description="설비 ID"),
    feature_names: List[str] = Body(...),
    lookback_days: int = Query(90, ge=30, le=365),
):
    """고장 예측 모델 학습"""
    from services.analytics_service import analytics_service

    try:
        result = await analytics_service.train_failure_model(
            equipment_id=equipment_id,
            feature_names=feature_names,
            lookback_days=lookback_days,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/prediction/train/quality/{process_id}")
async def train_quality_model(
    process_id: str = Path(..., description="공정 ID"),
    target_name: str = Query("yield", description="예측 대상"),
    feature_names: Optional[List[str]] = Body(None),
    lookback_days: int = Query(30, ge=7, le=365),
):
    """품질 예측 모델 학습"""
    from services.analytics_service import analytics_service

    try:
        result = await analytics_service.train_quality_model(
            process_id=process_id,
            target_name=target_name,
            feature_names=feature_names,
            lookback_days=lookback_days,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================
# 대시보드 API
# ============================================================

@router.get("/dashboard/summary")
async def get_dashboard_summary():
    """대시보드 요약 정보"""
    from services.analytics_service import analytics_service

    summary = await analytics_service.get_dashboard_summary()
    return summary


@router.get("/dashboard/alerts")
async def get_recent_alerts(
    hours: int = Query(24, ge=1, le=168),
    limit: int = Query(50, ge=1, le=200),
):
    """최근 알람 목록"""
    from services.analytics_service import analytics_service

    alerts = await analytics_service.get_recent_alerts(hours=hours, limit=limit)
    return alerts


@router.get("/dashboard/equipment-status")
async def get_equipment_status_summary():
    """설비 상태 요약"""
    from services.analytics_service import analytics_service

    status = await analytics_service.get_equipment_status_summary()
    return status
