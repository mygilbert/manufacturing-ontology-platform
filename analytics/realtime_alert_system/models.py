"""
Pydantic 데이터 모델 - Alert, Measurement, SystemStatus 등
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class SeverityLevel(str, Enum):
    """경보 심각도 레벨"""
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class Alert(BaseModel):
    """경보 데이터 모델"""
    id: str = Field(default="", description="경보 고유 ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="경보 발생 시간")
    severity: SeverityLevel = Field(default=SeverityLevel.NORMAL, description="심각도")
    ensemble_score: float = Field(default=0.0, description="앙상블 이상 점수 (0~1)")
    algorithm_votes: int = Field(default=0, description="이상 판정 알고리즘 수")
    sensor_values: Dict[str, float] = Field(default_factory=dict, description="센서 값")
    individual_scores: Dict[str, float] = Field(default_factory=dict, description="알고리즘별 점수")
    message: str = Field(default="", description="경보 메시지")
    window_index: int = Field(default=0, description="집계 윈도우 인덱스")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Measurement(BaseModel):
    """센서 측정값 모델"""
    timestamp: datetime = Field(default_factory=datetime.now)
    window_index: int = Field(default=0, description="집계 윈도우 인덱스")
    values: Dict[str, float] = Field(default_factory=dict, description="센서별 평균값")
    std_values: Dict[str, float] = Field(default_factory=dict, description="센서별 표준편차")
    min_values: Dict[str, float] = Field(default_factory=dict, description="센서별 최소값")
    max_values: Dict[str, float] = Field(default_factory=dict, description="센서별 최대값")
    sample_count: int = Field(default=0, description="집계된 샘플 수")


class DetectionResult(BaseModel):
    """이상 감지 결과 모델"""
    timestamp: datetime = Field(default_factory=datetime.now)
    window_index: int = Field(default=0)
    ensemble_score: float = Field(default=0.0)
    ensemble_prediction: int = Field(default=0, description="0: 정상, 1: 이상")
    individual_scores: Dict[str, float] = Field(default_factory=dict)
    individual_predictions: Dict[str, int] = Field(default_factory=dict)
    severity: SeverityLevel = Field(default=SeverityLevel.NORMAL)
    sensor_values: Dict[str, float] = Field(default_factory=dict)


class SystemStatus(BaseModel):
    """시스템 상태 모델"""
    is_running: bool = Field(default=False, description="모니터링 실행 중 여부")
    is_model_fitted: bool = Field(default=False, description="모델 학습 완료 여부")
    total_processed: int = Field(default=0, description="처리된 총 윈도우 수")
    total_alerts: int = Field(default=0, description="발생한 총 경보 수")
    alerts_by_severity: Dict[str, int] = Field(
        default_factory=lambda: {"WARNING": 0, "CRITICAL": 0, "EMERGENCY": 0}
    )
    current_window: int = Field(default=0, description="현재 윈도우 인덱스")
    last_update: Optional[datetime] = Field(default=None, description="마지막 업데이트 시간")
    start_time: Optional[datetime] = Field(default=None, description="모니터링 시작 시간")
    data_source: str = Field(default="", description="데이터 소스")


class DashboardData(BaseModel):
    """대시보드 전송 데이터"""
    type: str = Field(default="update", description="메시지 타입")
    timestamp: datetime = Field(default_factory=datetime.now)
    measurement: Optional[Measurement] = None
    detection: Optional[DetectionResult] = None
    alert: Optional[Alert] = None
    status: Optional[SystemStatus] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ControlCommand(BaseModel):
    """제어 명령 모델"""
    command: str = Field(..., description="start, stop, reset 등")
    params: Dict[str, Any] = Field(default_factory=dict, description="추가 파라미터")


class AlertHistoryRequest(BaseModel):
    """경보 이력 조회 요청"""
    limit: int = Field(default=100, ge=1, le=1000)
    severity: Optional[SeverityLevel] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
